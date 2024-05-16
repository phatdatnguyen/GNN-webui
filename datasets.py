import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, rdmolops
import deepchem as dc
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Dataset, Data

class MoleculeDatasetForRegression(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, gcn_featurizer_name, mol_featurizer_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)        
        self.target_column = target_column
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.output_scaler = MinMaxScaler(feature_range=(0, 1))
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForRegression, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def process(self):
        self.data = pd.read_csv('.\\Datasets\\' + self.dataset_name + '\\raw\\' + self.data_file_name)
        self.output_scaler.fit(self.data[self.target_column].to_numpy().reshape(-1, 1))

        if len(os.listdir(self.processed_dir)) > 2: # If these files are found in 'processed', processing is skipped.
            mol_features_list = np.loadtxt('Datasets\\' + self.dataset_name + '\\raw\\mol_features.csv', delimiter=',')
            self.mol_features_scaler.fit(np.array(mol_features_list))

            return
        
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            gcn_featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            gcn_featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            gcn_featurizer = dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        if (self.mol_featurizer_name == 'Mordred descriptors'):
            mol_featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
        elif (self.mol_featurizer_name == 'RDKit descriptors'):
            mol_featurizer = dc.feat.RDKitDescriptors()
        elif (self.mol_featurizer_name == 'MACCS keys'):
            mol_featurizer = dc.feat.MACCSKeysFingerprint()
        elif (self.mol_featurizer_name == 'Morgan fingerprint'):
            mol_featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        else:
            mol_featurizer = None
        
        mol_features_list = []
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                mol_obj = Chem.MolFromSmiles(mol['smiles'])
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                if mol_featurizer is not None:
                    mol_features = mol_featurizer.featurize(mol['smiles'])
                else:
                    if self.mol_featurizer_name == 'Avalon fingerprint':
                        avalon_fp = pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)
                        mol_features = np.array(avalon_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Atom-pairs fingerprint':
                        atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(atom_pairs_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Topological-torsion fingerprint':
                        topological_torsion_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(topological_torsion_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Layered fingerprint':
                        layered_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(layered_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Pattern fingerprint':
                        pattern_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(pattern_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'RDKit fingerprint':
                        rdk_fp = rdmolops.RDKFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(rdk_fp).reshape(1, -1)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features = torch.tensor(mol_features)
                self.n_mol_features = mol_features.shape[1]
                output = self.get_output(mol[self.target_column])

                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=output,
                            smiles=mol['smiles'],
                            mol_features=mol_features
                            )
                
                torch.save(data, os.path.join(self.processed_dir, 
                                 f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue
        
        mol_features_arr = np.array(mol_features_list)
        np.savetxt('Datasets\\' + self.dataset_name + '\\raw\\mol_features.csv', mol_features_arr, delimiter=',')
        self.mol_features_scaler.fit(mol_features_arr)

    def get_output(self, output):
        output = self.output_scaler.transform(np.array([output]).reshape(-1, 1))
        return torch.tensor(output)

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data
    
    @property
    def num_node_features(self):
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            return 33
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            return 94
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            return 133
        
    @property
    def num_edge_features(self):
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            return 11
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            return 42
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            return 14
    
    @property
    def num_mol_features(self):
        if (self.mol_featurizer_name == 'Mordred descriptors'):
            return 1613
        elif (self.mol_featurizer_name == 'RDKit descriptors'):
            return 210
        elif (self.mol_featurizer_name == 'MACCS keys'):
            return 167
        elif (self.mol_featurizer_name == 'Morgan fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Avalon fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Atom-pairs fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Topological-torsion fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Layered fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Pattern fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'RDKit fingerprint'):
            return 2048

class MoleculeDatasetForRegression3D(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)        
        self.target_column = target_column
        self.output_scaler = MinMaxScaler(feature_range=(0, 1))
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForRegression3D, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def ProcessData3D(self, smiles):
        # Get the molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Prepare the molecule
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        
        # Get z and pos
        z = []
        pos_x = []
        pos_y = []
        pos_z = []
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            z.append(atom.GetAtomicNum())
            atom_pos = mol.GetConformer().GetAtomPosition(atom_idx)
            pos_x.append(atom_pos[0])
            pos_y.append(atom_pos[1])
            pos_z.append(atom_pos[2])
            
        z = np.array(z)
        pos_x = np.array(pos_x)
        pos_y = np.array(pos_y)
        pos_z = np.array(pos_z)
        return z, pos_x, pos_y, pos_z

    def process(self):
        self.data = pd.read_csv('.\\Datasets\\' + self.dataset_name + '\\raw\\' + self.data_file_name)
        self.output_scaler.fit(self.data[self.target_column].to_numpy().reshape(-1, 1))
                
        if len(os.listdir(self.processed_dir)) > 2:
            return
        
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                smiles = mol["smiles"]
                z, pos_x, pos_y, pos_z = self.ProcessData3D(smiles)
                z = torch.tensor(z)
                pos = torch.tensor(np.array([pos_x, pos_y, pos_z]))
                output = self.get_output(mol[self.target_column])

                data = Data(z=z, 
                            pos=pos.T,
                            smiles=mol["smiles"],
                            y = output
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue

    def get_output(self, output):
        output = self.output_scaler.transform(np.array([output]).reshape(-1, 1))
        return torch.tensor(output)

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data

class MoleculeDatasetForRegressionPrediction(Dataset):
    def __init__(self, data_file_path, dataset_name, train_dataset_name, gcn_featurizer_name, mol_featurizer_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.train_dataset_name = train_dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForRegressionPrediction, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
        
    def download(self):
        pass
 
    def process(self):
        if len(os.listdir(self.processed_dir)) > 2: # If these files are found in 'processed', processing is skipped.
            mol_features_list = np.loadtxt('Datasets\\' + self.train_dataset_name + '\\raw\\mol_features.csv', delimiter=',')
            self.mol_features_scaler.fit(np.array(mol_features_list))

            return
        
        self.data = pd.read_csv('.\\Datasets\\' + self.dataset_name + '\\raw\\' + self.data_file_name)
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            gcn_featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            gcn_featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            gcn_featurizer = dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        if (self.mol_featurizer_name == 'Mordred descriptors'):
            mol_featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
        elif (self.mol_featurizer_name == 'RDKit descriptors'):
            mol_featurizer = dc.feat.RDKitDescriptors()
        elif (self.mol_featurizer_name == 'MACCS keys'):
            mol_featurizer = dc.feat.MACCSKeysFingerprint()
        elif (self.mol_featurizer_name == 'Morgan fingerprint'):
            mol_featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        else:
            mol_featurizer = None
        
        mol_features_list = []
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                mol_obj = Chem.MolFromSmiles(mol['smiles'])
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                if mol_featurizer is not None:
                    mol_features = mol_featurizer.featurize(mol['smiles'])
                else:
                    if self.mol_featurizer_name == 'Avalon fingerprint':
                        avalon_fp = pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)
                        mol_features = np.array(avalon_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Atom-pairs fingerprint':
                        atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(atom_pairs_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Topological-torsion fingerprint':
                        topological_torsion_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(topological_torsion_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Layered fingerprint':
                        layered_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(layered_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Pattern fingerprint':
                        pattern_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(pattern_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'RDKit fingerprint':
                        rdk_fp = rdmolops.RDKFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(rdk_fp).reshape(1, -1)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features = torch.tensor(mol_features)
                self.n_mol_features = mol_features.shape[1]

                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            smiles=mol['smiles'],
                            mol_features=mol_features
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue
        
        mol_features_arr = np.array(mol_features_list)
        np.savetxt('Datasets\\' + self.train_dataset_name + '\\raw\\mol_features.csv', mol_features_arr, delimiter=',')
        self.mol_features_scaler.fit(mol_features_arr)

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data

class MoleculeDatasetForRegressionPrediction3D(Dataset):
    def __init__(self, data_file_path, dataset_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForRegressionPrediction3D, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def ProcessData3D(self, smiles):
        # Get the molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Prepare the molecule
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        
        # Get z and pos
        z = []
        pos_x = []
        pos_y = []
        pos_z = []
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            z.append(atom.GetAtomicNum())
            atom_pos = mol.GetConformer().GetAtomPosition(atom_idx)
            pos_x.append(atom_pos[0])
            pos_y.append(atom_pos[1])
            pos_z.append(atom_pos[2])
            
        z = np.array(z)
        pos_x = np.array(pos_x)
        pos_y = np.array(pos_y)
        pos_z = np.array(pos_z)
        return z, pos_x, pos_y, pos_z

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2:
            return
        
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                smiles = mol["smiles"]
                z, pos_x, pos_y, pos_z = self.ProcessData3D(smiles)
                
                z = torch.tensor(z)
                pos = torch.tensor(np.array([pos_x, pos_y, pos_z]))

                data = Data(z=z, 
                            pos=pos.T,
                            smiles=mol["smiles"]
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data

class MoleculeDatasetForBinaryClassification(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, gcn_featurizer_name, mol_featurizer_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)        
        self.target_column = target_column
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForBinaryClassification, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def process(self):
        if len(os.listdir(self.processed_dir)) > 2: # If these files are found in 'processed', processing is skipped.
            mol_features_list = np.loadtxt('Datasets\\' + self.dataset_name + '\\raw\\mol_features.csv', delimiter=',')
            self.mol_features_scaler.fit(np.array(mol_features_list))

            return
        
        self.data = pd.read_csv('.\\Datasets\\' + self.dataset_name + '\\raw\\' + self.data_file_name)
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            gcn_featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            gcn_featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            gcn_featurizer = dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        if (self.mol_featurizer_name == 'Mordred descriptors'):
            mol_featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
        elif (self.mol_featurizer_name == 'RDKit descriptors'):
            mol_featurizer = dc.feat.RDKitDescriptors()
        elif (self.mol_featurizer_name == 'MACCS keys'):
            mol_featurizer = dc.feat.MACCSKeysFingerprint()
        elif (self.mol_featurizer_name == 'Morgan fingerprint'):
            mol_featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        else:
            mol_featurizer = None
        
        mol_features_list = []
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                mol_obj = Chem.MolFromSmiles(mol['smiles'])
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                if mol_featurizer is not None:
                    mol_features = mol_featurizer.featurize(mol['smiles'])
                else:
                    if self.mol_featurizer_name == 'Avalon fingerprint':
                        avalon_fp = pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)
                        mol_features = np.array(avalon_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Atom-pairs fingerprint':
                        atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(atom_pairs_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Topological-torsion fingerprint':
                        topological_torsion_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(topological_torsion_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Layered fingerprint':
                        layered_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(layered_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Pattern fingerprint':
                        pattern_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(pattern_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'RDKit fingerprint':
                        rdk_fp = rdmolops.RDKFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(rdk_fp).reshape(1, -1)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features = torch.tensor(mol_features)
                self.n_mol_features = mol_features.shape[1]
                output = self.get_output(mol[self.target_column])

                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=output,
                            smiles=mol['smiles'],
                            mol_features=mol_features
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue
        
        mol_features_arr = np.array(mol_features_list)
        np.savetxt('Datasets\\' + self.dataset_name + '\\raw\\mol_features.csv', mol_features_arr, delimiter=',')
        self.mol_features_scaler.fit(mol_features_arr)

    def get_output(self, output):
        output = np.array([output])
        return torch.tensor(output)

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data
    
    @property
    def num_node_features(self):
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            return 33
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            return 94
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            return 133
        
    @property
    def num_edge_features(self):
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            return 11
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            return 42
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            return 14
    
    @property
    def num_mol_features(self):
        if (self.mol_featurizer_name == 'Mordred descriptors'):
            return 1613
        elif (self.mol_featurizer_name == 'RDKit descriptors'):
            return 210
        elif (self.mol_featurizer_name == 'MACCS keys'):
            return 167
        elif (self.mol_featurizer_name == 'Morgan fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Avalon fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Atom-pairs fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Topological-torsion fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Layered fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'Pattern fingerprint'):
            return 2048
        elif (self.mol_featurizer_name == 'RDKit fingerprint'):
            return 2048

class MoleculeDatasetForBinaryClassification3D(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)        
        self.target_column = target_column
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForBinaryClassification3D, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def ProcessData3D(self, smiles):
        # Get the molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Prepare the molecule
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        
        # Get z and pos
        z = []
        pos_x = []
        pos_y = []
        pos_z = []
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            z.append(atom.GetAtomicNum())
            atom_pos = mol.GetConformer().GetAtomPosition(atom_idx)
            pos_x.append(atom_pos[0])
            pos_y.append(atom_pos[1])
            pos_z.append(atom_pos[2])
            
        z = np.array(z)
        pos_x = np.array(pos_x)
        pos_y = np.array(pos_y)
        pos_z = np.array(pos_z)
        return z, pos_x, pos_y, pos_z

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2: # If these files are found in 'processed', processing is skipped.
            return
        
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                smiles = mol["smiles"]
                z, pos_x, pos_y, pos_z = self.ProcessData3D(smiles)
                z = torch.tensor(z)
                pos = torch.tensor(np.array([pos_x, pos_y, pos_z]))
                output = self.get_output(mol[self.target_column])

                data = Data(z=z, 
                            pos=pos.T,
                            smiles=mol["smiles"],
                            y = output
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue

    def get_output(self, output):
        output = np.array([output])
        return torch.tensor(output)

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data

class MoleculeDatasetForBinaryClassificationPrediction(Dataset):
    def __init__(self, data_file_path, dataset_name, train_dataset_name, gcn_featurizer_name, mol_featurizer_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.train_dataset_name = train_dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForBinaryClassificationPrediction, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def process(self):
        if len(os.listdir(self.processed_dir)) > 2: # If these files are found in 'processed', processing is skipped.
            mol_features_list = np.loadtxt('Datasets\\' + self.train_dataset_name + '\\raw\\mol_features.csv', delimiter=',')
            self.mol_features_scaler.fit(np.array(mol_features_list))

            return
        
        self.data = pd.read_csv('.\\Datasets\\' + self.dataset_name + '\\raw\\' + self.data_file_name)
        if (self.gcn_featurizer_name == 'MolGraphConvFeaturizer'):
            gcn_featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif (self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer'):
            gcn_featurizer = dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif (self.gcn_featurizer_name == 'DMPNNFeaturizer'):
            gcn_featurizer = dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        if (self.mol_featurizer_name == 'Mordred descriptors'):
            mol_featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
        elif (self.mol_featurizer_name == 'RDKit descriptors'):
            mol_featurizer = dc.feat.RDKitDescriptors()
        elif (self.mol_featurizer_name == 'MACCS keys'):
            mol_featurizer = dc.feat.MACCSKeysFingerprint()
        elif (self.mol_featurizer_name == 'Morgan fingerprint'):
            mol_featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
        else:
            mol_featurizer = None

        mol_features_list = []
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                mol_obj = Chem.MolFromSmiles(mol['smiles'])
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                if mol_featurizer is not None:
                    mol_features = mol_featurizer.featurize(mol['smiles'])
                else:
                    if self.mol_featurizer_name == 'Avalon fingerprint':
                        avalon_fp = pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)
                        mol_features = np.array(avalon_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Atom-pairs fingerprint':
                        atom_pairs_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(atom_pairs_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Topological-torsion fingerprint':
                        topological_torsion_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol_obj, nBits=2048)
                        mol_features = np.array(topological_torsion_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Layered fingerprint':
                        layered_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(layered_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'Pattern fingerprint':
                        pattern_fp = rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(pattern_fp).reshape(1, -1)
                    elif self.mol_featurizer_name == 'RDKit fingerprint':
                        rdk_fp = rdmolops.RDKFingerprint(mol_obj, fpSize=2048)
                        mol_features = np.array(rdk_fp).reshape(1, -1)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features = torch.tensor(mol_features)
                self.n_mol_features = mol_features.shape[1]
                output = self.get_output(mol[self.target_column])

                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=output,
                            smiles=mol['smiles'],
                            mol_features=mol_features
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue
        
        mol_features_arr = np.array(mol_features_list)
        np.savetxt('Datasets\\' + self.train_dataset_name + '\\raw\\mol_features.csv', mol_features_arr, delimiter=',')
        self.mol_features_scaler.fit(mol_features_arr)

    def get_output(self, output):
        output = np.array([output])
        return torch.tensor(output)

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data

class MoleculeDatasetForBinaryClassificationPrediction3D(Dataset):
    def __init__(self, data_file_path, dataset_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        os.makedirs('.\\Datasets', exist_ok=True)
        root = '.\\Datasets\\' + dataset_name # Where the dataset should be stored. This folder is split into 'raw' and 'processed'.
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + '\\raw', exist_ok=True)
        shutil.copy2(data_file_path, root + '\\raw')
        super(MoleculeDatasetForBinaryClassificationPrediction3D, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self): # If this file exists in 'raw', the download is not triggered.
        return self.data_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index() # If these files are found in 'processed', processing is skipped.
        return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def ProcessData3D(self, smiles):
        # Get the molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Prepare the molecule
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        
        # Get z and pos
        z = []
        pos_x = []
        pos_y = []
        pos_z = []
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            z.append(atom.GetAtomicNum())
            atom_pos = mol.GetConformer().GetAtomPosition(atom_idx)
            pos_x.append(atom_pos[0])
            pos_y.append(atom_pos[1])
            pos_z.append(atom_pos[2])
            
        z = np.array(z)
        pos_x = np.array(pos_x)
        pos_y = np.array(pos_y)
        pos_z = np.array(pos_z)
        return z, pos_x, pos_y, pos_z

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2: # If these files are found in 'processed', processing is skipped.
            return
        
        graph_index = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            try:
                smiles = mol["smiles"]
                z, pos_x, pos_y, pos_z = self.ProcessData3D(smiles)
                z = torch.tensor(z)
                pos = torch.tensor(np.array([pos_x, pos_y, pos_z]))
                output = self.get_output(mol[self.target_column])

                data = Data(z=z, 
                            pos=pos.T,
                            smiles=mol["smiles"]
                            )
                torch.save(data, os.path.join(self.processed_dir, 
                                    f'{self.dataset_name}_{graph_index}.pt'))
                
                graph_index += 1
            except:
                continue

    def len(self):
        _, _, files = next(os.walk('.\\Datasets\\' + self.dataset_name + '\\processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'))   
        return data
