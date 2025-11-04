import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, SaltRemover, rdFingerprintGenerator, rdmolops
import deepchem as dc
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Dataset, Data

class MoleculeDatasetForRegressionHybrid(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, gcn_featurizer_name, mol_featurizer_name, dimensionality_reduction=False, variance_threshold=0.01, pca_num_components=32, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.target_column = target_column
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.dimensionality_reduction = dimensionality_reduction
        self.variance_threshold = VarianceThreshold(threshold=variance_threshold) if dimensionality_reduction else None
        self.pca = PCA(n_components=pca_num_components) if dimensionality_reduction else None
        self.pca_num_components = pca_num_components if dimensionality_reduction else None
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        self.output_scaler = MinMaxScaler(feature_range=(0, 1))
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        self.output_scaler.fit(self.data[self.target_column].to_numpy().reshape(-1, 1))

        if len(os.listdir(self.processed_dir)) > 2:
            mol_features_arr = np.loadtxt(os.path.join('./Datasets', self.dataset_name, 'raw', 'mol_features.csv'), delimiter=',')
            if self.dimensionality_reduction:
                mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
                mol_features_arr = self.pca.fit_transform(mol_features_arr)
            self.mol_features_scaler.fit(mol_features_arr)
            return

        gcn_featurizer = self._get_gcn_featurizer()
        mol_featurizer = self._get_mol_featurizer()
        mol_features_list = []
        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                mol_obj = Chem.MolFromSmiles(row['SMILES'])
                salt_remover = SaltRemover.SaltRemover()
                mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                mol_obj = Chem.AddHs(mol_obj)
                carbon_count = sum(1 for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() == 6)
                if carbon_count < 2:
                    raise Exception()
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                mol_features = self._featurize_molecule(mol_obj, row['SMILES'], mol_featurizer)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features_tensor = torch.tensor(mol_features)
                self.n_mol_features = mol_features_tensor.shape[1]
                output = self.get_output(row[self.target_column])

                data = Data(x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=output,
                            smiles=row['SMILES'],
                            mol_features=mol_features_tensor)
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue
        mol_features_arr = np.array(mol_features_list)
        np.savetxt(os.path.join('./Datasets', self.dataset_name, 'raw', 'mol_features.csv'), mol_features_arr, delimiter=',')
        if self.dimensionality_reduction:
            mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
            mol_features_arr = self.pca.fit_transform(mol_features_arr)
        self.mol_features_scaler.fit(mol_features_arr)

    def _get_gcn_featurizer(self):
        if self.gcn_featurizer_name == 'MolGraphConvFeaturizer':
            return dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer':
            return dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif self.gcn_featurizer_name == 'DMPNNFeaturizer':
            return dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        else:
            raise ValueError(f"Unknown GCN featurizer: {self.gcn_featurizer_name}")

    def _get_mol_featurizer(self):
        if self.mol_featurizer_name == 'Mordred descriptors':
            return dc.feat.MordredDescriptors(ignore_3D=True)
        elif self.mol_featurizer_name == 'RDKit descriptors':
            return dc.feat.RDKitDescriptors()
        elif self.mol_featurizer_name == 'MACCS keys':
            return dc.feat.MACCSKeysFingerprint()
        elif self.mol_featurizer_name == 'Morgan fingerprint':
            return dc.feat.CircularFingerprint(size=2048, radius=3)
        else:
            return None

    def _featurize_molecule(self, mol_obj, smiles, mol_featurizer):
        if mol_featurizer is not None:
            return mol_featurizer.featurize(smiles)
        name = self.mol_featurizer_name
        if name == 'Avalon fingerprint':
            return np.array(pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)).reshape(1, -1)
        elif name == 'Atom-pairs fingerprint':
            return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Topological-torsion fingerprint':
            return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Layered fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'Pattern fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'RDKit fingerprint':
            return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        else:
            raise ValueError(f"Unknown molecule featurizer: {name}")

    def get_output(self, output):
        return torch.tensor(self.output_scaler.transform(np.array([output]).reshape(-1, 1)))

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)

    @property
    def num_node_features(self):
        return {
            'MolGraphConvFeaturizer': 33,
            'PagtnMolGraphFeaturizer': 94,
            'DMPNNFeaturizer': 133
        }.get(self.gcn_featurizer_name, None)

    @property
    def num_edge_features(self):
        return {
            'MolGraphConvFeaturizer': 11,
            'PagtnMolGraphFeaturizer': 42,
            'DMPNNFeaturizer': 14
        }.get(self.gcn_featurizer_name, None)

    @property
    def num_mol_features(self):
        if self.dimensionality_reduction:
            return self.pca_num_components
        return {
            'Mordred descriptors': 1613,
            'RDKit descriptors': 210,
            'MACCS keys': 167,
            'Morgan fingerprint': 2048,
            'Avalon fingerprint': 2048,
            'Atom-pairs fingerprint': 2048,
            'Topological-torsion fingerprint': 2048,
            'Layered fingerprint': 2048,
            'Pattern fingerprint': 2048,
            'RDKit fingerprint': 2048
        }.get(self.mol_featurizer_name, None)

class MoleculeDatasetForRegressionPredictionHybrid(Dataset):
    def __init__(self, data_file_path, dataset_name, train_dataset_name, gcn_featurizer_name, mol_featurizer_name, dimensionality_reduction=False, variance_threshold=0.01, pca_num_components=32, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.train_dataset_name = train_dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.dimensionality_reduction = dimensionality_reduction
        self.variance_threshold = VarianceThreshold(threshold=variance_threshold) if dimensionality_reduction else None
        self.pca = PCA(n_components=pca_num_components) if dimensionality_reduction else None
        self.pca_num_components = pca_num_components if dimensionality_reduction else None
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def download(self):
        pass

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2:
            mol_features_arr = np.loadtxt(os.path.join('./Datasets', self.train_dataset_name, 'raw', 'mol_features.csv'), delimiter=',')
            if self.dimensionality_reduction:
                mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
                mol_features_arr = self.pca.fit_transform(mol_features_arr)
            self.mol_features_scaler.fit(mol_features_arr)
            return

        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        gcn_featurizer = self._get_gcn_featurizer()
        mol_featurizer = self._get_mol_featurizer()
        mol_features_list = []
        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                mol_obj = Chem.MolFromSmiles(row['SMILES'])
                salt_remover = SaltRemover.SaltRemover()
                mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                mol_obj = Chem.AddHs(mol_obj)
                carbon_count = sum(1 for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() == 6)
                if carbon_count < 2:
                    raise Exception()
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                mol_features = self._featurize_molecule(mol_obj, row['SMILES'], mol_featurizer)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features_tensor = torch.tensor(mol_features)
                self.n_mol_features = mol_features_tensor.shape[1]

                data = Data(x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            smiles=row['SMILES'],
                            mol_features=mol_features_tensor)
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue
        mol_features_arr = np.array(mol_features_list)
        np.savetxt(os.path.join('./Datasets', self.train_dataset_name, 'raw', 'mol_features.csv'), mol_features_arr, delimiter=',')
        if self.dimensionality_reduction:
            mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
            mol_features_arr = self.pca.fit_transform(mol_features_arr)
        self.mol_features_scaler.fit(mol_features_arr)

    def _get_gcn_featurizer(self):
        if self.gcn_featurizer_name == 'MolGraphConvFeaturizer':
            return dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer':
            return dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif self.gcn_featurizer_name == 'DMPNNFeaturizer':
            return dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        else:
            raise ValueError(f"Unknown GCN featurizer: {self.gcn_featurizer_name}")

    def _get_mol_featurizer(self):
        if self.mol_featurizer_name == 'Mordred descriptors':
            return dc.feat.MordredDescriptors(ignore_3D=True)
        elif self.mol_featurizer_name == 'RDKit descriptors':
            return dc.feat.RDKitDescriptors()
        elif self.mol_featurizer_name == 'MACCS keys':
            return dc.feat.MACCSKeysFingerprint()
        elif self.mol_featurizer_name == 'Morgan fingerprint':
            return dc.feat.CircularFingerprint(size=2048, radius=3)
        else:
            return None

    def _featurize_molecule(self, mol_obj, smiles, mol_featurizer):
        if mol_featurizer is not None:
            return mol_featurizer.featurize(smiles)
        name = self.mol_featurizer_name
        if name == 'Avalon fingerprint':
            return np.array(pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)).reshape(1, -1)
        elif name == 'Atom-pairs fingerprint':
            return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Topological-torsion fingerprint':
            return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Layered fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'Pattern fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'RDKit fingerprint':
            return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        else:
            raise ValueError(f"Unknown molecule featurizer: {name}")

    def len(self):
        _, _, files = next(os.walk('./Datasets/' + self.dataset_name + '/processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)   
        return data

class MoleculeDatasetForRegression3D(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, num_conformers=1, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.target_column = target_column
        self.num_conformers = num_conformers
        self.output_scaler = MinMaxScaler(feature_range=(0, 1))
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        self.output_scaler.fit(self.data[self.target_column].to_numpy().reshape(-1, 1))

        if len(os.listdir(self.processed_dir)) > 2:
            return
        
        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                smiles = row["SMILES"]
                z, pos = self._process_3d(smiles)
                output = self.get_output(row[self.target_column])
                data = Data(z=z, pos=pos, smiles=row["SMILES"], y=output)
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue

    def _process_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if carbon_count < 2:
            raise Exception()
        num_confs = self.num_conformers
        params = AllChem.ETKDGv3()
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        energies = []
        for conf_id in ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        min_conf = min(energies, key=lambda x: x[1])[0]
        conformer = mol.GetConformer(id=min_conf)
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pos = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        z = torch.tensor(z)
        pos = torch.tensor(pos)
        return z, pos

    def get_output(self, output):
        return torch.tensor(self.output_scaler.transform(np.array([output]).reshape(-1, 1)))

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)

class MoleculeDatasetForRegressionPrediction3D(Dataset):
    def __init__(self, data_file_path, dataset_name, num_conformers=1, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.num_conformers = num_conformers
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2:
            return

        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                smiles = row["SMILES"]
                z, pos = self._process_3d(smiles)
                data = Data(z=z, pos=pos, smiles=row["SMILES"])
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue

    def _process_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if carbon_count < 2:
            raise Exception()
        num_confs = self.num_conformers
        params = AllChem.ETKDGv3()
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        energies = []
        for conf_id in ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        min_conf = min(energies, key=lambda x: x[1])[0]
        conformer = mol.GetConformer(id=min_conf)
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pos = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        z = torch.tensor(z)
        pos = torch.tensor(pos)
        return z, pos

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)

class MoleculeDatasetForBinaryClassificationHybrid(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, gcn_featurizer_name, mol_featurizer_name, dimensionality_reduction=False, variance_threshold=0.01, pca_num_components=32, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.target_column = target_column
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.dimensionality_reduction = dimensionality_reduction
        self.variance_threshold = VarianceThreshold(threshold=variance_threshold) if dimensionality_reduction else None
        self.pca = PCA(n_components=pca_num_components) if dimensionality_reduction else None
        self.pca_num_components = pca_num_components if dimensionality_reduction else None
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        if len(os.listdir(self.processed_dir)) > 2:
            mol_features_arr = np.loadtxt(os.path.join('./Datasets', self.dataset_name, 'raw', 'mol_features.csv'), delimiter=',')
            if self.dimensionality_reduction:
                mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
                mol_features_arr = self.pca.fit_transform(mol_features_arr)
            self.mol_features_scaler.fit(mol_features_arr)
            return

        gcn_featurizer = self._get_gcn_featurizer()
        mol_featurizer = self._get_mol_featurizer()
        mol_features_list = []
        graph_index = 0
        for _, (_, mol) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                mol_obj = Chem.MolFromSmiles(mol['SMILES'])
                salt_remover = SaltRemover.SaltRemover()
                mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                mol_obj = Chem.AddHs(mol_obj)
                carbon_count = sum(1 for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() == 6)
                if carbon_count < 2:
                    raise Exception()
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                mol_features = self._featurize_molecule(mol_obj, mol['SMILES'], mol_featurizer)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features_tensor = torch.tensor(mol_features)
                self.n_mol_features = mol_features_tensor.shape[1]
                output = self.get_output(mol[self.target_column])

                data = Data(x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=output,
                            smiles=mol['SMILES'],
                            mol_features=mol_features_tensor)
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue
        mol_features_arr = np.array(mol_features_list)
        np.savetxt(os.path.join('./Datasets', self.dataset_name, 'raw', 'mol_features.csv'), mol_features_arr, delimiter=',')
        if self.dimensionality_reduction:
            mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
            mol_features_arr = self.pca.fit_transform(mol_features_arr)
        self.mol_features_scaler.fit(mol_features_arr)

    def _get_gcn_featurizer(self):
        if self.gcn_featurizer_name == 'MolGraphConvFeaturizer':
            return dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer':
            return dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif self.gcn_featurizer_name == 'DMPNNFeaturizer':
            return dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        else:
            raise ValueError(f"Unknown GCN featurizer: {self.gcn_featurizer_name}")

    def _get_mol_featurizer(self):
        if self.mol_featurizer_name == 'Mordred descriptors':
            return dc.feat.MordredDescriptors(ignore_3D=True)
        elif self.mol_featurizer_name == 'RDKit descriptors':
            return dc.feat.RDKitDescriptors()
        elif self.mol_featurizer_name == 'MACCS keys':
            return dc.feat.MACCSKeysFingerprint()
        elif self.mol_featurizer_name == 'Morgan fingerprint':
            return dc.feat.CircularFingerprint(size=2048, radius=3)
        else:
            return None

    def _featurize_molecule(self, mol_obj, smiles, mol_featurizer):
        if mol_featurizer is not None:
            return mol_featurizer.featurize(smiles)
        name = self.mol_featurizer_name
        if name == 'Avalon fingerprint':
            return np.array(pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)).reshape(1, -1)
        elif name == 'Atom-pairs fingerprint':
            return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Topological-torsion fingerprint':
            return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Layered fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'Pattern fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'RDKit fingerprint':
            return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        else:
            raise ValueError(f"Unknown molecule featurizer: {name}")

    def get_output(self, output):
        return torch.tensor(np.array([output]))

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)

    @property
    def num_node_features(self):
        return {
            'MolGraphConvFeaturizer': 33,
            'PagtnMolGraphFeaturizer': 94,
            'DMPNNFeaturizer': 133
        }.get(self.gcn_featurizer_name, None)

    @property
    def num_edge_features(self):
        return {
            'MolGraphConvFeaturizer': 11,
            'PagtnMolGraphFeaturizer': 42,
            'DMPNNFeaturizer': 14
        }.get(self.gcn_featurizer_name, None)

    @property
    def num_mol_features(self):
        if self.dimensionality_reduction:
            return self.pca_num_components
        return {
            'Mordred descriptors': 1613,
            'RDKit descriptors': 210,
            'MACCS keys': 167,
            'Morgan fingerprint': 2048,
            'Avalon fingerprint': 2048,
            'Atom-pairs fingerprint': 2048,
            'Topological-torsion fingerprint': 2048,
            'Layered fingerprint': 2048,
            'Pattern fingerprint': 2048,
            'RDKit fingerprint': 2048
        }.get(self.mol_featurizer_name, None)

class MoleculeDatasetForBinaryClassificationPredictionHybrid(Dataset):
    def __init__(self, data_file_path, dataset_name, train_dataset_name, gcn_featurizer_name, mol_featurizer_name, dimensionality_reduction=False, variance_threshold=0.01, pca_num_components=32, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.train_dataset_name = train_dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.gcn_featurizer_name = gcn_featurizer_name
        self.mol_featurizer_name = mol_featurizer_name
        self.dimensionality_reduction = dimensionality_reduction
        self.variance_threshold = VarianceThreshold(threshold=variance_threshold) if dimensionality_reduction else None
        self.pca = PCA(n_components=pca_num_components) if dimensionality_reduction else None
        self.pca_num_components = pca_num_components if dimensionality_reduction else None
        self.mol_features_scaler = MinMaxScaler(feature_range=(0, 1))
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2:
            mol_features_arr = np.loadtxt(os.path.join('./Datasets', self.train_dataset_name, 'raw', 'mol_features.csv'), delimiter=',')
            if self.dimensionality_reduction:
                mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
                mol_features_arr = self.pca.fit_transform(mol_features_arr)
            self.mol_features_scaler.fit(mol_features_arr)
            return

        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        gcn_featurizer = self._get_gcn_featurizer()
        mol_featurizer = self._get_mol_featurizer()
        mol_features_list = []
        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                mol_obj = Chem.MolFromSmiles(row['SMILES'])
                salt_remover = SaltRemover.SaltRemover()
                mol_obj = salt_remover.StripMol(mol_obj, dontRemoveEverything=True)
                mol_obj = Chem.AddHs(mol_obj)
                carbon_count = sum(1 for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() == 6)
                if carbon_count < 2:
                    raise Exception()
                gcn_features = gcn_featurizer.featurize(mol_obj)
                node_feats = torch.tensor(np.array(gcn_features[0].node_features))
                self.n_gcn_features = node_feats.shape[1]
                edge_index = torch.tensor(np.array(gcn_features[0].edge_index))
                edge_attr = torch.tensor(np.array(gcn_features[0].edge_features))
                mol_features = self._featurize_molecule(mol_obj, row['SMILES'], mol_featurizer)
                mol_features_list.append(mol_features.reshape(-1))
                mol_features_tensor = torch.tensor(mol_features)
                self.n_mol_features = mol_features_tensor.shape[1]

                data = Data(x=node_feats,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            smiles=row['SMILES'],
                            mol_features=mol_features_tensor)
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue
        mol_features_arr = np.array(mol_features_list)
        np.savetxt(os.path.join('./Datasets', self.train_dataset_name, 'raw', 'mol_features.csv'), mol_features_arr, delimiter=',')
        if self.dimensionality_reduction:
            mol_features_arr = self.variance_threshold.fit_transform(mol_features_arr)
            mol_features_arr = self.pca.fit_transform(mol_features_arr)
        self.mol_features_scaler.fit(mol_features_arr)

    def _get_gcn_featurizer(self):
        if self.gcn_featurizer_name == 'MolGraphConvFeaturizer':
            return dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
        elif self.gcn_featurizer_name == 'PagtnMolGraphFeaturizer':
            return dc.feat.PagtnMolGraphFeaturizer(max_length=5)
        elif self.gcn_featurizer_name == 'DMPNNFeaturizer':
            return dc.feat.DMPNNFeaturizer(is_adding_hs=False)
        else:
            raise ValueError(f"Unknown GCN featurizer: {self.gcn_featurizer_name}")

    def _get_mol_featurizer(self):
        if self.mol_featurizer_name == 'Mordred descriptors':
            return dc.feat.MordredDescriptors(ignore_3D=True)
        elif self.mol_featurizer_name == 'RDKit descriptors':
            return dc.feat.RDKitDescriptors()
        elif self.mol_featurizer_name == 'MACCS keys':
            return dc.feat.MACCSKeysFingerprint()
        elif self.mol_featurizer_name == 'Morgan fingerprint':
            return dc.feat.CircularFingerprint(size=2048, radius=3)
        else:
            return None

    def _featurize_molecule(self, mol_obj, smiles, mol_featurizer):
        if mol_featurizer is not None:
            return mol_featurizer.featurize(smiles)
        name = self.mol_featurizer_name
        if name == 'Avalon fingerprint':
            return np.array(pyAvalonTools.GetAvalonFP(mol_obj, nBits=2048)).reshape(1, -1)
        elif name == 'Atom-pairs fingerprint':
            return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Topological-torsion fingerprint':
            return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        elif name == 'Layered fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'Pattern fingerprint':
            return np.array(rdmolops.LayeredFingerprint(mol_obj, fpSize=2048)).reshape(1, -1)
        elif name == 'RDKit fingerprint':
            return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprintAsNumPy(mol_obj).reshape(1, -1)
        else:
            raise ValueError(f"Unknown molecule featurizer: {name}")

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)

    def len(self):
        _, _, files = next(os.walk('./Datasets/' + self.dataset_name + '/processed'))
        return len(files) - 2

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)   
        return data

class MoleculeDatasetForBinaryClassification3D(Dataset):
    def __init__(self, data_file_path, dataset_name, target_column, num_conformers=1, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.target_column = target_column
        self.num_conformers = num_conformers
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)
        if len(os.listdir(self.processed_dir)) > 2:
            return

        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                smiles = row["SMILES"]
                z, pos = self._process_3d(smiles)
                output = self.get_output(row[self.target_column])
                data = Data(z=z, pos=pos, smiles=row["SMILES"], y=output)
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue

    def _process_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if carbon_count < 2:
            raise Exception()
        num_confs = self.num_conformers
        params = AllChem.ETKDGv3()
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        energies = []
        for conf_id in ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        min_conf = min(energies, key=lambda x: x[1])[0]
        conformer = mol.GetConformer(id=min_conf)
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pos = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        z = torch.tensor(z)
        pos = torch.tensor(pos)
        return z, pos

    def get_output(self, output):
        return torch.tensor(np.array([output]))

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)

class MoleculeDatasetForBinaryClassificationPrediction3D(Dataset):
    def __init__(self, data_file_path, dataset_name, num_conformers=1, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.data_file_name = os.path.basename(data_file_path)
        self.num_conformers = num_conformers
        root = os.path.join('./Datasets', dataset_name)
        os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
        shutil.copy2(data_file_path, os.path.join(root, 'raw'))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.data_file_name

    @property
    def processed_file_names(self):
        data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f'data_{i}.pt' for i in data.index]

    def process(self):
        if len(os.listdir(self.processed_dir)) > 2:
            return

        data_path = os.path.join('./Datasets', self.dataset_name, 'raw', self.data_file_name)
        self.data = pd.read_csv(data_path)

        graph_index = 0
        for _, (_, row) in enumerate(tqdm(self.data.iterrows(), total=self.data.shape[0])):
            try:
                smiles = row["SMILES"]
                z, pos = self._process_3d(smiles)
                data = Data(z=z, pos=pos, smiles=row["SMILES"])
                torch.save(data, os.path.join(self.processed_dir, f'{self.dataset_name}_{graph_index}.pt'))
                graph_index += 1
            except:
                continue

    def _process_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
        mol = Chem.AddHs(mol)
        carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if carbon_count < 2:
            raise Exception()
        num_confs = self.num_conformers
        params = AllChem.ETKDGv3()
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        energies = []
        for conf_id in ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energy = ff.CalcEnergy()
            energies.append((conf_id, energy))
        min_conf = min(energies, key=lambda x: x[1])[0]
        conformer = mol.GetConformer(id=min_conf)
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        pos = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        z = torch.tensor(z)
        pos = torch.tensor(pos)
        return z, pos

    def len(self):
        _, _, files = next(os.walk(os.path.join('./Datasets', self.dataset_name, 'processed')))
        return len(files) - 2

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'{self.dataset_name}_{idx}.pt'), weights_only=False)
