import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class MolecularDataset(Dataset):
    def __init__(self, dataset_file_path, load_data, graph_directory_path, molecular_fingerprint_directory_path, datatype):
        self.dataset_file_path = dataset_file_path
        self.df = pd.read_csv(dataset_file_path)
        self.target_column_names = self.df.columns.tolist()[1:]
        self.load_data = load_data
        self.datatype = datatype
        
        # Validate that the graph and molecular fingerprint data are present if needed
        if load_data == 'Graphs' or load_data == 'Both':
            self.graph_directory_path = graph_directory_path
            num_graphs = len(os.listdir(graph_directory_path))
            if num_graphs == 0:
                raise ValueError(f"No graph files found in {graph_directory_path}")
        if load_data == 'Molecular fingerprints' or load_data == 'Both':
            self.fingerprint_df = pd.read_csv(os.path.join(molecular_fingerprint_directory_path, 'molecular_fingerprints.csv'))
            num_fingerprints = len(self.fingerprint_df)
            if num_fingerprints == 0:
                raise ValueError(f"No molecular fingerprint files found in {molecular_fingerprint_directory_path}")
        if load_data == 'Both':
            num_graphs = len(os.listdir(graph_directory_path))
            self.fingerprint_df = pd.read_csv(os.path.join(molecular_fingerprint_directory_path, 'molecular_fingerprints.csv'))
            num_fingerprints = len(self.fingerprint_df)
            if num_graphs != num_fingerprints:
                raise ValueError(f"Number of graph files ({num_graphs}) does not match number of molecular fingerprint files ({num_fingerprints})")

    def  __len__(self):
        if self.load_data == 'Graphs' or self.load_data == 'Both':
            num_graphs = len(os.listdir(self.graph_directory_path))
            return num_graphs
        else:
            num_fingerprints = len(self.fingerprint_df)
            return num_fingerprints

    def __getitem__(self, idx):
        if self.load_data == 'Graphs':
            data = torch.load(os.path.join(self.graph_directory_path, f'{idx}.pt'), weights_only=False)
        elif self.load_data == 'Molecular fingerprints':
            data = Data(x = None, edge_index = None, edge_attr = None)
            data.smiles = self.df['SMILES'].iloc[idx]
            data.fp = torch.tensor(self.fingerprint_df.iloc[idx][1:].astype(int).to_numpy(), dtype=self.datatype).unsqueeze(0)
        else:  # Both
            data = torch.load(os.path.join(self.graph_directory_path, f'{idx}.pt'), weights_only=False)
            data.fp = torch.tensor(self.fingerprint_df.iloc[idx][1:].astype(int).to_numpy(), dtype=self.datatype).unsqueeze(0)
        
        if self.df.shape[1] == 1: # Only SMILES column, no target columns
            data.y = None
        else:
            data.y = torch.tensor(self.df.drop(columns=['SMILES']).iloc[idx].values, dtype=self.datatype).unsqueeze(0)
        
        return data
    
    @property
    def num_node_features(self):
        if self.load_data == 'Graphs' or self.load_data == 'Both':
            data = torch.load(os.path.join(self.graph_directory_path, f'0.pt'), weights_only=False)
            return data.x.shape[1]
        else:
            return 0

    @property
    def num_edge_features(self):
        if self.load_data == 'Graphs' or self.load_data == 'Both':
            data = torch.load(os.path.join(self.graph_directory_path, f'0.pt'), weights_only=False)
            return data.edge_attr.shape[1]
        else:
            return 0
        
    @property
    def num_fingerprint_bits(self):
        if self.load_data == 'Molecular fingerprints' or self.load_data == 'Both':
            return self.fingerprint_df.shape[1] - 1
        else:
            return 0
        
    @property
    def num_outputs(self):
        return self.df.shape[1] - 1
    
class MolecularDataset3D(Dataset):
    def __init__(self, dataset_file_path, graph_directory_path, datatype):
        self.dataset_file_path = dataset_file_path
        self.df = pd.read_csv(dataset_file_path)
        self.target_column_name = self.df.columns[1] if self.df.shape[1] > 1 else None
        self.datatype = datatype
        
        # Validate that the graph data are present if needed
        self.graph_directory_path = graph_directory_path
        num_graphs = len(os.listdir(graph_directory_path))
        if num_graphs == 0:
            raise ValueError(f"No graph files found in {graph_directory_path}")

    def  __len__(self):
        return len(os.listdir(self.graph_directory_path))

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.graph_directory_path, f'{idx}.pt'), weights_only=False)
        
        if self.df.shape[1] == 1: # Only SMILES column, no target columns
            data.y = None
        else:
            data.y = torch.tensor(self.df.drop(columns=['SMILES']).iloc[idx].values, dtype=self.datatype).unsqueeze(0)
        
        return data
            
    @property
    def num_outputs(self):
        return 1