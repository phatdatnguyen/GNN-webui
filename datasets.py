import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import list_graph_files, read_graph_manifest

def _report_mismatch(index, graph_smiles, dataset_smiles):
    raise ValueError(
        f"Graph {index}.pt is for '{graph_smiles}' but row {index} of the dataset file is "
        f"'{dataset_smiles}'. The graphs are out of sync with the dataset - please re-run "
        "graph extraction."
    )

def _check_graphs_match_dataset(graph_directory_path, df):
    """Validate that graph <i>.pt lines up with row <i> of the dataset file.

    Graph and fingerprint extraction both rewrite the dataset CSV when a molecule fails, so
    running one after the other can leave the older artifacts shifted by a row. Comparing the
    SMILES the graphs were built from against the CSV turns that silent mismatch into an error.
    """
    indices = list_graph_files(graph_directory_path)
    if len(indices) == 0:
        raise ValueError(f"No graph files found in {graph_directory_path}")
    if indices != list(range(len(indices))):
        raise ValueError(
            f"Graph files in {graph_directory_path} are not numbered 0..{len(indices) - 1}. "
            "Please re-run graph extraction."
        )
    if len(indices) != len(df):
        raise ValueError(
            f"Number of graph files ({len(indices)}) does not match the number of rows in the "
            f"dataset file ({len(df)}). Please re-run graph extraction."
        )

    dataset_smiles = df['SMILES'].tolist()
    manifest = read_graph_manifest(graph_directory_path)
    if manifest is not None:
        # Extractions since the manifest was introduced can be checked in full, for the cost
        # of reading one small text file.
        if len(manifest) != len(indices):
            raise ValueError(
                f"The graph manifest in {graph_directory_path} lists {len(manifest)} molecules "
                f"but there are {len(indices)} graph files. Please re-run graph extraction."
            )
        for index, (graph_smiles, row_smiles) in enumerate(zip(manifest, dataset_smiles)):
            if graph_smiles != row_smiles:
                _report_mismatch(index, graph_smiles, row_smiles)
        return len(indices)

    # Older extractions have no manifest, so fall back to spot-checking the ends and middle:
    # a dropped row shifts everything after it.
    for index in sorted({0, len(indices) // 2, len(indices) - 1}):
        data = torch.load(os.path.join(graph_directory_path, f'{index}.pt'), weights_only=False)
        graph_smiles = getattr(data, 'smiles', None)
        if graph_smiles is not None and graph_smiles != dataset_smiles[index]:
            _report_mismatch(index, graph_smiles, dataset_smiles[index])
    return len(indices)

def _check_fingerprints_match_dataset(fingerprint_df, df, molecular_fingerprint_directory_path):
    if len(fingerprint_df) == 0:
        raise ValueError(f"No molecular fingerprint files found in {molecular_fingerprint_directory_path}")
    if len(fingerprint_df) != len(df):
        raise ValueError(
            f"Number of molecular fingerprints ({len(fingerprint_df)}) does not match the number of "
            f"rows in the dataset file ({len(df)}). Please re-run molecular fingerprint extraction."
        )
    mismatched = (fingerprint_df['SMILES'].to_numpy() != df['SMILES'].to_numpy()).nonzero()[0]
    if len(mismatched) > 0:
        row = int(mismatched[0])
        raise ValueError(
            f"Molecular fingerprint row {row} is for '{fingerprint_df['SMILES'].iloc[row]}' but row "
            f"{row} of the dataset file is '{df['SMILES'].iloc[row]}'. The fingerprints are out of "
            "sync with the dataset - please re-run molecular fingerprint extraction."
        )

class MolecularDataset(Dataset):
    def __init__(self, dataset_file_path, load_data, graph_directory_path, molecular_fingerprint_directory_path, datatype):
        self.dataset_file_path = dataset_file_path
        self.df = pd.read_csv(dataset_file_path)
        self.target_column_names = self.df.columns.tolist()[1:]
        self.load_data = load_data
        self.datatype = datatype
        self.graph_directory_path = graph_directory_path
        self.fingerprint_df = None
        self._length = len(self.df)

        # Validate that the graph and molecular fingerprint data are present and aligned with the dataset
        if load_data == 'Graphs' or load_data == 'Both':
            self._length = _check_graphs_match_dataset(graph_directory_path, self.df)
        if load_data == 'Molecular fingerprints' or load_data == 'Both':
            self.fingerprint_df = pd.read_csv(os.path.join(molecular_fingerprint_directory_path, 'molecular_fingerprints.csv'))
            _check_fingerprints_match_dataset(self.fingerprint_df, self.df, molecular_fingerprint_directory_path)

    def  __len__(self):
        return self._length

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

        # Graphs extracted by older versions stored edge_index as a float tensor.
        if getattr(data, 'edge_index', None) is not None and data.edge_index.is_floating_point():
            data.edge_index = data.edge_index.long()

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

        # Validate that the graph data are present and aligned with the dataset
        self.graph_directory_path = graph_directory_path
        self._length = _check_graphs_match_dataset(graph_directory_path, self.df)

    def  __len__(self):
        return self._length

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.graph_directory_path, f'{idx}.pt'), weights_only=False)

        # 3D graphs extracted by older versions stored atomic numbers as a float tensor.
        if getattr(data, 'z', None) is not None and data.z.is_floating_point():
            data.z = data.z.long()

        if self.df.shape[1] == 1: # Only SMILES column, no target columns
            data.y = None
        else:
            data.y = torch.tensor(self.df.drop(columns=['SMILES']).iloc[idx].values, dtype=self.datatype).unsqueeze(0)

        return data

    @property
    def num_outputs(self):
        return 1
