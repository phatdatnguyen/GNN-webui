import os
import copy
import operator
import sys
from collections import OrderedDict
from numbers import Integral

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, get_worker_info
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

DEFAULT_GRAPH_CACHE_BYTES = 128 * 1024 * 1024


def _can_clone_graph_tensors(graph):
    """Only independent, ordinary feature tensors use the faster copy path."""
    storages = {}
    for _, value in graph:
        if isinstance(value, torch.Tensor):
            if (value.requires_grad or value.layout != torch.strided
                    or not value.is_contiguous() or value.storage_offset()):
                return False
            storage = value.untyped_storage()
            if value.numel() * value.element_size() != storage.nbytes():
                return False
            identity = (storage.data_ptr(), storage.nbytes())
            if identity in storages and storages[identity] != id(value):
                return False
            storages[identity] = id(value)
        elif not isinstance(value, (str, bytes, int, float, bool, type(None))):
            # Nested metadata can contain other views into tensor storage.
            return False
    return True


def _copy_graph(graph, clone_tensors=None):
    """Copy graph values without Torch's slower storage-deepcopy path.

    Tensor-storage aliases, views and arbitrary metadata use the original full
    deepcopy semantics. Standard extracted graphs have only independent feature
    tensors and a SMILES string, so their eligibility is recorded once on caching.
    """
    if clone_tensors is None:
        clone_tensors = _can_clone_graph_tensors(graph)
    if not clone_tensors:
        return copy.deepcopy(graph)
    tensors = {}
    for key, value in graph:
        if isinstance(value, torch.Tensor) and id(value) not in tensors:
            tensors[id(value)] = value.clone()
    return copy.deepcopy(graph, tensors)


def _gather_fingerprints(values, indices, datatype):
    """Avoid waking the whole Torch CPU thread pool for a modest batch cast."""
    selected = values[indices]
    numpy_datatype = {torch.float16: np.float16, torch.float32: np.float32,
                      torch.float64: np.float64}.get(datatype)
    if numpy_datatype is not None:
        # Match Torch's overflow-to-infinity conversion without NumPy warnings.
        with np.errstate(over="ignore"):
            return torch.from_numpy(selected.astype(numpy_datatype, copy=False))
    source = torch.from_numpy(selected)
    result = torch.empty(source.shape, dtype=datatype)
    source, destination = source.reshape(-1), result.reshape(-1)
    # NumPy cannot represent bfloat16. Small copies retain Torch's exact casting
    # rules without triggering parallel-copy overhead on high-core-count hosts.
    for start in range(0, source.numel(), 16384):
        destination[start:start + 16384].copy_(source[start:start + 16384])
    return result


def _gather_tensor_rows(values, rows):
    """Keep large CPU gathers from paying high-core-count thread-pool overhead."""
    if rows.numel() * values.shape[1] <= 16384:
        return values.index_select(0, rows)
    if values.dtype == torch.bfloat16:
        # Select the stored bits without rounding through a different float type.
        selected = values.view(torch.int16).numpy()[rows.numpy()]
        return torch.from_numpy(selected).view(torch.bfloat16)
    return torch.from_numpy(values.numpy()[rows.numpy()])


def _graph_memory_bytes(graph):
    """Count tensor storage plus Python objects, including shared-storage views once.

    Allocator fragmentation and the temporary graph returned to a caller are not
    included. The per-entry allowance covers the LRU bookkeeping conservatively.
    """
    objects, storages = set(), set()

    def size(value):
        if id(value) in objects:
            return 0
        objects.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr(), storage.nbytes())
            if identity not in storages:
                storages.add(identity)
                # UntypedStorage.__sizeof__ already includes its backing bytes.
                result += max(storage.nbytes(), sys.getsizeof(storage))
        elif isinstance(value, dict):
            result += sum(size(key) + size(item) for key, item in value.items())
        elif isinstance(value, (tuple, list, set)):
            result += sum(size(item) for item in value)
        elif hasattr(value, "__dict__"):
            result += size(vars(value))
        return result

    return size(graph) + 256


class _PreparedDataset(Dataset):
    """Own the CPU caches shared by all train/validation/test Subsets."""

    def _initialize_cache(self, graph_cache_bytes):
        self._graph_cache = OrderedDict()
        self._graph_cache_size = 0
        self._graph_dimensions = None
        self._fp_prepared = self._y_prepared = None
        self._fp_prepared_key = self._y_prepared_key = None
        self.set_graph_cache_budget(graph_cache_bytes)
        self._captured_columns = tuple(self.df.columns)
        self._smiles = self.df["SMILES"].tolist()
        targets = self.df.drop(columns=["SMILES"]).to_numpy(dtype=np.float64, copy=True)
        self._targets = torch.from_numpy(np.ascontiguousarray(targets)).to(self.target_datatype)

    def __len__(self):
        return self._length

    def _row_index(self, idx):
        idx = operator.index(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Dataset row index is out of range.")
        return idx

    @property
    def graph_cache_info(self):
        return {"budget_bytes": self.graph_cache_bytes, "size_bytes": self._graph_cache_size,
                "entries": len(self._graph_cache)}

    def set_graph_cache_budget(self, budget):
        if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 0:
            raise ValueError("Graph cache size must be a nonnegative integer number of bytes.")
        self.graph_cache_bytes = int(budget)
        while self._graph_cache and self._graph_cache_size > self.graph_cache_bytes:
            _, (_, size, _) = self._graph_cache.popitem(last=False)
            self._graph_cache_size -= size

    def clear_cache(self):
        """Release optional caches; raw numerical inputs remain usable."""
        self._graph_cache.clear()
        self._graph_cache_size = 0
        self._fp_prepared = self._y_prepared = None
        self._fp_prepared_key = self._y_prepared_key = None

    def __getstate__(self):
        # Spawned workers must not receive another copy of the parent's graph cache.
        state = self.__dict__.copy()
        state["_graph_cache"] = OrderedDict()
        state["_graph_cache_size"] = 0
        return state

    def _load_graph(self, idx):
        idx = int(idx)
        # Worker lifetime is one iterator; caching there wastes memory and does not
        # survive epochs. Keep the shared parent cache for the default worker=0 path.
        cache_enabled = self.graph_cache_bytes > 0 and get_worker_info() is None
        if cache_enabled and idx in self._graph_cache:
            graph, size, clone_tensors = self._graph_cache.pop(idx)
            self._graph_cache[idx] = graph, size, clone_tensors
            return _copy_graph(graph, clone_tensors)
        graph = torch.load(os.path.join(self.graph_directory_path, f"{idx}.pt"),
                           weights_only=False, map_location="cpu")
        for name in ("edge_index", "z"):
            indices = getattr(graph, name, None)
            if indices is not None and indices.is_floating_point():
                setattr(graph, name, indices.long())
        if cache_enabled:
            size = _graph_memory_bytes(graph)
            if size <= self.graph_cache_bytes:
                while self._graph_cache and self._graph_cache_size + size > self.graph_cache_bytes:
                    _, (_, removed_size, _) = self._graph_cache.popitem(last=False)
                    self._graph_cache_size -= removed_size
                clone_tensors = _can_clone_graph_tensors(graph)
                self._graph_cache[idx] = graph, size, clone_tensors
                self._graph_cache_size += size
                return _copy_graph(graph, clone_tensors)
        return graph

    def _graph_for_row(self, idx):
        return self._load_graph(idx)

    def __getitems__(self, indices):
        """Gather numeric columns once per batch while returning normal samples.

        PyTorch and its Subset wrapper use this supported batched-fetch method.
        Advanced indexing owns new storage, including when indices repeat, so no
        returned sample can modify cached numerical inputs or another row's values.
        The ordinary ``__getitem__`` API stays available for model summaries and
        integrations that request individual molecules.
        """
        indices = [self._row_index(index) for index in indices]
        if not indices:
            return []
        rows = torch.tensor(indices, dtype=torch.long)
        targets = _gather_tensor_rows(self._targets, rows) if self._targets.shape[1] else None
        fingerprint_values = getattr(self, "fingerprint_values", None)
        fingerprints = (_gather_fingerprints(fingerprint_values, indices, self.datatype)
                        if fingerprint_values is not None else None)
        prepared_fp = (_gather_tensor_rows(self._fp_prepared, rows)
                       if self._fp_prepared is not None else None)
        prepared_y = (_gather_tensor_rows(self._y_prepared, rows)
                      if self._y_prepared is not None else None)
        samples = []
        for offset, index in enumerate(indices):
            data = self._graph_for_row(index)
            data.y = targets[offset:offset + 1] if targets is not None else None
            if fingerprints is not None:
                data.fp = fingerprints[offset:offset + 1]
            if prepared_fp is not None:
                data.fp_prepared = prepared_fp[offset:offset + 1]
            if prepared_y is not None:
                data.y_prepared = prepared_y[offset:offset + 1]
            samples.append(data)
        return samples

    def _attach_targets(self, data, idx):
        data.y = self._targets[idx:idx + 1].clone() if self._targets.shape[1] else None
        if self._y_prepared is not None:
            data.y_prepared = self._y_prepared[idx:idx + 1].clone()
        return data


def _transform_in_chunks(values, transform, datatype, chunk_size=2048):
    """Allocate only the output and one temporary transform chunk."""
    result = None
    for start in range(0, len(values), chunk_size):
        source = values[start:start + chunk_size]
        transformed = np.asarray(transform(source))
        if transformed.ndim != 2 or transformed.shape[0] != len(source):
            raise ValueError("Prepared feature shape does not match the dataset rows.")
        if result is None:
            result = torch.empty((len(values), transformed.shape[1]), dtype=datatype)
        result[start:start + len(source)] = torch.as_tensor(transformed, dtype=datatype)
    return result


def prepare_dataset(dataset, variance_threshold=None, pca=None, output_scaler=None, *, datatype=None,
                    prepare_targets=True):
    """Prepare fitted transforms once, rebuilding after checkpoint transformer changes.

    The transformers are never fitted here. Hash their learned contents, not their
    object IDs, so in-place refitting and checkpoint-restored preprocessing are safe.
    Generic datasets used by integrations keep their existing batch-time fallback.
    """
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, _PreparedDataset):
        return dataset
    if tuple(dataset.df.columns) != dataset._captured_columns or len(dataset.df) != len(dataset):
        raise ValueError("Dataset target/output shape changed after processing. "
                         "Please process data and re-create the model.")
    prepared_datatype = dataset.datatype if datatype is None else datatype
    use_fp = (getattr(dataset, "fingerprint_values", None) is not None
              and variance_threshold is not None and pca is not None)
    fp_key = joblib.hash((variance_threshold, pca, str(dataset.datatype), str(prepared_datatype))) if use_fp else None
    if fp_key != dataset._fp_prepared_key:
        def transform_fingerprints(values):
            # Match the former batch.fp -> double -> sklearn path, including
            # selected-dtype rounding for legacy nonbinary fingerprint files.
            source = torch.as_tensor(values).to(dataset.datatype).double().numpy()
            return pca.transform(variance_threshold.transform(source))
        prepared = (_transform_in_chunks(dataset.fingerprint_values, transform_fingerprints,
                                         prepared_datatype) if use_fp else None)
        dataset._fp_prepared, dataset._fp_prepared_key = prepared, fp_key
    if prepare_targets:
        use_targets = output_scaler is not None and dataset._targets.shape[1] > 0
        target_key = joblib.hash((output_scaler, str(prepared_datatype))) if use_targets else None
        if target_key != dataset._y_prepared_key:
            prepared = (_transform_in_chunks(dataset._targets,
                                             lambda values: output_scaler.transform(values.double().numpy().copy()),
                                             prepared_datatype) if use_targets else None)
            dataset._y_prepared, dataset._y_prepared_key = prepared, target_key
    return dataset


class MolecularDataset(_PreparedDataset):
    def __init__(self, dataset_file_path, load_data, graph_directory_path, molecular_fingerprint_directory_path, datatype, *, target_datatype=None, graph_cache_bytes=DEFAULT_GRAPH_CACHE_BYTES):
        self.dataset_file_path = dataset_file_path
        self.df = pd.read_csv(dataset_file_path)
        self.target_column_names = self.df.columns.tolist()[1:]
        self.load_data = load_data
        self.datatype = datatype
        self.target_datatype = target_datatype if target_datatype is not None else datatype
        self.graph_directory_path = graph_directory_path
        self.fingerprint_df = None
        self.fingerprint_values = None
        self._length = len(self.df)

        # Validate that the graph and molecular fingerprint data are present and aligned with the dataset
        if load_data == 'Graphs' or load_data == 'Both':
            self._length = _check_graphs_match_dataset(graph_directory_path, self.df)
        if load_data == 'Molecular fingerprints' or load_data == 'Both':
            self.fingerprint_df = pd.read_csv(os.path.join(molecular_fingerprint_directory_path, 'molecular_fingerprints.csv'))
            _check_fingerprints_match_dataset(self.fingerprint_df, self.df, molecular_fingerprint_directory_path)
            numeric = self.fingerprint_df.iloc[:, 1:].to_numpy(copy=True)
            binary = np.isin(numeric, (0, 1)).all()
            # Nonbinary legacy inputs retain the old integer conversion semantics.
            if not binary:
                # pandas rejects NaN/infinity instead of casting them to a large
                # sentinel integer, as an unchecked NumPy conversion would do.
                numeric = self.fingerprint_df.iloc[:, 1:].astype(np.int64).to_numpy()
            self.fingerprint_values = np.array(numeric, dtype=np.uint8 if binary else np.int64,
                                               order="C", copy=True)
            if binary and all(pd.api.types.is_integer_dtype(dtype)
                              for dtype in self.fingerprint_df.dtypes.iloc[1:]):
                # Keep provenance/public DataFrame access, sharing compact storage
                # without changing the CSV representation of integer bit columns.
                compact = pd.DataFrame(self.fingerprint_values, columns=self.fingerprint_df.columns[1:],
                                       copy=False)
                compact.insert(0, "SMILES", self.fingerprint_df["SMILES"].to_numpy(copy=True))
                self.fingerprint_df = compact
        self._initialize_cache(graph_cache_bytes)

    def __getitem__(self, idx):
        idx = self._row_index(idx)
        data = self._graph_for_row(idx)
        if self.fingerprint_values is not None:
            data.fp = torch.from_numpy(self.fingerprint_values[idx:idx + 1]).to(self.datatype)
        if self._fp_prepared is not None:
            data.fp_prepared = self._fp_prepared[idx:idx + 1].clone()
        return self._attach_targets(data, idx)

    def _graph_for_row(self, idx):
        if self.load_data == 'Molecular fingerprints':
            return Data(smiles=self._smiles[idx])
        return self._load_graph(idx)

    def _feature_dimensions(self):
        if self._graph_dimensions is None:
            data = self._load_graph(0)
            self._graph_dimensions = (data.x.shape[1] if data.x is not None else 0,
                                      data.edge_attr.shape[1] if data.edge_attr is not None else 0)
        return self._graph_dimensions

    @property
    def num_node_features(self):
        if self.load_data == 'Graphs' or self.load_data == 'Both':
            return self._feature_dimensions()[0]
        else:
            return 0

    @property
    def num_edge_features(self):
        if self.load_data == 'Graphs' or self.load_data == 'Both':
            return self._feature_dimensions()[1]
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

class MolecularDataset3D(_PreparedDataset):
    def __init__(self, dataset_file_path, graph_directory_path, datatype, *, target_datatype=None, graph_cache_bytes=DEFAULT_GRAPH_CACHE_BYTES):
        self.dataset_file_path = dataset_file_path
        self.df = pd.read_csv(dataset_file_path)
        if self.df.shape[1] > 2:
            raise ValueError("3D models accept only one target column. Please select a dataset "
                             "with SMILES and one target column, or SMILES only for prediction.")
        self.target_column_name = self.df.columns[1] if self.df.shape[1] > 1 else None
        self.datatype = datatype
        self.target_datatype = target_datatype if target_datatype is not None else datatype

        # Validate that the graph data are present and aligned with the dataset
        self.graph_directory_path = graph_directory_path
        self._length = _check_graphs_match_dataset(graph_directory_path, self.df)
        self._initialize_cache(graph_cache_bytes)

    def __getitem__(self, idx):
        idx = self._row_index(idx)
        return self._attach_targets(self._load_graph(idx), idx)

    @property
    def num_outputs(self):
        return 1
