"""Numerical equivalence, bounded graph ownership, and fitted-transform invalidation."""

import pickle
import weakref

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import datasets
from datasets import MolecularDataset, MolecularDataset3D, prepare_dataset
from utils import write_graph_manifest


@pytest.fixture
def prepared_files(tmp_path):
    smiles = ["CCO", "CCN", "CCC", "CCS"]
    pd.DataFrame({"SMILES": smiles, "target": [100000.125, 100000.25, 100000.5, 100001.0]}).to_csv(
        tmp_path / "dataset.csv", index=False)
    graph_dir, fp_dir = tmp_path / "GRAPH_graphs", tmp_path / "FP_fps"
    graph_dir.mkdir()
    fp_dir.mkdir()
    for index, molecule in enumerate(smiles):
        graph = Data(x=torch.arange(6, dtype=torch.float32).reshape(3, 2),
                     edge_index=torch.tensor([[0, 1], [1, 2]]),
                     edge_attr=torch.ones(2, 1), z=torch.tensor([6, 6, 8]),
                     pos=torch.arange(9, dtype=torch.float32).reshape(3, 3), smiles=molecule,
                     metadata={"labels": ["original"], "tensor": torch.ones(2)})
        torch.save(graph, graph_dir / f"{index}.pt")
    write_graph_manifest(graph_dir, smiles)
    fp = pd.DataFrame({"SMILES": smiles, "Bit_1": [0, 1, 1, 0],
                       "Bit_2": [1, 0, 1, 0], "Bit_3": [1, 1, 1, 1]})
    fp.to_csv(fp_dir / "molecular_fingerprints.csv", index=False)
    return tmp_path


def make_dataset(directory, *, is_3d=False, datatype=torch.float32, graph_cache_bytes=128 * 1024**2):
    arguments = (str(directory / "dataset.csv"), str(directory / "GRAPH_graphs"))
    if is_3d:
        return MolecularDataset3D(*arguments, datatype, target_datatype=torch.float64,
                                  graph_cache_bytes=graph_cache_bytes)
    return MolecularDataset(arguments[0], "Both", arguments[1], str(directory / "FP_fps"), datatype,
                            target_datatype=torch.float64, graph_cache_bytes=graph_cache_bytes)


@pytest.mark.parametrize("is_3d", [False, True])
def test_cached_graphs_load_once_and_never_share_mutable_samples(prepared_files, monkeypatch, is_3d):
    dataset = make_dataset(prepared_files, is_3d=is_3d)
    original_load = torch.load
    reads = []

    def load(*args, **kwargs):
        reads.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    first = dataset[0]
    first.x.zero_()
    first.metadata["labels"][0] = "changed"
    first.metadata["tensor"].zero_()
    first.y.zero_()
    second = Subset(dataset, [0])[0]
    assert len(reads) == 1
    assert second.x.sum() > 0
    assert second.metadata["labels"] == ["original"]
    assert second.metadata["tensor"].sum() == 2
    assert second.y.item() == 100000.125
    assert dataset.graph_cache_info["entries"] == 1
    assert 0 < dataset.graph_cache_info["size_bytes"] <= dataset.graph_cache_info["budget_bytes"]


def test_cache_eviction_disabled_mode_and_clear_release_storage(prepared_files, monkeypatch):
    dataset = make_dataset(prepared_files)
    dataset[0]
    tensor = weakref.ref(dataset._graph_cache[0][0].x)
    one_graph_bytes = dataset.graph_cache_info["size_bytes"]
    assert one_graph_bytes > dataset[0].x.numel() * 4
    dataset.set_graph_cache_budget(one_graph_bytes)
    dataset[1]
    assert dataset.graph_cache_info["entries"] == 1
    assert tensor() is None
    assert list(dataset._graph_cache) == [1]
    dataset.clear_cache()
    assert dataset.graph_cache_info["size_bytes"] == 0
    dataset.set_graph_cache_budget(0)
    dataset[0]
    assert dataset.graph_cache_info["entries"] == 0
    dataset.set_graph_cache_budget(1)
    dataset[0]
    assert dataset.graph_cache_info["entries"] == 0


def test_spawn_does_not_duplicate_parent_graph_cache(prepared_files, monkeypatch):
    dataset = make_dataset(prepared_files)
    dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert dataset.graph_cache_info["entries"] == 1
    assert restored.graph_cache_info["entries"] == 0
    assert torch.equal(restored[0].x, dataset[0].x)
    restored.clear_cache()
    monkeypatch.setattr(datasets, "get_worker_info", lambda: object())
    restored[0]
    assert restored.graph_cache_info["entries"] == 0


def test_fingerprint_arrays_are_compact_and_match_original_values(prepared_files, monkeypatch):
    dataset = make_dataset(prepared_files)
    assert dataset.fingerprint_values.dtype == np.uint8
    assert dataset.fingerprint_values.flags.c_contiguous
    assert dataset.fingerprint_values.flags.writeable
    assert dataset.fingerprint_values.nbytes == 4 * 3
    assert all(dtype == np.uint8 for dtype in dataset.fingerprint_df.dtypes.iloc[1:])
    expected = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=torch.float32)
    # Dataset iteration no longer selects pandas rows or drops target columns.
    monkeypatch.setattr(pd.DataFrame, "drop", lambda *args, **kwargs: pytest.fail("pandas in row access"))
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert torch.equal(batch.fp, expected)
    assert batch.y.dtype == torch.float64
    dataset[0].fp.zero_()
    assert torch.equal(dataset[0].fp, expected[:1])


def test_compaction_preserves_feature_provenance(prepared_files):
    from feature_provenance import capture_feature_sources

    directory = str(prepared_files / "FP_fps")
    from_file = capture_feature_sources(fingerprint_directory=directory)
    dataset = make_dataset(prepared_files)
    from_dataset = capture_feature_sources(fingerprint_directory=directory,
                                          fingerprint_df=dataset.fingerprint_df)
    assert from_file == from_dataset


def test_nonbinary_values_keep_legacy_integer_conversion(prepared_files):
    path = prepared_files / "FP_fps" / "molecular_fingerprints.csv"
    frame = pd.read_csv(path)
    frame["Bit_1"] = [0.5, 2.9, -1.1, 2049]
    frame.to_csv(path, index=False)
    dataset = make_dataset(prepared_files)
    assert dataset.fingerprint_values.dtype == np.int64
    assert dataset.fingerprint_values[:, 0].tolist() == [0, 2, -1, 2049]
    assert torch.equal(dataset[3].fp, torch.tensor([[2049., 0., 1.]]))
    frame.loc[0, "Bit_1"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises((ValueError, pd.errors.IntCastingNaNError)):
        make_dataset(prepared_files)


@pytest.mark.parametrize("datatype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_prepared_features_match_batch_transforms_and_invalidate_on_refit(prepared_files, datatype):
    dataset = make_dataset(prepared_files, datatype=datatype)
    raw = torch.cat([dataset[i].fp for i in range(len(dataset))]).double().numpy()
    variance = VarianceThreshold().fit(raw[:3])
    pca = PCA(n_components=2).fit(variance.transform(raw[:3]))
    scaler = MinMaxScaler().fit(dataset._targets[:2].numpy())
    prepare_dataset(Subset(dataset, [0, 1]), variance, pca, scaler)
    before = dataset._fp_prepared
    expected = torch.tensor(pca.transform(variance.transform(raw)), dtype=datatype)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert torch.equal(batch.fp_prepared, expected)
    assert torch.equal(batch.y_prepared, torch.tensor(scaler.transform(batch.y.numpy()), dtype=datatype))
    assert batch.y.flatten().tolist() == [100000.125, 100000.25, 100000.5, 100001.0]
    # Identical fitted contents reuse storage, even if the transformer was unpickled.
    prepare_dataset(dataset, pickle.loads(pickle.dumps(variance)), pickle.loads(pickle.dumps(pca)), scaler)
    assert dataset._fp_prepared is before
    pca.components_ *= 2
    prepare_dataset(dataset, variance, pca, scaler)
    assert dataset._fp_prepared is not before
    assert torch.equal(dataset[0].fp_prepared, torch.tensor(pca.transform(variance.transform(raw[:1])), dtype=datatype))
    targets_before = dataset._y_prepared
    scaler.fit(dataset._targets.numpy())
    prepare_dataset(dataset, variance, pca, scaler)
    assert dataset._y_prepared is not targets_before
    assert dataset[3].y_prepared.item() == 1
    prepare_dataset(dataset)
    assert "fp_prepared" not in dataset[0] and "y_prepared" not in dataset[0]


def test_graph_only_and_3d_scaled_targets_keep_precision(prepared_files):
    dataset = make_dataset(prepared_files, is_3d=True, datatype=torch.bfloat16)
    scaler = MinMaxScaler().fit(dataset._targets[:2].numpy())
    prepare_dataset(dataset, output_scaler=scaler)
    assert dataset[0].y.item() == 100000.125
    assert dataset[0].y_prepared.item() == 0
    assert dataset[1].y_prepared.item() == 1
    assert dataset[1].y_prepared.dtype == torch.bfloat16
    dataset.clear_cache()
    assert dataset._y_prepared is None


def test_prepared_output_dtype_can_follow_new_model_precision(prepared_files):
    dataset = make_dataset(prepared_files, datatype=torch.float16)
    raw = dataset.fingerprint_values.astype(np.float64)
    variance = VarianceThreshold().fit(raw[:3])
    pca = PCA(n_components=2).fit(variance.transform(raw[:3]))
    scaler = MinMaxScaler().fit(dataset._targets[:2].numpy())
    prepare_dataset(dataset, variance, pca, scaler)
    old_prepared = dataset._fp_prepared
    prepare_dataset(dataset, variance, pca, scaler, datatype=torch.float64)
    assert dataset._fp_prepared is not old_prepared
    assert dataset[0].fp.dtype == torch.float16
    assert dataset[0].fp_prepared.dtype == torch.float64
    assert dataset[0].y_prepared.dtype == torch.float64
    expected = pca.transform(variance.transform(raw[:1]))
    np.testing.assert_allclose(dataset[0].fp_prepared.numpy(), expected, rtol=0, atol=0)


def test_inplace_scaler_cannot_modify_raw_targets(prepared_files):
    dataset = make_dataset(prepared_files)
    original = dataset._targets.clone()
    scaler = MinMaxScaler(copy=False).fit(original.numpy())
    prepare_dataset(dataset, output_scaler=scaler)
    assert torch.equal(dataset._targets, original)
    assert dataset[0].y_prepared.item() == 0


def test_feature_dimensions_read_graph_only_once_even_without_cache(prepared_files, monkeypatch):
    dataset = make_dataset(prepared_files, graph_cache_bytes=0)
    original_load = torch.load
    reads = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: reads.append(args[0]) or original_load(*args, **kwargs))
    for _ in range(3):
        assert dataset.num_node_features == 2
        assert dataset.num_edge_features == 1
    assert len(reads) == 1


def test_prepare_generic_datasets_keeps_integration_fallbacks():
    rows = [Data(y=torch.tensor([[1.]]))]
    assert prepare_dataset(rows) is rows


def test_changed_target_schema_cannot_silently_reuse_prepared_values(prepared_files):
    dataset = make_dataset(prepared_files)
    dataset.df["new_target"] = 1
    with pytest.raises(ValueError, match="output shape"):
        prepare_dataset(dataset)


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_cache_budget_is_rejected(prepared_files, budget):
    with pytest.raises(ValueError, match="cache size"):
        make_dataset(prepared_files, graph_cache_bytes=budget)
