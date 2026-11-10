"""Batched fetch must preserve molecule order, values, and sample ownership."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data

from datasets import (MolecularDataset, MolecularDataset3D, _copy_graph,
                      _gather_fingerprints, _gather_tensor_rows, prepare_dataset)
from gnn_runtime import make_loader
from utils import write_graph_manifest


@pytest.fixture
def batch_files(tmp_path):
    smiles = ["CCO", "CCN", "CCC", "CCS"]
    pd.DataFrame({"SMILES": smiles, "target": [100000.125, 100000.25, 100000.5, 100001.0]}).to_csv(
        tmp_path / "dataset.csv", index=False)
    graphs, fps = tmp_path / "GRAPH_graphs", tmp_path / "FP_fps"
    graphs.mkdir()
    fps.mkdir()
    for i, molecule in enumerate(smiles):
        x = torch.arange(2 * (i + 2), dtype=torch.float32).reshape(-1, 2) + 1
        torch.save(Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]),
                        edge_attr=torch.ones(2, 1), z=torch.full((i + 2,), 6),
                        pos=torch.arange(3 * (i + 2), dtype=torch.float32).reshape(-1, 3),
                        smiles=molecule, metadata={"labels": ["original"], "alias": x}),
                   graphs / f"{i}.pt")
    write_graph_manifest(graphs, smiles)
    pd.DataFrame({"SMILES": smiles, "Bit_1": [0, 1, 1, 0],
                  "Bit_2": [1, 0, 1, 0], "Bit_3": [1, 1, 1, 1]}).to_csv(
        fps / "molecular_fingerprints.csv", index=False)
    return tmp_path


def make_dataset(path, mode="Both", datatype=torch.float32):
    args = [str(path / "dataset.csv")]
    kwargs = dict(target_datatype=torch.float64)
    if mode == "3D":
        return MolecularDataset3D(*args, str(path / "GRAPH_graphs"), datatype, **kwargs)
    return MolecularDataset(*args, mode, str(path / "GRAPH_graphs"), str(path / "FP_fps"),
                            datatype, **kwargs)


def assert_values_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_values_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            assert_values_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("mode", ["Graphs", "Both", "Molecular fingerprints", "3D"])
def test_batched_rows_match_individual_rows_and_preserve_graph_offsets(batch_files, mode):
    dataset = make_dataset(batch_files, mode)
    indices = [3, 0, 0, 2]
    expected = Batch.from_data_list([dataset[i] for i in indices])
    actual = Batch.from_data_list(dataset.__getitems__(indices))
    assert_values_equal(actual.to_dict(), expected.to_dict())
    assert actual.num_graphs == 4
    assert actual.y.dtype == torch.float64


def test_batched_samples_cannot_mutate_cached_or_repeated_rows(batch_files):
    dataset = make_dataset(batch_files, datatype=torch.bfloat16)
    variance = VarianceThreshold().fit(dataset.fingerprint_values[:3])
    pca = PCA(n_components=2).fit(variance.transform(dataset.fingerprint_values[:3]))
    scaler = MinMaxScaler().fit(dataset._targets[:3].numpy())
    prepare_dataset(dataset, variance, pca, scaler)
    rows = dataset.__getitems__([0, 0, 3])
    expected = dataset[0]
    # Internal aliases are preserved within one copy, without aliasing the cache.
    assert rows[0].metadata["alias"] is rows[0].x
    for field in ("x", "edge_index", "edge_attr", "z", "pos", "fp", "y", "fp_prepared", "y_prepared"):
        rows[0][field].zero_()
        torch.testing.assert_close(rows[1][field], expected[field], rtol=0, atol=0)
        torch.testing.assert_close(dataset[0][field], expected[field], rtol=0, atol=0)
    rows[0].metadata["labels"].append("changed")
    assert rows[1].metadata["labels"] == dataset[0].metadata["labels"] == ["original"]
    assert rows[1].y.item() == 100000.125
    assert rows[1].fp.dtype == rows[1].fp_prepared.dtype == torch.bfloat16


def test_nested_subsets_use_batched_fetch_and_keep_seeded_order(batch_files, monkeypatch):
    dataset = make_dataset(batch_files)
    subset = Subset(Subset(dataset, [3, 1, 0, 2]), [2, 0, 2, 3])
    expected = [dataset._smiles[i] for i in [0, 3, 0, 2]]
    # The optimized loader path must not silently return to per-row allocation.
    monkeypatch.setattr(MolecularDataset, "__getitem__", lambda *args: pytest.fail("per-row fetch"))
    assert [smiles for batch in make_loader(subset, 3) for smiles in batch.smiles] == expected

    def orders():
        loader = make_loader(subset, 3, shuffle=True, seed=19)
        return [[smiles for batch in loader for smiles in batch.smiles] for _ in range(2)]

    assert orders() == orders()


def test_reused_loader_observes_new_preprocessing_and_spawn_copy(batch_files):
    dataset = make_dataset(batch_files)
    variance = VarianceThreshold().fit(dataset.fingerprint_values[:3])
    pca = PCA(n_components=2).fit(variance.transform(dataset.fingerprint_values[:3]))
    loader = make_loader(dataset, 4)
    prepare_dataset(dataset, variance, pca)
    original = next(iter(loader)).fp_prepared.clone()
    pca.components_ *= -1
    prepare_dataset(dataset, variance, pca)
    torch.testing.assert_close(next(iter(loader)).fp_prepared, -original)
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.graph_cache_info["entries"] == 0
    assert_values_equal(next(iter(make_loader(restored, 4))).to_dict(), next(iter(loader)).to_dict())
    prepare_dataset(dataset)
    assert "fp_prepared" not in next(iter(loader))


def test_unlabeled_and_nonbinary_batches_keep_existing_values(batch_files):
    path = batch_files / "dataset.csv"
    pd.read_csv(path)[["SMILES"]].to_csv(path, index=False)
    fp_path = batch_files / "FP_fps" / "molecular_fingerprints.csv"
    fp = pd.read_csv(fp_path)
    fp["Bit_1"] = [0.5, 2.9, -1.1, 2049]
    fp.to_csv(fp_path, index=False)
    dataset = make_dataset(batch_files, "Molecular fingerprints", torch.float64)
    batch = next(iter(make_loader(Subset(dataset, [3, 1, 0]), 3)))
    assert batch.y is None
    assert batch.fp[:, 0].tolist() == [2049, 2, 0]
    assert dataset.__getitems__([]) == []
    torch.testing.assert_close(dataset.__getitems__([-1])[0].fp, dataset[3].fp)
    with pytest.raises(IndexError):
        dataset.__getitems__([len(dataset)])


def test_graph_copy_preserves_storage_aliases_and_autograd_copy_rules():
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    graph = Data(x=source, metadata={"view": source[1:]})
    copied = _copy_graph(graph)
    copied.x[1, 0] = -100
    assert copied.metadata["view"][0, 0].item() == -100
    assert source[1, 0].item() == 3
    top_level_views = _copy_graph(Data(x=source, alternate=source[1:]))
    top_level_views.x[1, 0] = -200
    assert top_level_views.alternate[0, 0].item() == -200
    leaf = torch.ones(2, requires_grad=True)
    leaf.grad = torch.full_like(leaf, 2)
    copied_leaf = _copy_graph(Data(x=leaf)).x
    assert copied_leaf.is_leaf and copied_leaf.requires_grad
    torch.testing.assert_close(copied_leaf.grad, leaf.grad)
    with pytest.raises(RuntimeError, match="deepcopy"):
        _copy_graph(Data(x=leaf * 2))


@pytest.mark.parametrize("datatype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
def test_large_batch_casts_and_gathers_keep_exact_values_without_storage_aliases(datatype):
    values = np.random.default_rng(31).integers(-100000, 100000, (80, 1024), dtype=np.int64)
    indices = [3, 7, 3, 0] * 16
    rows = torch.tensor(indices)
    expected = torch.from_numpy(values[indices]).to(datatype)
    actual = _gather_fingerprints(values, indices, datatype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    source = torch.linspace(-.5, .5, 80 * 1024, dtype=datatype).reshape(80, 1024)
    gathered = _gather_tensor_rows(source, rows)
    torch.testing.assert_close(gathered, source.index_select(0, rows), rtol=0, atol=0)
    gathered[0].zero_()
    torch.testing.assert_close(gathered[2], source[3], rtol=0, atol=0)
    assert torch.count_nonzero(source[3]) > 0
