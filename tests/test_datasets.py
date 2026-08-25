"""Tests for MolecularDataset / MolecularDataset3D."""
import os

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from datasets import MolecularDataset, MolecularDataset3D
from conftest import SMILES


def make_dataset(workdir, load_data="Both", datatype=torch.float32):
    return MolecularDataset(
        os.path.join(workdir, "dataset.csv"), load_data,
        os.path.join(workdir, "graphs"), os.path.join(workdir, "fps"), datatype,
    )


# ------------------------------------------------------------------------------- length

@pytest.mark.parametrize("load_data", ["Graphs", "Molecular fingerprints", "Both"])
def test_length_matches_the_dataset_file(classification_workdir, load_data):
    assert len(make_dataset(classification_workdir, load_data)) == len(SMILES)


def test_length_ignores_unrelated_files_in_the_graph_directory(classification_workdir):
    """Regression: __len__ used to count directory entries, so a stray file made the
    dataset report one item too many and indexing it raised FileNotFoundError."""
    dataset = make_dataset(classification_workdir)
    before = len(dataset)

    graph_dir = os.path.join(classification_workdir, "graphs")
    open(os.path.join(graph_dir, "notes.txt"), "w").write("scratch")
    os.makedirs(os.path.join(graph_dir, "subdir"), exist_ok=True)

    assert len(dataset) == before
    assert dataset[len(dataset) - 1] is not None


def test_length_does_not_rescan_the_directory_on_every_call(classification_workdir, monkeypatch):
    """DataLoader calls len() every epoch; it must not cost a syscall each time."""
    dataset = make_dataset(classification_workdir)
    import datasets as datasets_module

    calls = []
    monkeypatch.setattr(datasets_module, "list_graph_files",
                        lambda p: calls.append(p) or list(range(len(SMILES))))
    for _ in range(50):
        len(dataset)
    assert calls == []


# ------------------------------------------------------------------------------ getitem

def test_getitem_graphs_mode(classification_workdir):
    dataset = make_dataset(classification_workdir, "Graphs")
    item = dataset[0]
    assert item.x.shape[0] > 0
    assert item.edge_index.dtype == torch.int64
    assert item.edge_index.shape[0] == 2
    assert item.smiles == SMILES[0]
    assert item.y.shape == (1, 1)
    assert not hasattr(item, "fp") or item.fp is None


def test_getitem_fingerprints_mode(classification_workdir):
    dataset = make_dataset(classification_workdir, "Molecular fingerprints")
    item = dataset[0]
    assert item.fp.shape == (1, 128)
    assert item.fp.dtype == torch.float32
    assert item.smiles == SMILES[0]
    assert item.y.shape == (1, 1)


def test_getitem_both_mode(classification_workdir):
    dataset = make_dataset(classification_workdir, "Both")
    item = dataset[0]
    assert item.x.shape[0] > 0
    assert item.fp.shape == (1, 128)


def test_getitem_respects_the_datatype(classification_workdir):
    dataset = make_dataset(classification_workdir, "Both", torch.float64)
    item = dataset[0]
    assert item.fp.dtype == torch.float64
    assert item.y.dtype == torch.float64


def test_targets_are_none_when_the_dataset_has_no_target_column(classification_workdir):
    """Prediction datasets carry SMILES only."""
    path = os.path.join(classification_workdir, "dataset.csv")
    pd.read_csv(path)[["SMILES"]].to_csv(path, index=False)

    dataset = make_dataset(classification_workdir, "Graphs")
    assert dataset[0].y is None
    assert dataset.num_outputs == 0


def test_multi_target_y_shape(tmp_path, progress):
    import utils
    source = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "a": [1, 0], "b": [0, 1]}).to_csv(source, index=False)
    utils.process_dataset_file(str(source), str(tmp_path), "dataset", "smiles", "a,b")
    utils.extract_graphs(str(tmp_path), "dataset.csv", "graphs",
                         "MolGraphConvFeaturizer", torch.float32, progress)

    dataset = MolecularDataset(str(tmp_path / "dataset.csv"), "Graphs",
                               str(tmp_path / "graphs"), "", torch.float32)
    assert dataset.num_outputs == 2
    assert dataset[0].y.shape == (1, 2)
    assert dataset.target_column_names == ["a", "b"]


def test_legacy_float_edge_index_is_cast_on_load(classification_workdir):
    """Graphs written before the dtype fix must still load."""
    path = os.path.join(classification_workdir, "graphs", "0.pt")
    data = torch.load(path, weights_only=False)
    data.edge_index = data.edge_index.float()
    torch.save(data, path)

    dataset = make_dataset(classification_workdir, "Graphs")
    assert dataset[0].edge_index.dtype == torch.int64


# --------------------------------------------------------------------------- properties

def test_feature_dimension_properties(classification_workdir):
    both = make_dataset(classification_workdir, "Both")
    assert both.num_node_features > 0
    assert both.num_edge_features > 0
    assert both.num_fingerprint_bits == 128
    assert both.num_outputs == 1

    fp_only = make_dataset(classification_workdir, "Molecular fingerprints")
    assert fp_only.num_node_features == 0
    assert fp_only.num_edge_features == 0
    assert fp_only.num_fingerprint_bits == 128

    graphs_only = make_dataset(classification_workdir, "Graphs")
    assert graphs_only.num_fingerprint_bits == 0


# ------------------------------------------------------------------- alignment checking

def test_rejects_fewer_graphs_than_dataset_rows(classification_workdir):
    """Regression: re-running one extractor after the other can drop rows from the CSV and
    leave the other artifact stale. That used to go unnoticed."""
    path = os.path.join(classification_workdir, "dataset.csv")
    pd.read_csv(path).drop(index=3).to_csv(path, index=False)

    with pytest.raises(ValueError, match="graph"):
        make_dataset(classification_workdir, "Graphs")


def test_rejects_fingerprints_shifted_against_the_dataset(classification_workdir):
    """Same row count, wrong order - a count check alone would pass this."""
    fp_path = os.path.join(classification_workdir, "fps", "molecular_fingerprints.csv")
    fp = pd.read_csv(fp_path)
    pd.concat([fp.iloc[3:], fp.iloc[:3]]).reset_index(drop=True).to_csv(fp_path, index=False)

    with pytest.raises(ValueError, match="out of sync|does not match"):
        make_dataset(classification_workdir, "Molecular fingerprints")


def test_rejects_graphs_shifted_against_the_dataset(classification_workdir):
    """Rotate the CSV rows; counts still match but graph i no longer describes row i."""
    path = os.path.join(classification_workdir, "dataset.csv")
    df = pd.read_csv(path)
    pd.concat([df.iloc[5:], df.iloc[:5]]).reset_index(drop=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match="out of sync"):
        make_dataset(classification_workdir, "Graphs")


def test_manifest_catches_a_shift_confined_to_the_middle(classification_workdir):
    """The old spot check only sampled the first, middle and last graph. The manifest lets
    the whole ordering be verified for the cost of one small file read."""
    path = os.path.join(classification_workdir, "dataset.csv")
    df = pd.read_csv(path)
    swapped = df.copy()
    swapped.iloc[3], swapped.iloc[4] = df.iloc[4].copy(), df.iloc[3].copy()
    swapped.to_csv(path, index=False)

    with pytest.raises(ValueError, match="out of sync"):
        make_dataset(classification_workdir, "Graphs")


def test_extraction_writes_a_manifest(classification_workdir):
    import utils
    manifest = utils.read_graph_manifest(os.path.join(classification_workdir, "graphs"))
    assert manifest == SMILES


def test_alignment_check_falls_back_when_there_is_no_manifest(classification_workdir):
    """Graph directories written before the manifest existed must still load."""
    import utils
    os.remove(os.path.join(classification_workdir, "graphs", utils.GRAPH_MANIFEST))
    assert len(make_dataset(classification_workdir, "Graphs")) == len(SMILES)

    path = os.path.join(classification_workdir, "dataset.csv")
    df = pd.read_csv(path)
    pd.concat([df.iloc[5:], df.iloc[:5]]).reset_index(drop=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="out of sync"):
        make_dataset(classification_workdir, "Graphs")


def test_rejects_a_gap_in_the_graph_numbering(classification_workdir):
    os.remove(os.path.join(classification_workdir, "graphs", "4.pt"))
    with pytest.raises(ValueError):
        make_dataset(classification_workdir, "Graphs")


def test_rejects_an_empty_graph_directory(empty_workdir, classification_workdir):
    os.makedirs(os.path.join(classification_workdir, "no_graphs"), exist_ok=True)
    with pytest.raises(ValueError, match="No graph files"):
        MolecularDataset(os.path.join(classification_workdir, "dataset.csv"), "Graphs",
                         os.path.join(classification_workdir, "no_graphs"), "", torch.float32)


# ------------------------------------------------------------------------------ batching

def test_batches_collate_with_correct_node_offsets(classification_workdir):
    dataset = make_dataset(classification_workdir, "Both")
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    assert batch.edge_index.dtype == torch.int64
    assert int(batch.edge_index.max()) < batch.num_nodes
    assert batch.y.shape == (len(dataset), 1)
    assert batch.fp.shape == (len(dataset), 128)
    assert len(batch.smiles) == len(dataset)


def test_batch_size_one_still_produces_a_batch_dimension(classification_workdir):
    dataset = make_dataset(classification_workdir, "Both")
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch.y.shape == (1, 1)


# ---------------------------------------------------------------------------- 3D dataset

@pytest.mark.slow
def test_3d_dataset_basics(workdir_3d):
    from conftest import SMILES_3D
    dataset = MolecularDataset3D(os.path.join(workdir_3d, "dataset.csv"),
                                 os.path.join(workdir_3d, "g3d"), torch.float32)
    assert len(dataset) == len(SMILES_3D)
    assert dataset.num_outputs == 1
    assert dataset.target_column_name == "target"

    item = dataset[0]
    assert item.z.dtype == torch.int64
    assert item.pos.shape == (item.z.shape[0], 3)
    assert item.y.shape == (1, 1)


@pytest.mark.slow
def test_3d_dataset_detects_desync(workdir_3d):
    path = os.path.join(workdir_3d, "dataset.csv")
    pd.read_csv(path).drop(index=2).to_csv(path, index=False)
    with pytest.raises(ValueError):
        MolecularDataset3D(path, os.path.join(workdir_3d, "g3d"), torch.float32)


@pytest.mark.slow
def test_3d_legacy_float_z_is_cast_on_load(workdir_3d):
    path = os.path.join(workdir_3d, "g3d", "0.pt")
    data = torch.load(path, weights_only=False)
    data.z = data.z.float()
    torch.save(data, path)

    dataset = MolecularDataset3D(os.path.join(workdir_3d, "dataset.csv"),
                                 os.path.join(workdir_3d, "g3d"), torch.float32)
    assert dataset[0].z.dtype == torch.int64
