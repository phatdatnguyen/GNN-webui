"""Tests for the dataset preparation and feature extraction helpers in utils.py."""
import os

import pandas as pd
import pytest
import torch

import utils
from conftest import SMILES


# --------------------------------------------------------------------------- sanitising

@pytest.mark.parametrize("name", ["wd", "dataset", "my-run_2", "run 3", "a.b", "graphs"])
def test_sanitize_accepts_plain_names(name):
    assert utils.sanitize_path_component(name) == name


def test_sanitize_strips_surrounding_whitespace():
    assert utils.sanitize_path_component("  wd  ") == "wd"


@pytest.mark.parametrize("name", [
    "..", ".", "", "   ", None,
    "../escape", "../../escape", "/etc", "/etc/passwd", "a/b", "a\\b",
    "sub/dir", "~/x", "name\x00null", "wd;rm -rf /", "$(whoami)",
])
def test_sanitize_rejects_unsafe_names(name):
    """Rejecting rather than coercing: silently turning '/etc' into 'etc' would write
    the user's data somewhere they never asked for."""
    with pytest.raises(ValueError):
        utils.sanitize_path_component(name, "working directory name")


def test_sanitize_error_mentions_the_label():
    with pytest.raises(ValueError, match="checkpoint name"):
        utils.sanitize_path_component("../x", "checkpoint name")


# ------------------------------------------------------------------------ graph listing

def test_list_graph_files_orders_numerically(tmp_path):
    for i in [0, 1, 2, 10, 11]:
        (tmp_path / f"{i}.pt").write_text("x")
    assert utils.list_graph_files(str(tmp_path)) == [0, 1, 2, 10, 11]


def test_list_graph_files_ignores_unrelated_entries(tmp_path):
    (tmp_path / "0.pt").write_text("x")
    (tmp_path / "1.pt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "checkpoint.ckpt").write_text("x")
    (tmp_path / "backup.pt.bak").write_text("x")
    (tmp_path / "graph_a.pt").write_text("x")
    (tmp_path / "subdir").mkdir()
    assert utils.list_graph_files(str(tmp_path)) == [0, 1]


def test_list_graph_files_on_missing_directory(tmp_path):
    assert utils.list_graph_files(str(tmp_path / "nope")) == []


# --------------------------------------------------------------- process_dataset_file

def _source(tmp_path, smiles, targets, column="target"):
    path = tmp_path / "source.csv"
    pd.DataFrame({"smiles": smiles, column: targets}).to_csv(path, index=False)
    return str(path)


def test_process_dataset_file_writes_named_csv_with_standard_column(tmp_path):
    source = _source(tmp_path, ["CCO", "CCN"], [1, 0])
    status, df = utils.process_dataset_file(source, str(tmp_path), "mydata", "smiles", "target")

    assert "color: green" in status
    assert list(df.columns) == ["SMILES", "target"]
    written = pd.read_csv(tmp_path / "mydata.csv")
    assert list(written.columns) == ["SMILES", "target"]
    assert len(written) == 2


def test_process_dataset_file_drops_invalid_and_carbon_free_rows(tmp_path):
    source = _source(tmp_path, ["CCO", "NOT_A_SMILES", "O", "CCN"], [1, 1, 0, 0])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")

    assert "color: green" in status
    assert df["SMILES"].tolist() == ["CCO", "CCN"]


def test_process_dataset_file_keeps_duplicate_smiles(tmp_path):
    """Duplicates are legitimate data (e.g. replicate measurements) and must survive."""
    source = _source(tmp_path, ["CCO", "CCO", "CCN"], [1, 0, 1])
    _, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert df["SMILES"].tolist() == ["CCO", "CCO", "CCN"]


def test_process_dataset_file_drops_rows_with_missing_values(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "target": [1, None, 0]}).to_csv(path, index=False)
    _, df = utils.process_dataset_file(str(path), str(tmp_path), "dataset", "smiles", "target")
    assert df["SMILES"].tolist() == ["CCO", "CCC"]


def test_process_dataset_file_supports_multiple_targets(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "a": [1, 0], "b": [0, 1]}).to_csv(path, index=False)
    _, df = utils.process_dataset_file(str(path), str(tmp_path), "dataset", "smiles", "a, b")
    assert list(df.columns) == ["SMILES", "a", "b"]


def test_process_dataset_file_allows_no_target_column(tmp_path):
    """Prediction-only datasets have SMILES and nothing else."""
    source = _source(tmp_path, ["CCO", "CCN"], [1, 0])
    _, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "")
    assert list(df.columns) == ["SMILES"]


def test_process_dataset_file_rejects_missing_smiles_column(tmp_path):
    source = _source(tmp_path, ["CCO"], [1])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "nope", "target")
    assert "color: red" in status and df is None
    assert "nope" in status


def test_process_dataset_file_rejects_missing_target_column(tmp_path):
    source = _source(tmp_path, ["CCO"], [1])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "missing")
    assert "color: red" in status and df is None


def test_process_dataset_file_rejects_multiple_targets_for_3d(tmp_path):
    """The 3D models have a single output head."""
    path = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["CCO"], "a": [1], "b": [0]}).to_csv(path, index=False)
    status, df = utils.process_dataset_file(str(path), str(tmp_path), "dataset", "smiles", "a,b", True)
    assert "color: red" in status and df is None


def test_process_dataset_file_rejects_unsafe_dataset_name(tmp_path):
    source = _source(tmp_path, ["CCO"], [1])
    status, df = utils.process_dataset_file(source, str(tmp_path), "../escape", "smiles", "target")
    assert "color: red" in status and df is None
    assert not (tmp_path.parent / "escape.csv").exists()


# ---------------------------------------------------------------------- extract_graphs

def test_extract_graphs_writes_one_file_per_molecule(classification_workdir):
    assert utils.list_graph_files(os.path.join(classification_workdir, "graphs")) == list(range(len(SMILES)))


def test_extract_graphs_stores_edge_index_as_long(classification_workdir):
    """Regression: edge_index used to be saved with the model dtype. Under float16 that
    silently corrupts every node index above 2048 once batches are collated."""
    data = torch.load(os.path.join(classification_workdir, "graphs", "0.pt"), weights_only=False)
    assert data.edge_index.dtype == torch.int64


@pytest.mark.parametrize("datatype", [torch.float16, torch.float32, torch.float64])
def test_extract_graphs_feature_dtype_follows_setting_but_edge_index_does_not(
        tmp_path, datatype, progress):
    source = _source(tmp_path, ["CCO", "c1ccccc1"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    status = utils.extract_graphs(str(tmp_path), "dataset.csv", "graphs",
                                  "MolGraphConvFeaturizer", datatype, progress)
    assert "color: green" in status

    data = torch.load(os.path.join(tmp_path, "graphs", "0.pt"), weights_only=False)
    assert data.x.dtype == datatype
    assert data.edge_attr.dtype == datatype
    assert data.edge_index.dtype == torch.int64


@pytest.mark.parametrize("featurizer", [
    "MolGraphConvFeaturizer", "PagtnMolGraphFeaturizer", "DMPNNFeaturizer",
])
def test_every_offered_featurizer_works(tmp_path, featurizer, progress):
    """Regression: two of the three dropdown options did not exist in deepchem < 2.6."""
    source = _source(tmp_path, ["CCO", "c1ccccc1"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    status = utils.extract_graphs(str(tmp_path), "dataset.csv", featurizer,
                                  featurizer, torch.float32, progress)
    assert "color: green" in status, status
    assert utils.list_graph_files(os.path.join(tmp_path, featurizer)) == [0, 1]


def test_extract_graphs_records_the_source_smiles(classification_workdir):
    data = torch.load(os.path.join(classification_workdir, "graphs", "0.pt"), weights_only=False)
    assert data.smiles == SMILES[0]


def test_extract_graphs_rejects_unsafe_directory_name(tmp_path, progress):
    """This path is handed to shutil.rmtree, so traversal here is destructive."""
    source = _source(tmp_path, ["CCO"], [1])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    victim = tmp_path.parent / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("do not delete")

    status = utils.extract_graphs(str(tmp_path), "dataset.csv", "../victim",
                                  "MolGraphConvFeaturizer", torch.float32, progress)
    assert "color: red" in status
    assert (victim / "precious.txt").exists()


def test_extract_graphs_replaces_previous_output(tmp_path, progress):
    source = _source(tmp_path, ["CCO", "CCN", "CCC"], [1, 0, 1])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    utils.extract_graphs(str(tmp_path), "dataset.csv", "graphs",
                         "MolGraphConvFeaturizer", torch.float32, progress)
    stale = tmp_path / "graphs" / "99.pt"
    stale.write_text("stale")

    utils.extract_graphs(str(tmp_path), "dataset.csv", "graphs",
                         "MolGraphConvFeaturizer", torch.float32, progress)
    assert not stale.exists()
    assert utils.list_graph_files(os.path.join(tmp_path, "graphs")) == [0, 1, 2]


# --------------------------------------------------------- extract_molecule_fingerprints

@pytest.mark.parametrize("fingerprint", [
    "Morgan fingerprint", "Atom-pair fingerprint", "Topological-torsion fingerprint",
    "Avalon fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint",
])
def test_every_offered_fingerprint_works(tmp_path, fingerprint, progress):
    source = _source(tmp_path, ["CCO", "c1ccccc1"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    status = utils.extract_molecule_fingerprints(str(tmp_path), "dataset.csv", "fps",
                                                 fingerprint, 2, 64, progress)
    assert "color: green" in status, status

    df = pd.read_csv(os.path.join(tmp_path, "fps", "molecular_fingerprints.csv"))
    assert df.shape == (2, 65)                       # SMILES + 64 bits
    assert df.columns[0] == "SMILES"
    assert df["SMILES"].tolist() == ["CCO", "c1ccccc1"]


def test_fingerprint_bit_width_follows_setting(tmp_path, progress):
    source = _source(tmp_path, ["CCO"], [1])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    utils.extract_molecule_fingerprints(str(tmp_path), "dataset.csv", "fps",
                                        "Morgan fingerprint", 2, 256, progress)
    df = pd.read_csv(os.path.join(tmp_path, "fps", "molecular_fingerprints.csv"))
    assert df.shape[1] - 1 == 256


def test_extract_fingerprints_rejects_unsafe_directory_name(tmp_path, progress):
    source = _source(tmp_path, ["CCO"], [1])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    victim = tmp_path.parent / "victim_fp"
    victim.mkdir()
    (victim / "precious.txt").write_text("do not delete")

    status = utils.extract_molecule_fingerprints(str(tmp_path), "dataset.csv", "../victim_fp",
                                                 "Morgan fingerprint", 2, 64, progress)
    assert "color: red" in status
    assert (victim / "precious.txt").exists()


# ------------------------------------------------------------------- 3D graph extraction

@pytest.mark.slow
def test_extract_3d_graphs_stores_integral_atomic_numbers(workdir_3d):
    """Regression: z indexes an embedding table, so it must not be stored as a float."""
    data = torch.load(os.path.join(workdir_3d, "g3d", "0.pt"), weights_only=False)
    assert data.z.dtype == torch.int64
    assert data.pos.dtype == torch.float32
    assert data.pos.shape == (data.z.shape[0], 3)


@pytest.mark.slow
def test_extract_3d_graphs_covers_every_molecule(workdir_3d):
    from conftest import SMILES_3D
    assert utils.list_graph_files(os.path.join(workdir_3d, "g3d")) == list(range(len(SMILES_3D)))


@pytest.mark.slow
@pytest.mark.parametrize("force_field", ["MMFF", "UFF"])
def test_both_force_fields_work(tmp_path, force_field, progress):
    source = _source(tmp_path, ["CCO", "CCN"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target", True)
    status = utils.extract_3d_graphs(str(tmp_path), "dataset.csv", "g3d", 2,
                                     force_field, torch.float32, progress)
    assert "color: green" in status, status
    assert utils.list_graph_files(os.path.join(tmp_path, "g3d")) == [0, 1]
