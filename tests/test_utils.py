"""Tests for the dataset preparation and feature extraction helpers in utils.py."""
import os
import json
from itertools import product

import pandas as pd
import pytest
import torch

import utils
from conftest import SMILES


# --------------------------------------------------------------------------- sanitising

@pytest.mark.parametrize("name", ["wd", "dataset", "my-run_2", "run 3", "a.b", "GRAPH_graphs"])
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


@pytest.mark.parametrize("name", ["metrics", "metrics.csv", "metrics.CSV"])
def test_result_export_writes_csv_without_an_extra_index_or_extension(tmp_path, name):
    frame = pd.DataFrame({"Target": ["assay"], "Metric": ["ROC-AUC score"], "Value": [0.875]})
    filename = utils.export_result_csv(str(tmp_path), name, frame)
    expected = name if name.lower().endswith(".csv") else name + ".csv"
    assert filename == expected
    assert [path.name for path in tmp_path.iterdir()] == [expected]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / expected), frame)


@pytest.mark.parametrize("frame", [None, pd.DataFrame(), pd.DataFrame(columns=["SMILES"])])
def test_result_export_rejects_absent_rows(tmp_path, frame):
    with pytest.raises(ValueError, match="produce results"):
        utils.export_result_csv(str(tmp_path), "evaluation", frame)
    assert not list(tmp_path.iterdir())


def test_result_export_requires_open_directory(tmp_path):
    frame = pd.DataFrame({"target": [1]})
    for directory in (None, str(tmp_path / "missing")):
        with pytest.raises(ValueError, match="working directory"):
            utils.export_result_csv(directory, "results", frame)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "..", "", "<bad>", "<unsafe>/file"])
def test_result_export_rejects_unsafe_filenames(tmp_path, name):
    # Exhaustive filename variants belong to the shared writer. Workflow tests
    # separately verify that each handler turns its failures into safe UI updates.
    with pytest.raises(ValueError, match="export file name"):
        utils.export_result_csv(str(tmp_path), name, pd.DataFrame({"target": [1]}))
    assert not list(tmp_path.iterdir())


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
    (tmp_path / "2.pt").mkdir()
    (tmp_path / "03.pt").write_text("not a canonical shard name")
    (tmp_path / "².pt").write_text("not an integer filename")
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
    """Replicate measurements remain available when duplicate removal is disabled."""
    source = _source(tmp_path, ["CCO", "CCO", "CCN"], [1, 0, 1])
    _, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target",
                                        remove_duplicates=False)
    assert df["SMILES"].tolist() == ["CCO", "CCO", "CCN"]
    assert df["target"].tolist() == [1, 0, 1]


@pytest.mark.parametrize("canonicalize_smiles,remove_duplicates,remove_salt", product([False, True], repeat=3))
def test_dataset_cleanup_options_are_independent(tmp_path, canonicalize_smiles, remove_duplicates, remove_salt):
    original = ["OCC", "CCO", "OCC.[Na+]", "CCN", "NCC", "CCN.Cl"]
    source = _source(tmp_path, original, [10, 11, 12, 20, 21, 22])
    status, df = utils.process_dataset_file(
        source, str(tmp_path), "dataset", "smiles", "target",
        canonicalize_smiles=canonicalize_smiles, remove_duplicates=remove_duplicates,
        remove_salt=remove_salt,
    )
    assert "color: green" in status, status
    expected = (["CCO", "CCO", "CCO", "CCN", "CCN", "CCN"] if remove_salt
                else ["CCO", "CCO", "CCO.[Na+]", "CCN", "CCN", "CCN.Cl"])
    if not canonicalize_smiles:
        expected = ["OCC", "CCO", "OCC", "CCN", "NCC", "CCN"] if remove_salt else original
    kept = ([0, 3] if remove_salt else [0, 2, 3, 5]) if remove_duplicates else list(range(6))
    assert df["SMILES"].tolist() == [expected[index] for index in kept]
    assert df["target"].tolist() == [[10, 11, 12, 20, 21, 22][index] for index in kept]
    assert f"Imported {len(kept)} molecules; removed {6 - len(kept)} rows" in status
    assert f"0 missing values, 0 invalid SMILES, 0 without carbon, {6 - len(kept)} duplicates" in status
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "dataset.csv"), df)


def test_default_cleanup_counts_each_removed_source_row_once(tmp_path):
    smiles = [None, " ", "NOT_A_SMILES", "NOT_A_SMILES", "O", "[C]", "OCC", "CCO", "OCC.[Na+]", "CCN"]
    source = _source(tmp_path, smiles, [1, 0, None, 1, 0, 0, 7, 8, 9, 10])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert "Imported 2 molecules; removed 8 rows (3 missing values, 1 invalid SMILES, 2 without carbon, 2 duplicates)." in status
    assert df["SMILES"].tolist() == ["CCO", "CCN"]
    assert df["target"].tolist() == [7, 10]


def test_import_counts_blank_csv_records_as_missing_source_rows(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("smiles,target\nCCO,1\n\nCCN,0\n")
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert "Imported 2 molecules; removed 1 rows (1 missing values, 0 invalid SMILES, 0 without carbon, 0 duplicates)." in status
    assert df["SMILES"].tolist() == ["CCO", "CCN"]


def test_nonstring_and_blank_smiles_are_filtered_per_row(tmp_path, monkeypatch):
    frame = pd.DataFrame({"smiles": [123, False, None, "  ", "CCO"], "target": [1, 0, 0, 1, 7]})
    monkeypatch.setattr(utils.pd, "read_csv", lambda *args, **kwargs: frame.copy())
    status, df = utils.process_dataset_file("source.csv", str(tmp_path), "dataset", "smiles", "target")
    assert "Imported 1 molecules; removed 4 rows (2 missing values, 2 invalid SMILES, 0 without carbon, 0 duplicates)." in status
    assert df["SMILES"].tolist() == ["CCO"] and df["target"].tolist() == [7]


def test_equivalent_salts_keep_first_rows_targets_and_noncanonical_representation(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["OCC.[Na+]", "CCO", "CCN"],
                  "first": [3, 8, 1], "second": [4.5, 9.5, 2.5]}).to_csv(source, index=False)
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "first, second",
                                           canonicalize_smiles=False)
    assert "color: green" in status
    assert df.to_dict("list") == {"SMILES": ["OCC", "CCN"], "first": [3, 1], "second": [4.5, 2.5]}


def test_deduplication_preserves_stereoisomers_and_isotopes(tmp_path):
    smiles = ["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "[13CH3]CO", "CCO", "OCC"]
    source = _source(tmp_path, smiles, [0, 1, 2, 3, 4])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert "Imported 4 molecules; removed 1 rows" in status
    assert df["target"].tolist() == [0, 1, 2, 3]
    assert len(set(df["SMILES"].iloc[:2])) == 2
    assert all("@" in smiles for smiles in df["SMILES"].iloc[:2])
    assert "13" in df["SMILES"].iloc[2] and "13" not in df["SMILES"].iloc[3]


def test_cleanup_does_not_merge_tautomers_or_neutralize_charge(tmp_path):
    smiles = ["CC(=O)C", "CC(O)=C", "CC(=O)O", "CC(=O)[O-]"]
    source = _source(tmp_path, smiles, [0, 1, 2, 3])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert "Imported 4 molecules; removed 0 rows" in status
    assert df["target"].tolist() == [0, 1, 2, 3]
    assert utils.Chem.GetFormalCharge(utils.Chem.MolFromSmiles(df["SMILES"].iloc[3])) == -1


@pytest.mark.parametrize("remove_salt", [False, True])
def test_cleanup_rechecks_carbon_after_removing_organic_salt_fragments(tmp_path, remove_salt):
    source = _source(tmp_path, ["CC(=O)[O-].[Fe+2]", "CCN"], [1, 0])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target",
                                           remove_salt=remove_salt)
    assert "color: green" in status
    assert df["target"].tolist() == ([0] if remove_salt else [1, 0])
    assert f"{int(remove_salt)} without carbon" in status


def test_process_dataset_file_drops_rows_with_missing_values(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "target": [1, None, 0]}).to_csv(path, index=False)
    _, df = utils.process_dataset_file(str(path), str(tmp_path), "dataset", "smiles", "target")
    assert df["SMILES"].tolist() == ["CCO", "CCC"]


@pytest.mark.parametrize("smiles, targets", [(["NOT_A_SMILES", "O"], [1, 0]), (["CCO"], [None]), ([], [])])
def test_empty_upload_does_not_overwrite_existing_dataset(tmp_path, smiles, targets):
    destination = tmp_path / "dataset.csv"
    previous = "SMILES,target\nCCO,1\n"
    destination.write_text(previous)
    source = _source(tmp_path, smiles, targets)

    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")

    assert "color: red" in status and "No valid molecules" in status
    assert f"Imported 0 molecules; removed {len(smiles)} rows" in status
    assert df is None
    assert destination.read_text() == previous


@pytest.mark.parametrize("failure", ["serialize", "publish"])
def test_failed_dataset_publication_preserves_existing_csv(tmp_path, monkeypatch, failure):
    destination = tmp_path / "dataset.csv"
    previous = "SMILES,target\nCCN,17\n"
    destination.write_text(previous)
    source = _source(tmp_path, ["OCC"], [1])

    def fail_serialization(self, stream, *args, **kwargs):
        stream.write("partially written CSV")
        raise OSError("simulated disk failure")

    def fail_publication(*args, **kwargs):
        raise OSError("simulated disk failure")

    if failure == "serialize":
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_serialization)
    else:
        monkeypatch.setattr(utils.os, "replace", fail_publication)
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert "color: red" in status and "simulated disk failure" in status
    assert df is None and destination.read_text() == previous
    assert not list(tmp_path.glob(".dataset-*"))


def test_dataset_import_can_atomically_replace_its_own_source_file(tmp_path):
    source = tmp_path / "dataset.csv"
    pd.DataFrame({"smiles": ["OCC", "CCO"], "target": [1, 0]}).to_csv(source, index=False)
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    assert "color: green" in status
    pd.testing.assert_frame_equal(pd.read_csv(source), df)
    assert df.to_dict("list") == {"SMILES": ["CCO"], "target": [1]}


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


@pytest.mark.parametrize("targets", ["target,target", "target,", ",target", "smiles", "SMILES", ["target"]])
def test_dataset_import_rejects_ambiguous_target_selections_without_replacing_data(tmp_path, targets):
    destination = tmp_path / "dataset.csv"
    previous = "SMILES,target\nCCN,9\n"
    destination.write_text(previous)
    source = _source(tmp_path, ["CCO"], [1])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", targets)
    assert "color: red" in status and df is None
    assert destination.read_text() == previous


def test_dataset_import_status_escapes_user_supplied_column_names(tmp_path):
    source = _source(tmp_path, ["CCO"], [1])
    status, df = utils.process_dataset_file(source, str(tmp_path), "dataset", "<missing>", "target")
    assert "color: red" in status and df is None
    assert "&lt;missing&gt;" in status and "<missing>" not in status


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
    assert utils.list_graph_files(os.path.join(classification_workdir, "GRAPH_graphs")) == list(range(len(SMILES)))


@pytest.mark.parametrize("datatype", [torch.float16, torch.float32, torch.float64])
def test_extract_graphs_feature_dtype_follows_setting_but_edge_index_does_not(
        tmp_path, datatype, progress):
    """Regression: edge_index used to be saved with the model dtype. Under float16 that
    silently corrupts every node index above 2048 once batches are collated."""
    source = _source(tmp_path, ["CCO", "c1ccccc1"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    status = utils.extract_graphs(str(tmp_path), "dataset.csv", "GRAPH_graphs",
                                  "MolGraphConvFeaturizer", datatype, progress)
    assert "color: green" in status

    data = torch.load(os.path.join(tmp_path, "GRAPH_graphs", "0.pt"), weights_only=False)
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
    assert utils.list_graph_files(os.path.join(tmp_path, "GRAPH_" + featurizer)) == [0, 1]


def test_extract_graphs_records_the_source_smiles(classification_workdir):
    data = torch.load(os.path.join(classification_workdir, "GRAPH_graphs", "0.pt"), weights_only=False)
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
    utils.extract_graphs(str(tmp_path), "dataset.csv", "GRAPH_graphs",
                         "MolGraphConvFeaturizer", torch.float32, progress)
    stale = tmp_path / "GRAPH_graphs" / "99.pt"
    stale.write_text("stale")

    utils.extract_graphs(str(tmp_path), "dataset.csv", "GRAPH_graphs",
                         "MolGraphConvFeaturizer", torch.float32, progress)
    assert not stale.exists()
    assert utils.list_graph_files(os.path.join(tmp_path, "GRAPH_graphs")) == [0, 1, 2]


# --------------------------------------------------------- extract_molecule_fingerprints

@pytest.mark.parametrize("fingerprint", [
    "Morgan fingerprint", "Atom-pair fingerprint", "Topological-torsion fingerprint",
    "Avalon fingerprint", "Layered fingerprint", "Pattern fingerprint", "RDKit fingerprint",
])
def test_every_offered_fingerprint_works(tmp_path, fingerprint, progress):
    source = _source(tmp_path, ["CCO", "c1ccccc1"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    status = utils.extract_molecule_fingerprints(str(tmp_path), "dataset.csv", "FP_fps",
                                                 fingerprint, 2, 64, progress)
    assert "color: green" in status, status

    df = pd.read_csv(os.path.join(tmp_path, "FP_fps", "molecular_fingerprints.csv"))
    assert df.shape == (2, 65)                       # SMILES + 64 bits
    assert df.columns[0] == "SMILES"
    assert df["SMILES"].tolist() == ["CCO", "c1ccccc1"]
    with open(tmp_path / "FP_fps" / utils.FINGERPRINT_CONFIG) as handle:
        assert json.load(handle) == {
            "name": fingerprint,
            "radius": 2 if fingerprint == "Morgan fingerprint" else None,
            "number_of_bits": 64,
        }


def test_fingerprint_bit_width_follows_setting(tmp_path, progress):
    source = _source(tmp_path, ["CCO"], [1])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target")
    utils.extract_molecule_fingerprints(str(tmp_path), "dataset.csv", "FP_fps",
                                        "Morgan fingerprint", 2, 256, progress)
    df = pd.read_csv(os.path.join(tmp_path, "FP_fps", "molecular_fingerprints.csv"))
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
    data = torch.load(os.path.join(workdir_3d, "GRAPH_g3d", "0.pt"), weights_only=False)
    assert data.z.dtype == torch.int64
    assert data.pos.dtype == torch.float32
    assert data.pos.shape == (data.z.shape[0], 3)


@pytest.mark.slow
def test_extract_3d_graphs_covers_every_molecule(workdir_3d):
    from conftest import SMILES_3D
    assert utils.list_graph_files(os.path.join(workdir_3d, "GRAPH_g3d")) == list(range(len(SMILES_3D)))


@pytest.mark.slow
@pytest.mark.parametrize("force_field", ["MMFF", "UFF"])
def test_both_force_fields_work(tmp_path, force_field, progress):
    source = _source(tmp_path, ["CCO", "CCN"], [1, 0])
    utils.process_dataset_file(source, str(tmp_path), "dataset", "smiles", "target", True)
    status = utils.extract_3d_graphs(str(tmp_path), "dataset.csv", "GRAPH_g3d", 2,
                                     force_field, torch.float32, progress)
    assert "color: green" in status, status
    assert utils.list_graph_files(os.path.join(tmp_path, "GRAPH_g3d")) == [0, 1]


# -------------------------------------------------------------- extraction preservation

def _feature_output(directory, kind="graphs"):
    return directory / ("FP_features" if kind == "fingerprints" else "GRAPH_features")


def _extract(kind, directory, progress, dataset_file_name="dataset.csv"):
    if kind == "graphs":
        return utils.extract_graphs(str(directory), dataset_file_name, "features",
                                    "MolGraphConvFeaturizer", torch.float32, progress)
    if kind == "3d":
        return utils.extract_3d_graphs(str(directory), dataset_file_name, "features",
                                       1, "MMFF", torch.float32, progress)
    return utils.extract_molecule_fingerprints(str(directory), dataset_file_name, "features",
                                               "Morgan fingerprint", 2, 64, progress)


def _existing_extraction(tmp_path, smiles, kind="graphs"):
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame({"SMILES": smiles, "target": range(len(smiles))}).to_csv(dataset_path, index=False)
    output = _feature_output(tmp_path, kind)
    output.mkdir()
    (output / "previous.pt").write_bytes(b"previous extracted features")
    return dataset_path.read_bytes()


def _assert_extraction_preserved(tmp_path, dataset_contents, kind="graphs"):
    assert (tmp_path / "dataset.csv").read_bytes() == dataset_contents
    assert (_feature_output(tmp_path, kind) / "previous.pt").read_bytes() == b"previous extracted features"
    assert sorted(p.name for p in (_feature_output(tmp_path, kind)).iterdir()) == ["previous.pt"]
    assert not list(tmp_path.glob(".extraction-*"))
    assert not list(tmp_path.glob(f".{_feature_output(tmp_path, kind).name}-backup-*"))


@pytest.mark.parametrize("kind", ["graphs", "3d", "fingerprints"])
def test_failed_extraction_preserves_dataset_and_previous_features(tmp_path, progress, monkeypatch, kind):
    """Regression: if every molecule failed, extraction erased the dataset and still reported success."""
    contents = _existing_extraction(tmp_path, ["CCO", "CCN"], kind)
    monkeypatch.setattr(utils.Chem, "MolFromSmiles", lambda smiles: None)

    status = _extract(kind, tmp_path, progress)

    assert "color: red" in status and "No molecules" in status
    _assert_extraction_preserved(tmp_path, contents, kind)


@pytest.mark.parametrize("kind", ["graphs", "3d", "fingerprints"])
def test_interrupted_extraction_preserves_previous_features(tmp_path, kind):
    contents = _existing_extraction(tmp_path, ["CCO"], kind)

    class InterruptedProgress:
        def tqdm(self, iterable, **kwargs):
            raise RuntimeError("extraction interrupted")

    status = _extract(kind, tmp_path, InterruptedProgress())

    assert "color: red" in status and "extraction interrupted" in status
    _assert_extraction_preserved(tmp_path, contents, kind)


def test_fingerprint_metadata_write_failure_preserves_previous_extraction(tmp_path, progress, monkeypatch):
    contents = _existing_extraction(tmp_path, ["CCO", "NOT_A_SMILES"], "fingerprints")

    def fail_metadata_save(*args, **kwargs):
        raise OSError("metadata disk full")

    monkeypatch.setattr(utils.json, "dump", fail_metadata_save)
    status = _extract("fingerprints", tmp_path, progress)

    assert "color: red" in status and "metadata disk full" in status
    _assert_extraction_preserved(tmp_path, contents, "fingerprints")


@pytest.mark.parametrize("kind", ["graphs", "fingerprints", pytest.param("3d", marks=pytest.mark.slow)])
def test_partial_extraction_publishes_aligned_features(tmp_path, progress, kind):
    _existing_extraction(tmp_path, ["CCO", "NOT_A_SMILES", "CCN"], kind)

    status = _extract(kind, tmp_path, progress)

    assert "color: green" in status, status
    df = pd.read_csv(tmp_path / "dataset.csv")
    assert df["SMILES"].tolist() == ["CCO", "CCN"]
    assert df["target"].tolist() == [0, 2]
    assert not (_feature_output(tmp_path, kind) / "previous.pt").exists()
    if kind == "fingerprints":
        assert pd.read_csv(_feature_output(tmp_path, kind) / "molecular_fingerprints.csv")["SMILES"].tolist() == ["CCO", "CCN"]
    else:
        assert utils.read_graph_manifest(_feature_output(tmp_path, kind)) == ["CCO", "CCN"]
        assert utils.list_graph_files(_feature_output(tmp_path, kind)) == [0, 1]
        for i, smiles in enumerate(["CCO", "CCN"]):
            assert torch.load(_feature_output(tmp_path, kind) / f"{i}.pt", weights_only=False).smiles == smiles


@pytest.mark.parametrize("kind", ["graphs", pytest.param("3d", marks=pytest.mark.slow)])
def test_graph_save_failure_does_not_remove_valid_molecules(tmp_path, progress, monkeypatch, kind):
    contents = _existing_extraction(tmp_path, ["CCO", "CCN"], kind)
    original_save = utils.torch.save

    def fail_second_save(data, path):
        if os.path.basename(path) == "1.pt":
            raise OSError("disk full")
        return original_save(data, path)

    monkeypatch.setattr(utils.torch, "save", fail_second_save)
    status = _extract(kind, tmp_path, progress)

    assert "color: red" in status and "disk full" in status
    _assert_extraction_preserved(tmp_path, contents, kind)


def test_dataset_save_failure_restores_previous_features(tmp_path, progress, monkeypatch):
    contents = _existing_extraction(tmp_path, ["CCO", "NOT_A_SMILES"], "graphs")
    original_replace = utils.os.replace

    def fail_dataset_replace(source, destination):
        if str(destination) == str(tmp_path / "dataset.csv"):
            raise OSError("dataset is read-only")
        return original_replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", fail_dataset_replace)
    status = _extract("graphs", tmp_path, progress)

    assert "color: red" in status and "dataset is read-only" in status
    _assert_extraction_preserved(tmp_path, contents, "graphs")


@pytest.mark.parametrize("interrupt_phase", ["backup", "before_publish", "after_publish"])
def test_publication_interrupt_restores_previous_features(tmp_path, progress, monkeypatch, interrupt_phase):
    contents = _existing_extraction(tmp_path, ["CCO", "NOT_A_SMILES"], "graphs")
    original_replace = utils.os.replace

    def interrupt_publication(source, destination):
        if interrupt_phase == "backup" and ".GRAPH_features-backup-" in str(destination):
            original_replace(source, destination)
            raise KeyboardInterrupt("publication interrupted")
        if (interrupt_phase != "backup" and str(destination) == str(_feature_output(tmp_path, "graphs"))
                and ".extraction-" in str(source)):
            if interrupt_phase == "after_publish":
                original_replace(source, destination)
            raise KeyboardInterrupt("publication interrupted")
        return original_replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", interrupt_publication)
    with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
        _extract("graphs", tmp_path, progress)

    _assert_extraction_preserved(tmp_path, contents, "graphs")


def test_failed_restoration_preserves_backup_outside_staging_directory(tmp_path, progress, monkeypatch):
    contents = _existing_extraction(tmp_path, ["CCO", "NOT_A_SMILES"], "graphs")
    original_replace = utils.os.replace

    def fail_publication_and_restoration(source, destination):
        if str(destination) == str(tmp_path / "dataset.csv"):
            raise OSError("dataset is read-only")
        if str(destination) == str(_feature_output(tmp_path, "graphs")) and ".GRAPH_features-backup-" in str(source):
            raise OSError("restoration denied")
        return original_replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", fail_publication_and_restoration)
    status = _extract("graphs", tmp_path, progress)

    assert "color: red" in status and "restoration denied" in status
    assert (tmp_path / "dataset.csv").read_bytes() == contents
    backups = list(tmp_path.glob(".GRAPH_features-backup-*/previous"))
    assert len(backups) == 1
    assert str(backups[0]) in status
    assert (backups[0] / "previous.pt").read_bytes() == b"previous extracted features"
    assert not list(tmp_path.glob(".extraction-*"))


def test_interrupt_after_dataset_publication_keeps_features_aligned(tmp_path, progress, monkeypatch):
    _existing_extraction(tmp_path, ["CCO", "NOT_A_SMILES"], "graphs")
    original_replace = utils.os.replace

    def interrupt_after_dataset_replace(source, destination):
        result = original_replace(source, destination)
        if str(destination) == str(tmp_path / "dataset.csv"):
            raise KeyboardInterrupt("dataset was published")
        return result

    monkeypatch.setattr(utils.os, "replace", interrupt_after_dataset_replace)
    with pytest.raises(KeyboardInterrupt, match="dataset was published"):
        _extract("graphs", tmp_path, progress)

    assert pd.read_csv(tmp_path / "dataset.csv")["SMILES"].tolist() == ["CCO"]
    assert utils.read_graph_manifest(_feature_output(tmp_path, "graphs")) == ["CCO"]
    assert utils.list_graph_files(_feature_output(tmp_path, "graphs")) == [0]
    assert not list(tmp_path.glob(".extraction-*"))


@pytest.mark.parametrize("kind", ["graphs", "3d", "fingerprints"])
def test_extractors_reject_dataset_path_traversal(tmp_path, progress, kind):
    contents = _existing_extraction(tmp_path, ["CCO"], kind)

    status = _extract(kind, tmp_path, progress, "../dataset.csv")

    assert "color: red" in status and "dataset file name" in status
    _assert_extraction_preserved(tmp_path, contents, kind)


def test_fingerprint_csv_quotes_smiles_with_molecule_names(tmp_path, progress):
    """RDKit accepts a name after SMILES; commas in that name must be CSV-escaped."""
    smiles = "CCO ethanol, sample"
    pd.DataFrame({"SMILES": [smiles], "target": [1]}).to_csv(tmp_path / "dataset.csv", index=False)

    status = _extract("fingerprints", tmp_path, progress)

    assert "color: green" in status, status
    df = pd.read_csv(_feature_output(tmp_path, "fingerprints") / "molecular_fingerprints.csv")
    assert df.shape == (1, 65)
    assert df["SMILES"].tolist() == [smiles]
