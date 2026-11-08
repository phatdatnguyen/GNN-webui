"""Content snapshots distinguish changed GNN features without repeated training I/O."""

import builtins
from copy import deepcopy
import json
import os
import shutil

import pandas as pd
import pytest

import feature_provenance as provenance


@pytest.fixture
def feature_directories(tmp_path):
    graphs = tmp_path / "GRAPH_graphs"
    graphs.mkdir()
    (graphs / "0.pt").write_bytes(b"graph-zero-features")
    (graphs / "1.pt").write_bytes(b"graph-one-features")
    (graphs / "smiles.txt").write_text("CCO\nCCN\n", encoding="utf-8")
    fingerprints = tmp_path / "FP_fps"
    fingerprints.mkdir()
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN"], "Bit_1": [0, 1], "Bit_2": [1, 0]})
    frame.to_csv(fingerprints / "molecular_fingerprints.csv", index=False)
    config = {"name": "Morgan fingerprint", "radius": 2, "number_of_bits": 2}
    (fingerprints / "fingerprint_config.json").write_text(json.dumps(config), encoding="utf-8")
    return graphs, fingerprints, frame, config


def test_feature_identity_survives_workdir_copy_and_file_timestamp_changes(feature_directories, tmp_path):
    graphs, fingerprints, frame, _ = feature_directories
    original = provenance.capture_feature_sources(graphs, fingerprints, frame)
    copied = tmp_path / "copied"
    copied.mkdir()
    shutil.copytree(graphs, copied / "GRAPH_graphs")
    shutil.copytree(fingerprints, copied / "FP_fps")
    os.utime(copied / "GRAPH_graphs" / "0.pt", (1, 1))
    restored = provenance.capture_feature_sources(copied / "GRAPH_graphs", copied / "FP_fps")
    assert restored == original
    assert original["graph_directory"] == "GRAPH_graphs"
    assert original["fingerprint_directory"] == "FP_fps"
    assert len(original["graph_hash"]) == len(original["fingerprint_hash"]) == 64


@pytest.mark.parametrize("change", ["values", "row_order", "column_order", "config"])
def test_changed_fingerprint_inputs_change_identity_in_the_same_directory(feature_directories, change):
    _, fingerprints, frame, config = feature_directories
    original = provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame)
    changed = frame.copy()
    if change == "values":
        changed.loc[0, "Bit_1"] = 1
    elif change == "row_order":
        changed = changed.iloc[::-1]
    elif change == "column_order":
        changed = changed[["SMILES", "Bit_2", "Bit_1"]]
    else:
        config["radius"] = 3
        (fingerprints / "fingerprint_config.json").write_text(json.dumps(config), encoding="utf-8")
    replacement = provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=changed)
    assert replacement["fingerprint_directory"] == original["fingerprint_directory"]
    assert replacement["fingerprint_hash"] != original["fingerprint_hash"]


def test_fingerprint_identity_uses_loaded_rows_and_ignores_dataframe_index(feature_directories, monkeypatch):
    _, fingerprints, frame, _ = feature_directories
    original = provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame)
    renamed_index = frame.copy()
    renamed_index.index = [100, 200]
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("Loaded fingerprints must not be reread"))
    assert provenance.capture_feature_sources(fingerprint_directory=fingerprints,
                                               fingerprint_df=renamed_index) == original


def test_fingerprint_config_formatting_does_not_change_semantic_identity(feature_directories):
    _, fingerprints, frame, config = feature_directories
    original = provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame)
    (fingerprints / "fingerprint_config.json").write_text(json.dumps(config, indent=4, sort_keys=True), encoding="utf-8")
    assert provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame) == original


@pytest.mark.parametrize("filename", ["0.pt", "1.pt", "smiles.txt"])
def test_changed_graph_artifacts_change_identity_in_the_same_directory(feature_directories, filename):
    graphs, _, _, _ = feature_directories
    original = provenance.capture_feature_sources(graph_directory=graphs)
    path = graphs / filename
    path.write_bytes(path.read_bytes() + b" changed")
    changed = provenance.capture_feature_sources(graph_directory=graphs)
    assert changed["graph_directory"] == original["graph_directory"]
    assert changed["graph_hash"] != original["graph_hash"]


def test_graph_snapshot_reads_each_shard_once_and_ignores_unrelated_files(feature_directories, monkeypatch):
    graphs, _, _, _ = feature_directories
    original = provenance.capture_feature_sources(graph_directory=graphs)
    (graphs / "notes.txt").write_text("unrelated notes")
    (graphs / "model.ckpt").write_bytes(b"unrelated model")
    reads = []
    original_hash = provenance._hash_file

    def record_hash(path):
        reads.append(os.path.basename(path))
        return original_hash(path)

    monkeypatch.setattr(provenance, "_hash_file", record_hash)
    assert provenance.capture_feature_sources(graph_directory=graphs) == original
    assert reads == ["0.pt", "1.pt", "smiles.txt"]


def test_large_graph_files_are_read_in_bounded_chunks(tmp_path, monkeypatch):
    graphs = tmp_path / "GRAPH_graphs"
    graphs.mkdir()
    (graphs / "0.pt").write_bytes(b"a" * (provenance._READ_SIZE * 2 + 17))
    original_open = builtins.open
    read_sizes = []

    class RecordedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            read_sizes.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(provenance, "open", lambda path, mode: RecordedFile(original_open(path, mode)), raising=False)
    assert provenance.capture_feature_sources(graph_directory=graphs)["graph_hash"]
    assert len(read_sizes) == 4 and set(read_sizes) == {1024 * 1024}


def test_fingerprint_serialization_uses_a_stream_and_bounded_formatting_chunks(feature_directories, monkeypatch):
    _, fingerprints, frame, _ = feature_directories
    original_to_csv = pd.DataFrame.to_csv
    observed = []

    def record_csv(self, destination, *args, **kwargs):
        observed.append((destination, kwargs.get("chunksize")))
        return original_to_csv(self, destination, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", record_csv)
    provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame)
    assert len(observed) == 1 and hasattr(observed[0][0], "write")
    assert observed[0][1] == 256


def test_legacy_feature_directories_without_metadata_remain_supported(feature_directories):
    graphs, fingerprints, frame, _ = feature_directories
    (graphs / "smiles.txt").unlink()
    (fingerprints / "fingerprint_config.json").unlink()
    snapshot = provenance.capture_feature_sources(graphs, fingerprints, frame)
    assert snapshot["graph_hash"] and snapshot["fingerprint_hash"]
    assert snapshot["fingerprint_config"] is None


@pytest.mark.parametrize("config", ["{broken", "null", '{"radius": NaN}'])
def test_malformed_configuration_is_rejected(feature_directories, config):
    _, fingerprints, frame, _ = feature_directories
    (fingerprints / "fingerprint_config.json").write_text(config, encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame)


def test_returned_snapshot_does_not_change_with_later_data_or_config_mutations(feature_directories):
    _, fingerprints, frame, config = feature_directories
    snapshot = provenance.capture_feature_sources(fingerprint_directory=fingerprints, fingerprint_df=frame)
    before = deepcopy(snapshot)
    frame.loc[0, "Bit_1"] = 1
    config["radius"] = 3
    (fingerprints / "fingerprint_config.json").write_text(json.dumps(config), encoding="utf-8")
    assert snapshot == before
