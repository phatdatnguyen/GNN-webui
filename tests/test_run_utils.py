"""Run provenance must survive slider changes, reloads and directory copies."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
import torch

from run_utils import (
    ensure_run_directory, ensure_split_directory, list_saved_models, make_run_context,
    make_split_context, read_run_context_for_file, rebase_run_context, resolve_saved_model,
    run_directory, validate_run_context,
)
from utils import export_result_csv, load_model_checkpoint


@pytest.fixture
def split(tmp_path):
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC", "CCCC"], "target": [0, 1, 0, 1]})
    return make_split_context(str(tmp_path), "dataset.csv", frame, [0, 2], [3], 17, "Random", [1])


def test_run_and_split_folders_are_siblings_with_captured_provenance(split):
    run = make_run_context(split, "forest", 42, "classification_classical", "RandomForestClassifier",
                           {"n_estimators": 5})
    # Changing the caller's original split cannot relabel a fitted model's run.
    split["split_seed"] = 99
    folder = Path(ensure_run_directory(run))
    assert folder == Path(run["working_directory"]) / "MODEL_forest_42"
    seed_folder = folder.parent / "random_seed_17"
    assert seed_folder.is_dir()
    recorded = json.loads((folder / "run.json").read_text())
    assert recorded["model_seed"] == 42 and recorded["split_seed"] == 17
    assert recorded["split_context"]["test_indices"] == [3]
    assert "working_directory" not in recorded
    assert len(list(seed_folder.glob("split_*.json"))) == 1
    assert ensure_run_directory(run) == str(folder)


def test_same_model_name_and_seed_cannot_silently_mix_splits(split):
    first = make_run_context(split, "experiment", 3, "regression_hybrid", "GCNModel")
    folder = Path(ensure_run_directory(first))
    original = (folder / "run.json").read_bytes()
    frame = pd.DataFrame({"SMILES": split["smiles"], "target": [0, 1, 0, 1]})
    other = make_split_context(split["working_directory"], "dataset.csv", frame,
                               [0, 1], [2], 18, "Random", [3])
    with pytest.raises(ValueError, match="different data or model settings"):
        ensure_run_directory(make_run_context(other, "experiment", 3, "regression_hybrid", "GCNModel"))
    assert (folder / "run.json").read_bytes() == original


def test_same_data_seed_can_capture_different_split_settings(split):
    directory = Path(ensure_split_directory(split))
    frame = pd.DataFrame({"SMILES": split["smiles"], "target": [0, 1, 0, 1]})
    other = make_split_context(split["working_directory"], "dataset.csv", frame,
                               [0, 1, 2], [3], 17, "Random")
    assert ensure_split_directory(other) == str(directory)
    assert len(list(directory.glob("split_*.json"))) == 2


def test_model_files_are_discoverable_and_portable(split, tmp_path):
    run = make_run_context(split, "my_model", 5, "classification_hybrid", "GCNModel")
    folder = Path(ensure_run_directory(run))
    (folder / "checkpoint.ckpt").write_bytes(b"example")
    (tmp_path / "legacy.ckpt").write_bytes(b"old")
    (tmp_path / "notes.csv").write_text("x\n1\n")
    assert list_saved_models(str(tmp_path), ".ckpt") == ["MODEL_my_model_5/checkpoint.ckpt"]
    assert resolve_saved_model(str(tmp_path), "MODEL_my_model_5/checkpoint.ckpt", ".ckpt") == str(folder / "checkpoint.ckpt")
    assert read_run_context_for_file(str(tmp_path / "legacy.ckpt")) is None
    copied = tmp_path / "copied"
    copied.mkdir()
    shutil.copytree(folder, copied / folder.name)
    restored = read_run_context_for_file(str(copied / folder.name / "checkpoint.ckpt"))
    assert restored["working_directory"] == str(copied)
    assert restored["run_id"] == run["run_id"]
    assert ensure_run_directory(restored) == str(copied / folder.name)
    assert rebase_run_context(run, str(copied)) == restored


@pytest.mark.parametrize("name", ["../escape", "/tmp/outside", "a/b", ""])
def test_run_names_cannot_escape_the_working_directory(split, name):
    with pytest.raises(ValueError):
        make_run_context(split, name, 0, "classification_hybrid", "GCNModel")


def test_model_prefix_is_added_once_without_changing_the_captured_name(split):
    plain = make_run_context(split, "forest", 4, "classification_classical", "RandomForestClassifier")
    prefixed = make_run_context(split, "MODEL_forest", 4, "classification_classical", "RandomForestClassifier")
    assert run_directory(plain) == run_directory(prefixed)
    assert Path(run_directory(prefixed)).name == "MODEL_forest_4"
    assert prefixed["model_name"] == "MODEL_forest"
    ensure_run_directory(plain)
    # These are distinct user labels, even though their folder names coincide.
    with pytest.raises(ValueError, match="different data or model settings"):
        ensure_run_directory(prefixed)


def test_model_prefix_prevents_collision_with_data_seed_directory(split):
    run = make_run_context(split, "random_seed", 17, "classification_hybrid", "GCNModel")
    model_directory = Path(ensure_run_directory(run))
    assert model_directory.name == "MODEL_random_seed_17"
    assert model_directory.parent.joinpath("random_seed_17").is_dir()


@pytest.mark.parametrize("suffix", [".ckpt", ".joblib"])
def test_saved_model_discovery_only_uses_model_directories(tmp_path, suffix):
    (tmp_path / f"root{suffix}").write_bytes(b"ignore")
    for name in ["MODEL_second", "MODEL_first", "GRAPH_features", "FP_features", "random_seed_7", "old_run", "model_lowercase"]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / f"weights{suffix}").write_bytes(b"model")
        (directory / "notes.csv").write_text("ignored\n")
    (tmp_path / "MODEL_first" / f"linked{suffix}").symlink_to(tmp_path / f"root{suffix}")
    (tmp_path / "MODEL_link").symlink_to(tmp_path / "MODEL_first", target_is_directory=True)
    (tmp_path / f"MODEL_root_file{suffix}").write_bytes(b"ignore")
    assert list_saved_models(str(tmp_path), suffix) == [
        f"MODEL_first/weights{suffix}", f"MODEL_second/weights{suffix}"]
    # Root files are omitted from the selector but remain usable by direct callers.
    assert resolve_saved_model(str(tmp_path), f"root{suffix}", suffix) == str(tmp_path / f"root{suffix}")
    for relative in [f"old_run/weights{suffix}", f"FP_features/weights{suffix}"]:
        with pytest.raises(ValueError, match="MODEL_"):
            resolve_saved_model(str(tmp_path), relative, suffix)


@pytest.mark.parametrize("name", ["../file.ckpt", "/tmp/file.ckpt", "folder/../file.ckpt", "folder\\file.ckpt"])
def test_load_paths_cannot_escape_working_directory(tmp_path, name):
    with pytest.raises(ValueError):
        resolve_saved_model(str(tmp_path), name, ".ckpt")


def test_run_metadata_and_output_directories_reject_symlinks(split, tmp_path):
    run = make_run_context(split, "model", 1, "classification_hybrid", "GCNModel")
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "MODEL_model_1").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        ensure_run_directory(run)
    (other / "file.ckpt").write_bytes(b"example")
    assert "MODEL_model_1/file.ckpt" not in list_saved_models(str(tmp_path), ".ckpt")
    with pytest.raises(ValueError, match="symbolic"):
        resolve_saved_model(str(tmp_path), "MODEL_model_1/file.ckpt", ".ckpt")


def test_csv_export_cannot_follow_a_link_outside_its_run(split, tmp_path):
    run = make_run_context(split, "model", 1, "classification_hybrid", "GCNModel")
    directory = Path(ensure_run_directory(run))
    original = tmp_path / "original.csv"
    original.write_text("preserve me\n")
    (directory / "metrics.csv").symlink_to(original)
    with pytest.raises(ValueError, match="symbolic"):
        export_result_csv(str(directory), "metrics", pd.DataFrame({"value": [1]}))
    assert original.read_text() == "preserve me\n"


def test_run_context_rejects_mutated_seed_and_split(split):
    run = make_run_context(split, "model", 1, "classification_hybrid", "GCNModel")
    changed = deepcopy(run)
    changed["model_seed"] = 2
    with pytest.raises(ValueError, match="changed"):
        validate_run_context(changed)
    changed = deepcopy(run)
    changed["split_context"]["train_indices"] = [0, 1]
    with pytest.raises(ValueError, match="partition"):
        validate_run_context(changed)


def test_checkpoint_metadata_validation_precedes_weight_changes(tmp_path):
    model = torch.nn.Linear(2, 1)
    old = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "model.ckpt"
    torch.save({"model_class": "Linear", "state_dict": {key: value + 10 for key, value in old.items()}}, path)

    def reject(checkpoint):
        raise ValueError("Wrong run metadata")

    with pytest.raises(ValueError, match="Wrong run metadata"):
        load_model_checkpoint(model, path, validate_checkpoint=reject)
    assert all(torch.equal(value, old[key]) for key, value in model.state_dict().items())
