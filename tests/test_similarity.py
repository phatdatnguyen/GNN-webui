"""Exact similarity, bounded comparison chunks, and captured-split UI/export contracts."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import pytest
from gradio.state_holder import SessionState
from rdkit import Chem, DataStructs
from rdkit.Chem import SaltRemover, rdFingerprintGenerator

import similarity
from similarity_ui import bind_similarity_controls, build_similarity_controls


@pytest.fixture
def split_context(tmp_path):
    return {"working_directory": str(tmp_path), "dataset_file": "dataset.csv", "split_seed": 17,
            "split_method": "Random", "train_indices": [2, 1, 0], "validation_indices": [3],
            "test_indices": [6, 5, 4, 7],
            "smiles": ["CCO", "CCN", "OCC", "invalid-validation-smiles", "c1ccccc1",
                       "CC(=O)O", "CCO", "C"], "dataset_hash": "a" * 64, "split_id": "b" * 64}


def reference_fingerprint(smiles, radius=2, bits=128):
    molecule = Chem.MolFromSmiles(smiles)
    molecule = SaltRemover.SaltRemover().StripMol(molecule, dontRemoveEverything=True)
    molecule = Chem.AddHs(molecule)
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits).GetFingerprint(molecule)


def test_exact_nearest_rows_and_all_pair_histogram_match_direct_rdkit(split_context):
    result = similarity.calculate_similarity(split_context, number_of_bits=128, bins=10, chunk_size=2)
    train_rows = sorted(split_context["train_indices"])
    test_rows = split_context["test_indices"]
    expected = np.array([
        [DataStructs.TanimotoSimilarity(reference_fingerprint(split_context["smiles"][test]),
                                       reference_fingerprint(split_context["smiles"][train]))
         for train in train_rows] for test in test_rows])
    assert result.nearest["Test row"].tolist() == test_rows
    assert result.nearest["Test SMILES"].tolist() == [split_context["smiles"][row] for row in test_rows]
    assert result.nearest["Training row"].tolist() == [train_rows[index] for index in expected.argmax(axis=1)]
    assert result.nearest["Training SMILES"].tolist() == [
        split_context["smiles"][row] for row in result.nearest["Training row"]]
    np.testing.assert_allclose(result.nearest["Tanimoto similarity"], expected.max(axis=1))
    np.testing.assert_array_equal(result.distribution["Pair count"],
                                  np.histogram(expected.ravel(), bins=np.linspace(0, 1, 11))[0])
    assert result.distribution["Pair count"].sum() == expected.size
    assert result.distribution["Proportion"].sum() == pytest.approx(1)
    assert result.summary["mean_pair_similarity"] == pytest.approx(expected.mean())
    assert result.summary["mean_nearest_similarity"] == pytest.approx(expected.max(axis=1).mean())
    assert result.summary["validation_molecules_excluded"] == 1
    # Two equivalent training rows tie exactly; choose the lowest original row ID,
    # even when captured random-split order lists the other equivalent row first.
    assert result.nearest.iloc[0]["Training row"] == 0
    assert result.nearest.iloc[0]["Tanimoto similarity"] == 1


def test_similarity_buffers_are_chunk_bounded_and_progress_finishes(split_context, monkeypatch):
    original = similarity.DataStructs.BulkTanimotoSimilarity
    chunk_lengths, progress_values = [], []

    def record(query, training):
        chunk_lengths.append(len(training))
        assert len(training) <= 2
        return original(query, training)

    monkeypatch.setattr(similarity.DataStructs, "BulkTanimotoSimilarity", record)
    result = similarity.calculate_similarity(
        split_context, number_of_bits=128, chunk_size=2,
        progress=lambda value, **kwargs: progress_values.append(value))
    assert chunk_lengths == [2, 1] * len(split_context["test_indices"])
    assert sum(chunk_lengths) == result.summary["pair_count"]
    assert progress_values[0] == 0 and progress_values[-1] == 1
    assert progress_values == sorted(progress_values)
    assert all(0 <= value <= 1 for value in progress_values)


def test_histogram_includes_zero_internal_boundaries_and_one(split_context, monkeypatch):
    split_context.update(smiles=["CCO"] * 5, train_indices=[0, 1, 2, 3],
                         test_indices=[4], validation_indices=[])
    monkeypatch.setattr(similarity.DataStructs, "BulkTanimotoSimilarity",
                        lambda query, training: [0.0, 0.2, 0.5, 1.0])
    result = similarity.calculate_similarity(split_context, number_of_bits=128, bins=5)
    assert result.distribution["Pair count"].tolist() == [1, 1, 1, 0, 1]
    assert result.nearest["Training row"].tolist() == [3]
    assert result.nearest["Tanimoto similarity"].tolist() == [1.0]


@pytest.mark.parametrize("train,test", [
    ("CC[NH3+].[Cl-]", "CC[NH3+]"),
    ("C[C@H](O)F", "C[C@@H](O)F"),
])
def test_salt_and_chirality_rules_match_existing_morgan_features(split_context, train, test):
    split_context.update(smiles=[train, test], train_indices=[0], test_indices=[1], validation_indices=[])
    result = similarity.calculate_similarity(split_context, number_of_bits=128)
    assert result.nearest["Tanimoto similarity"].tolist() == [1.0]
    assert result.summary["pair_count"] == 1
    assert result.distribution.iloc[-1]["Pair count"] == 1
    assert result.settings["remove_salts"] is True
    assert result.settings["explicit_hydrogens"] is True
    assert result.settings["include_chirality"] is False


def test_analysis_keeps_a_snapshot_when_caller_mutates_context(split_context):
    expected = deepcopy(split_context)
    result = similarity.calculate_similarity(split_context, number_of_bits=128)
    split_context["split_seed"] = 999
    split_context["smiles"][0] = "CCC"
    split_context["test_indices"].reverse()
    assert result.context == expected


@pytest.mark.parametrize("part,row", [("train_indices", 0), ("test_indices", 6)])
def test_invalid_used_smiles_reports_original_row_and_drops_nothing(split_context, part, row):
    split_context["smiles"][row] = "not-a-smiles"
    with pytest.raises(ValueError, match=f"dataset row {row}: invalid SMILES"):
        similarity.calculate_similarity(split_context, number_of_bits=128)
    assert row in split_context[part]


@pytest.mark.parametrize("partition", ["train_indices", "test_indices"])
def test_empty_training_or_test_split_is_a_readable_error(split_context, partition):
    split_context["validation_indices"].extend(split_context[partition])
    split_context[partition] = []
    with pytest.raises(ValueError, match="split is empty"):
        similarity.calculate_similarity(split_context)


@pytest.mark.parametrize("context", [None, {}, {"smiles": []}])
def test_missing_processed_context_is_a_readable_error(context):
    with pytest.raises(ValueError, match="process the data|captured split"):
        similarity.calculate_similarity(context)


def test_overlapping_split_rows_are_rejected(split_context):
    split_context["test_indices"][0] = split_context["train_indices"][0]
    with pytest.raises(ValueError, match="without overlap"):
        similarity.calculate_similarity(split_context)


@pytest.mark.parametrize("options", [dict(radius=-1), dict(radius=True), dict(radius=1.5),
                                     dict(number_of_bits=0), dict(bins=0), dict(chunk_size=0)])
def test_invalid_settings_are_rejected(split_context, options):
    with pytest.raises(ValueError, match="must be an integer"):
        similarity.calculate_similarity(split_context, **options)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_similarity_results_fail_before_publishing(split_context, monkeypatch, bad):
    monkeypatch.setattr(similarity.DataStructs, "BulkTanimotoSimilarity",
                        lambda query, training: [bad] * len(training))
    with pytest.raises(ValueError, match="invalid Tanimoto"):
        similarity.calculate_similarity(split_context)


def test_export_rejects_a_different_current_split_before_creating_files(split_context, tmp_path):
    result = similarity.calculate_similarity(split_context, number_of_bits=128)
    changed = deepcopy(split_context)
    changed["split_seed"] += 1
    with pytest.raises(ValueError, match="split changed"):
        similarity.export_similarity(result, "nearest", "neighbors", current_context=changed)
    assert list(tmp_path.iterdir()) == []


def captured_context(split_context):
    from run_utils import make_split_context

    frame = pd.DataFrame({"SMILES": split_context["smiles"], "target": range(len(split_context["smiles"]))})
    return make_split_context(split_context["working_directory"], "dataset.csv", frame,
                              split_context["train_indices"], split_context["test_indices"],
                              split_context["split_seed"], "Random", split_context["validation_indices"])


def test_exports_retain_actual_split_seed_and_settings_with_portable_manifest(split_context):
    context = captured_context(split_context)
    result = similarity.calculate_similarity(context, radius=3, number_of_bits=256)
    context["split_seed"] = 999  # A later UI edit cannot redirect this captured result.
    for kind, expected in [("distribution", result.distribution), ("nearest", result.nearest)]:
        csv_path, metadata_path = similarity.export_similarity(result, kind, f"{kind}.csv")
        csv_path, metadata_path = Path(csv_path), Path(metadata_path)
        assert csv_path.parent == Path(split_context["working_directory"]) / "random_seed_17"
        assert csv_path.name == f"{kind}.csv"
        pd.testing.assert_frame_equal(pd.read_csv(csv_path), expected)
        metadata = json.loads(metadata_path.read_text())
        assert metadata["split_seed"] == 17
        assert metadata["split_id"] == result.context["split_id"]
        assert metadata["dataset_hash"] == result.context["dataset_hash"]
        assert metadata["settings"]["radius"] == 3
        assert metadata["settings"]["number_of_bits"] == 256
        assert metadata["settings"]["metric"] == "Tanimoto"
        assert metadata["summary"]["pair_count"] == 12
        manifest = csv_path.parent / f"split_{result.context['split_id']}.json"
        assert json.loads(manifest.read_text())["test_indices"] == split_context["test_indices"]
    assert not (Path(split_context["working_directory"]) / "random_seed_999").exists()


def test_export_refuses_modified_manifest_and_preserves_previous_csv(split_context):
    context = captured_context(split_context)
    result = similarity.calculate_similarity(context, number_of_bits=128)
    csv_path, _ = similarity.export_similarity(result, "nearest", "neighbors")
    csv_path = Path(csv_path)
    before = csv_path.read_bytes()
    manifest = csv_path.parent / f"split_{context['split_id']}.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="different data"):
        similarity.export_similarity(result, "nearest", "neighbors")
    assert csv_path.read_bytes() == before


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", ""])
def test_similarity_export_rejects_unsafe_names(split_context, name):
    result = similarity.calculate_similarity(captured_context(split_context), number_of_bits=128)
    with pytest.raises(ValueError, match="file name"):
        similarity.export_similarity(result, "nearest", name)


@pytest.mark.parametrize("failure", ["serialization", "publication", "interruption"])
def test_export_failure_preserves_previous_csv_and_metadata(split_context, monkeypatch, failure):
    result = similarity.calculate_similarity(captured_context(split_context), number_of_bits=128)
    paths = [Path(path) for path in similarity.export_similarity(result, "nearest", "neighbors")]
    original = [path.read_bytes() for path in paths]
    result.nearest = result.nearest.iloc[::-1].reset_index(drop=True)
    result.settings["radius"] = 3
    if failure == "serialization":
        def fail_json(*args, **kwargs):
            raise OSError("metadata serialization failed")
        monkeypatch.setattr(similarity.json, "dump", fail_json)
    else:
        replace = similarity.os.replace
        def fail_metadata(source, destination):
            if Path(source).name == "neighbors.csv.metadata.json":
                if failure == "interruption":
                    raise KeyboardInterrupt("publication interrupted")
                raise OSError("metadata publication failed")
            return replace(source, destination)
        monkeypatch.setattr(similarity.os, "replace", fail_metadata)
    with pytest.raises(KeyboardInterrupt if failure == "interruption" else OSError):
        similarity.export_similarity(result, "nearest", "neighbors")
    assert [path.read_bytes() for path in paths] == original
    assert not list(paths[0].parent.glob(".similarity-*"))


def test_failed_restore_preserves_recovery_files_and_reports_location(split_context, monkeypatch):
    result = similarity.calculate_similarity(captured_context(split_context), number_of_bits=128)
    csv_path, metadata_path = map(Path, similarity.export_similarity(result, "nearest", "neighbors"))
    previous_csv = csv_path.read_bytes()
    previous_metadata = metadata_path.read_bytes()
    result.nearest = result.nearest.iloc[::-1].reset_index(drop=True)
    replace = similarity.os.replace

    def fail_publish_and_restore(source, destination):
        if Path(source).name in {"neighbors.csv.metadata.json", "previous_0"}:
            raise OSError("simulated failure")
        return replace(source, destination)

    monkeypatch.setattr(similarity.os, "replace", fail_publish_and_restore)
    with pytest.raises(RuntimeError, match="Recovery files are preserved at") as error:
        similarity.export_similarity(result, "nearest", "neighbors")
    recovery, = csv_path.parent.glob(".similarity-*")
    assert str(recovery) in str(error.value)
    assert (recovery / "previous_0").read_bytes() == previous_csv
    assert metadata_path.read_bytes() == previous_metadata


@pytest.mark.parametrize("suffix", [".csv", ".csv.metadata.json"])
def test_export_rejects_final_symlink_without_changing_target(split_context, suffix):
    from run_utils import ensure_split_directory

    result = similarity.calculate_similarity(captured_context(split_context), number_of_bits=128)
    directory = Path(ensure_split_directory(result.context))
    target = Path(split_context["working_directory"]) / "keep.txt"
    target.write_text("original content")
    destination = directory / f"neighbors{suffix}"
    destination.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        similarity.export_similarity(result, "nearest", "neighbors")
    assert destination.is_symlink()
    assert target.read_text() == "original content"
    assert not list(directory.glob(".similarity-*"))


def test_shared_controls_use_only_captured_context_and_reset_every_result(split_context):
    with gr.Blocks() as blocks:
        context_state = gr.State()
        status = gr.HTML()
        controls = build_similarity_controls()
        bind_similarity_controls(controls, context_state, status)
    calculation = next(callback for callback in blocks.fns.values()
                       if callback.fn.__name__ == "on_calculate_similarity")
    assert calculation.inputs == [context_state, controls.radius, controls.number_of_bits]
    session = SessionState(blocks)
    session[context_state._id] = split_context

    async def run():
        response = await asyncio.wait_for(
            blocks.process_api(calculation, [None, 2, 128], state=session, session_hash="similarity"), 20)
        assert len(response["data"]) == len(calculation.outputs)
        assert "color: green" in response["data"][0]
        result = session[controls.state._id]
        assert result.context["split_seed"] == 17
        for component, payload in zip(calculation.outputs, response["data"], strict=True):
            if component in (controls.pair_plot, controls.nearest_plot):
                assert payload["type"] == "matplotlib" and payload["plot"].startswith("data:image/")
            elif component in (controls.distribution_export_button, controls.nearest_export_button):
                assert payload["interactive"] is True
        # A missing processed context produces a status update and leaves any earlier
        # result intact, rather than publishing an incomplete analysis.
        session[context_state._id] = None
        response = await asyncio.wait_for(
            blocks.process_api(calculation, [None, 2, 128], state=session, session_hash="similarity"), 20)
        assert "color: red" in response["data"][0]
        assert session[controls.state._id] is result

    asyncio.run(run())
    resets = controls.reset_updates()
    assert controls.reset_outputs == list(resets)
    assert resets[controls.state] is None
    assert resets[controls.pair_plot]["value"] is None
    assert resets[controls.nearest_plot]["value"] is None
    assert resets[controls.summary]["value"] is None
    assert resets[controls.preview]["value"] is None
    assert resets[controls.distribution_export_button]["interactive"] is False
    assert resets[controls.nearest_export_button]["interactive"] is False


def test_ui_exports_use_cached_analysis_and_captured_split_not_live_controls(split_context, monkeypatch):
    context = captured_context(split_context)
    with gr.Blocks() as blocks:
        context_state = gr.State()
        status = gr.HTML()
        live_split_seed = gr.Number(value=999)
        live_model_seed = gr.Number(value=41)
        controls = build_similarity_controls()
        bind_similarity_controls(controls, context_state, status)
    session = SessionState(blocks)
    session[context_state._id] = context
    result = similarity.calculate_similarity(context, radius=2, number_of_bits=128)
    session[controls.state._id] = result

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Export must reuse cached similarity results.")

    monkeypatch.setattr(similarity, "calculate_similarity", unexpected_calculation)

    async def run():
        for name, filename in [("on_export_similarity_distribution", "distribution"),
                               ("on_export_nearest_training_molecules", "nearest")]:
            callback = next(callback for callback in blocks.fns.values() if callback.fn.__name__ == name)
            assert callback.inputs[:2] == [controls.state, context_state]
            assert live_split_seed not in callback.inputs and live_model_seed not in callback.inputs
            response = await asyncio.wait_for(
                blocks.process_api(callback, [None, None, filename], state=session,
                                   session_hash="similarity-export"), 20)
            assert "color: green" in response["data"][0]
            saved = Path(split_context["working_directory"]) / "random_seed_17" / f"{filename}.csv"
            assert saved.is_file()
        session[context_state._id] = None
        response = await asyncio.wait_for(
            blocks.process_api(callback, [None, None, "no_split"], state=session,
                               session_hash="similarity-export"), 20)
        assert "color: red" in response["data"][0]
        assert not list(Path(split_context["working_directory"]).rglob("no_split.csv"))

    asyncio.run(run())
