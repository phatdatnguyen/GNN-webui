"""Fingerprint workflows, model controls, and invalidation of classical ML state."""

import os
from copy import deepcopy
from pathlib import Path
import shutil

import gradio as gr
import numpy as np
import pandas as pd
import pytest

import binary_classification_classical as classification
import regression_classical as regression
import classical_ml as ml
import classical_ui
import run_utils
from classical_ui import ClassicalWorkflow
from conftest import FakeProgress, is_error, is_ok


@pytest.fixture(params=["classification", "regression"])
def classical_case(tmp_path, request):
    task = request.param
    workflow = ClassicalWorkflow(task, lambda: str(tmp_path))
    message, directory = workflow.on_open_working_directory("experiment")
    assert is_ok(message)
    smiles = ["CCO", "CCN", "CCC", "CCS", "CCCl", "CCBr", "CCCC", "CCCCO"] * 2
    targets = np.arange(16) % 2 if task == "classification" else np.arange(16) * 2.5
    df = pd.DataFrame({"SMILES": smiles, "target": targets})
    df.to_csv(os.path.join(directory, "dataset.csv"), index=False)
    fingerprints = pd.DataFrame(np.eye(16, dtype=int), columns=[f"Bit_{i + 1}" for i in range(16)])
    fingerprints.insert(0, "SMILES", smiles)
    os.mkdir(os.path.join(directory, "FP_fps"))
    fingerprints.to_csv(os.path.join(directory, "FP_fps", "molecular_fingerprints.csv"), index=False)
    df[["SMILES"]].iloc[:4].to_csv(os.path.join(directory, "prediction.csv"), index=False)
    os.mkdir(os.path.join(directory, "FP_prediction_fps"))
    fingerprints.iloc[:4].to_csv(os.path.join(directory, "FP_prediction_fps", "molecular_fingerprints.csv"), index=False)
    return workflow, directory


def fitted_case(workflow, directory):
    message, processed = workflow.on_process_data(directory, "dataset.csv", "FP_fps", 0.25, 0)
    assert is_ok(message), message
    message, fitted, summary = workflow.on_fit(
        processed, "Random Forest", {"n_estimators": 5, "max_depth": 3}, 0)
    assert is_ok(message), message
    return processed, fitted, summary


def test_classical_data_and_model_seeds_capture_independent_contexts(classical_case):
    workflow, directory = classical_case
    message, processed = workflow.on_process_data(directory, "dataset.csv", "FP_fps", 0.25, 7)
    assert is_ok(message), message
    original_indices = processed.train_indices.copy(), processed.test_indices.copy()
    context_before = deepcopy(processed.split_context)
    for name, seed in [("first_run", 11), ("second_run", 29)]:
        message, fitted, summary = workflow.on_fit(processed, "Random Forest", {"n_estimators": 3}, seed,
                                                  run_name=name)
        assert is_ok(message), message
        assert fitted.run_context["model_name"] == name
        assert fitted.run_context["model_seed"] == fitted.random_seed == seed
        assert fitted.run_context["split_seed"] == 7
        assert fitted.run_context["split_context"] == context_before
        assert fitted.model_name == "Random Forest"
        assert fitted.pipelines[0].named_steps["estimator"].random_state == seed
        assert f"Model name: {name}" in summary and f"Model random seed: {seed}" in summary
        assert not Path(run_utils.run_directory(fitted.run_context)).exists()
    np.testing.assert_array_equal(processed.train_indices, original_indices[0])
    np.testing.assert_array_equal(processed.test_indices, original_indices[1])
    _, other = workflow.on_process_data(directory, "dataset.csv", "FP_fps", 0.25, 8)
    assert other.split_context["split_id"] != processed.split_context["split_id"]
    assert processed.split_context == context_before


def test_classical_legacy_models_adopt_only_an_honest_processed_split(classical_case):
    workflow, directory = classical_case
    processed, fitted, _ = fitted_case(workflow, directory)
    fitted.run_context = None
    ml.save_model(fitted, Path(directory) / "legacy.joblib")
    message, loaded, _ = workflow.on_load_model(directory, "legacy.joblib", run_name="new_run")
    assert is_ok(message) and loaded.run_context is None
    _, predictions = workflow.on_predict(loaded, directory, "prediction.csv", "FP_prediction_fps")
    assert predictions is not None
    assert is_error(workflow.on_export(directory, "predictions", predictions, loaded))
    assert is_error(workflow.on_save_model(directory, "copy", loaded))
    _, other = workflow.on_process_data(directory, "dataset.csv", "FP_fps", 0.25, 1)
    message, loaded, _ = workflow.on_load_model(directory, "legacy.joblib", processed=other, run_name="new_run")
    assert is_ok(message) and loaded.run_context is None
    message, loaded, _ = workflow.on_load_model(directory, "legacy.joblib", processed=processed, run_name="new_run")
    assert is_ok(message), message
    assert loaded.run_context["model_name"] == "new_run"
    assert loaded.run_context["model_seed"] == fitted.random_seed
    assert loaded.run_context["split_context"] == processed.split_context
    assert is_ok(workflow.on_export(directory, "predictions", predictions, loaded))
    assert (Path(run_utils.run_directory(loaded.run_context)) / "predictions.csv").is_file()


def test_classical_loaded_run_context_is_portable_and_ignores_live_run_name(classical_case, tmp_path):
    workflow, directory = classical_case
    _, fitted, _ = fitted_case(workflow, directory)
    original_context = deepcopy(fitted.run_context)
    assert is_ok(workflow.on_save_model(directory, "fitted", fitted))
    model_path = Path(run_utils.run_directory(fitted.run_context)) / "fitted.joblib"
    relative = os.path.relpath(model_path, directory)
    copied = tmp_path / "copied_workdir"
    shutil.copytree(directory, copied)
    message, restored, _ = workflow.on_load_model(str(copied), relative, run_name="ignored_live_name")
    assert is_ok(message), message
    assert restored.run_context["model_name"] == original_context["model_name"]
    assert restored.run_context["model_seed"] == original_context["model_seed"]
    assert restored.run_context["run_id"] == original_context["run_id"]
    assert restored.run_context["working_directory"] == str(copied)
    assert fitted.run_context == original_context
    rows = pd.DataFrame({"target (Prediction)": [1]})
    assert is_ok(workflow.on_export(str(copied), "portable", rows, restored))
    assert (Path(run_utils.run_directory(restored.run_context)) / "portable.csv").is_file()
    assert not (Path(run_utils.run_directory(original_context)) / "portable.csv").exists()
    assert is_error(workflow.on_export(directory, "wrong_workdir", rows, restored))


def test_classical_fit_rejects_unsafe_run_names_before_training(classical_case, monkeypatch):
    workflow, directory = classical_case
    _, processed = workflow.on_process_data(directory, "dataset.csv", "FP_fps")
    monkeypatch.setattr(ml, "fit_model", lambda *args, **kwargs: pytest.fail("Invalid model name must be rejected before training"))
    message, model, _ = workflow.on_fit(processed, "Random Forest", {}, 0, run_name="../outside")
    assert is_error(message) and model is None


def test_classical_named_runs_do_not_mix_changed_fingerprints_or_foreign_models(classical_case):
    workflow, directory = classical_case
    processed, original, _ = fitted_case(workflow, directory)
    rows = pd.DataFrame({"target (Prediction)": [1]})
    assert is_ok(workflow.on_export(directory, "predictions", rows, original))
    run_directory = Path(run_utils.run_directory(original.run_context))
    original_bytes = (run_directory / "predictions.csv").read_bytes()
    fingerprints_path = Path(directory) / "FP_fps" / "molecular_fingerprints.csv"
    fingerprints = pd.read_csv(fingerprints_path)
    fingerprints.loc[0, "Bit_1"] = 1 - fingerprints.loc[0, "Bit_1"]
    fingerprints.to_csv(fingerprints_path, index=False)
    changed, replacement, _ = fitted_case(workflow, directory)
    assert changed.split_context["split_id"] == processed.split_context["split_id"]
    assert replacement.training_identity != original.training_identity
    assert is_error(workflow.on_export(directory, "predictions", rows * 0, replacement))
    assert (run_directory / "predictions.csv").read_bytes() == original_bytes
    # A checkpoint copied into another run must not adopt that folder's unrelated provenance.
    replacement.run_context = None
    ml.save_model(replacement, run_directory / "foreign.joblib")
    relative = os.path.relpath(run_directory / "foreign.joblib", directory)
    message, model, _ = workflow.on_load_model(directory, relative)
    assert is_error(message) and model is None


def test_classical_handlers_fit_test_predict_save_load_export(classical_case):
    workflow, directory = classical_case
    processed, fitted, summary = fitted_case(workflow, directory)
    assert len(processed.train_indices) == 12 and len(processed.test_indices) == 4
    assert len(processed.val_indices) == 0
    assert "Random Forest" in summary and '"n_estimators": 5' in summary
    assert "target" in summary
    message, metrics, rows, selected, first_plot, second_plot = workflow.on_evaluate(fitted, processed)
    assert is_ok(message), message
    assert list(metrics.columns) == ["Target", "Metric", "Value"]
    assert not metrics.empty
    assert len(rows) == 4 and selected == "target"
    assert first_plot is not None
    assert (second_plot is not None) == (workflow.task == "classification")
    run_directory = Path(run_utils.run_directory(fitted.run_context))
    assert not run_directory.exists()
    assert is_ok(workflow.on_save_model(directory, "fitted", fitted))
    relative_model = os.path.relpath(run_directory / "fitted.joblib", directory)
    message, loaded, loaded_summary = workflow.on_load_model(directory, relative_model)
    assert is_ok(message), message
    assert loaded_summary == summary
    message, predictions = workflow.on_predict(loaded, directory, "prediction.csv", "FP_prediction_fps")
    assert is_ok(message), message
    assert len(predictions) == 4
    assert "target (Prediction)" in predictions
    if workflow.task == "classification":
        assert predictions["target (Probability)"].between(0, 1).all()
    assert is_ok(workflow.on_export(directory, "predictions", predictions, loaded))
    assert len(pd.read_csv(run_directory / "predictions.csv")) == 4
    assert not os.path.exists(os.path.join(directory, "predictions.csv"))


def test_classical_handler_errors_preserve_existing_model_and_results(classical_case):
    workflow, directory = classical_case
    processed, fitted, summary = fitted_case(workflow, directory)
    _, metrics, rows, *_ = workflow.on_evaluate(fitted, processed)
    message, current, _ = workflow.on_fit(processed, "Unknown", {}, 0, fitted)
    assert is_error(message) and current is fitted
    message, current, _ = workflow.on_load_model(directory, "missing.joblib", fitted)
    assert is_error(message) and current is fitted
    message, previous = workflow.on_predict(fitted, directory, "<bad>.csv", "FP_fps", rows)
    assert is_error(message) and previous is rows
    assert "<bad>" not in message and "&lt;bad&gt;" in message
    _, other_split = workflow.on_process_data(directory, "dataset.csv", "FP_fps", 0.25, 1)
    message, previous, previous_rows, *unchanged = workflow.on_evaluate(fitted, other_split, metrics, rows)
    assert is_error(message) and "split" in message
    assert previous is metrics and previous_rows is rows
    assert all(value == gr.update() for value in unchanged)
    assert is_error(workflow.on_save_model(directory, "../outside", fitted))


MODULES = [classification, regression]


def build(module):
    builder = (module.binary_classification_classical_tab_content if module is classification
               else module.regression_classical_tab_content)
    with gr.Blocks() as blocks:
        builder()
    return blocks


def callback(blocks, name):
    matches = [event for event in blocks.fns.values() if getattr(event.fn, "__name__", None) == name]
    assert len(matches) == 1
    return matches[0]


def ancestor(component, kind):
    parent = component.parent
    while parent is not None and not isinstance(parent, kind):
        parent = parent.parent
    return parent


def assert_evaluation_cleared(updates):
    plots = [component for component in updates if isinstance(component, gr.Plot)]
    assert plots
    assert all(updates[plot]["value"] is None for plot in plots)
    output = next(component for component in updates if getattr(component, "label", None) == "Output")
    assert updates[output]["choices"] == [] and updates[output]["value"] is None


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_classical_controls_only_show_selected_model_parameters(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    blocks = build(module)
    change = callback(blocks, "on_model_change")
    task = "classification" if module is classification else "regression"
    choices = change.inputs[0].choices
    assert [value for _, value in choices] == list(ml.MODEL_NAMES[task])
    groups = [component for component in change.outputs if isinstance(component, gr.Group)]
    assert len(groups) == len(ml.MODEL_NAMES[task])
    assert sum(group.visible for group in groups) == 1
    for index, model_name in enumerate(ml.MODEL_NAMES[task]):
        updates = change.fn(model_name, None)
        assert [updates[group]["visible"] for group in groups] == [i == index for i in range(len(groups))]
    labels = {getattr(component, "label", None) for component in blocks.blocks.values()}
    assert not labels.intersection({"Device", "Epochs", "Graph directory", "Validation size", "Validation metrics"})
    # Some estimators expose their own optimizer; there is no separate GNN training setup.
    buttons = {component.value for component in blocks.blocks.values() if isinstance(component, gr.Button)}
    assert not buttons.intersection({"Create optimizer", "Create scheduler", "Train model"})
    assert "Fitted model" in labels
    notes = getattr(ml, "MODEL_DESCRIPTIONS", getattr(ml, "MODEL_NOTES", {})).get(task, {})
    visible_help = {component.value for component in blocks.blocks.values() if isinstance(component, gr.Markdown)}
    assert all(note in visible_help for note in notes.values() if note)
    assert len({callback(blocks, name).concurrency_id for name in (
        "on_process_data", "on_fit", "on_load_model", "on_upload_dataset",
        "on_extract_fingerprints", "on_predict", "on_evaluate")}) == 1
    process, fit = callback(blocks, "on_process_data"), callback(blocks, "on_fit")
    split_seed, model_seed, run_name = process.inputs[4], fit.inputs[2], fit.inputs[4]
    assert split_seed.label == "Data split random seed"
    assert model_seed.label == "Model random seed" and model_seed is not split_seed
    assert run_name.label == "Model name" and run_name.value == f"{task}_classical"
    assert ancestor(run_name, gr.Row) is ancestor(model_seed, gr.Row)
    split_group = ancestor(split_seed, gr.Group)
    assert split_group is not None and ancestor(process.inputs[-1], gr.Group) is split_group
    calculate = callback(blocks, "on_calculate_similarity")
    assert calculate.concurrency_id == process.concurrency_id
    assert isinstance(calculate.inputs[0], gr.State)
    assert split_seed not in calculate.inputs and model_seed not in calculate.inputs


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_classical_processing_and_model_changes_clear_old_results(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    blocks = build(module)
    process = callback(blocks, "on_process_data")
    updates = process.fn(str(tmp_path), "missing.csv", "FP_fps", 0.2, 0, "Random")
    assert is_error(updates[process.outputs[0]])
    assert all(updates[component] is None for component in process.outputs if isinstance(component, gr.State))
    for component, value in updates.items():
        if isinstance(component, gr.Button):
            assert value["interactive"] is False
    assert_evaluation_cleared(updates)
    change = callback(blocks, "on_model_change")
    updates = change.fn("Extra Trees", object())
    assert all(value is None for component, value in updates.items() if isinstance(component, gr.State))
    fit_button = next(component for component in updates if isinstance(component, gr.Button) and component.value == "Fit model")
    assert updates[fit_button]["interactive"] is True
    assert_evaluation_cleared(updates)


def test_evaluation_switches_cached_targets_and_errors_preserve_results(classical_case, monkeypatch):
    workflow, directory = classical_case
    frame = pd.read_csv(os.path.join(directory, "dataset.csv"))
    frame["second"] = 1 - frame["target"] if workflow.task == "classification" else frame["target"] + 3
    frame.to_csv(os.path.join(directory, "dataset.csv"), index=False)
    processed, fitted, _ = fitted_case(workflow, directory)
    module = classification if workflow.task == "classification" else regression
    monkeypatch.setattr(module, "DATA_DIR", workflow.data_root())
    blocks = build(module)
    evaluate = callback(blocks, "on_evaluate")
    updates = evaluate.fn(fitted, processed, None, None, None)
    assert is_ok(updates[evaluate.outputs[0]])
    metrics, rows = updates[evaluate.inputs[2]], updates[evaluate.inputs[3]]
    assert set(metrics["Target"]) == {"target", "second"}
    output = evaluate.inputs[4]
    assert updates[output]["choices"] == ["target", "second"]
    table = next(component for component in updates if getattr(component, "label", None) == "Evaluation metrics")
    assert set(updates[table]["Target"]) == {"target"}
    monkeypatch.setattr(ml, "evaluate_predictions", lambda *args, **kwargs: pytest.fail("Target selection must reuse cached predictions"))
    select = callback(blocks, "on_output_change")
    switched = select.fn("second", rows, metrics)
    assert set(switched[table]["Target"]) == {"second"}
    assert all(switched[component] is not None for component in select.outputs if isinstance(component, gr.Plot))
    assert select.fn(None, None, None) == {table: gr.update()}
    bad_output = select.fn("missing", rows, metrics)
    assert set(bad_output) == {evaluate.outputs[0]}
    assert is_error(bad_output[evaluate.outputs[0]])
    failed = evaluate.fn(None, processed, metrics, rows, "second")
    assert is_error(failed[evaluate.outputs[0]])
    assert failed[evaluate.inputs[2]] is metrics and failed[evaluate.inputs[3]] is rows
    assert all(component not in failed for component in select.outputs if isinstance(component, gr.Plot))


def test_fit_load_and_open_clear_previous_evaluation(classical_case, monkeypatch):
    workflow, directory = classical_case
    processed, fitted, _ = fitted_case(workflow, directory)
    module = classification if workflow.task == "classification" else regression
    monkeypatch.setattr(module, "DATA_DIR", workflow.data_root())
    blocks = build(module)
    fit = callback(blocks, "on_fit")
    # Refit resets plots and cached evaluation rows, even when the estimator type stays the same.
    monkeypatch.setattr(ml, "fit_model", lambda *args, **kwargs: fitted)
    monkeypatch.setattr(ml, "evaluate_predictions", lambda *args, **kwargs: pytest.fail("Fit must not evaluate held-out rows"))
    monkeypatch.setattr(ml, "evaluate_model", lambda *args, **kwargs: pytest.fail("Fit must not evaluate held-out rows"))
    result = fit.fn(processed, "Random Forest", 0, fitted, *[component.value for component in fit.inputs[4:]])
    assert is_ok(result[fit.outputs[0]])
    assert_evaluation_cleared(result)
    assert is_ok(workflow.on_save_model(directory, "fitted", fitted))
    load = callback(blocks, "on_load_model")
    relative_model = os.path.relpath(Path(run_utils.run_directory(fitted.run_context)) / "fitted.joblib", directory)
    result = load.fn(directory, relative_model, fitted, processed)
    assert is_ok(result[load.outputs[0]])
    assert_evaluation_cleared(result)
    failed = load.fn(directory, "missing.joblib", fitted, processed)
    assert is_error(failed[load.outputs[0]])
    assert all(component not in failed for component in load.outputs if isinstance(component, gr.Plot))
    open_event = next(event for event in blocks.fns.values() if getattr(event.fn, "__name__", None) == "on_open_working_directory")
    result = open_event.fn("experiment")
    assert_evaluation_cleared(result)
    assert all(result[component] is None for component in open_event.outputs
               if isinstance(component, gr.State) and component in result and result[component] != directory)


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_prediction_upload_and_extraction_preserve_model_unless_training_artifacts_change(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    blocks = build(module)
    upload = callback(blocks, "on_upload_dataset")
    extraction = callback(blocks, "on_extract_fingerprints")
    message, workdir = module.on_open_working_directory("experiment")
    assert is_ok(message)
    source = tmp_path / "source.csv"
    pd.DataFrame({"SMILES": ["CCO", "CCN"], "target": [0, 1]}).to_csv(source, index=False)
    training_source = (workdir, "dataset.csv", "FP_fps")
    model_state = callback(blocks, "on_fit").inputs[3]

    prediction_upload = upload.fn(str(source), workdir, "prediction", "SMILES", "", training_source)
    assert is_ok(prediction_upload[upload.outputs[0]])
    assert model_state not in prediction_upload
    # A model loaded without processing any training data has no training-source state.
    loaded_model_upload = upload.fn(str(source), workdir, "prediction", "SMILES", "", None)
    assert model_state not in loaded_model_upload
    prediction_extraction = extraction.fn(workdir, "prediction.csv", "FP_prediction_fps", "Morgan fingerprint",
                                          2, 128, training_source, FakeProgress())
    assert is_ok(prediction_extraction[extraction.outputs[0]])
    assert model_state not in prediction_extraction

    training_upload = upload.fn(str(source), workdir, "dataset", "SMILES", "target", training_source)
    assert training_upload[model_state] is None
    overwritten_training_fps = extraction.fn(workdir, "prediction.csv", "FP_fps", "Morgan fingerprint",
                                             2, 128, training_source, FakeProgress())
    assert overwritten_training_fps[model_state] is None


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_classical_upload_forwards_cleanup_choices_and_preserves_model_on_failure(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    blocks = build(module)
    upload = callback(blocks, "on_upload_dataset")
    model_state = callback(blocks, "on_fit").inputs[3]
    cleanup = upload.inputs[-3:]
    assert [component.label for component in cleanup] == ["Canonicalize SMILES", "Remove duplicates", "Remove salt"]
    assert all(isinstance(component, gr.Checkbox) and component.value is True for component in cleanup)
    _, workdir = module.on_open_working_directory("experiment")
    source = tmp_path / "source.csv"
    source.write_text("SMILES,target\nCCO,1\n", encoding="utf-8")
    calls = []

    def process_file(*args, **kwargs):
        calls.append(kwargs)
        return "<span style='color: green;'>Uploaded</span>", pd.DataFrame({"SMILES": ["CCO"], "target": [1]})

    monkeypatch.setattr(classical_ui.utils, "process_dataset_file", process_file)
    training_source = (workdir, "dataset.csv", "FP_fps")
    result = upload.fn(str(source), workdir, "prediction", "SMILES", "", training_source, False, True, False)
    assert is_ok(result[upload.outputs[0]])
    assert calls[-1] == {"canonicalize_smiles": False, "remove_duplicates": True, "remove_salt": False}
    assert model_state not in result
    result = upload.fn(str(source), workdir, "dataset", "SMILES", "target", training_source)
    assert calls[-1] == {"canonicalize_smiles": True, "remove_duplicates": True, "remove_salt": True}
    assert result[model_state] is None

    monkeypatch.setattr(classical_ui.utils, "process_dataset_file",
                        lambda *args, **kwargs: ("<span style='color: red;'>Invalid data</span>", None))
    result = upload.fn(str(source), workdir, "dataset", "SMILES", "target", training_source, True, False, True)
    assert is_error(result[upload.outputs[0]])
    assert result[upload.outputs[1]] == gr.update()
    assert model_state not in result


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_classical_upload_missing_inputs_preserve_preview(tmp_path, task):
    workflow = ClassicalWorkflow(task, lambda: str(tmp_path))
    for file_path, workdir in [(None, str(tmp_path)), ("missing.csv", None),
                               ("missing.csv", str(tmp_path / "missing"))]:
        message, preview = workflow.on_upload_dataset(file_path, workdir, "dataset", "SMILES", "target")
        assert is_error(message)
        assert preview == gr.update()


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_classical_workdir_helpers_use_workflow_root_and_reject_traversal(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    message, path = module.on_open_working_directory("experiment")
    expected_root = "Classification" if module is classification else "Regression"
    assert is_ok(message)
    assert path == str(tmp_path / expected_root / "experiment")
    assert module.get_working_directories() == ["experiment"]
    message, path = module.on_open_working_directory("../outside")
    assert is_error(message) and path is None
