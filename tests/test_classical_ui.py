"""Fingerprint workflows, model controls, and invalidation of classical ML state."""

import os

import gradio as gr
import numpy as np
import pandas as pd
import pytest

import binary_classification_classical as classification
import regression_classical as regression
import classical_ml as ml
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
    os.mkdir(os.path.join(directory, "fps"))
    fingerprints.to_csv(os.path.join(directory, "fps", "molecular_fingerprints.csv"), index=False)
    df[["SMILES"]].iloc[:4].to_csv(os.path.join(directory, "prediction.csv"), index=False)
    os.mkdir(os.path.join(directory, "prediction_fps"))
    fingerprints.iloc[:4].to_csv(os.path.join(directory, "prediction_fps", "molecular_fingerprints.csv"), index=False)
    return workflow, directory


def fitted_case(workflow, directory):
    message, processed = workflow.on_process_data(directory, "dataset.csv", "fps", 0.25, 0)
    assert is_ok(message), message
    message, fitted, summary = workflow.on_fit(
        processed, "Random Forest", {"n_estimators": 5, "max_depth": 3}, 0)
    assert is_ok(message), message
    return processed, fitted, summary


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
    assert is_ok(workflow.on_save_model(directory, "fitted", fitted))
    message, loaded, loaded_summary = workflow.on_load_model(directory, "fitted.joblib")
    assert is_ok(message), message
    assert loaded_summary == summary
    message, predictions = workflow.on_predict(loaded, directory, "prediction.csv", "prediction_fps")
    assert is_ok(message), message
    assert len(predictions) == 4
    assert "target (Prediction)" in predictions
    if workflow.task == "classification":
        assert predictions["target (Probability)"].between(0, 1).all()
    assert is_ok(workflow.on_export(directory, "predictions", predictions))
    assert len(pd.read_csv(os.path.join(directory, "predictions.csv"))) == 4


def test_classical_handler_errors_preserve_existing_model_and_results(classical_case):
    workflow, directory = classical_case
    processed, fitted, summary = fitted_case(workflow, directory)
    _, metrics, rows, *_ = workflow.on_evaluate(fitted, processed)
    message, current, _ = workflow.on_fit(processed, "Unknown", {}, 0, fitted)
    assert is_error(message) and current is fitted
    message, current, _ = workflow.on_load_model(directory, "missing.joblib", fitted)
    assert is_error(message) and current is fitted
    message, previous = workflow.on_predict(fitted, directory, "<bad>.csv", "fps", rows)
    assert is_error(message) and previous is rows
    assert "<bad>" not in message and "&lt;bad&gt;" in message
    _, other_split = workflow.on_process_data(directory, "dataset.csv", "fps", 0.25, 1)
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


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_classical_processing_and_model_changes_clear_old_results(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    blocks = build(module)
    process = callback(blocks, "on_process_data")
    updates = process.fn(str(tmp_path), "missing.csv", "fps", 0.2, 0, "Random")
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
    result = load.fn(directory, "fitted.joblib", fitted, processed)
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
    training_source = (workdir, "dataset.csv", "fps")
    model_state = callback(blocks, "on_fit").inputs[3]

    prediction_upload = upload.fn(str(source), workdir, "prediction", "SMILES", "", training_source)
    assert is_ok(prediction_upload[upload.outputs[0]])
    assert model_state not in prediction_upload
    # A model loaded without processing any training data has no training-source state.
    loaded_model_upload = upload.fn(str(source), workdir, "prediction", "SMILES", "", None)
    assert model_state not in loaded_model_upload
    prediction_extraction = extraction.fn(workdir, "prediction.csv", "prediction_fps", "Morgan fingerprint",
                                          2, 128, training_source, FakeProgress())
    assert is_ok(prediction_extraction[extraction.outputs[0]])
    assert model_state not in prediction_extraction

    training_upload = upload.fn(str(source), workdir, "dataset", "SMILES", "target", training_source)
    assert training_upload[model_state] is None
    overwritten_training_fps = extraction.fn(workdir, "prediction.csv", "fps", "Morgan fingerprint",
                                             2, 128, training_source, FakeProgress())
    assert overwritten_training_fps[model_state] is None


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
