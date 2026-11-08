"""Exercise classical workflows through Gradio preprocessing and session state."""

import asyncio
from copy import deepcopy
from pathlib import Path
import shutil

import gradio as gr
import pandas as pd
import pytest
from gradio.state_holder import SessionState

from classical_ui import ClassicalWorkflow, build_classical_tab
import classical_ml as ml
import run_utils


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_classical_session_process_fit_save_reload_predict(tmp_path, request, monkeypatch, task):
    source = request.getfixturevalue(f"{task}_workdir")
    root = tmp_path / "Data"
    workdir = root / ("Classification" if task == "classification" else "Regression") / "run"
    shutil.copytree(source, workdir)
    dataset = pd.read_csv(workdir / "dataset.csv")
    first_target = dataset.columns[1]
    dataset["second_target"] = (1 - dataset[first_target] if task == "classification"
                                else 2 * dataset[first_target] + 1)
    dataset.to_csv(workdir / "dataset.csv", index=False)
    workflow = ClassicalWorkflow(task, lambda: str(root))
    with gr.Blocks() as blocks:
        build_classical_tab(workflow, task)
    session = SessionState(blocks)
    values = {component._id: getattr(component, "value", None)
              for component in blocks.blocks.values()}
    plot_components = [component for component in blocks.blocks.values()
                       if isinstance(component, gr.Plot)
                       and component.label in {"Confusion matrix", "ROC curve", "Actual vs. predicted"}]
    inference_calls = []
    original_predict = ml._predict_arrays

    def record_predict(*args, **kwargs):
        inference_calls.append(True)
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(ml, "_predict_arrays", record_predict)

    def assert_rendered_plots():
        assert len(plot_components) == (2 if task == "classification" else 1)
        for component in plot_components:
            payload = values[component._id]
            assert payload["type"] == "matplotlib"
            assert payload["plot"].startswith("data:image/")

    async def call(name, overrides=None, *, export_label=None):
        callback = next(callback for callback in blocks.fns.values()
                        if getattr(callback.fn, "__name__", None) == name
                        and (export_label is None or callback.inputs[1].label == export_label))
        for index, value in (overrides or {}).items():
            values[callback.inputs[index]._id] = value
        inputs = [None if isinstance(component, gr.State) else values[component._id]
                  for component in callback.inputs]
        result = await asyncio.wait_for(
            blocks.process_api(callback, inputs, state=session, session_hash=f"test-{task}"),
            timeout=20,
        )
        assert len(result["data"]) == len(callback.outputs)
        messages = []
        for component, output in zip(callback.outputs, result["data"]):
            if isinstance(component, gr.State):
                continue
            if isinstance(output, dict) and output.get("__type__") == "update":
                if "value" in output:
                    values[component._id] = output["value"]
            else:
                values[component._id] = output
            if isinstance(component, gr.HTML) and isinstance(output, str):
                messages.append(output)
        assert not any("color: red" in message for message in messages), messages
        return callback

    async def run():
        await call("on_open_working_directory", {0: "run"})
        # Use the UI's default 20% test fraction; every other row is training data.
        process = await call("on_process_data", {1: "dataset.csv", 2: "FP_fps", 4: 3})
        assert process.inputs[3].value == 0.2
        analysis = await call("on_calculate_similarity")
        analysis_state = next(component for component in analysis.outputs if isinstance(component, gr.State))
        captured_analysis = session[analysis_state._id]
        fit = await call("on_fit", {2: 17, 4: "named_run", 5: 5})
        processed = session[fit.inputs[0]._id]
        assert len(processed.test_indices) == int(len(processed.data) * 0.2)
        assert len(processed.train_indices) + len(processed.test_indices) == len(processed.data)
        model_state = fit.inputs[3]
        fitted = session[model_state._id]
        assert fitted is not None and len(fitted.pipelines) == 2
        assert fitted.run_context["split_seed"] == processed.random_seed == 3
        assert fitted.run_context["model_seed"] == fitted.random_seed == 17
        assert fitted.run_context["model_name"] == "named_run"
        assert session[analysis_state._id] is captured_analysis
        captured_run = deepcopy(fitted.run_context)
        run_directory = Path(run_utils.run_directory(captured_run))
        assert not run_directory.exists()
        # Editing visible controls does not relabel a fitted model or its cached split.
        values[fit.inputs[2]._id] = 999
        values[fit.inputs[4]._id] = "edited_name"
        values[process.inputs[4]._id] = 888
        evaluation = await call("on_evaluate")
        assert_rendered_plots()
        calls_before_selection = len(inference_calls)
        await call("on_output_change", {0: "second_target"})
        assert len(inference_calls) == calls_before_selection
        assert_rendered_plots()
        await call("on_export", {1: "metrics.csv"}, export_label="Metrics export file")
        await call("on_export", {1: "evaluation.csv"}, export_label="Evaluation export file")
        exported_metrics = pd.read_csv(run_directory / "metrics.csv")
        exported_rows = pd.read_csv(run_directory / "evaluation.csv")
        assert set(exported_metrics["Target"]) == {first_target, "second_target"}
        pd.testing.assert_frame_equal(exported_metrics, session[evaluation.inputs[2]._id])
        pd.testing.assert_frame_equal(exported_rows, session[evaluation.inputs[3]._id])
        assert len(inference_calls) == calls_before_selection
        assert not (run_directory / "metrics.csv.csv").exists()
        assert not (run_directory / "evaluation.csv.csv").exists()
        assert not (workdir / "metrics.csv").exists()
        assert fitted.run_context == captured_run
        prediction_callback = await call("on_predict", {2: "dataset.csv", 3: "FP_fps"})
        prediction_state = prediction_callback.inputs[-1]
        original_predictions = session[prediction_state._id].copy()
        await call("on_export", {1: "predictions.csv"}, export_label="Prediction export file")
        pd.testing.assert_frame_equal(pd.read_csv(run_directory / "predictions.csv"), original_predictions)
        await call("on_save_model", {1: "round_trip"})
        assert (run_directory / "round_trip.joblib").is_file()
        await call("refresh_directory")
        relative_model = str((run_directory / "round_trip.joblib").relative_to(workdir))
        assert relative_model in run_utils.list_saved_models(str(workdir), ".joblib")
        await call("on_load_model", {1: relative_model})
        assert session[model_state._id].run_context == captured_run
        assert values[fit.inputs[2]._id] == 17 and values[fit.inputs[4]._id] == "named_run"
        assert session[analysis_state._id] is captured_analysis
        # Clearing the output selector fires its change event in the browser.
        await call("on_output_change", {0: None})
        assert all(values[component._id] is None for component in plot_components)
        await call("on_predict")
        assert original_predictions.equals(session[prediction_state._id])
        await call("on_evaluate")
        assert_rendered_plots()
        await call("on_process_data")
        await call("on_output_change", {0: None})
        assert session[model_state._id] is None
        assert session[analysis_state._id] is None
        assert all(values[component._id] is None for component in plot_components)

    asyncio.run(run())
