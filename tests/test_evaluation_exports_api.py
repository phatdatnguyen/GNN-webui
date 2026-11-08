"""Check the GNN export buttons through Gradio's real callback/session pipeline."""

import asyncio
import inspect
from types import SimpleNamespace
from pathlib import Path

import gradio as gr
import pandas as pd
import pytest
import torch
from gradio.state_holder import SessionState

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
from evaluation_metrics import metrics_from_predictions
from run_utils import make_run_context, make_split_context, run_directory


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content, "classification", False),
    (bc3, bc3.binary_classification_3d_tab_content, "classification", True),
    (rg, rg.regression_hybrid_tab_content, "regression", False),
    (rg3, rg3.regression_3d_tab_content, "regression", True),
]


@pytest.mark.parametrize("module,builder,task,is_3d", CASES, ids=[case[0].__name__ for case in CASES])
def test_evaluation_exports_match_cached_results_and_clear_after_model_actions(
        tmp_path, monkeypatch, module, builder, task, is_3d):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    targets = ["assay"] if is_3d else ["assay", "second_assay"]
    actual = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])[:, :len(targets)]
    probabilities = torch.tensor([[0.1, 0.2], [0.8, 0.7], [0.4, 0.3], [0.6, 0.9]])[:, :len(targets)]
    outputs = torch.logit(probabilities) if task == "classification" else probabilities.numpy()
    inference_calls = []

    def infer(*args):
        inference_calls.append(True)
        return actual, outputs, ["CCO", "CCN", "CCC", "CCCC"]

    monkeypatch.setattr(module, "test", infer)
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC", "CCCC"] * 2,
                          **{target: actual[:, index].tolist() * 2 for index, target in enumerate(targets)}})
    split = make_split_context(str(tmp_path), "dataset.csv", frame, [0, 1, 2, 3], [4, 5, 6, 7], 17, "Random")
    dataset = SimpleNamespace(target_column_name=targets[0], target_column_names=targets, split_context=split)
    context = make_run_context(split, "export_test", 42, module.__name__, "Linear")
    run_path = Path(run_directory(context))
    with gr.Blocks() as blocks:
        builder()
    session = SessionState(blocks)
    values = {component._id: getattr(component, "value", None) for component in blocks.blocks.values()}
    properties = {}

    def callback_named(name):
        return next(callback for callback in blocks.fns.values()
                    if getattr(callback.fn, "__name__", None) == name)

    async def call(name, arguments=None, *, success=True):
        callback = callback_named(name)
        names = list(inspect.signature(callback.fn).parameters)
        for name, value in (arguments or {}).items():
            component = callback.inputs[names.index(name)]
            if isinstance(component, gr.State):
                session[component._id] = value
            else:
                values[component._id] = value
        inputs = [None if isinstance(component, gr.State) else values[component._id]
                  for component in callback.inputs]
        result = await asyncio.wait_for(
            blocks.process_api(callback, inputs, state=session, session_hash="gnn-export"), 20)
        assert len(result["data"]) == len(callback.outputs)
        messages = []
        for component, output in zip(callback.outputs, result["data"]):
            if isinstance(component, gr.State):
                continue
            if isinstance(output, dict) and output.get("__type__") == "update":
                properties.setdefault(component._id, {}).update(output)
                if "value" in output:
                    values[component._id] = output["value"]
            else:
                values[component._id] = output
            if isinstance(component, gr.HTML) and isinstance(output, str):
                messages.append(output)
        assert any(("color: green" if success else "color: red") in message for message in messages), messages
        return callback

    async def run():
        evaluation = callback_named("on_evaluate")
        export_buttons = evaluation.outputs[-2:]
        for action in ("on_create_model", "on_load_checkpoint", "on_train"):
            model = torch.nn.Linear(1, 1)
            model._run_context = context
            arguments = dict(model=model, dataset=dataset, test_dataloader=object(),
                             device="cpu", datatype=torch.float32)
            if task == "regression":
                arguments["output_scaler"] = object()  # The inference stub already returns original units.
            await call("on_evaluate", arguments)
            rows = session[evaluation.outputs[1]._id]
            assert all(properties[button._id]["interactive"] for button in export_buttons)
            await call("on_export_metrics", dict(working_directory_path=str(tmp_path), file_name="metrics.csv"))
            await call("on_export_evaluation", dict(working_directory_path=str(tmp_path), file_name="evaluation.csv"))
            pd.testing.assert_frame_equal(pd.read_csv(run_path / "evaluation.csv"), rows)
            expected = metrics_from_predictions(task, rows, targets)
            pd.testing.assert_frame_equal(pd.read_csv(run_path / "metrics.csv"), expected)
            assert set(expected["Target"]) == set(targets)
            assert not (run_path / "metrics.csv.csv").exists()
            assert not (tmp_path / "metrics.csv").exists()

            # A failed model action can still discard a model or change some weights.
            # Its real UI wrapper must invalidate the results in the same response.
            mutation_args = ({"dataset": None} if action == "on_create_model"
                             else {"checkpoint_file_name": None} if action == "on_load_checkpoint"
                             else {"model": None})
            # Named runs require a captured dataset split, including for 3D models.
            await call(action, mutation_args, success=False)
            assert session[evaluation.outputs[1]._id] is None
            assert all(properties[button._id]["interactive"] is False for button in export_buttons)
            exports_before = (run_path / "metrics.csv").read_bytes()
            await call("on_export_metrics", success=False)
            await call("on_export_evaluation", success=False)
            assert (run_path / "metrics.csv").read_bytes() == exports_before
        assert len(inference_calls) == 3

    asyncio.run(run())
