"""Checkpoint discovery must update in the action response, even for unchanged roots."""

import asyncio
import inspect
import gradio as gr
import pandas as pd
import pytest
import torch
from gradio.state_holder import SessionState

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import gnn_run_ui
import regression_hybrid as rg
import regression_3d as rg3
from run_utils import make_run_context, make_split_context


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content, "Classification"),
    (bc3, bc3.binary_classification_3d_tab_content, "Classification"),
    (rg, rg.regression_hybrid_tab_content, "Regression"),
    (rg3, rg3.regression_3d_tab_content, "Regression"),
]


@pytest.mark.parametrize("module,builder,workflow", CASES, ids=[case[0].__name__ for case in CASES])
def test_checkpoint_save_reopen_and_load_update_the_same_response(tmp_path, monkeypatch, module, builder, workflow):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    workdir = tmp_path / workflow / "first"
    workdir.mkdir(parents=True)
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCC", "CCCC"], "target": [0, 1, 0, 1]})
    frame.to_csv(workdir / "dataset.csv", index=False)
    split = make_split_context(str(workdir), "dataset.csv", frame, [0, 1], [2, 3], 0, "Random")
    model = torch.nn.Linear(1, 1)
    model._run_context = make_run_context(split, "run", 7, module.__name__, "Linear")

    with gr.Blocks() as blocks:
        builder()
    session = SessionState(blocks)
    values = {component._id: getattr(component, "value", None) for component in blocks.blocks.values()}
    properties = {}
    selector = next(component for component in blocks.blocks.values()
                    if isinstance(component, gr.Dropdown) and component.label == "Checkpoint file")
    train_button = next(component for component in blocks.blocks.values()
                        if isinstance(component, gr.Button) and component.value == "Train")

    def named(name):
        return next(callback for callback in blocks.fns.values()
                    if getattr(callback.fn, "__name__", None) == name)

    async def call(name, arguments=None):
        callback = named(name)
        parameters = list(inspect.signature(callback.fn).parameters)
        for parameter, value in (arguments or {}).items():
            component = callback.inputs[parameters.index(parameter)]
            if isinstance(component, gr.State):
                session[component._id] = value
            else:
                values[component._id] = value
        inputs = [None if isinstance(component, gr.State) else values[component._id]
                  for component in callback.inputs]
        result = await asyncio.wait_for(
            blocks.process_api(callback, inputs, state=session, session_hash="checkpoint-refresh"), 20)
        assert len(result["data"]) == len(callback.outputs)
        for component, output in zip(callback.outputs, result["data"]):
            if isinstance(component, gr.State):
                continue
            if isinstance(output, dict) and output.get("__type__") == "update":
                properties.setdefault(component._id, {}).update(output)
                if "value" in output:
                    values[component._id] = output["value"]
            else:
                values[component._id] = output
        return result

    def choices():
        return [choice[1] if isinstance(choice, (tuple, list)) else choice
                for choice in properties[selector._id]["choices"]]

    async def run():
        await call("on_open_working_directory", {"working_directory_name": "first"})
        assert choices() == []
        for epoch in (1, 2):
            saved = await call("on_save_checkpoint", {
                "working_directory_path": str(workdir), "checkpoint_name": "weights",
                "model": model, "trained_epochs": epoch,
            })
            assert "color: green" in saved["data"][0]
            assert choices() == [f"MODEL_run_7/weights_{index}.ckpt" for index in range(1, epoch + 1)]
            assert values[selector._id] == f"MODEL_run_7/weights_{epoch}.ckpt"
            if epoch == 1:
                root_contents = module.get_working_directory_contents(str(workdir))
        assert module.get_working_directory_contents(str(workdir)) == root_contents

        # Failed saves and unrelated refreshes must retain a non-first selection.
        failed = await call("on_save_checkpoint", {"checkpoint_name": "../invalid"})
        assert "color: red" in failed["data"][0]
        assert values[selector._id] == "MODEL_run_7/weights_2.ckpt"
        await call("on_content_list_change")
        assert values[selector._id] == "MODEL_run_7/weights_2.ckpt"

        # A post-save discovery failure must not turn a completed save into a traceback.
        def unavailable_listing(*args):
            raise PermissionError("listing denied")

        with monkeypatch.context() as patch:
            patch.setattr(gnn_run_ui, "list_saved_models", unavailable_listing)
            saved = await call("on_save_checkpoint", {"checkpoint_name": "weights"})
        assert "saved" in saved["data"][0].lower()
        assert "refresh" in saved["data"][0].lower() and "listing denied" in saved["data"][0]
        assert values[selector._id] == "MODEL_run_7/weights_2.ckpt"
        assert torch.load(workdir / "MODEL_run_7" / "weights_2.ckpt", weights_only=False)["trained_epochs"] == 2

        # Successful load must discard old optimizer moments; failed load must not.
        optimizer_callback = named("on_create_optimizer")
        optimizer_state, scheduler_state = optimizer_callback.outputs[1:3]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        session[optimizer_state._id] = optimizer
        session[scheduler_state._id] = scheduler
        properties[train_button._id] = {"interactive": True}
        training_plot, export_losses = named("on_train").outputs[1:3]
        values[training_plot._id] = "previous training plot"
        properties[export_losses._id] = {"interactive": True}
        failed = await call("on_load_checkpoint", {"checkpoint_file_name": None})
        assert "color: red" in failed["data"][0]
        assert session[optimizer_state._id] is optimizer
        assert session[scheduler_state._id] is scheduler
        assert properties[train_button._id]["interactive"] is True
        assert values[training_plot._id] == "previous training plot"
        assert properties[export_losses._id]["interactive"] is True
        loaded = await call("on_load_checkpoint", {"checkpoint_file_name": "MODEL_run_7/weights_2.ckpt"})
        assert "color: green" in loaded["data"][0]
        assert session[optimizer_state._id] is None
        assert session[scheduler_state._id] is None
        assert properties[train_button._id]["interactive"] is False
        assert values[training_plot._id] is None
        assert properties[export_losses._id]["interactive"] is False

        # Reopening must discover changes inside a folder whose root name is unchanged.
        extra = workdir / "MODEL_run_7" / "external.ckpt"
        extra.write_bytes((workdir / "MODEL_run_7" / "weights_2.ckpt").read_bytes())
        await call("on_open_working_directory", {"working_directory_name": "first"})
        assert "MODEL_run_7/external.ckpt" in choices()
        extra.unlink()
        await call("on_open_working_directory", {"working_directory_name": "first"})
        assert "MODEL_run_7/external.ckpt" not in choices()

        # Equal root filenames in two directories cannot suppress checkpoint discovery.
        other = workdir.parent / "second"
        other.mkdir()
        for name in root_contents:
            source = workdir / name
            if source.is_dir():
                (other / name).mkdir()
            else:
                (other / name).write_bytes(source.read_bytes())
        (other / "MODEL_run_7" / "other.ckpt").write_bytes(b"external checkpoint")
        assert sorted(module.get_working_directory_contents(str(other))) == sorted(root_contents)
        await call("on_open_working_directory", {"working_directory_name": "second"})
        assert choices() == ["MODEL_run_7/other.ckpt"]
        assert values[selector._id] == "MODEL_run_7/other.ckpt"

    asyncio.run(run())


@pytest.mark.parametrize("module,builder,workflow", CASES, ids=[case[0].__name__ for case in CASES])
def test_directory_refresh_tolerates_disappeared_entries(tmp_path, module, builder, workflow):
    (tmp_path / "broken.csv").symlink_to(tmp_path / "missing.csv")
    run = tmp_path / "MODEL_run_0"
    run.mkdir()
    (run / "weights.ckpt").write_bytes(b"checkpoint")
    arguments = [str(tmp_path), "dataset", "graphs"]
    if module in (bc, rg):
        arguments.append("fps")
    arguments.append(None)
    refreshed = module.on_content_list_change(*arguments)
    assert any(isinstance(update, dict) and update.get("choices") == ["MODEL_run_0/weights.ckpt"]
               for update in refreshed)
    assert module.get_working_directory_contents(str(tmp_path / "deleted")) == []
