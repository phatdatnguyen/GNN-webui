"""Resumed checkpoints retain cumulative epoch numbers in plots and loss exports."""

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pandas as pd
import pytest
import torch
from gradio.state_holder import SessionState

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
from conftest import is_error, is_ok
from run_utils import make_run_context, make_split_context, run_directory


CASES = [
    (bc, bc.binary_classification_hybrid_tab_content),
    (bc3, bc3.binary_classification_3d_tab_content),
    (rg, rg.regression_hybrid_tab_content),
    (rg3, rg3.regression_3d_tab_content),
]


@pytest.fixture(params=CASES, ids=[module.__name__ for module, _ in CASES])
def workflow(request):
    return request.param


@pytest.mark.parametrize("fail_second_epoch", [False, True])
def test_resumed_training_plots_and_exports_completed_epoch_numbers(
        workflow, monkeypatch, tmp_path, progress, fail_second_epoch):
    module, _ = workflow
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler_steps = []
    scheduler = SimpleNamespace(step=lambda: scheduler_steps.append(True))
    validation_calls = []

    def validate(*args):
        validation_calls.append(True)
        if fail_second_epoch and len(validation_calls) == 2:
            raise ValueError("Simulated validation failure")
        return 0.6

    monkeypatch.setattr(module, "train", lambda *args: 0.5)
    monkeypatch.setattr(module, "validation", validate)
    values = dict(model=model, dataset=object(), train_dataloader=object(), val_dataloader=object(),
                  device="cpu", datatype=torch.float32, output_scaler=object(), optimizer=optimizer,
                  scheduler=scheduler, epochs=2, trained_epochs=10, train_losses=[], val_losses=[],
                  gnn_model_tab="GCN", customgcn_convolutional_layer_name="GCNConv",
                  variance_threshold=None, pca=None, progress=progress)
    result = module.on_train(**{name: values[name] for name in inspect.signature(module.on_train).parameters})
    message, figure, export_button, completed, train_losses, val_losses = result
    expected_epochs = [11] if fail_second_epoch else [11, 12]
    assert (is_error if fail_second_epoch else is_ok)(message), message
    assert completed == expected_epochs[-1]
    assert len(scheduler_steps) == len(expected_epochs)
    assert export_button.interactive is True
    assert [line.get_xdata().tolist() for line in figure.axes[0].lines] == [expected_epochs, expected_epochs]
    assert train_losses == [0.5] * len(expected_epochs)
    assert val_losses == [0.6] * len(expected_epochs)

    message, _ = module.on_export_losses(str(tmp_path), "resumed", train_losses, val_losses, completed)
    assert is_ok(message), message
    exported = pd.read_csv(tmp_path / "resumed.csv")
    assert exported["epoch"].tolist() == expected_epochs
    assert exported["train_losses"].tolist() == train_losses
    assert exported["val_losses"].tolist() == val_losses


def test_legacy_direct_loss_export_uses_one_based_epochs(workflow, tmp_path):
    module, _ = workflow
    message, _ = module.on_export_losses(str(tmp_path), "legacy", [0.6, 0.5], [0.7, 0.6])
    assert is_ok(message), message
    assert pd.read_csv(tmp_path / "legacy.csv")["epoch"].tolist() == [1, 2]


def test_loss_export_rejects_inconsistent_epoch_count(workflow, tmp_path):
    module, _ = workflow
    message, _ = module.on_export_losses(str(tmp_path), "invalid", [0.6, 0.5], [0.7, 0.6], 1)
    assert is_error(message) and "epoch count" in message, message
    assert not (tmp_path / "invalid.csv").exists()


def test_loss_export_ui_passes_captured_cumulative_epochs(workflow, tmp_path, monkeypatch):
    module, builder = workflow
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path / "Data"))
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN"], "target": [0, 1]})
    split = make_split_context(str(tmp_path), "dataset.csv", frame, [0], [1], 0, "Random")
    model = torch.nn.Linear(1, 1)
    model._run_context = make_run_context(split, "resumed", 7, module.__name__, "Linear")
    with gr.Blocks() as blocks:
        builder()
    callback = next(callback for callback in blocks.fns.values()
                    if getattr(callback.fn, "__name__", None) == "on_export_losses")
    session = SessionState(blocks)
    values = dict(working_directory_path=str(tmp_path), file_name="resumed_losses", train_losses=[0.5, 0.4],
                  val_losses=[0.6, 0.5], trained_epochs=12, model=model)
    inputs = []
    for name, component in zip(inspect.signature(callback.fn).parameters, callback.inputs, strict=True):
        if isinstance(component, gr.State):
            session[component._id] = values[name]
            inputs.append(None)
        else:
            inputs.append(values[name])

    async def run():
        return await asyncio.wait_for(blocks.process_api(
            callback, inputs, state=session, session_hash="resumed-loss-export"), 20)

    response = asyncio.run(run())
    assert is_ok(response["data"][0]), response["data"][0]
    exported = pd.read_csv(Path(run_directory(model._run_context)) / "resumed_losses.csv")
    assert exported["epoch"].tolist() == [11, 12]
    assert not (tmp_path / "resumed_losses.csv").exists()
