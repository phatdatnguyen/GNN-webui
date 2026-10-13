"""A newly selected data split must not reuse a model trained on the previous split."""

import inspect

import gradio as gr
import pandas as pd
import pytest
import torch

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
from conftest import is_error, is_ok
from test_split_handlers import split_workdir  # Reuse aligned synthetic data without conformers.


BUILDERS = [bc.binary_classification_hybrid_tab_content, rg.regression_hybrid_tab_content,
            bc3.binary_classification_3d_tab_content, rg3.regression_3d_tab_content]


def callback_named(blocks, name):
    callbacks = [callback for callback in blocks.fns.values()
                 if getattr(callback.fn, "__name__", None) == name]
    assert len(callbacks) == 1
    return callbacks[0]


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda builder: builder.__name__)
@pytest.mark.parametrize("success", [True, False], ids=["successful_processing", "failed_processing"])
def test_processing_new_split_invalidates_previous_training_and_results(split_workdir, builder, success):
    with gr.Blocks() as blocks:
        builder()
    processing = callback_named(blocks, "on_process_data")
    creation = callback_named(blocks, "on_create_model")
    optimizer_creation = callback_named(blocks, "on_create_optimizer")
    evaluation = callback_named(blocks, "on_evaluate")
    prediction_processing = callback_named(blocks, "on_process_prediction_data")
    prediction = callback_named(blocks, "on_predict")
    # The creation wrapper appends result resets after the handler's original outputs.
    creation_states = [component for component in creation.outputs if isinstance(component, gr.State)]
    model_state, train_losses_state, val_losses_state, epoch_state = creation_states[:4]
    optimizer_state, scheduler_state = optimizer_creation.outputs[1:3]
    evaluation_state = evaluation.outputs[1]
    prediction_states = prediction_processing.outputs[1:3]

    values = dict(working_directory_path=str(split_workdir), dataset_file_name="dataset.csv",
                  load_data="Molecular fingerprints", graph_directory="graphs",
                  molecular_fingerprint_directory="fps", datatype=torch.float32,
                  dimensionality_reduction=False, variance_threshold=0.0, pca_num_components=2,
                  test_ratio=0.25, val_ratio=0.25, batch_size=2, shuffle_training_data=False,
                  random_seed=17, split_method="Random")
    arguments = {name: values[name] for name in inspect.signature(processing.fn).parameters}
    previous = processing.fn(**arguments)
    assert isinstance(previous, dict)
    assert is_ok(previous[processing.outputs[0]])

    # Model, history and results from a completed run on the previous Random split.
    old_model = torch.nn.Linear(1, 1)
    old_optimizer = torch.optim.SGD(old_model.parameters(), lr=0.1)
    old_scheduler = torch.optim.lr_scheduler.ExponentialLR(old_optimizer, gamma=0.9)
    stale_train_losses, stale_val_losses = [0.5], [0.6]
    old_states = {model_state: old_model, optimizer_state: old_optimizer,
                  scheduler_state: old_scheduler, train_losses_state: stale_train_losses,
                  val_losses_state: stale_val_losses, epoch_state: 1,
                  evaluation_state: pd.DataFrame({"old_prediction": [1.0]}),
                  **{component: object() for component in prediction_states}}

    arguments["split_method"] = "Scaffold"
    if not success:
        arguments["dataset_file_name"] = "missing.csv"
    result = processing.fn(**arguments)
    assert set(result) == set(processing.outputs)
    assert (is_ok if success else is_error)(result[processing.outputs[0]])
    assert result[processing.outputs[1]].get("interactive") is success
    session = {**previous, **old_states, **result}

    for component in (model_state, optimizer_state, scheduler_state,
                      evaluation_state, *prediction_states):
        assert session[component] is None
    assert session[train_losses_state] == session[val_losses_state] == []
    assert session[train_losses_state] is not stale_train_losses
    assert session[val_losses_state] is not stale_val_losses
    assert session[train_losses_state] is not session[val_losses_state]
    assert session[epoch_state] == 0

    # A reset must be part of this click's outputs, so no stale model can be used
    # between separate processing and cleanup events.
    for label in ("Train", "Evaluate", "Predict", "Create optimizer", "Save checkpoint",
                  "Load checkpoint", "Process prediction data"):
        button = next(component for component in blocks.blocks.values()
                      if isinstance(component, gr.Button) and component.value == label)
        assert result[button].get("interactive") is False
    export_buttons = [component for component in blocks.blocks.values()
                      if isinstance(component, gr.Button) and component.value.startswith("Export")]
    assert len(export_buttons) == 4
    assert all(result[button].get("interactive") is False for button in export_buttons)
    for component in blocks.blocks.values():
        if isinstance(component, gr.Plot) or (isinstance(component, gr.Dataframe)
                                             and component.label in ("Evaluation metrics", "Prediction")) \
                or getattr(component, "label", None) == "Model summary":
            assert result[component].get("value", "not cleared") is None
    assert result[prediction.outputs[1]].get("value", "not cleared") is None

    # Original processing outputs still carry the newly selected dataset/loaders.
    dataset_state = creation.inputs[0]
    assert (result[dataset_state] is not None) is success
    next_result = processing.fn(**arguments)
    assert next_result[train_losses_state] == next_result[val_losses_state] == []
    assert next_result[train_losses_state] is not result[train_losses_state]
    assert next_result[val_losses_state] is not result[val_losses_state]
