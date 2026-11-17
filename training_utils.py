"""Small training policies shared by the four GNN workflows."""

import inspect
import math
from functools import wraps

import gradio as gr
from matplotlib.figure import Figure
import torch


class TrainingUpdate(tuple):
    """The original six handler outputs, plus lifecycle information for the UI."""

    def __new__(cls, values, *, finished):
        update = super().__new__(cls, values)
        update.finished = finished
        return update


def training_handler(stream):
    """Keep direct callers synchronous while exposing a Gradio streaming callback."""
    @wraps(stream)
    def complete_training(*args, **kwargs):
        for update in stream(*args, **kwargs):
            result = update
        return result

    complete_training.stream = stream
    return complete_training


def build_training_update(status, loss_epochs, trained_epochs, train_losses, val_losses,
                          ylabel, *, finished):
    """Publish independent histories and a figure without pyplot's global registry."""
    figure = Figure(figsize=(10, 5))
    axes = figure.subplots()
    axes.plot(loss_epochs or [], train_losses if loss_epochs is not None else [], label="Train loss")
    axes.plot(loss_epochs or [], val_losses if loss_epochs is not None else [], label="Validation loss")
    axes.set_xlabel("Epoch")
    axes.set_ylabel(ylabel)
    axes.legend()
    axes.grid(True)
    export_button = gr.Button(value="Export", interactive=bool(train_losses) and loss_epochs is not None)
    # Gradio may still be serializing the previous update when another epoch finishes.
    # Its completed histories must not refer to the lists the loop keeps appending to.
    return TrainingUpdate((status, figure, export_button, trained_epochs,
                           list(train_losses) if train_losses is not None else None,
                           list(val_losses) if val_losses is not None else None), finished=finished)


def create_optimizer(model, name, learning_rate, use_fused=False):
    """Opt into fused updates for supported float32 CUDA Adam/AdamW parameters."""
    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}
    optimizer_class = optimizers.get(name)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer '{name}'.")
    parameters = list(model.parameters())
    options = {}
    if (use_fused and name in ("Adam", "AdamW") and parameters
            and all(parameter.is_cuda and parameter.dtype == torch.float32 for parameter in parameters)
            and "fused" in inspect.signature(optimizer_class).parameters):
        options["fused"] = True
    return optimizer_class(parameters, lr=learning_rate, **options)


def validation_loss(total_loss, num_samples, *, regression=False):
    """Read one weighted validation loss per epoch instead of every CUDA batch."""
    mean = total_loss.item() / num_samples if num_samples else 0.0
    if not math.isfinite(mean):
        raise ValueError("Validation loss is not finite. Try a lower learning rate or float32 data type.")
    return math.sqrt(mean) if regression else mean
