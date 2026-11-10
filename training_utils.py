"""Small training policies shared by the four GNN workflows."""

import inspect
import math

import torch


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
