"""Shared GNN loading and device-transfer policy.

Keep workers optional: small cached datasets often iterate faster in the main
process. Spawn avoids forking a process after CUDA has been initialized.
"""

import random
from numbers import Integral, Real

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary


def cache_budget_bytes(graph_cache_mb):
    if (isinstance(graph_cache_mb, bool) or not isinstance(graph_cache_mb, Real)
            or not np.isfinite(graph_cache_mb) or not 0 <= graph_cache_mb <= 2048):
        raise ValueError("Graph cache size must be between 0 and 2048 MiB.")
    return int(graph_cache_mb * 1024 ** 2)


def seed_loader_worker(worker_id):
    """Top-level callable so seeded workers also work on Windows/spawn."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, batch_size, *, shuffle=False, seed=0,
                loader_workers=0, pin_memory=False, loader_class=DataLoader):
    if (isinstance(loader_workers, bool) or not isinstance(loader_workers, Integral)
            or loader_workers not in (0, 2, 4)):
        raise ValueError("Loader workers must be 0, 2, or 4.")
    if not isinstance(pin_memory, bool):
        raise ValueError("Pin memory must be enabled or disabled.")
    options = dict(batch_size=batch_size, shuffle=bool(shuffle),
                   num_workers=int(loader_workers), pin_memory=pin_memory,
                   generator=torch.Generator().manual_seed(seed))
    if loader_workers:
        options.update(multiprocessing_context="spawn", worker_init_fn=seed_loader_worker,
                       prefetch_factor=2, persistent_workers=False)
    return loader_class(dataset, **options)


def hybrid_graph_inputs(batch, load_data, need_edge_attr, device, datatype):
    if load_data not in ("Graphs", "Both"):
        return None, None, None, None
    return (
        batch.x.to(device=device, dtype=datatype, non_blocking=True),
        batch.edge_index.to(device=device, dtype=torch.long, non_blocking=True),
        batch.edge_attr.to(device=device, dtype=datatype, non_blocking=True)
        if need_edge_attr else None,
        batch.batch.to(device=device, non_blocking=True),
    )


def batch_fingerprints(batch, device, datatype, variance_threshold=None, pca=None):
    value = getattr(batch, "fp_prepared", None)
    if value is None:
        value = batch.fp
        # Compatibility for direct callers using generic PyG datasets.
        if variance_threshold is not None and pca is not None:
            value = torch.from_numpy(np.asarray(pca.transform(
                variance_threshold.transform(value.detach().cpu().double().numpy()))))
    return value.to(device=device, dtype=datatype, non_blocking=True)


def batch_targets(batch, output_scaler, device, datatype):
    value = getattr(batch, "y_prepared", None)
    if value is None:
        value = torch.from_numpy(np.asarray(output_scaler.transform(
            batch.y.detach().cpu().double().numpy())))
    return value.to(device=device, dtype=datatype, non_blocking=True)


def safe_model_summary(model, *args):
    """Remove summary hooks and restore training flags even if forward fails."""
    before = [(module, set(module._forward_hooks), module.training)
              for module in model.modules()]
    try:
        return summary(model, *args)
    finally:
        for module, original_hooks, training in before:
            for key in set(module._forward_hooks).difference(original_hooks):
                del module._forward_hooks[key]
            module.training = training
