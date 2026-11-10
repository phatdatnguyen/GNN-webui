"""Loader reproducibility, transfer policy and failed model-summary cleanup."""

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

import gnn_runtime
from gnn_runtime import cache_budget_bytes, hybrid_graph_inputs, make_loader, safe_model_summary


def _samples():
    return [Data(x=torch.tensor([[float(i)]]), edge_index=torch.empty((2, 0), dtype=torch.long),
                 y=torch.tensor([[float(i)]])) for i in range(13)]


def _orders(loader):
    return [torch.cat([batch.y.flatten() for batch in loader]).tolist() for _ in range(2)]


def test_loaders_reproduce_shuffle_without_using_model_random_stream():
    samples = _samples()
    torch.manual_seed(47)
    before = torch.random.get_rng_state().clone()
    first = _orders(make_loader(samples, 4, shuffle=True, seed=11))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert first == _orders(make_loader(samples, 4, shuffle=True, seed=11))
    assert first[0] != first[1]
    assert first != _orders(make_loader(samples, 4, shuffle=True, seed=12))
    assert _orders(make_loader(samples, 4)) == [list(range(13)), list(range(13))]


@pytest.mark.parametrize("workers", [2, 4])
def test_worker_options_use_spawn_and_do_not_keep_workers_between_epochs(workers):
    calls = []

    def capture(dataset, **kwargs):
        calls.append(kwargs)

    make_loader(_samples(), 4, loader_workers=workers, pin_memory=True, loader_class=capture)
    assert calls[0]["multiprocessing_context"] == "spawn"
    assert calls[0]["persistent_workers"] is False
    assert calls[0]["prefetch_factor"] == 2
    assert calls[0]["pin_memory"] is True


@pytest.mark.parametrize("workers", [-1, 1, 2.5, True])
def test_invalid_worker_configuration_is_rejected(workers):
    with pytest.raises(ValueError, match="workers"):
        make_loader(_samples(), 4, loader_workers=workers)


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), True, 2049])
def test_invalid_graph_cache_limits_are_rejected(budget):
    with pytest.raises(ValueError, match="cache size"):
        cache_budget_bytes(budget)


def test_graph_inputs_do_not_touch_unused_edge_features():
    class WithoutEdgeFeatures:
        x = torch.zeros(2, 3)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch = torch.zeros(2, dtype=torch.long)

        @property
        def edge_attr(self):
            raise AssertionError("A model without edge attributes must not transfer them.")

    x, edges, attrs, batch = hybrid_graph_inputs(
        WithoutEdgeFeatures(), "Graphs", False, "cpu", torch.float32)
    assert x.shape == (2, 3) and edges.dtype == torch.long and attrs is None
    assert batch.shape == (2,)
    assert hybrid_graph_inputs(SimpleNamespace(), "Molecular fingerprints", False,
                               "cpu", torch.float32) == (None, None, None, None)


def test_failed_summary_removes_its_hooks_and_restores_training_flags(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout())
    model.train()
    model[0].eval()
    preserved = model[0].register_forward_hook(lambda *args: None)
    before = {module: (set(module._forward_hooks), module.training) for module in model.modules()}

    def failing_summary(model, *args):
        model.eval()
        for module in model.modules():
            module.register_forward_hook(lambda *args: None)
        raise RuntimeError("summary forward failed")

    monkeypatch.setattr(gnn_runtime, "summary", failing_summary)
    with pytest.raises(RuntimeError, match="summary forward failed"):
        safe_model_summary(model, torch.zeros(1, 2))
    for module, (hooks, training) in before.items():
        assert set(module._forward_hooks) == hooks
        assert module.training == training
    preserved.remove()
