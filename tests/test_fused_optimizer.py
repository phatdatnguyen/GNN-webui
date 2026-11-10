"""Optional optimizer acceleration keeps update and fallback behavior explicit."""

from copy import deepcopy

import pytest
import torch

from training_utils import create_optimizer


@pytest.mark.parametrize("name", ["Adam", "AdamW", "SGD"])
def test_cpu_optimizer_fallback_updates_weights_and_supports_scheduler(name):
    model = torch.nn.Linear(2, 1)
    optimizer = create_optimizer(model, name, 0.01, use_fused=True)
    assert not optimizer.defaults.get("fused")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    before = model.weight.detach().clone()
    model(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    assert not torch.equal(before, model.weight)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("name", ["Adam", "AdamW"])
def test_fused_cuda_updates_match_standard_optimizer(name):
    torch.manual_seed(31)
    reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1)).cuda()
    accelerated = deepcopy(reference)
    standard = create_optimizer(reference, name, 0.003)
    fused = create_optimizer(accelerated, name, 0.003, use_fused=True)
    assert not standard.defaults.get("fused")
    assert fused.defaults.get("fused") is True
    samples = torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3) / 10
    target = torch.tensor([[0.1], [0.5], [0.7], [0.9]], device="cuda")
    for _ in range(4):
        for model, optimizer in ((reference, standard), (accelerated, fused)):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(samples), target).backward()
            optimizer.step()
    for actual, expected in zip(accelerated.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_non_float32_cuda_keeps_standard_updates_when_acceleration_requested():
    model = torch.nn.Linear(2, 1, device="cuda", dtype=torch.float64)
    optimizer = create_optimizer(model, "AdamW", 0.01, use_fused=True)
    assert not optimizer.defaults.get("fused")
    before = model.weight.detach().clone()
    model(torch.ones(2, 2, device="cuda", dtype=torch.float64)).square().mean().backward()
    optimizer.step()
    assert torch.isfinite(model.weight).all()
    assert not torch.equal(before, model.weight)
