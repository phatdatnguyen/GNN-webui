"""Keep failure reporting and pooling correct when reducing CUDA scalar reads."""

from types import SimpleNamespace

import pytest
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
import models


@pytest.mark.parametrize("module", [bc, rg, bc3, rg3])
def test_validation_rejects_nonfinite_loss_without_changing_weights(module):
    class Nonfinite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, *args):
            return (self.value * float("nan")).reshape(1, 1)

    model = Nonfinite()
    before = model.value.detach().clone()
    sample = Data(fp=torch.ones(1, 2), y=torch.ones(1, 1), z=torch.tensor([6]),
                  pos=torch.zeros(1, 3), smiles="C")
    loader = DataLoader([sample, sample], batch_size=1)
    args = [model]
    if module in (bc, rg):
        args.append(SimpleNamespace(load_data="Molecular fingerprints"))
    args.extend([loader, "cpu", torch.float32])
    if module in (rg, rg3):
        args.append(MinMaxScaler().fit([[0.0], [1.0]]))
    if module in (bc, rg):
        args.extend(["GCN", "GCNConv", None, None])
    with pytest.raises(ValueError, match="Validation loss is not finite"):
        module.validation(*args)
    torch.testing.assert_close(model.value, before)
    assert model.value.grad is None


def test_loader_graph_count_reaches_pool_without_inference(monkeypatch):
    original = models.global_mean_pool
    sizes = []

    def sized_pool(x, batch, size=None):
        sizes.append(size)
        return original(x, batch, size=size)

    monkeypatch.setattr(models, "global_mean_pool", sized_pool)
    model = models.GCNModel(2, 4, 1, 2, 0.0, 0, 4, 1, 0, 4, 1, 1)
    samples = [Data(x=torch.ones(2, 2), edge_index=torch.tensor([[0, 1], [1, 0]]),
                    y=torch.zeros(1, 1), smiles="CC") for _ in range(3)]
    bc.validation(model, SimpleNamespace(load_data="Graphs"), DataLoader(samples, batch_size=2),
                  "cpu", torch.float32, "GCN", "GCNConv", None, None)
    assert sizes == [2, 1]
