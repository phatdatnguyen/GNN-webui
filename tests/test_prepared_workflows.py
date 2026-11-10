"""Exercise prepared and batch-time preprocessing through all four training paths."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import binary_classification_hybrid as classification_hybrid
import binary_classification_3d as classification_3d
import regression_hybrid
import regression_3d
from datasets import MolecularDataset, MolecularDataset3D, prepare_dataset
from utils import write_graph_manifest


WORKFLOWS = [classification_hybrid, regression_hybrid, classification_3d, regression_3d]


class TinyHybrid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 1)

    def forward(self, x, edge_index, batch, fingerprints):
        return self.head(torch.cat([global_mean_pool(x, batch), fingerprints], dim=1))


class Tiny3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 1)

    def forward(self, z, pos, batch):
        return self.head(global_mean_pool(pos, batch))


def build_case(tmp_path, module, datatype):
    is_3d = module in (classification_3d, regression_3d)
    is_regression = module in (regression_hybrid, regression_3d)
    smiles = ["CCO", "CCN", "CCC", "CCS", "CCCl", "CCBr", "CNC", "COC"]
    targets = [100000.125 + index * .125 if is_regression else index % 2 for index in range(8)]
    pd.DataFrame({"SMILES": smiles, "target": targets}).to_csv(tmp_path / "dataset.csv", index=False)
    graph_dir, fp_dir = tmp_path / "GRAPH_graphs", tmp_path / "FP_fps"
    graph_dir.mkdir()
    fp_dir.mkdir()
    for index, molecule in enumerate(smiles):
        torch.save(Data(x=torch.tensor([[0.1, 0.2], [0.3, 0.4]]) + index / 10,
                        edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 1),
                        z=torch.tensor([6, 8]), pos=torch.tensor([[0., 0., 0.], [0., 1., 1.]]) + index / 10,
                        smiles=molecule), graph_dir / f"{index}.pt")
    write_graph_manifest(graph_dir, smiles)
    fingerprints = pd.DataFrame([[index >> bit & 1 for bit in range(3)] for index in range(8)],
                                columns=["Bit_1", "Bit_2", "Bit_3"])
    fingerprints.insert(0, "SMILES", smiles)
    fingerprints.to_csv(fp_dir / "molecular_fingerprints.csv", index=False)
    kwargs = {"target_datatype": torch.float64 if is_regression else datatype}
    if is_3d:
        dataset = MolecularDataset3D(str(tmp_path / "dataset.csv"), str(graph_dir), datatype, **kwargs)
        variance, pca = None, None
    else:
        dataset = MolecularDataset(str(tmp_path / "dataset.csv"), "Both", str(graph_dir), str(fp_dir),
                                   datatype, **kwargs)
        variance = VarianceThreshold().fit(dataset.fingerprint_values[:5])
        pca = PCA(n_components=2).fit(variance.transform(dataset.fingerprint_values[:5]))
    scaler = MinMaxScaler().fit(np.asarray(targets[:5]).reshape(-1, 1)) if is_regression else None
    # Capture raw rows before preparation, exercising the actual fallback code.
    fallback = [dataset[index] for index in range(len(dataset))]
    torch.manual_seed(11)
    model = (Tiny3D() if is_3d else TinyHybrid()).to(datatype)
    return dataset, fallback, model, variance, pca, scaler


def invoke(module, action, model, dataset, loader, datatype, variance, pca, scaler, optimizer=None):
    arguments = [model]
    if module in (classification_hybrid, regression_hybrid):
        arguments.append(dataset)
    arguments.extend([loader, "cpu", datatype])
    if module in (regression_hybrid, regression_3d):
        arguments.append(scaler)
    if action == "train":
        arguments.append(optimizer)
    if module in (classification_hybrid, regression_hybrid):
        arguments.extend(["Custom", "GCNConv", variance, pca])
    return getattr(module, action)(*arguments)


@pytest.mark.parametrize("module", WORKFLOWS, ids=lambda value: value.__name__)
@pytest.mark.parametrize("datatype", [torch.float32, torch.float64])
def test_prepared_and_batch_time_paths_match_training_evaluation_and_prediction(tmp_path, module, datatype):
    dataset, fallback, model, variance, pca, scaler = build_case(tmp_path, module, datatype)
    reference_model = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    reference_optimizer = torch.optim.Adam(reference_model.parameters(), lr=.005)
    train_indices, test_indices = [0, 1, 2, 3, 4], [5, 6, 7]
    prepared_train = DataLoader(Subset(dataset, train_indices), batch_size=3)
    raw_train = DataLoader(Subset(fallback, train_indices), batch_size=3)
    prepared_test = DataLoader(Subset(dataset, test_indices), batch_size=2)
    raw_test = DataLoader(Subset(fallback, test_indices), batch_size=2)
    for _ in range(2):
        actual = invoke(module, "train", model, dataset, prepared_train, datatype, variance, pca, scaler, optimizer)
        expected = invoke(module, "train", reference_model, dataset, raw_train, datatype,
                          variance, pca, scaler, reference_optimizer)
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
        actual_validation = invoke(module, "validation", model, dataset, prepared_test, datatype, variance, pca, scaler)
        expected_validation = invoke(module, "validation", reference_model, dataset, raw_test, datatype, variance, pca, scaler)
        assert actual_validation == pytest.approx(expected_validation, rel=1e-6, abs=1e-7)
    for parameter, reference in zip(model.parameters(), reference_model.parameters(), strict=True):
        torch.testing.assert_close(parameter, reference, atol=1e-7, rtol=1e-6)
    actual = invoke(module, "test", model, dataset, prepared_test, datatype, variance, pca, scaler)
    expected = invoke(module, "test", reference_model, dataset, raw_test, datatype, variance, pca, scaler)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-7, rtol=1e-6)
    assert actual[2] == expected[2]
    actual = invoke(module, "predict", model, dataset, prepared_test, datatype, variance, pca, scaler)
    expected = invoke(module, "predict", reference_model, dataset, raw_test, datatype, variance, pca, scaler)
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("module", [classification_hybrid, regression_hybrid, regression_3d],
                         ids=lambda value: value.__name__)
def test_checkpoint_restored_transformers_replace_prepared_cache_before_use(tmp_path, module):
    datatype = torch.float64
    dataset, fallback, model, variance, pca, scaler = build_case(tmp_path, module, datatype)
    loader = DataLoader(dataset, batch_size=3)
    raw_loader = DataLoader(fallback, batch_size=3)
    prepare_dataset(dataset, variance, pca, scaler)
    previous_fp, previous_y = dataset._fp_prepared, dataset._y_prepared
    saved_pca, saved_scaler = deepcopy(pca), deepcopy(scaler)
    if saved_pca is not None:
        saved_pca.components_ *= -1
    if saved_scaler is not None:
        saved_scaler.fit(np.array([[99999.0], [100002.0]]))
    checkpoint = {"model_class": type(model).__name__, "state_dict": model.state_dict(),
                  "trained_epochs": 0, "variance_threshold": variance, "pca": saved_pca,
                  "output_scaler": saved_scaler}
    torch.save(checkpoint, tmp_path / "restored.ckpt")
    arguments = [str(tmp_path), "restored.ckpt", model, [], [], 0]
    if module in (classification_hybrid, regression_hybrid):
        arguments.extend([variance, pca])
    if module in (regression_hybrid, regression_3d):
        arguments.append(scaler)
    loaded = module.on_load_checkpoint(*arguments)
    assert "color: green" in loaded[0], loaded[0]
    if module in (classification_hybrid, regression_hybrid):
        variance, pca = loaded[5:7]
    if module is regression_hybrid:
        scaler = loaded[7]
    elif module is regression_3d:
        scaler = loaded[5]
    actual = invoke(module, "predict", model, dataset, loader, datatype, variance, pca, scaler)
    expected = invoke(module, "predict", model, dataset, raw_loader, datatype, variance, pca, scaler)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-10, rtol=1e-12)
    assert actual[0] == expected[0]
    if pca is not None:
        assert dataset._fp_prepared is not previous_fp
    if scaler is not None:
        assert dataset._y_prepared is not previous_y
        actual_loss = invoke(module, "validation", model, dataset, loader, datatype, variance, pca, scaler)
        expected_loss = invoke(module, "validation", model, dataset, raw_loader, datatype, variance, pca, scaler)
        assert actual_loss == pytest.approx(expected_loss, rel=1e-12)
