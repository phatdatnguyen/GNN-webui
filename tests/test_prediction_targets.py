"""Checkpoint predictions retain the model's target labels across dataset changes."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import binary_classification_hybrid as bc
import regression_hybrid as rg
import binary_classification_3d as bc3
import regression_3d as rg3
from conftest import is_error, is_ok
from utils import prediction_target_names


@pytest.fixture(params=[bc, rg, bc3, rg3], ids=lambda module: module.__name__)
def workflow(request):
    return request.param


def _predict(module, monkeypatch, names, *, recorded=True, shape=None):
    is_3d = module in (bc3, rg3)
    width = 1 if is_3d else 2
    current_names = [f"current_{index}" for index in range(width)]
    dataset = (SimpleNamespace(target_column_name=current_names[0]) if is_3d
               else SimpleNamespace(target_column_names=current_names))
    model = torch.nn.Linear(1, width)
    if recorded:
        model._run_context = {"parameters": {"target_names": names}}
    values = np.arange(np.prod(shape or (2, width)), dtype=np.float32).reshape(shape or (2, width))
    predictions = torch.from_numpy(values) if module in (bc, bc3) else values
    monkeypatch.setattr(module, "predict", lambda *args: (["CCO", "CCN"], predictions))
    arguments = {
        "model": model, "dataset": dataset, "prediction_dataset": dataset,
        "predict_dataloader": object(), "device": "cpu", "datatype": torch.float32,
        "output_scaler": object(), "gnn_model_tab": "GCN",
        "customgcn_convolutional_layer_name": "GCNConv", "variance_threshold": None, "pca": None,
    }
    result = module.on_predict(**{name: arguments[name] for name in inspect.signature(module.on_predict).parameters})
    return result, current_names, predictions


def test_prediction_uses_checkpoint_targets_in_recorded_order(workflow, monkeypatch):
    names = ["saved_target"] if workflow in (bc3, rg3) else ["saved_second", "saved_first"]
    (status, frame, export), current_names, predictions = _predict(workflow, monkeypatch, names)
    assert is_ok(status), status
    suffixes = ["Prediction", "Probability"] if workflow in (bc, bc3) else ["Prediction"]
    assert frame.columns.tolist() == ["SMILES", *[f"{name} ({suffix})" for name in names for suffix in suffixes]]
    assert not any(name in column for name in current_names for column in frame.columns)
    assert frame["SMILES"].tolist() == ["CCO", "CCN"]
    for index, name in enumerate(names):
        if workflow in (bc, bc3):
            np.testing.assert_allclose(frame[f"{name} (Probability)"], torch.sigmoid(predictions[:, index]))
        else:
            np.testing.assert_allclose(frame[f"{name} (Prediction)"], predictions[:, index])
    assert export["interactive"] is True


def test_legacy_prediction_keeps_dataset_target_labels(workflow, monkeypatch):
    (status, frame, _), names, _ = _predict(workflow, monkeypatch, None, recorded=False)
    assert is_ok(status), status
    assert [column for column in frame if column.endswith(" (Prediction)")] == [f"{name} (Prediction)" for name in names]


@pytest.mark.parametrize("names", [None, "saved", [], [""], ["   "], [123], ["same", "same"]])
def test_invalid_recorded_target_names_are_not_replaced_by_current_dataset(workflow, monkeypatch, names):
    (status, frame, export), _, _ = _predict(workflow, monkeypatch, names)
    assert is_error(status) and "target" in status.lower(), status
    assert frame is None
    assert export["interactive"] is False


def test_prediction_rejects_recorded_target_count_mismatch(workflow, monkeypatch):
    (status, frame, export), _, _ = _predict(workflow, monkeypatch, ["first", "second"], shape=(2, 1))
    assert is_error(status) and "output column" in status, status
    assert frame is None and export["interactive"] is False


@pytest.mark.parametrize("malformation", ["dimensions", "rows"])
def test_prediction_rejects_unexpected_output_shape(workflow, monkeypatch, malformation):
    is_3d = workflow in (bc3, rg3)
    names = ["saved"] if is_3d else ["first", "second"]
    shape = ((2, 2) if is_3d else (2,)) if malformation == "dimensions" else (3, 1 if is_3d else 2)
    (status, frame, export), _, _ = _predict(workflow, monkeypatch, names, shape=shape)
    assert is_error(status) and "shape" in status.lower(), status
    assert frame is None and export["interactive"] is False


def test_target_name_helper_preserves_legacy_metadata_fallback():
    model = SimpleNamespace(_run_context={"parameters": {}})
    dataset = SimpleNamespace(target_column_names=["legacy_target"])
    assert prediction_target_names(model, dataset, 1) == ["legacy_target"]
    assert prediction_target_names(SimpleNamespace(), None, 1) == ["target"]


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_target_name_helper_rejects_invalid_prediction_width(width):
    with pytest.raises(ValueError, match="output column"):
        prediction_target_names(SimpleNamespace(), None, width)


@pytest.mark.parametrize("context", [[], {"parameters": []}])
def test_target_name_helper_rejects_malformed_run_metadata(context):
    with pytest.raises(ValueError, match="target names"):
        prediction_target_names(SimpleNamespace(_run_context=context), None, 1)
