"""Hybrid predictions require the trained fingerprint definition, not its molecules."""

from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import binary_classification_hybrid as bc
import regression_hybrid as rg
from conftest import is_error, is_ok
from feature_provenance import capture_prediction_features, ensure_prediction_features


CONFIG = {"name": "Morgan fingerprint", "radius": 2, "number_of_bits": 2048}


def _model(config=CONFIG, mode="Molecular fingerprints"):
    model = torch.nn.Linear(2048, 1)
    model._run_context = {"parameters": {
        "load_data": mode, "target_names": ["saved_target"],
        "feature_sources": {"fingerprint_config": deepcopy(config), "fingerprint_hash": "training-content"},
    }}
    return model


def _prediction(module, monkeypatch, model, dataset):
    calls = []

    def predict(*args):
        calls.append(args)
        values = np.asarray([[0.25], [0.75]], dtype=np.float32)
        return ["CCO", "CCC"], torch.from_numpy(values) if module is bc else values

    monkeypatch.setattr(module, "predict", predict)
    arguments = {
        "model": model, "dataset": SimpleNamespace(target_column_names=["active_target"]),
        "prediction_dataset": dataset, "predict_dataloader": object(), "device": "cpu",
        "datatype": torch.float32, "output_scaler": object(), "gnn_model_tab": "GCN",
        "customgcn_convolutional_layer_name": "GCNConv", "variance_threshold": None, "pca": None,
    }
    result = module.on_predict(**{name: arguments[name] for name in inspect.signature(module.on_predict).parameters})
    return result, calls


@pytest.mark.parametrize("module", [bc, rg], ids=lambda module: module.__name__)
@pytest.mark.parametrize("change", ["algorithm", "radius", "missing"])
def test_prediction_rejects_incompatible_fingerprints_before_inference(module, monkeypatch, change):
    config = deepcopy(CONFIG)
    if change == "algorithm":
        config.update(name="Avalon fingerprint", radius=None)
    elif change == "radius":
        config["radius"] = 3
    else:
        config = None
    dataset = SimpleNamespace(load_data="Molecular fingerprints", feature_sources={"fingerprint_config": config})
    (status, frame, export), calls = _prediction(module, monkeypatch, _model(), dataset)
    assert is_error(status) and "re-extract" in status, status
    assert frame is None and export["interactive"] is False
    assert calls == []


@pytest.mark.parametrize("module", [bc, rg], ids=lambda module: module.__name__)
def test_prediction_accepts_new_molecules_with_matching_settings(module, monkeypatch, tmp_path):
    directory = tmp_path / "FP_new_molecules"
    directory.mkdir()
    (directory / "fingerprint_config.json").write_text(json.dumps(CONFIG))
    dataset = SimpleNamespace(load_data="Molecular fingerprints",
                              df=pd.DataFrame({"SMILES": ["CCO", "CCC"]}))
    capture_prediction_features(dataset, directory)
    # No training fingerprint contents or molecule identities are needed or read.
    assert "fingerprint_hash" not in dataset.feature_sources
    (status, frame, _), calls = _prediction(module, monkeypatch, _model(), dataset)
    assert is_ok(status), status
    assert len(calls) == 1 and frame["SMILES"].tolist() == ["CCO", "CCC"]


@pytest.mark.parametrize("module", [bc, rg], ids=lambda module: module.__name__)
def test_prediction_processing_captures_configuration_once(module, monkeypatch, tmp_path):
    directory = tmp_path / "FP_prediction"
    directory.mkdir()
    config_path = directory / "fingerprint_config.json"
    config_path.write_text(json.dumps(CONFIG))
    dataset = SimpleNamespace(load_data="Molecular fingerprints")
    monkeypatch.setattr(module, "MolecularDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, "DataLoader", lambda *args, **kwargs: object())
    status, processed, _, _ = module.on_process_prediction_data(
        str(tmp_path), "prediction.csv", dataset, None, directory.name, torch.float32, 32)
    assert is_ok(status), status
    assert processed.feature_sources["fingerprint_config"] == CONFIG
    config_path.write_text(json.dumps({**CONFIG, "radius": 3}))
    assert processed.feature_sources["fingerprint_config"] == CONFIG


@pytest.mark.parametrize("mode", ["Graphs", "Both"])
def test_prediction_rejects_model_input_mode_changes(mode):
    dataset = SimpleNamespace(load_data=mode, feature_sources={"fingerprint_config": CONFIG})
    with pytest.raises(ValueError, match="input mode"):
        ensure_prediction_features(_model(), dataset)


def test_graph_only_models_do_not_require_fingerprint_metadata():
    ensure_prediction_features(_model(None, "Graphs"), SimpleNamespace(load_data="Graphs"))


@pytest.mark.parametrize("legacy", ["unnamed", "missing_settings", "unknown_settings"])
def test_legacy_models_without_fingerprint_settings_remain_usable(legacy):
    model = _model(None)
    if legacy == "unnamed":
        del model._run_context
    elif legacy == "missing_settings":
        model._run_context["parameters"].pop("feature_sources")
    ensure_prediction_features(model, SimpleNamespace(load_data="Molecular fingerprints"))


@pytest.mark.parametrize("config_text", ["{broken", "null", '{"radius": NaN}'])
def test_prediction_capture_rejects_malformed_fingerprint_metadata(tmp_path, config_text):
    (tmp_path / "fingerprint_config.json").write_text(config_text)
    with pytest.raises(ValueError, match="configuration"):
        capture_prediction_features(SimpleNamespace(), tmp_path)


def test_prediction_capture_allows_missing_legacy_metadata(tmp_path):
    dataset = SimpleNamespace(load_data="Molecular fingerprints")
    capture_prediction_features(dataset, tmp_path)
    assert dataset.feature_sources["fingerprint_config"] is None
    with pytest.raises(ValueError, match="no extraction metadata"):
        ensure_prediction_features(_model(), dataset)
