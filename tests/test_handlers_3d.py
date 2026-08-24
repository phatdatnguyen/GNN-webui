"""Tests for the 3D tabs. Marked slow: every fixture needs conformer generation."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

import binary_classification_3d as bc3
import regression_3d as rg3
from conftest import FakeProgress, SMILES_3D, create_model_3d_args, is_error, is_ok

pytestmark = pytest.mark.slow


def process_3d_classification(workdir):
    return bc3.on_process_data(workdir, "dataset.csv", "g3d", torch.float32, 0.25, 0.25, 4, 0)


@pytest.fixture
def classifier_3d(workdir_3d):
    status, _, dataset, train_dl, val_dl, test_dl = process_3d_classification(workdir_3d)
    assert is_ok(status), status

    result = bc3.on_create_model(*create_model_3d_args(dataset, "SchNet"))
    assert is_ok(result[0]), result[0]
    model, train_losses, val_losses, trained_epochs = result[-4:]

    _, optimizer, scheduler, _ = bc3.on_create_optimizer(model, "Adam", 0.001, 0.99)
    status, _, _, trained_epochs, train_losses, val_losses = bc3.on_train(
        model, train_dl, val_dl, "cpu", torch.float32, optimizer, scheduler, 2,
        trained_epochs, train_losses, val_losses, FakeProgress())
    assert is_ok(status), status

    return dict(workdir=workdir_3d, dataset=dataset, model=model, test_dl=test_dl,
                train_losses=train_losses, val_losses=val_losses, trained_epochs=trained_epochs)


# ----------------------------------------------------------------------- data processing

def test_process_3d_data(workdir_3d):
    status, create_btn, dataset, train_dl, val_dl, test_dl = process_3d_classification(workdir_3d)
    assert is_ok(status)
    assert len(dataset) == len(SMILES_3D)
    assert sum(len(dl.dataset) for dl in (train_dl, val_dl, test_dl)) == len(SMILES_3D)


def test_process_3d_data_rejects_non_binary_targets(workdir_3d):
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = range(len(df))
    df.to_csv(path, index=False)

    status, create_btn = process_3d_classification(workdir_3d)[:2]
    assert is_error(status)
    assert create_btn is not None and create_btn.get("interactive") is False


# ------------------------------------------------------------------------ model creation

@pytest.mark.parametrize("architecture", ["SchNet", "DimeNet++", "ViSNet"])
def test_create_each_3d_architecture(workdir_3d, architecture):
    dataset = process_3d_classification(workdir_3d)[2]
    result = bc3.on_create_model(*create_model_3d_args(dataset, architecture))
    assert is_ok(result[0]), result[0]
    assert result[-4] is not None


def test_unknown_architecture_is_reported_as_an_error(workdir_3d):
    """Regression: an unmatched tab name returned 'created successfully' with model=None,
    and the failure only surfaced later as an AttributeError."""
    dataset = process_3d_classification(workdir_3d)[2]
    result = bc3.on_create_model(*create_model_3d_args(dataset, "NotAModel"))
    assert is_error(result[0])
    assert result[-4] is None


# ---------------------------------------------------------------- training and evaluation

def test_3d_training_records_losses(classifier_3d):
    assert len(classifier_3d["train_losses"]) == 2
    assert classifier_3d["trained_epochs"] == 2
    assert all(np.isfinite(classifier_3d["train_losses"]))


def test_3d_evaluate(classifier_3d):
    c = classifier_3d
    status, eval_df, metrics, cm_figure, roc_figure, export_btn = bc3.on_evaluate(
        c["model"], c["dataset"], c["test_dl"], "cpu", torch.float32)

    assert is_ok(status)
    assert len(eval_df) == len(c["test_dl"].dataset)
    assert list(metrics["Metric"]) == ["Accuracy", "Precision", "Recall", "Specificity",
                                       "F1 score", "ROC-AUC score"]
    assert cm_figure is not None and roc_figure is not None


def test_3d_evaluate_without_a_model_reports_an_error(classifier_3d):
    status, eval_df, metrics, cm, roc, _ = bc3.on_evaluate(
        None, classifier_3d["dataset"], classifier_3d["test_dl"], "cpu", torch.float32)
    assert is_error(status) and eval_df is None


def test_3d_prediction(classifier_3d):
    c = classifier_3d
    status, prediction_dataset, dataloader, _ = bc3.on_process_prediction_data(
        c["workdir"], "dataset.csv", "g3d", torch.float32, 4)
    assert is_ok(status)

    status, prediction_df, _ = bc3.on_predict(c["model"], c["dataset"], dataloader,
                                              "cpu", torch.float32)
    assert is_ok(status)
    assert len(prediction_df) == len(SMILES_3D)


# --------------------------------------------------------------------------- checkpoints

def test_3d_checkpoint_round_trip(classifier_3d):
    c = classifier_3d
    status, _ = bc3.on_save_checkpoint(c["workdir"], "ckpt", c["model"], c["trained_epochs"])
    assert is_ok(status)

    status, model, train_losses, val_losses, trained_epochs = bc3.on_load_checkpoint(
        c["workdir"], f"ckpt_{c['trained_epochs']}.ckpt", c["model"],
        c["train_losses"], c["val_losses"], c["trained_epochs"])
    assert is_ok(status)
    assert trained_epochs == c["trained_epochs"]


def test_3d_failed_load_preserves_state(classifier_3d):
    c = classifier_3d
    status, _, train_losses, val_losses, trained_epochs = bc3.on_load_checkpoint(
        c["workdir"], "missing.ckpt", c["model"], [0.1], [0.2], 5)

    assert is_error(status)
    assert train_losses == [0.1] and val_losses == [0.2] and trained_epochs == 5


def test_3d_architecture_mismatch_is_rejected(workdir_3d):
    dataset = process_3d_classification(workdir_3d)[2]
    schnet = bc3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]
    visnet = bc3.on_create_model(*create_model_3d_args(dataset, "ViSNet"))[-4]

    bc3.on_save_checkpoint(workdir_3d, "ckpt", schnet, 1)
    status, _, train_losses, _, trained_epochs = bc3.on_load_checkpoint(
        workdir_3d, "ckpt_1.ckpt", visnet, [0.3], [0.4], 2)

    assert is_error(status) and "SchNet" in status
    assert train_losses == [0.3] and trained_epochs == 2


# ------------------------------------------------------------------------- 3D regression

def test_3d_regression_flow(tmp_path, workdir_3d):
    """Targets must be numeric here, so rewrite the binary column with real values."""
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = [float(i) * 2.5 for i in range(len(df))]
    df.to_csv(path, index=False)

    status, _, dataset, train_dl, val_dl, test_dl, scaler = rg3.on_process_data(
        workdir_3d, "dataset.csv", "g3d", torch.float32, 0.25, 0.25, 4, 0)
    assert is_ok(status), status
    assert scaler is not None

    model = rg3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]
    _, optimizer, scheduler, _ = rg3.on_create_optimizer(model, "Adam", 0.001, 0.99)
    status, _, _, trained_epochs, train_losses, _ = rg3.on_train(
        model, train_dl, val_dl, "cpu", torch.float32, scaler, optimizer, scheduler, 2,
        0, [], [], FakeProgress())
    assert is_ok(status) and len(train_losses) == 2

    status, eval_df, metrics, scatter, _ = rg3.on_evaluate(
        model, dataset, test_dl, "cpu", torch.float32, scaler)
    assert is_ok(status)
    assert list(metrics["Metric"]) == ["MAE", "MSE", "RMSE", "R²"]


def test_3d_regression_checkpoint_carries_the_scaler(workdir_3d):
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = [float(i) * 2.5 for i in range(len(df))]
    df.to_csv(path, index=False)

    _, _, dataset, _, _, _, scaler = rg3.on_process_data(
        workdir_3d, "dataset.csv", "g3d", torch.float32, 0.25, 0.25, 4, 0)
    model = rg3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]

    rg3.on_save_checkpoint(workdir_3d, "ckpt", model, 0, scaler)
    restored_scaler = rg3.on_load_checkpoint(workdir_3d, "ckpt_0.ckpt", model, [], [], 0, None)[-1]

    assert restored_scaler is not None
    assert np.allclose(restored_scaler.data_min_, scaler.data_min_)


def test_3d_regression_evaluate_without_a_scaler_reports_an_error(workdir_3d):
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = [float(i) * 2.5 for i in range(len(df))]
    df.to_csv(path, index=False)

    _, _, dataset, _, _, test_dl, _ = rg3.on_process_data(
        workdir_3d, "dataset.csv", "g3d", torch.float32, 0.25, 0.25, 4, 0)
    model = rg3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]

    status, eval_df = rg3.on_evaluate(model, dataset, test_dl, "cpu", torch.float32, None)[:2]
    assert is_error(status) and eval_df is None
