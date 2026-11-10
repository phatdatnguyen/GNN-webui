"""Tests for the Gradio callbacks in binary_classification_hybrid.py and regression_hybrid.py.

These are the functions wired to buttons, so they return status HTML plus component
updates rather than raising. The tests assert on that contract.
"""
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import binary_classification_hybrid as bc
import regression_hybrid as rg
from conftest import FakeProgress, SMILES, create_model_args, is_error, is_ok
from gnn_run_ui import route_model_export
from run_utils import run_directory, list_saved_models


def process_classification(workdir, load_data="Both", dimensionality_reduction=False,
                           test_ratio=0.25, val_ratio=0.25, batch_size=4, shuffle=False, seed=0):
    return bc.on_process_data(
        workdir, "dataset.csv", load_data, "GRAPH_graphs", "FP_fps", torch.float32,
        dimensionality_reduction, 0.0, 4, test_ratio, val_ratio, batch_size, shuffle, seed,
    )


def process_regression(workdir, load_data="Both", dimensionality_reduction=False, shuffle=False):
    return rg.on_process_data(
        workdir, "dataset.csv", load_data, "GRAPH_graphs", "FP_fps", torch.float32,
        dimensionality_reduction, 0.0, 4, 0.25, 0.25, 4, shuffle, 0,
    )


@pytest.fixture
def trained_classifier(classification_workdir):
    """A trained GCNConv classifier plus everything needed to evaluate it."""
    status, _, _, dataset, train_dl, val_dl, test_dl, vt, pca = process_classification(
        classification_workdir)
    assert is_ok(status), status

    result = bc.on_create_model(*create_model_args(dataset, "Custom", vt, pca))
    assert is_ok(result[0]), result[0]
    model, train_losses, val_losses, trained_epochs = result[-4:]

    _, optimizer, scheduler, _ = bc.on_create_optimizer(model, "Adam", 0.01, 0.99)
    status, _, _, trained_epochs, train_losses, val_losses = bc.on_train(
        model, dataset, train_dl, val_dl, "cpu", torch.float32, optimizer, scheduler, 2,
        trained_epochs, train_losses, val_losses, "Custom", "GCNConv", vt, pca, FakeProgress())
    assert is_ok(status), status

    return dict(workdir=classification_workdir, dataset=dataset, model=model,
                train_dl=train_dl, test_dl=test_dl, vt=vt, pca=pca,
                train_losses=train_losses, val_losses=val_losses,
                trained_epochs=trained_epochs)


# ----------------------------------------------------------------------- data processing

@pytest.mark.parametrize("load_data", ["Graphs", "Molecular fingerprints", "Both"])
def test_process_data_succeeds_for_each_input_mode(classification_workdir, load_data):
    status, create_btn, target_dd, dataset, train_dl, val_dl, test_dl, vt, pca = \
        process_classification(classification_workdir, load_data)

    assert is_ok(status)
    assert f"{len(SMILES)} molecules" in status
    assert dataset.load_data == load_data
    assert all(dl is not None for dl in (train_dl, val_dl, test_dl))


def test_process_data_rejects_non_binary_targets(classification_workdir):
    path = os.path.join(classification_workdir, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = range(len(df))
    df.to_csv(path, index=False)

    status = process_classification(classification_workdir)[0]
    assert is_error(status) and "not binary" in status


def test_regression_rejects_non_numeric_targets(regression_workdir):
    path = os.path.join(regression_workdir, "dataset.csv")
    df = pd.read_csv(path)
    df["value"] = ["a"] * len(df)
    df.to_csv(path, index=False)

    status = process_regression(regression_workdir)[0]
    assert is_error(status) and "not numeric" in status


def test_process_data_rejects_a_dataset_with_no_targets(classification_workdir):
    path = os.path.join(classification_workdir, "dataset.csv")
    pd.read_csv(path)[["SMILES"]].to_csv(path, index=False)

    status = process_classification(classification_workdir)[0]
    assert is_error(status) and "No target columns" in status


def test_process_data_rejects_an_empty_training_split(classification_workdir):
    status = process_classification(classification_workdir, test_ratio=0.6, val_ratio=0.6)[0]
    assert is_error(status) and "split" in status.lower()


def test_process_data_error_path_disables_the_create_button(classification_workdir):
    """Regression: the failure path used to return None into component slots."""
    path = os.path.join(classification_workdir, "dataset.csv")
    pd.read_csv(path)[["SMILES"]].to_csv(path, index=False)

    status, create_btn, target_dd = process_classification(classification_workdir)[:3]
    assert is_error(status)
    assert create_btn is not None and create_btn.get("interactive") is False
    assert target_dd is not None


def test_stale_dimensionality_reduction_flag_is_ignored_without_fingerprints(classification_workdir):
    """Regression: the checkbox is only hidden - not reset - when switching to Graphs, and
    the handler then tried to read a fingerprint file that may not exist."""
    status, _, _, _, _, _, _, vt, pca = process_classification(
        classification_workdir, "Graphs", dimensionality_reduction=True)
    assert is_ok(status)
    assert vt is None and pca is None


def test_dimensionality_reduction_is_fitted_on_the_training_split_only(classification_workdir):
    """Fitting PCA on the whole dataset would leak test information into training."""
    _, _, _, dataset, train_dl, _, _, vt, pca = process_classification(
        classification_workdir, "Both", dimensionality_reduction=True)
    assert vt is not None and pca.n_components_ == 4
    assert pca.n_samples_ == len(train_dl.dataset)


def test_training_data_is_reshuffled_between_epochs_when_enabled(classification_workdir):
    """Regression: the training loader never shuffled, so every epoch saw the same batches
    in the same order."""
    train_dl = process_classification(classification_workdir, "Graphs", shuffle=True)[4]
    first = [batch.smiles for batch in train_dl]
    second = [batch.smiles for batch in train_dl]
    assert first != second
    assert sorted(sum(first, [])) == sorted(sum(second, []))     # same molecules, new order


def test_shuffling_can_be_turned_off_for_a_fixed_batch_order(classification_workdir):
    train_dl = process_classification(classification_workdir, "Graphs", shuffle=False)[4]
    assert [b.smiles for b in train_dl] == [b.smiles for b in train_dl]


def test_shuffled_runs_are_reproducible_for_a_fixed_seed(classification_workdir):
    """Shuffling is seeded from the random seed, so it does not cost reproducibility."""
    first = [b.smiles for b in process_classification(
        classification_workdir, "Graphs", shuffle=True, seed=7)[4]]
    second = [b.smiles for b in process_classification(
        classification_workdir, "Graphs", shuffle=True, seed=7)[4]]
    other = [b.smiles for b in process_classification(
        classification_workdir, "Graphs", shuffle=True, seed=8)[4]]

    assert first == second
    assert first != other


def test_validation_and_test_order_stay_deterministic(classification_workdir):
    _, _, _, _, _, val_dl, test_dl, _, _ = process_classification(
        classification_workdir, "Graphs", shuffle=True)
    assert [b.smiles for b in val_dl] == [b.smiles for b in val_dl]
    assert [b.smiles for b in test_dl] == [b.smiles for b in test_dl]


# ------------------------------------------------------------------------ model creation

@pytest.mark.parametrize("tab", ["GCN", "GraphSAGE", "GIN", "GAT", "EdgeCNN", "AttentiveFP"])
def test_create_model_for_each_architecture(classification_workdir, tab):
    dataset = process_classification(classification_workdir)[3]
    result = bc.on_create_model(*create_model_args(dataset, tab))
    assert is_ok(result[0]), result[0]
    assert result[-4] is not None
    assert result[1]                       # a model summary string


@pytest.mark.parametrize("layer", ["GCNConv", "ChebConv", "GATConv", "TransformerConv", "PDNConv"])
@pytest.mark.parametrize("n_layers", [1, 3])
def test_create_custom_model_at_every_layer_count(classification_workdir, layer, n_layers):
    dataset = process_classification(classification_workdir)[3]
    result = bc.on_create_model(*create_model_args(
        dataset, "Custom", customgcn_convolutional_layer_name=layer,
        customgcn_n_layers=n_layers))
    assert is_ok(result[0]), result[0]


def test_creating_a_model_disables_training_until_an_optimizer_exists(classification_workdir):
    """Regression: Train stayed enabled after creating a new model, so it could run with an
    optimizer bound to the previous model - which updates nothing and reports success."""
    dataset = process_classification(classification_workdir)[3]
    result = bc.on_create_model(*create_model_args(dataset, "GCN"))
    train_button_update = result[7]
    assert train_button_update.get("interactive") is False

    model = result[-4]
    _, optimizer, scheduler, train_button = bc.on_create_optimizer(model, "Adam", 0.01, 0.99)
    assert train_button.get("interactive") is True


def test_training_with_an_optimizer_from_another_model_is_refused(classification_workdir):
    """The guard behind the UI change: if the states ever do get out of step, training must
    report an error rather than silently doing nothing."""
    _, _, _, dataset, train_dl, val_dl, _, vt, pca = process_classification(classification_workdir)
    model_a = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    _, optimizer, scheduler, _ = bc.on_create_optimizer(model_a, "Adam", 0.1, 0.99)
    model_b = bc.on_create_model(*create_model_args(dataset, "GCN", gcn_num_layers=3))[-4]

    before = [p.detach().clone() for p in model_b.parameters()]
    status, _, _, trained_epochs, train_losses, _ = bc.on_train(
        model_b, dataset, train_dl, val_dl, "cpu", torch.float32, optimizer, scheduler, 2,
        0, [], [], "GCN", "GCNConv", vt, pca, FakeProgress())

    assert is_error(status)
    assert "Create optimizer" in status
    assert trained_epochs == 0 and train_losses == []
    assert all(torch.equal(p.detach(), q) for p, q in zip(model_b.parameters(), before))


def test_training_actually_updates_the_model(classification_workdir):
    """The positive counterpart: with a matching optimizer, parameters must move."""
    _, _, _, dataset, train_dl, val_dl, _, vt, pca = process_classification(classification_workdir)
    model = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    _, optimizer, scheduler, _ = bc.on_create_optimizer(model, "Adam", 0.1, 0.99)

    before = [p.detach().clone() for p in model.parameters()]
    status, _, _, trained_epochs, train_losses, _ = bc.on_train(
        model, dataset, train_dl, val_dl, "cpu", torch.float32, optimizer, scheduler, 2,
        0, [], [], "GCN", "GCNConv", vt, pca, FakeProgress())

    assert is_ok(status)
    assert trained_epochs == 2
    assert any(not torch.equal(p.detach(), q) for p, q in zip(model.parameters(), before))


def test_training_reports_failures_instead_of_raising(classification_workdir):
    """Regression: on_train had no try/except, so any failure produced a raw traceback and
    left train_losses and trained_epochs disagreeing."""
    _, _, _, dataset, train_dl, _, _, vt, pca = process_classification(classification_workdir)
    model = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    _, optimizer, scheduler, _ = bc.on_create_optimizer(model, "Adam", 0.01, 0.99)

    losses = []
    status, figure, _, trained_epochs, train_losses, val_losses = bc.on_train(
        model, dataset, train_dl, None, "cpu", torch.float32, optimizer, scheduler, 2,
        0, losses, [], "GCN", "GCNConv", vt, pca, FakeProgress())

    assert is_error(status)
    assert len(train_losses) == len(val_losses) == trained_epochs   # state stays consistent
    assert figure is not None


def test_training_keeps_completed_epochs_when_a_later_one_fails(classification_workdir):
    """A failure part-way through must not discard the epochs that did finish."""
    _, _, _, dataset, train_dl, val_dl, _, vt, pca = process_classification(classification_workdir)
    model = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    _, optimizer, scheduler, _ = bc.on_create_optimizer(model, "Adam", 0.01, 0.99)

    st, _, _, epochs_done, train_losses, val_losses = bc.on_train(
        model, dataset, train_dl, val_dl, "cpu", torch.float32, optimizer, scheduler, 2,
        0, [], [], "GCN", "GCNConv", vt, pca, FakeProgress())
    assert is_ok(st) and epochs_done == 2

    # Now break the validation loader and ask for more epochs.
    st, _, _, epochs_after, train_losses, val_losses = bc.on_train(
        model, dataset, train_dl, None, "cpu", torch.float32, optimizer, scheduler, 3,
        epochs_done, train_losses, val_losses, "GCN", "GCNConv", vt, pca, FakeProgress())

    assert is_error(st)
    assert epochs_after == 2
    assert len(train_losses) == len(val_losses) == 2


def test_training_refuses_a_model_on_another_device(classification_workdir):
    """Regression: switching the Device dropdown after creating a model raised an uncaught
    RuntimeError about mismatched tensors."""
    _, _, _, dataset, train_dl, val_dl, _, vt, pca = process_classification(classification_workdir)
    model = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]   # built on cpu
    _, optimizer, scheduler, _ = bc.on_create_optimizer(model, "Adam", 0.01, 0.99)

    status = bc.on_train(model, dataset, train_dl, val_dl, "cuda", torch.float32,
                         optimizer, scheduler, 1, 0, [], [], "GCN", "GCNConv",
                         vt, pca, FakeProgress())[0]
    assert is_error(status)
    assert "cpu" in status and "cuda" in status


def test_create_model_resets_training_state(classification_workdir):
    dataset = process_classification(classification_workdir)[3]
    result = bc.on_create_model(*create_model_args(dataset, "GCN"))
    model, train_losses, val_losses, trained_epochs = result[-4:]
    assert train_losses == [] and val_losses == [] and trained_epochs == 0


def test_create_model_reports_failure_instead_of_raising(classification_workdir):
    """Both output widths at zero is a configuration the sliders allow."""
    dataset = process_classification(classification_workdir)[3]
    result = bc.on_create_model(*create_model_args(
        dataset, "GCN", gcn_n_outputs=0, mlp_n_outputs=0))
    assert is_error(result[0])
    assert result[-4] is None


def test_fingerprint_only_mode_builds_an_mlp(classification_workdir):
    dataset = process_classification(classification_workdir, "Molecular fingerprints")[3]
    result = bc.on_create_model(*create_model_args(dataset, "GCN"))
    assert is_ok(result[0]), result[0]
    assert type(result[-4]).__name__ == "MLPModel"


def test_model_input_width_follows_pca_when_enabled(classification_workdir):
    _, _, _, dataset, _, _, _, vt, pca = process_classification(
        classification_workdir, "Both", dimensionality_reduction=True)
    result = bc.on_create_model(*create_model_args(dataset, "GCN", vt, pca))
    assert is_ok(result[0]), result[0]
    assert result[-4].fingerprint_encoder[0].in_features == pca.n_components_


# ------------------------------------------------------------------------------ training

def test_training_records_one_loss_per_epoch(trained_classifier):
    assert len(trained_classifier["train_losses"]) == 2
    assert len(trained_classifier["val_losses"]) == 2
    assert trained_classifier["trained_epochs"] == 2
    assert all(np.isfinite(trained_classifier["train_losses"]))


def test_training_resumes_from_the_existing_epoch_count(trained_classifier):
    t = trained_classifier
    _, optimizer, scheduler, _ = bc.on_create_optimizer(t["model"], "Adam", 0.01, 0.99)
    status, _, _, trained_epochs, train_losses, _ = bc.on_train(
        t["model"], t["dataset"], t["train_dl"], t["train_dl"], "cpu", torch.float32,
        optimizer, scheduler, 3, t["trained_epochs"], t["train_losses"], t["val_losses"],
        "Custom", "GCNConv", t["vt"], t["pca"], FakeProgress())

    assert trained_epochs == 5
    assert len(train_losses) == 5


def test_epoch_loss_is_independent_of_batch_size(classification_workdir):
    """Regression: the epoch loss was a mean of per-batch losses, so it shifted with the
    batch size purely because the last batch is short."""
    losses = []
    for batch_size in (4, 5):
        _, _, _, dataset, train_dl, val_dl, _, vt, pca = process_classification(
            classification_workdir, "Graphs", batch_size=batch_size)
        torch.manual_seed(0)
        result = bc.on_create_model(*create_model_args(dataset, "GCN", vt, pca))
        model = result[-4]
        losses.append(bc.validation(model, dataset, val_dl, "cpu", torch.float32,
                                    "GCN", "GCNConv", vt, pca))

    assert losses[0] == pytest.approx(losses[1], rel=1e-4)


def test_regression_epoch_loss_is_a_true_rmse(regression_workdir):
    """sqrt of the mean squared error over the epoch, not a mean of per-batch RMSEs."""
    _, _, _, dataset, _, val_dl, _, vt, pca, scaler = process_regression(regression_workdir, "Graphs")
    torch.manual_seed(0)
    model = rg.on_create_model(*create_model_args(dataset, "GCN", vt, pca))[-4]

    reported = rg.validation(model, dataset, val_dl, "cpu", torch.float32, scaler,
                             "GCN", "GCNConv", vt, pca)

    squared_error, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            output = model(batch.x.float(), batch.edge_index.long(), batch.batch, None)
            target = torch.tensor(scaler.transform(batch.y), dtype=torch.float32)
            squared_error += float(((output - target) ** 2).sum())
            n += target.numel()
    assert reported == pytest.approx((squared_error / n) ** 0.5, rel=1e-5)


# ---------------------------------------------------------------------------- evaluation

def test_evaluate_produces_one_row_per_test_molecule(trained_classifier):
    t = trained_classifier
    status, eval_df, export_btn, metrics_export_btn = bc.on_evaluate(
        t["model"], t["dataset"], t["test_dl"], "cpu", torch.float32,
        "Custom", "GCNConv", t["vt"], t["pca"])

    assert is_ok(status)
    assert len(eval_df) == len(t["test_dl"].dataset)
    assert list(eval_df.columns) == ["SMILES", "target", "target (Prediction)", "target (Probability)"]
    assert eval_df["target (Prediction)"].isin([0, 1]).all()
    assert eval_df["target (Probability)"].between(0, 1).all()
    assert export_btn["interactive"] is True and metrics_export_btn["interactive"] is True


def test_evaluate_reports_an_error_without_a_model(trained_classifier):
    status, eval_df, export_btn, metrics_export_btn = bc.on_evaluate(
        None, trained_classifier["dataset"], trained_classifier["test_dl"], "cpu",
        torch.float32, "Custom", "GCNConv", None, None)
    assert is_error(status) and eval_df is None
    assert export_btn["interactive"] is False and metrics_export_btn["interactive"] is False


def test_metrics_are_computed_for_a_normal_split():
    eval_df = pd.DataFrame({
        "SMILES": ["CCO", "CCN", "CCC", "CCCl"],
        "target": [1, 0, 1, 0],
        "target (Prediction)": [1, 0, 0, 0],
        "target (Probability)": [0.9, 0.2, 0.4, 0.1],
    })
    metrics, cm_figure, roc_figure = bc.on_target_column_change(eval_df, "target")

    assert list(metrics["Metric"]) == ["Accuracy", "Balanced accuracy", "Precision", "Recall", "Specificity",
                                       "F1 score", "MCC", "ROC-AUC score", "AUPRC", "Average precision"]
    assert metrics.set_index("Metric").loc["Accuracy", "Value"] == pytest.approx(0.75)
    assert metrics.set_index("Metric").loc["Specificity", "Value"] == pytest.approx(1.0)
    assert metrics.set_index("Metric").loc["Balanced accuracy", "Value"] == pytest.approx(0.75)
    assert metrics.set_index("Metric").loc["MCC", "Value"] == pytest.approx(0.5774)
    assert metrics.set_index("Metric").loc["AUPRC", "Value"] == pytest.approx(1.0)
    assert metrics.set_index("Metric").loc["Average precision", "Value"] == pytest.approx(1.0)
    assert cm_figure is not None and roc_figure is not None


def test_metrics_survive_a_single_class_test_split():
    """Regression: roc_auc_score raises when only one class is present, which used to
    surface as a raw traceback in the UI."""
    eval_df = pd.DataFrame({
        "SMILES": ["CCO", "CCN", "CCC"],
        "target": [1, 1, 1],
        "target (Prediction)": [1, 1, 0],
        "target (Probability)": [0.9, 0.8, 0.4],
    })
    metrics, cm_figure, roc_figure = bc.on_target_column_change(eval_df, "target")

    assert metrics is not None
    values = metrics.set_index("Metric")["Value"]
    for metric in ["ROC-AUC score", "AUPRC", "Average precision", "Balanced accuracy", "MCC", "Specificity"]:
        assert np.isnan(values[metric])
    assert cm_figure is not None and roc_figure is not None


def test_metrics_handle_predictions_that_never_fire():
    """precision/recall are undefined with no positive predictions; must not raise."""
    eval_df = pd.DataFrame({
        "SMILES": ["CCO", "CCN"],
        "target": [1, 0],
        "target (Prediction)": [0, 0],
        "target (Probability)": [0.4, 0.1],
    })
    metrics, _, _ = bc.on_target_column_change(eval_df, "target")
    assert metrics.set_index("Metric").loc["Precision", "Value"] == 0


def test_metrics_return_none_for_empty_input():
    assert bc.on_target_column_change(None, "target") == (None, None, None)
    assert bc.on_target_column_change(pd.DataFrame(), "target") == (None, None, None)


def test_regression_metrics():
    eval_df = pd.DataFrame({
        "SMILES": ["CCO", "CCN", "CCC"],
        "value": [1.0, 2.0, 3.0],
        "value (Prediction)": [1.5, 2.0, 2.5],
    })
    metrics, scatter = rg.on_target_column_change(eval_df, "value")
    by_name = metrics.set_index("Metric")["Value"]

    # The handler rounds every metric to 4 decimal places for display.
    assert by_name["MAE"] == pytest.approx((0.5 + 0.0 + 0.5) / 3, abs=5e-5)
    assert by_name["MSE"] == pytest.approx((0.25 + 0 + 0.25) / 3, abs=5e-5)
    assert by_name["RMSE"] == pytest.approx(((0.25 + 0 + 0.25) / 3) ** 0.5, abs=5e-5)
    assert by_name["R²"] == pytest.approx(0.75, abs=5e-5)
    assert scatter is not None


@pytest.mark.parametrize("value,count", [(2.0, 2), (0.1, 3), (0.1, 10)])
def test_regression_r2_is_nan_when_every_target_is_identical(value, count):
    eval_df = pd.DataFrame({
        "SMILES": ["CCO"] * count,
        "value": [value] * count,
        "value (Prediction)": [value + 1] * count,
    })
    metrics, _ = rg.on_target_column_change(eval_df, "value")
    assert np.isnan(metrics.set_index("Metric").loc["R²", "Value"])


# --------------------------------------------------------------------------- checkpoints

def test_checkpoint_round_trip_restores_weights(trained_classifier):
    t = trained_classifier
    status, _ = bc.on_save_checkpoint(t["workdir"], "ckpt", t["model"], t["trained_epochs"],
                                      t["vt"], t["pca"])
    assert is_ok(status)
    assert os.path.exists(os.path.join(t["workdir"], f"ckpt_{t['trained_epochs']}.ckpt"))

    before = {k: v.clone() for k, v in t["model"].state_dict().items()}
    for parameter in t["model"].parameters():
        with torch.no_grad():
            parameter.add_(1.0)

    status, model, train_losses, val_losses, trained_epochs, vt, pca = bc.on_load_checkpoint(
        t["workdir"], f"ckpt_{t['trained_epochs']}.ckpt", t["model"],
        t["train_losses"], t["val_losses"], t["trained_epochs"], None, None)

    assert is_ok(status)
    assert trained_epochs == t["trained_epochs"]
    for key, value in model.state_dict().items():
        assert torch.allclose(value, before[key])


def test_checkpoint_carries_the_fitted_transformers(classification_workdir):
    """Regression: only weights were saved, so a reloaded model silently received
    differently-preprocessed inputs."""
    _, _, _, dataset, _, _, _, vt, pca = process_classification(
        classification_workdir, "Both", dimensionality_reduction=True)
    model = bc.on_create_model(*create_model_args(dataset, "GCN", vt, pca))[-4]

    bc.on_save_checkpoint(classification_workdir, "ckpt", model, 0, vt, pca)
    _, _, _, _, _, restored_vt, restored_pca = bc.on_load_checkpoint(
        classification_workdir, "ckpt_0.ckpt", model, [], [], 0, None, None)

    assert restored_vt is not None and restored_pca is not None
    assert restored_pca.n_components_ == pca.n_components_
    assert np.allclose(restored_pca.components_, pca.components_)


def test_regression_checkpoint_carries_the_output_scaler(regression_workdir):
    """Without the scaler every prediction comes back denormalised on the wrong scale."""
    _, _, _, dataset, _, _, _, vt, pca, scaler = process_regression(regression_workdir, "Graphs")
    model = rg.on_create_model(*create_model_args(dataset, "GCN", vt, pca))[-4]

    rg.on_save_checkpoint(regression_workdir, "ckpt", model, 0, vt, pca, scaler)
    result = rg.on_load_checkpoint(regression_workdir, "ckpt_0.ckpt", model, [], [], 0,
                                   None, None, None)
    restored_scaler = result[-1]

    assert restored_scaler is not None
    assert np.allclose(restored_scaler.data_min_, scaler.data_min_)
    assert np.allclose(restored_scaler.data_max_, scaler.data_max_)


def test_loading_into_a_different_architecture_is_rejected(classification_workdir):
    """Regression: a mismatched load either crashed or half-applied silently."""
    dataset = process_classification(classification_workdir)[3]
    gcn = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    gat = bc.on_create_model(*create_model_args(dataset, "GAT"))[-4]

    bc.on_save_checkpoint(classification_workdir, "ckpt", gcn, 7, None, None)
    status, model, train_losses, val_losses, trained_epochs, _, _ = bc.on_load_checkpoint(
        classification_workdir, "ckpt_7.ckpt", gat, [0.5], [0.6], 3, None, None)

    assert is_error(status)
    assert "GCNModel" in status
    assert train_losses == [0.5] and val_losses == [0.6] and trained_epochs == 3


def test_failed_load_preserves_training_state(classification_workdir):
    """Regression: the error path returned None, so the next Train click crashed on
    range(None, ...)."""
    dataset = process_classification(classification_workdir)[3]
    model = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]

    status, _, train_losses, val_losses, trained_epochs, _, _ = bc.on_load_checkpoint(
        classification_workdir, "does_not_exist.ckpt", model, [0.1, 0.2], [0.3, 0.4], 9,
        None, None)

    assert is_error(status)
    assert train_losses == [0.1, 0.2] and val_losses == [0.3, 0.4] and trained_epochs == 9


def test_save_checkpoint_rejects_an_unsafe_name(trained_classifier):
    t = trained_classifier
    status, _ = bc.on_save_checkpoint(t["workdir"], "../escape", t["model"], 1, None, None)
    assert is_error(status)
    assert not os.path.exists(os.path.join(os.path.dirname(t["workdir"]), "escape_1.ckpt"))


def test_save_checkpoint_without_a_model_reports_an_error(classification_workdir):
    status, _ = bc.on_save_checkpoint(classification_workdir, "ckpt", None, 0, None, None)
    assert is_error(status)


# -------------------------------------------------------------------------- prediction

def test_prediction_covers_every_input_molecule(trained_classifier):
    t = trained_classifier
    status, prediction_dataset, dataloader, predict_btn = bc.on_process_prediction_data(
        t["workdir"], "dataset.csv", t["dataset"], "GRAPH_graphs", "FP_fps", torch.float32, 4)
    assert is_ok(status)

    status, prediction_df, export_btn = bc.on_predict(
        t["model"], t["dataset"], prediction_dataset, dataloader, "cpu", torch.float32,
        "Custom", "GCNConv", t["vt"], t["pca"])

    assert is_ok(status)
    assert len(prediction_df) == len(SMILES)
    assert prediction_df["SMILES"].tolist() == t["dataset"].df["SMILES"].tolist()
    assert prediction_df["target (Prediction)"].isin([0, 1]).all()


def test_predict_without_data_reports_an_error(trained_classifier):
    t = trained_classifier
    status, prediction_df, _ = bc.on_predict(
        t["model"], t["dataset"], None, None, "cpu", torch.float32,
        "Custom", "GCNConv", t["vt"], t["pca"])
    assert is_error(status) and prediction_df is None


def test_regression_predict_without_a_scaler_reports_an_error(regression_workdir):
    _, _, _, dataset, _, _, test_dl, vt, pca, scaler = process_regression(regression_workdir, "Graphs")
    model = rg.on_create_model(*create_model_args(dataset, "GCN", vt, pca))[-4]

    status, prediction_df, _ = rg.on_predict(model, dataset, dataset, test_dl, "cpu",
                                             torch.float32, None, "GCN", "GCNConv", vt, pca)
    assert is_error(status) and prediction_df is None


# ------------------------------------------------------------------------------- exports

def test_export_losses_writes_a_csv(trained_classifier):
    t = trained_classifier
    status, _ = bc.on_export_losses(t["workdir"], "losses", t["train_losses"], t["val_losses"])
    assert is_ok(status)

    df = pd.read_csv(os.path.join(t["workdir"], "losses.csv"))
    assert list(df.columns) == ["epoch", "train_losses", "val_losses"]
    assert len(df) == len(t["train_losses"])


def test_export_losses_before_training_reports_an_error(classification_workdir):
    status, _ = bc.on_export_losses(classification_workdir, "losses", [], [])
    assert is_error(status)


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", ".."])
def test_exports_reject_unsafe_file_names(trained_classifier, name):
    t = trained_classifier
    status, _ = bc.on_export_losses(t["workdir"], name, t["train_losses"], t["val_losses"])
    assert is_error(status)


@pytest.mark.parametrize("module", [bc, rg])
def test_evaluation_rejects_overlapping_target_and_prediction_column_names(module):
    dataset = SimpleNamespace(target_column_names=["target", "target (Prediction)"])
    scaler_arguments = [object()] if module is rg else []
    result = module.on_evaluate(object(), dataset, object(), "cpu", torch.float32, *scaler_arguments,
                                "Custom", "GCNConv", None, None)
    assert is_error(result[0]) and "overlap generated evaluation columns" in result[0]
    assert result[1] is None
    assert result[2]["interactive"] is False and result[3]["interactive"] is False


@pytest.mark.parametrize("module", [bc, rg])
def test_nonfinite_model_outputs_fail_evaluation_and_disable_both_exports(module, monkeypatch):
    dataset = SimpleNamespace(target_column_names=["target"])
    predicted = torch.tensor([[float("nan")], [0.0]]) if module is bc else np.array([[np.nan], [0.0]])
    monkeypatch.setattr(module, "test", lambda *args: (torch.tensor([[0.0], [1.0]]), predicted, ["CCO", "CCN"]))
    scaler_arguments = [object()] if module is rg else []
    result = module.on_evaluate(torch.nn.Linear(1, 1), dataset, object(), "cpu", torch.float32, *scaler_arguments,
                                "Custom", "GCNConv", None, None)
    assert is_error(result[0])
    assert "finite" in result[0]
    assert result[1] is None
    assert result[2]["interactive"] is False and result[3]["interactive"] is False


def test_prediction_export_without_data_reports_an_error(classification_workdir):
    assert is_error(bc.on_export_prediction(classification_workdir, "prediction", None)[0])


# ------------------------------------------------------------- working directory listing

def test_create_optimizer_without_a_model_reports_an_error():
    """Regression: this raised AttributeError on None.parameters()."""
    status, optimizer, scheduler, train_button = bc.on_create_optimizer(None, "Adam", 0.01, 0.99)
    assert is_error(status)
    assert optimizer is None and scheduler is None
    assert train_button.get("interactive") is False


def test_create_optimizer_rejects_an_unknown_optimizer(classification_workdir):
    dataset = process_classification(classification_workdir)[3]
    model = bc.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    status, optimizer, _, _ = bc.on_create_optimizer(model, "NotAnOptimizer", 0.01, 0.99)
    assert is_error(status) and optimizer is None


def test_content_listing_handles_a_missing_working_directory():
    """Regression: reachable after a rejected directory name, and raised TypeError."""
    result = bc.on_content_list_change(None, "dataset", "GRAPH_graphs", "FP_fps", "checkpoint")
    assert result[0].empty
    assert len(result) == 9


def test_working_directory_contents_of_none_is_empty():
    """os.listdir(None) lists the current working directory, which is not what we want."""
    assert bc.get_working_directory_contents(None) == []


def test_pca_sliders_follow_the_checkbox_not_just_the_input_mode():
    """Regression: choosing 'Both' revealed the PCA sliders even with the box unticked."""
    unticked = bc.on_load_data_option_change("Both", False)
    ticked = bc.on_load_data_option_change("Both", True)

    assert unticked[3].get("visible") is False and unticked[4].get("visible") is False
    assert ticked[3].get("visible") is True and ticked[4].get("visible") is True
    assert bc.on_load_data_option_change("Graphs", True)[3].get("visible") is False


def test_saving_over_an_existing_checkpoint_says_so(trained_classifier):
    """Regression: the filename is derived from the epoch count, so this silently replaced
    a previous file - including one from a different architecture."""
    t = trained_classifier
    first, _ = bc.on_save_checkpoint(t["workdir"], "ckpt", t["model"], t["trained_epochs"], None, None)
    second, _ = bc.on_save_checkpoint(t["workdir"], "ckpt", t["model"], t["trained_epochs"], None, None)

    assert is_ok(first) and "replaced" not in first
    assert is_ok(second) and "replaced an existing file" in second


def test_working_directory_listing_sorts_newest_first(classification_workdir):
    """Regression: rows were sorted on the ctime string, which orders by weekday name."""
    old = os.path.join(classification_workdir, "old.csv")
    new = os.path.join(classification_workdir, "new.csv")
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(old, index=False)
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(new, index=False)
    os.utime(old, (1_000_000_000, 1_000_000_000))     # 2001
    os.utime(new, (1_900_000_000, 1_900_000_000))     # 2030

    contents_df = bc.on_content_list_change(classification_workdir, "dataset",
                                            "GRAPH_graphs", "FP_fps", "checkpoint")[0]
    order = contents_df["File"].tolist()
    assert order.index("new.csv") < order.index("old.csv")


def test_working_directory_listing_classifies_entries(classification_workdir):
    open(os.path.join(classification_workdir, "model.ckpt"), "w").close()
    contents_df = bc.on_content_list_change(classification_workdir, "dataset",
                                            "GRAPH_graphs", "FP_fps", "checkpoint")[0]
    by_name = contents_df.set_index("File")["Type"]

    assert by_name["GRAPH_graphs"] == "Directory"
    assert by_name["dataset.csv"] == "Data file"
    assert by_name["model.ckpt"] == "Checkpoint file"


@pytest.mark.parametrize("module", [bc, rg])
@pytest.mark.parametrize("mode,graph_dir,fp_dir", [
    ("Graphs", "GRAPH_graphs", None), ("Molecular fingerprints", None, "FP_fps"),
])
def test_single_input_mode_ignores_unselected_directory(classification_workdir, module, mode, graph_dir, fp_dir):
    result = module.on_process_data(
        classification_workdir, "dataset.csv", mode, graph_dir, fp_dir, torch.float32,
        False, 0.0, 4, 0.25, 0.25, 4, False, 0)
    assert is_ok(result[0]), result[0]
    prediction = module.on_process_prediction_data(
        classification_workdir, "dataset.csv", result[3], graph_dir, fp_dir, torch.float32, 4)
    assert is_ok(prediction[0]), prediction[0]
    assert len(prediction[1]) == len(SMILES)


@pytest.mark.parametrize("module", [bc, rg])
def test_create_model_without_processed_data_reports_error(module):
    result = module.on_create_model(*create_model_args(None, "GCN"))
    assert is_error(result[0])
    assert "process the data" in result[0]
    assert result[7]["interactive"] is False


@pytest.mark.parametrize("module", [bc, rg])
@pytest.mark.parametrize("target", [float("nan"), float("inf"), -float("inf")])
def test_processing_rejects_nonfinite_targets(classification_workdir, module, target):
    path = os.path.join(classification_workdir, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = df["target"].astype(float)
    df.loc[0, "target"] = target
    df.to_csv(path, index=False)
    result = module.on_process_data(
        classification_workdir, "dataset.csv", "Both", "GRAPH_graphs", "FP_fps", torch.float32,
        False, 0.0, 4, 0.25, 0.25, 4, False, 0)
    assert is_error(result[0])
    assert result[3] is None


@pytest.mark.parametrize("module", [bc, rg])
@pytest.mark.parametrize("actual_tab,selected_tab", [("GAT", "GCN"), ("GCN", "GAT")])
def test_training_and_inference_follow_created_model_after_tab_change(classification_workdir, module, actual_tab, selected_tab):
    processed = module.on_process_data(
        classification_workdir, "dataset.csv", "Both", "GRAPH_graphs", "FP_fps", torch.float32,
        False, 0.0, 4, 0.25, 0.25, 4, False, 0)
    dataset, train_dl, val_dl, test_dl, vt, pca = processed[3:9]
    scaler_args = [processed[9]] if module is rg else []
    model = module.on_create_model(*create_model_args(dataset, actual_tab))[-4]
    expected = module.predict(model, dataset, test_dl, "cpu", torch.float32, *scaler_args,
                              actual_tab, "GCNConv", vt, pca)[1]
    actual = module.predict(model, dataset, test_dl, "cpu", torch.float32, *scaler_args,
                            selected_tab, "GCNConv", vt, pca)[1]
    np.testing.assert_allclose(actual, expected)
    evaluation = module.on_evaluate(model, dataset, test_dl, "cpu", torch.float32, *scaler_args,
                                    selected_tab, "GCNConv", vt, pca)
    assert is_ok(evaluation[0]), evaluation[0]
    _, optimizer, scheduler, _ = module.on_create_optimizer(model, "Adam", 0.01, 0.99)
    trained = module.on_train(model, dataset, train_dl, val_dl, "cpu", torch.float32,
                              *scaler_args, optimizer, scheduler, 1, 0, [], [],
                              selected_tab, "GCNConv", vt, pca, FakeProgress())
    assert is_ok(trained[0]), trained[0]
    assert trained[3] == 1


@pytest.mark.parametrize("module", [bc, rg])
def test_bfloat16_fingerprints_with_pca_train_evaluate_and_predict(classification_workdir, module):
    processed = module.on_process_data(
        classification_workdir, "dataset.csv", "Molecular fingerprints", None, "FP_fps", torch.bfloat16,
        True, 0.0, 4, 0.25, 0.25, 4, False, 0)
    assert is_ok(processed[0]), processed[0]
    dataset, train_dl, val_dl, test_dl, vt, pca = processed[3:9]
    scaler_args = [processed[9]] if module is rg else []
    args = create_model_args(dataset, "GCN", vt, pca)
    args[3] = torch.bfloat16
    created = module.on_create_model(*args)
    assert is_ok(created[0]), created[0]
    model = created[-4]
    _, optimizer, scheduler, _ = module.on_create_optimizer(model, "SGD", 0.01, 0.99)
    trained = module.on_train(model, dataset, train_dl, val_dl, "cpu", torch.bfloat16,
                              *scaler_args, optimizer, scheduler, 1, 0, [], [],
                              "GCN", "GCNConv", vt, pca, FakeProgress())
    assert is_ok(trained[0]), trained[0]
    assert np.isfinite(trained[4]).all()
    evaluation = module.on_evaluate(model, dataset, test_dl, "cpu", torch.bfloat16, *scaler_args,
                                    "GCN", "GCNConv", vt, pca)
    assert is_ok(evaluation[0]), evaluation[0]
    prediction = module.on_predict(model, dataset, dataset, test_dl, "cpu", torch.bfloat16,
                                    *scaler_args, "GCN", "GCNConv", vt, pca)
    assert is_ok(prediction[0]), prediction[0]


@pytest.mark.parametrize("module", [bc, rg])
def test_incompatible_checkpoint_preserves_all_model_weights(classification_workdir, module):
    dataset = process_classification(classification_workdir)[3]
    model = module.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    changed = {k: v + 10 for k, v in before.items()}
    # Earlier tensors fit, but the final tensor fails: PyTorch loads those earlier
    # tensors before raising unless the entire checkpoint is validated first.
    last_key = list(changed)[-1]
    changed[last_key] = torch.zeros(17)
    torch.save({"state_dict": changed, "model_class": type(model).__name__},
               os.path.join(classification_workdir, "bad.ckpt"))
    scaler_args = [None] if module is rg else []
    loaded = module.on_load_checkpoint(classification_workdir, "bad.ckpt", model,
                                       [0.1], [0.2], 3, None, None, *scaler_args)
    assert is_error(loaded[0])
    assert loaded[2:5] == ([0.1], [0.2], 3)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key]), key


def test_regression_reports_an_empty_test_split(regression_workdir):
    processed = rg.on_process_data(
        regression_workdir, "dataset.csv", "Graphs", "GRAPH_graphs", None, torch.float32,
        False, 0.0, 4, 0.0, 0.25, 4, False, 0)
    dataset, test_dl, scaler = processed[3], processed[6], processed[9]
    model = rg.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    result = rg.on_evaluate(model, dataset, test_dl, "cpu", torch.float32, scaler,
                            "GCN", "GCNConv", None, None)
    assert is_error(result[0])
    assert "test split is empty" in result[0]


@pytest.mark.parametrize("module", [bc, rg])
def test_scheduler_failure_preserves_completed_epoch_count(classification_workdir, module):
    processed = module.on_process_data(
        classification_workdir, "dataset.csv", "Graphs", "GRAPH_graphs", None, torch.float32,
        False, 0.0, 4, 0.25, 0.25, 4, False, 0)
    dataset, train_dl, val_dl = processed[3:6]
    scaler_args = [processed[9]] if module is rg else []
    model = module.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    _, optimizer, _, _ = module.on_create_optimizer(model, "SGD", 0.01, 0.99)

    class BrokenScheduler:
        def step(self):
            raise RuntimeError("scheduler failed")

    trained = module.on_train(model, dataset, train_dl, val_dl, "cpu", torch.float32,
                              *scaler_args, optimizer, BrokenScheduler(), 2, 0, [], [],
                              "GCN", "GCNConv", None, None, FakeProgress())
    assert is_error(trained[0])
    assert trained[3] == len(trained[4]) == len(trained[5]) == 1


@pytest.mark.parametrize("module", [bc, rg])
def test_nonfinite_loss_is_rejected_before_optimizer_updates(classification_workdir, module, monkeypatch):
    processed = module.on_process_data(
        classification_workdir, "dataset.csv", "Graphs", "GRAPH_graphs", None, torch.float32,
        False, 0.0, 4, 0.25, 0.25, 4, False, 0)
    dataset, train_dl, val_dl = processed[3:6]
    scaler_args = [processed[9]] if module is rg else []
    model = module.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    before = [p.detach().clone() for p in model.parameters()]
    _, optimizer, scheduler, _ = module.on_create_optimizer(model, "SGD", 0.01, 0.99)
    monkeypatch.setattr(module, "loss_fn", lambda output, target: output.sum() * float("nan"))
    trained = module.on_train(model, dataset, train_dl, val_dl, "cpu", torch.float32,
                              *scaler_args, optimizer, scheduler, 1, 0, [], [],
                              "GCN", "GCNConv", None, None, FakeProgress())
    assert is_error(trained[0])
    assert "not finite" in trained[0]
    assert trained[3] == 0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


@pytest.mark.parametrize("datatype", [torch.float16, torch.bfloat16])
def test_regression_scales_large_targets_before_reducing_precision(regression_workdir, datatype):
    path = os.path.join(regression_workdir, "dataset.csv")
    df = pd.read_csv(path)
    df["value"] = 100000.0 + np.arange(len(df)) * 0.125
    df.to_csv(path, index=False)
    processed = rg.on_process_data(
        regression_workdir, "dataset.csv", "Molecular fingerprints", None, "FP_fps", datatype,
        False, 0.0, 4, 0.25, 0.25, 4, False, 0)
    assert is_ok(processed[0]), processed[0]
    dataset, val_dl, scaler = processed[3], processed[5], processed[9]
    args = create_model_args(dataset, "GCN")
    args[3] = datatype
    model = rg.on_create_model(*args)[-4]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    actual_rmse = rg.validation(model, dataset, val_dl, "cpu", datatype, scaler,
                                "GCN", "GCNConv", None, None)
    expected_targets = scaler.transform(df[["value"]].iloc[val_dl.dataset.indices])
    assert actual_rmse == pytest.approx(np.sqrt(np.mean(expected_targets ** 2)), rel=0.01)
    evaluated = rg.on_evaluate(model, dataset, val_dl, "cpu", datatype, scaler,
                               "GCN", "GCNConv", None, None)
    assert is_ok(evaluated[0]), evaluated[0]
    assert evaluated[1]["value"].tolist() == df["value"].iloc[val_dl.dataset.indices].tolist()


def test_regression_refuses_broadcasting_mismatched_target_width(regression_workdir):
    processed = process_regression(regression_workdir, "Molecular fingerprints")
    dataset, train_dl, val_dl, scaler = processed[3], processed[4], processed[5], processed[9]
    model = rg.on_create_model(*create_model_args(dataset, "GCN"))[-4]
    # Reprocessing can change the target width while an old model is still present.
    dataset.df["second_value"] = dataset.df["value"]
    scaler.fit(dataset.df[["value", "second_value"]])
    _, optimizer, scheduler, _ = rg.on_create_optimizer(model, "SGD", 0.01, 0.99)
    before = [p.detach().clone() for p in model.parameters()]
    result = rg.on_train(model, dataset, train_dl, val_dl, "cpu", torch.float32,
                         scaler, optimizer, scheduler, 1, 0, [], [],
                         "GCN", "GCNConv", None, None, FakeProgress())
    assert is_error(result[0])
    assert "output shape" in result[0]
    assert result[3] == 0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


@pytest.mark.parametrize("module", [bc, rg])
def test_named_hybrid_runs_capture_separate_seeds_and_route_all_results(request, module):
    workdir = request.getfixturevalue("classification_workdir" if module is bc else "regression_workdir")
    processed = module.on_process_data(workdir, "dataset.csv", "Molecular fingerprints", "GRAPH_graphs", "FP_fps",
                                       torch.float32, True, 0.0, 4, 0.25, 0.25, 4, False, 7)
    assert is_ok(processed[0]), processed[0]
    dataset, train_dl, _, test_dl, vt, pca = processed[3:9]
    assert dataset.split_context["split_seed"] == 7 and pca.random_state == 7
    assert dataset.split_context["train_indices"] == list(train_dl.dataset.indices)
    assert dataset.split_context["test_indices"] == list(test_dl.dataset.indices)
    args = create_model_args(dataset, "GCN", vt, pca)
    args[4] = 13
    created = module.on_create_model(*args, model_name="experiment")
    assert is_ok(created[0]), created[0]
    model = created[-4]
    context = deepcopy(model._run_context)
    assert context["model_seed"] == 13 and context["split_seed"] == 7
    assert context["parameters"]["pca"]["random_state"] == 7
    assert context["parameters"]["feature_sources"]["fingerprint_directory"] == "FP_fps"
    assert not os.path.exists(run_directory(context))  # Creation records identity without writing outputs.
    scaler_args = [processed[9]] if module is rg else []
    saved = module.on_save_checkpoint(workdir, "weights", model, 2, vt, pca, *scaler_args)
    assert is_ok(saved[0]), saved[0]
    checkpoint = os.path.join(run_directory(context), "weights_2.ckpt")
    assert os.path.isfile(checkpoint) and not os.path.exists(os.path.join(workdir, "weights_2.ckpt"))
    relative = os.path.relpath(checkpoint, workdir)
    assert relative in list_saved_models(workdir, ".ckpt")
    args[4] = 99
    other = module.on_create_model(*args, model_name="edited_selection")[-4]
    assert not torch.equal(next(model.parameters()), next(other.parameters()))
    loaded = module.on_load_checkpoint(workdir, relative, other, [], [], 0, vt, pca, *scaler_args)
    assert is_ok(loaded[0]), loaded[0]
    assert other._run_context == context

    rows = dataset.df.iloc[test_dl.dataset.indices].copy()
    for target in dataset.target_column_names:
        rows[f"{target} (Prediction)"] = rows[target]
        if module is bc:
            rows[f"{target} (Probability)"] = rows[target] * 0.8 + 0.1
    exports = [(module.on_export_losses, ("losses", [0.2], [0.3])),
               (module.on_export_evaluation, ("evaluation", rows)),
               (module.on_export_metrics, ("metrics", rows, dataset)),
               (module.on_export_prediction, ("predictions", rows))]
    for callback, inputs in exports:
        status, _ = route_model_export(callback)(workdir, *inputs, other)
        assert is_ok(status), status
        assert os.path.isfile(os.path.join(run_directory(context), inputs[0] + ".csv"))
        assert not os.path.exists(os.path.join(workdir, inputs[0] + ".csv"))
    choices = module.on_content_list_change(workdir, "dataset", "GRAPH_graphs", "FP_fps", relative)
    assert choices[2]["choices"] == ["GRAPH_graphs"] and choices[3]["choices"] == ["FP_fps"]
    assert relative in choices[5]["choices"]


def test_legacy_checkpoint_import_keeps_named_run_and_marks_unknown_seed(classification_workdir):
    processed = process_classification(classification_workdir, "Molecular fingerprints")
    dataset = processed[3]
    args = create_model_args(dataset, "GCN")
    legacy = bc.on_create_model(*args)[-4]
    assert is_ok(bc.on_save_checkpoint(classification_workdir, "legacy", legacy, 0, None, None)[0])
    args[4] = 17
    imported = bc.on_create_model(*args, model_name="imported")[-4]
    loaded = bc.on_load_checkpoint(classification_workdir, "legacy_0.ckpt", imported, [], [], 0, None, None)
    assert is_ok(loaded[0]) and "unknown" in loaded[0]
    assert imported._run_context["model_seed"] == 17
    assert imported._run_context["legacy_checkpoint"] is True
    assert imported._run_context["initialization_seed_verified"] is False
    assert is_ok(bc.on_save_checkpoint(classification_workdir, "imported", imported, 0, None, None)[0])
    assert os.path.isfile(os.path.join(run_directory(imported._run_context), "imported_0.ckpt"))


def test_named_model_rejects_different_split_and_bad_checkpoint_metadata(classification_workdir):
    processed = process_classification(classification_workdir, "Molecular fingerprints")
    dataset = processed[3]
    model = bc.on_create_model(*create_model_args(dataset, "GCN"), model_name="named")[-4]
    other = process_classification(classification_workdir, "Molecular fingerprints", seed=1)
    result = bc.on_evaluate(model, other[3], other[6], "cpu", torch.float32, "GCN", "GCNConv", None, None)
    assert is_error(result[0]) and "original" in result[0]
    _, optimizer, scheduler, _ = bc.on_create_optimizer(model, "SGD", 0.01, 1.0)
    trained = bc.on_train(model, other[3], other[4], other[5], "cpu", torch.float32,
                          optimizer, scheduler, 1, 0, [], [], "GCN", "GCNConv", None, None, FakeProgress())
    assert is_error(trained[0]) and trained[3] == 0
    before = {name: value.clone() for name, value in model.state_dict().items()}
    bad_context = deepcopy(model._run_context)
    bad_context["model_seed"] = 999  # This no longer matches the recorded run identity.
    torch.save({"model_class": type(model).__name__, "run_context": bad_context,
                "state_dict": {name: value + 1 for name, value in before.items()}},
               os.path.join(classification_workdir, "bad_metadata.ckpt"))
    loaded = bc.on_load_checkpoint(classification_workdir, "bad_metadata.ckpt", model, [0.1], [0.2], 3, None, None)
    assert is_error(loaded[0])
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert loaded[2:5] == ([0.1], [0.2], 3)


@pytest.mark.parametrize("module", [bc, rg])
@pytest.mark.parametrize("seed", [None, 1.5, -1])
def test_hybrid_invalid_model_seed_reports_an_error(module, seed):
    args = create_model_args(SimpleNamespace(), "GCN")
    args[4] = seed
    result = module.on_create_model(*args)
    assert is_error(result[0]) and "seed" in result[0].lower()
    assert result[-4] is None


def test_named_run_rejects_changed_fingerprints_in_the_same_folder(classification_workdir):
    first = process_classification(classification_workdir, "Molecular fingerprints")
    original = bc.on_create_model(*create_model_args(first[3], "GCN"), model_name="reused")[-4]
    assert is_ok(bc.on_save_checkpoint(classification_workdir, "weights", original, 0, None, None)[0])
    checkpoint_path = os.path.join(run_directory(original._run_context), "weights_0.ckpt")
    with open(checkpoint_path, "rb") as stream:
        before = stream.read()
    fingerprint_path = os.path.join(classification_workdir, "FP_fps", "molecular_fingerprints.csv")
    frame = pd.read_csv(fingerprint_path)
    frame.iloc[0, 1] = 1 - frame.iloc[0, 1]
    frame.to_csv(fingerprint_path, index=False)
    second = process_classification(classification_workdir, "Molecular fingerprints")
    evaluated = bc.on_evaluate(original, second[3], second[6], "cpu", torch.float32,
                                "GCN", "GCNConv", None, None)
    assert is_error(evaluated[0]) and "features" in evaluated[0]
    changed = bc.on_create_model(*create_model_args(second[3], "GCN"), model_name="reused")[-4]
    assert changed._run_context["parameters"]["feature_sources"]["fingerprint_hash"] != original._run_context["parameters"]["feature_sources"]["fingerprint_hash"]
    result = bc.on_save_checkpoint(classification_workdir, "weights", changed, 0, None, None)
    assert is_error(result[0]) and "another model name" in result[0]
    with open(checkpoint_path, "rb") as stream:
        assert stream.read() == before


def test_named_checkpoint_rejects_different_dropout_before_copying_weights(classification_workdir):
    dataset = process_classification(classification_workdir)[3]
    original = bc.on_create_model(*create_model_args(dataset, "GCN", gcn_dropout=0.1), model_name="dropout")[-4]
    assert is_ok(bc.on_save_checkpoint(classification_workdir, "weights", original, 0, None, None)[0])
    other = bc.on_create_model(*create_model_args(dataset, "GCN", gcn_dropout=0.7), model_name="different")[-4]
    before = {name: value.clone() for name, value in other.state_dict().items()}
    filename = os.path.relpath(os.path.join(run_directory(original._run_context), "weights_0.ckpt"), classification_workdir)
    result = bc.on_load_checkpoint(classification_workdir, filename, other, [], [], 0, None, None)
    assert is_error(result[0]) and "architecture" in result[0].lower()
    assert all(torch.equal(value, before[name]) for name, value in other.state_dict().items())
