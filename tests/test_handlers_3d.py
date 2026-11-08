"""Tests for the 3D tabs. Marked slow: every fixture needs conformer generation."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import binary_classification_3d as bc3
import regression_3d as rg3
from conftest import FakeProgress, SMILES_3D, create_model_3d_args, is_error, is_ok
from gnn_run_ui import route_model_export
from run_utils import run_directory, list_saved_models

pytestmark = pytest.mark.slow


def process_3d_classification(workdir, shuffle=False):
    return bc3.on_process_data(workdir, "dataset.csv", "GRAPH_g3d", torch.float32, 0.25, 0.25, 4, shuffle, 0)


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


def test_3d_creating_a_model_disables_training(workdir_3d):
    dataset = process_3d_classification(workdir_3d)[2]
    result = bc3.on_create_model(*create_model_3d_args(dataset, "SchNet"))
    assert result[6].get("interactive") is False        # train_button

    _, _, _, train_button = bc3.on_create_optimizer(result[-4], "Adam", 0.001, 0.99)
    assert train_button.get("interactive") is True


def test_3d_training_with_a_stale_optimizer_is_refused(workdir_3d):
    _, _, dataset, train_dl, val_dl, _ = process_3d_classification(workdir_3d)
    model_a = bc3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]
    _, optimizer, scheduler, _ = bc3.on_create_optimizer(model_a, "Adam", 0.01, 0.99)
    model_b = bc3.on_create_model(*create_model_3d_args(dataset, "SchNet",
                                                        schnet_n_interactions=3))[-4]

    status, _, _, trained_epochs, train_losses, _ = bc3.on_train(
        model_b, train_dl, val_dl, "cpu", torch.float32, optimizer, scheduler, 1,
        0, [], [], FakeProgress())

    assert is_error(status) and "Create optimizer" in status
    assert trained_epochs == 0 and train_losses == []


def test_3d_training_reports_failures_instead_of_raising(workdir_3d):
    _, _, dataset, train_dl, _, _ = process_3d_classification(workdir_3d)
    model = bc3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]
    _, optimizer, scheduler, _ = bc3.on_create_optimizer(model, "Adam", 0.001, 0.99)

    status, figure, _, trained_epochs, train_losses, val_losses = bc3.on_train(
        model, train_dl, None, "cpu", torch.float32, optimizer, scheduler, 1,
        0, [], [], FakeProgress())

    assert is_error(status)
    assert len(train_losses) == len(val_losses) == trained_epochs
    assert figure is not None


def test_3d_shuffle_toggle(workdir_3d):
    shuffled = process_3d_classification(workdir_3d, shuffle=True)[3]
    assert [b.smiles for b in shuffled] != [b.smiles for b in shuffled]

    fixed = process_3d_classification(workdir_3d, shuffle=False)[3]
    assert [b.smiles for b in fixed] == [b.smiles for b in fixed]


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
    status, eval_df, metrics, cm_figure, roc_figure, export_btn, metrics_btn = bc3.on_evaluate(
        c["model"], c["dataset"], c["test_dl"], "cpu", torch.float32)

    assert is_ok(status)
    assert len(eval_df) == len(c["test_dl"].dataset)
    assert list(metrics["Metric"]) == ["Accuracy", "Balanced accuracy", "Precision", "Recall", "Specificity",
                                       "F1 score", "MCC", "ROC-AUC score", "AUPRC", "Average precision"]
    assert cm_figure is not None and roc_figure is not None
    assert export_btn["interactive"] is True and metrics_btn["interactive"] is True


def test_3d_evaluate_without_a_model_reports_an_error(classifier_3d):
    status, eval_df, metrics, cm, roc, export_btn, metrics_btn = bc3.on_evaluate(
        None, classifier_3d["dataset"], classifier_3d["test_dl"], "cpu", torch.float32)
    assert is_error(status) and eval_df is None
    assert export_btn["interactive"] is False and metrics_btn["interactive"] is False


def test_3d_prediction(classifier_3d):
    c = classifier_3d
    status, prediction_dataset, dataloader, _ = bc3.on_process_prediction_data(
        c["workdir"], "dataset.csv", "GRAPH_g3d", torch.float32, 4)
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
        workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32, 0.25, 0.25, 4, False, 0)
    assert is_ok(status), status
    assert scaler is not None

    model = rg3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]
    _, optimizer, scheduler, _ = rg3.on_create_optimizer(model, "Adam", 0.001, 0.99)
    status, _, _, trained_epochs, train_losses, _ = rg3.on_train(
        model, train_dl, val_dl, "cpu", torch.float32, scaler, optimizer, scheduler, 2,
        0, [], [], FakeProgress())
    assert is_ok(status) and len(train_losses) == 2

    status, eval_df, metrics, scatter, export_btn, metrics_btn = rg3.on_evaluate(
        model, dataset, test_dl, "cpu", torch.float32, scaler)
    assert is_ok(status)
    assert list(metrics["Metric"]) == ["MAE", "MSE", "RMSE", "R²"]
    assert export_btn["interactive"] is True and metrics_btn["interactive"] is True


def test_3d_regression_checkpoint_carries_the_scaler(workdir_3d):
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = [float(i) * 2.5 for i in range(len(df))]
    df.to_csv(path, index=False)

    _, _, dataset, _, _, _, scaler = rg3.on_process_data(
        workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32, 0.25, 0.25, 4, False, 0)
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
        workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32, 0.25, 0.25, 4, False, 0)
    model = rg3.on_create_model(*create_model_3d_args(dataset, "SchNet"))[-4]

    status, eval_df = rg3.on_evaluate(model, dataset, test_dl, "cpu", torch.float32, None)[:2]
    assert is_error(status) and eval_df is None


@pytest.fixture(params=[bc3, rg3], ids=["classification", "regression"])
def cached_3d_evaluation(request):
    module = request.param
    frame = pd.DataFrame({"SMILES": ["C", "CC", "CCC", "CCCC"],
                          "activity": [0, 1, 0, 1], "activity (Prediction)": [0, 1, 1, 1]})
    if module is bc3:
        frame["activity (Probability)"] = [0.1, 0.8, 0.6, 0.9]
    return module, frame, SimpleNamespace(target_column_name="activity")


@pytest.mark.parametrize("suffix", ["", ".csv"])
def test_3d_exports_cached_rows_and_all_metrics(tmp_path, cached_3d_evaluation, suffix):
    module, frame, dataset = cached_3d_evaluation
    status, contents = module.on_export_evaluation(str(tmp_path), f"evaluation{suffix}", frame)
    assert is_ok(status) and "evaluation.csv" in contents
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "evaluation.csv"), frame)
    status, contents = module.on_export_metrics(str(tmp_path), f"test_metrics{suffix}", frame, dataset)
    assert is_ok(status) and "test_metrics.csv" in contents
    metrics = pd.read_csv(tmp_path / "test_metrics.csv")
    assert list(metrics.columns) == ["Target", "Metric", "Value"]
    assert set(metrics["Target"]) == {"activity"}
    expected = (["Accuracy", "Balanced accuracy", "Precision", "Recall", "Specificity", "F1 score",
                 "MCC", "ROC-AUC score", "AUPRC", "Average precision"] if module is bc3
                else ["MAE", "MSE", "RMSE", "R²"])
    assert metrics["Metric"].tolist() == expected
    assert metrics.loc[metrics["Metric"] == ("Accuracy" if module is bc3 else "MAE"), "Value"].item() == (0.75 if module is bc3 else 0.25)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evaluation.csv", "test_metrics.csv"]


@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_3d_exports_refuse_missing_or_empty_evaluation(tmp_path, cached_3d_evaluation, frame):
    module, _, dataset = cached_3d_evaluation
    for result in (module.on_export_evaluation(str(tmp_path), "evaluation", frame),
                   module.on_export_metrics(str(tmp_path), "test_metrics", frame, dataset)):
        assert is_error(result[0])
        assert result[1] == gr.update()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["../outside", "<unsafe>/file", ""])
def test_3d_exports_refuse_unsafe_names(tmp_path, cached_3d_evaluation, name):
    module, frame, dataset = cached_3d_evaluation
    for result in (module.on_export_evaluation(str(tmp_path), name, frame),
                   module.on_export_metrics(str(tmp_path), name, frame, dataset)):
        assert is_error(result[0])
        assert "<unsafe>" not in result[0]
        assert result[1] == gr.update()
    assert not list(tmp_path.iterdir())


def test_3d_exports_report_missing_directory_or_targets_without_raising(tmp_path, cached_3d_evaluation):
    module, frame, dataset = cached_3d_evaluation
    for workdir in (None, str(tmp_path / "missing")):
        assert is_error(module.on_export_evaluation(workdir, "evaluation", frame)[0])
        assert is_error(module.on_export_metrics(workdir, "metrics", frame, dataset)[0])
    assert is_error(module.on_export_metrics(str(tmp_path), "metrics", frame, None)[0])
    assert is_error(module.on_export_metrics(str(tmp_path), "metrics", frame, SimpleNamespace(target_column_name=None))[0])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("module", [bc3, rg3])
def test_3d_processing_disables_both_evaluation_exports(tmp_path, monkeypatch, module):
    monkeypatch.setattr(module, "DATA_DIR", str(tmp_path))
    builder = module.binary_classification_3d_tab_content if module is bc3 else module.regression_3d_tab_content
    with gr.Blocks() as blocks:
        builder()
    process = next(event for event in blocks.fns.values() if getattr(event.fn, "__name__", None) == "on_process_data")
    updates = process.fn(str(tmp_path), "missing.csv", "GRAPH_graphs", torch.float32, 0.2, 0.2, 4, False, 0, "Random")
    buttons = [component for component in process.outputs if isinstance(component, gr.Button)
               and component.value in {"Export test metrics (all targets)", "Export evaluation rows (all targets)"}]
    assert len(buttons) == 2
    assert all(updates[button]["interactive"] is False for button in buttons)


@pytest.mark.parametrize("module", [bc3, rg3])
@pytest.mark.parametrize("value", [float('nan'), float('inf'), -float('inf')])
def test_3d_processing_rejects_nonfinite_targets(workdir_3d, module, value):
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = df["target"].astype(float)
    df.loc[0, "target"] = value
    df.to_csv(path, index=False)

    result = module.on_process_data(
        workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32, 0.25, 0.25, 4, False, 0)

    assert is_error(result[0]) and "non-finite" in result[0]
    assert result[2] is None


@pytest.mark.parametrize("module", [bc3, rg3])
def test_3d_processing_rejects_multiple_targets(workdir_3d, module):
    # A working directory can contain a CSV uploaded through one of the 2D tabs.
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["second_target"] = 1 - df["target"]
    df.to_csv(path, index=False)

    result = module.on_process_data(
        workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32, 0.25, 0.25, 4, False, 0)

    assert is_error(result[0]) and "target" in result[0]
    assert result[2] is None


@pytest.mark.parametrize("module", [bc3, rg3])
def test_3d_failed_checkpoint_load_preserves_every_weight(tmp_path, module):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    saved = {name: value + 10 for name, value in before.items()}
    # load_state_dict used to copy earlier compatible tensors before this late mismatch.
    saved["1.weight"] = torch.zeros(2, 3)
    torch.save({'model_class': 'Sequential', 'state_dict': saved}, tmp_path / "bad.ckpt")
    scaler_args = [None] if module is rg3 else []

    result = module.on_load_checkpoint(str(tmp_path), "bad.ckpt", model, [0.1], [0.2], 3, *scaler_args)

    assert is_error(result[0])
    assert result[2:5] == ([0.1], [0.2], 3)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


@pytest.mark.parametrize("module", [bc3, rg3])
def test_3d_cuda_checkpoint_loads_on_cpu(tmp_path, monkeypatch, module):
    model = torch.nn.Linear(2, 1)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    # Tag serialized CPU storage as CUDA to cover portability without requiring a GPU.
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, "location_tag", lambda storage: "cuda:0")
        torch.save({'model_class': 'Linear', 'state_dict': saved}, tmp_path / "cuda.ckpt")
    with torch.no_grad():
        model.weight.zero_()
    scaler_args = [None] if module is rg3 else []

    result = module.on_load_checkpoint(str(tmp_path), "cuda.ckpt", model, [], [], 0, *scaler_args)

    assert is_ok(result[0]), result[0]
    assert model.weight.device.type == "cpu"
    assert torch.equal(model.weight, saved["weight"])


class Constant3DModel(torch.nn.Module):
    """Small differentiable predictor to exercise handler dtype boundaries."""

    def __init__(self, dtype):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.25, dtype=dtype))

    def forward(self, z, pos, batch):
        assert z.dtype == torch.long
        return self.value.expand(int(batch.max()) + 1, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_3d_regression_dtype_training_evaluation_prediction(dtype):
    samples = [Data(z=torch.tensor([6]), pos=torch.zeros(1, 3, dtype=dtype),
                    y=torch.tensor([[value]], dtype=dtype), smiles="C") for value in (0.0, 1.0)]
    loader = DataLoader(samples, batch_size=2)
    dataset = SimpleNamespace(target_column_name="target")
    scaler = MinMaxScaler().fit([[0.0], [1.0]])
    model = Constant3DModel(dtype)
    _, optimizer, scheduler, _ = rg3.on_create_optimizer(model, "SGD", 0.1, 1.0)
    before = model.value.detach().clone()

    result = rg3.on_train(model, loader, loader, "cpu", dtype, scaler, optimizer,
                         scheduler, 1, 0, [], [], FakeProgress())
    assert is_ok(result[0]), result[0]
    assert model.value != before
    assert np.isfinite(result[-1]).all()
    evaluation = rg3.on_evaluate(model, dataset, loader, "cpu", dtype, scaler)
    assert is_ok(evaluation[0]), evaluation[0]
    prediction = rg3.on_predict(model, dataset, loader, "cpu", dtype, scaler)
    assert is_ok(prediction[0]), prediction[0]
    assert np.allclose(evaluation[1]["target (Prediction)"], prediction[1]["target (Prediction)"])


def test_3d_regression_empty_inference_reports_the_empty_dataset():
    model = Constant3DModel(torch.float32)
    dataset = SimpleNamespace(target_column_name="target")
    scaler = MinMaxScaler().fit([[0.0], [1.0]])

    evaluation = rg3.on_evaluate(model, dataset, [], "cpu", torch.float32, scaler)
    prediction = rg3.on_predict(model, dataset, [], "cpu", torch.float32, scaler)

    assert is_error(evaluation[0]) and "test split is empty" in evaluation[0]
    assert is_error(prediction[0]) and "prediction dataset is empty" in prediction[0]


def test_3d_regression_constant_targets_match_displayed_and_exported_metrics(tmp_path):
    # Repeated 0.1 values can have nonzero floating-point std despite being identical.
    samples = [Data(z=torch.tensor([6]), pos=torch.zeros(1, 3, dtype=torch.float64),
                    y=torch.tensor([[0.1]], dtype=torch.float64), smiles="C") for _ in range(3)]
    loader = DataLoader(samples, batch_size=3)
    dataset = SimpleNamespace(target_column_name="target")
    scaler = MinMaxScaler().fit([[0.0], [1.0]])
    model = Constant3DModel(torch.float64)
    status, rows, metrics, *_ = rg3.on_evaluate(model, dataset, loader, "cpu", torch.float64, scaler)
    assert is_ok(status), status
    assert np.isnan(metrics.loc[metrics["Metric"] == "R²", "Value"].item())
    status, _ = rg3.on_export_metrics(str(tmp_path), "test_metrics", rows, dataset)
    assert is_ok(status), status
    exported = pd.read_csv(tmp_path / "test_metrics.csv")
    assert np.isnan(exported.loc[exported["Metric"] == "R²", "Value"].item())
    pd.testing.assert_frame_equal(metrics, exported.drop(columns="Target").round({"Value": 4}))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_3d_regression_scales_before_casting_large_targets(workdir_3d, dtype):
    path = os.path.join(workdir_3d, "dataset.csv")
    df = pd.read_csv(path)
    df["target"] = np.arange(len(df), dtype=float) + 100000
    df.to_csv(path, index=False)
    result = rg3.on_process_data(
        workdir_3d, "dataset.csv", "GRAPH_g3d", dtype, 0.25, 0.25, 4, False, 0)
    assert is_ok(result[0]), result[0]
    _, _, dataset, train_dl, val_dl, test_dl, scaler = result
    assert dataset[0].y.item() == 100000
    assert dataset[1].y.item() == 100001
    model = Constant3DModel(dtype)
    _, optimizer, scheduler, _ = rg3.on_create_optimizer(model, "SGD", 0.1, 1.0)

    training = rg3.on_train(model, train_dl, val_dl, "cpu", dtype, scaler, optimizer,
                           scheduler, 1, 0, [], [], FakeProgress())
    evaluation = rg3.on_evaluate(model, dataset, test_dl, "cpu", dtype, scaler)

    assert is_ok(training[0]), training[0]
    assert is_ok(evaluation[0]), evaluation[0]
    assert evaluation[1]["target"].tolist() == df.iloc[test_dl.dataset.indices]["target"].tolist()


@pytest.mark.parametrize("module", [bc3, rg3])
def test_3d_nonfinite_loss_does_not_update_weights_or_report_success(module):
    class Nonfinite3DModel(Constant3DModel):
        def forward(self, z, pos, batch):
            return super().forward(z, pos, batch) * float('nan')

    loader = DataLoader([Data(z=torch.tensor([6]), pos=torch.zeros(1, 3),
                             y=torch.tensor([[1.0]]), smiles="C")])
    model = Nonfinite3DModel(torch.float32)
    _, optimizer, scheduler, _ = module.on_create_optimizer(model, "SGD", 0.1, 1.0)
    before = model.value.detach().clone()
    scaler_args = [MinMaxScaler().fit([[0.0], [1.0]])] if module is rg3 else []

    result = module.on_train(model, loader, loader, "cpu", torch.float32, *scaler_args,
                             optimizer, scheduler, 1, 0, [], [], FakeProgress())

    assert is_error(result[0]) and "not finite" in result[0]
    assert model.value == before
    assert result[3] == len(result[4]) == len(result[5]) == 0


@pytest.mark.parametrize("module", [bc3, rg3])
def test_3d_scheduler_failure_keeps_completed_epoch_count(monkeypatch, module):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(module, "train", lambda *args: 0.3)
    monkeypatch.setattr(module, "validation", lambda *args: 0.4)

    class FailingScheduler:
        def step(self):
            raise RuntimeError("scheduler failure")

    scaler_args = [MinMaxScaler().fit([[0.0], [1.0]])] if module is rg3 else []
    result = module.on_train(model, [], [], "cpu", torch.float32, *scaler_args, optimizer,
                             FailingScheduler(), 1, 0, [], [], FakeProgress())

    assert is_error(result[0]) and "scheduler failure" in result[0]
    assert result[3] == len(result[4]) == len(result[5]) == 1


@pytest.mark.parametrize("module", [bc3, rg3])
def test_named_3d_runs_capture_seeds_save_load_and_export(workdir_3d, module):
    processed = module.on_process_data(workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32,
                                       0.25, 0.25, 4, False, 7)
    assert is_ok(processed[0]), processed[0]
    dataset, train_dl, _, test_dl = processed[2:6]
    assert dataset.split_context["split_seed"] == 7
    assert dataset.split_context["train_indices"] == list(train_dl.dataset.indices)
    assert dataset.split_context["test_indices"] == list(test_dl.dataset.indices)
    args = create_model_3d_args(dataset, "SchNet")
    args[4] = 13
    created = module.on_create_model(*args, model_name="spatial")
    assert is_ok(created[0]), created[0]
    model = created[-4]
    assert model._run_context["model_seed"] == 13 and model._run_context["split_seed"] == 7
    scaler_args = [processed[6]] if module is rg3 else []
    assert is_ok(module.on_save_checkpoint(workdir_3d, "weights", model, 2, *scaler_args)[0])
    relative = os.path.relpath(os.path.join(run_directory(model._run_context), "weights_2.ckpt"), workdir_3d)
    assert relative in list_saved_models(workdir_3d, ".ckpt")
    args[4] = 99
    other = module.on_create_model(*args, model_name="changed_name")[-4]
    loaded = module.on_load_checkpoint(workdir_3d, relative, other, [], [], 0, *scaler_args)
    assert is_ok(loaded[0]), loaded[0]
    assert other._run_context == model._run_context
    rows = dataset.df.iloc[test_dl.dataset.indices].copy()
    rows["target (Prediction)"] = rows["target"]
    if module is bc3:
        rows["target (Probability)"] = rows["target"] * 0.8 + 0.1
    for callback, inputs in [(module.on_export_losses, ("losses", [0.1], [0.2])),
                             (module.on_export_evaluation, ("evaluation", rows)),
                             (module.on_export_metrics, ("metrics", rows, dataset)),
                             (module.on_export_prediction, ("predictions", rows))]:
        result = route_model_export(callback)(workdir_3d, *inputs, other)
        assert is_ok(result[0]), result[0]
        assert os.path.isfile(os.path.join(run_directory(model._run_context), inputs[0] + ".csv"))
        assert not os.path.exists(os.path.join(workdir_3d, inputs[0] + ".csv"))
    choices = module.on_content_list_change(workdir_3d, "dataset", "GRAPH_g3d", relative)
    assert choices[2]["choices"] == ["GRAPH_g3d"] and relative in choices[4]["choices"]
    other_split = module.on_process_data(workdir_3d, "dataset.csv", "GRAPH_g3d", torch.float32,
                                         0.25, 0.25, 4, False, 8)
    _, optimizer, scheduler, _ = module.on_create_optimizer(other, "SGD", 0.01, 1.0)
    result = module.on_train(other, other_split[3], other_split[4], "cpu", torch.float32,
                             *scaler_args, optimizer, scheduler, 1, 0, [], [], FakeProgress())
    assert is_error(result[0]) and "original" in result[0] and result[3] == 0


@pytest.mark.parametrize("module", [bc3, rg3])
@pytest.mark.parametrize("seed", [None, 1.5, -1])
def test_3d_invalid_model_seed_reports_an_error(module, seed):
    args = create_model_3d_args(SimpleNamespace(), "SchNet")
    args[4] = seed
    result = module.on_create_model(*args)
    assert is_error(result[0]) and "seed" in result[0].lower()
    assert result[-4] is None
