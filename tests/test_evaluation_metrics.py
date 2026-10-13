"""Classification metric definitions, degenerate splits, and shared tab behavior."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation_metrics import classification_metrics, metrics_from_predictions


def values(y_true, probabilities, y_pred=None):
    if y_pred is None:
        y_pred = (np.asarray(probabilities) > 0.5).astype(int)
    return classification_metrics(y_true, y_pred, probabilities).set_index("Metric")["Value"]


def test_known_precision_recall_area_distinguishes_average_precision():
    # PR points: (recall, precision) = (1, 1/2), (1, 2/3), (1/2, 1/2),
    # (1/2, 1), (0, 1). Trapezoids give 19/24; the step integral gives 5/6.
    actual = values([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    expected = {
        "Accuracy": 0.75, "Balanced accuracy": 0.75, "Precision": 1.0,
        "Recall": 0.5, "Specificity": 1.0, "F1 score": 0.6667,
        "MCC": 0.5774, "ROC-AUC score": 0.75, "AUPRC": 0.7917,
        "Average precision": 0.8333,
    }
    assert actual.to_dict() == pytest.approx(expected)


def test_balanced_metrics_do_not_hide_minority_class_errors():
    # TN=7, FP=1, FN=1, TP=1: accuracy .8, balanced accuracy (.875+.5)/2.
    actual = values([0] * 8 + [1, 1], [0.1] * 7 + [0.9, 0.8, 0.2])
    assert actual["Accuracy"] == 0.8
    assert actual["Balanced accuracy"] == 0.6875
    assert actual["F1 score"] == 0.5
    assert actual["MCC"] == 0.375


def test_extreme_imbalance_does_not_reward_predicting_only_negatives():
    actual = values([0] * 99 + [1], [0.1] * 99 + [0.4])
    assert actual["Accuracy"] == 0.99
    assert actual["Balanced accuracy"] == 0.5
    assert actual["F1 score"] == 0
    assert actual["MCC"] == 0
    # The ranking is still perfect; the fixed threshold misses the rare positive.
    assert actual["AUPRC"] == 1 and actual["Average precision"] == 1


@pytest.mark.parametrize("perfect", [True, False], ids=["perfect", "inverted"])
def test_perfect_and_inverted_predictions(perfect):
    probabilities = [0.1, 0.2, 0.8, 0.9] if perfect else [0.9, 0.8, 0.2, 0.1]
    actual = values([0, 0, 1, 1], probabilities)
    for name in ["Accuracy", "Balanced accuracy", "Precision", "Recall", "Specificity", "F1 score", "ROC-AUC score"]:
        assert actual[name] == float(perfect)
    assert actual["MCC"] == (1.0 if perfect else -1.0)
    assert actual["AUPRC"] == (1.0 if perfect else 0.2917)
    assert actual["Average precision"] == (1.0 if perfect else 0.4167)


def test_no_positive_predictions_keeps_ranking_scores_separate():
    actual = values([1, 0], [0.4, 0.1])
    for name in ["Precision", "Recall", "F1 score", "MCC"]:
        assert actual[name] == 0
    assert actual["Specificity"] == 1
    assert actual["Balanced accuracy"] == 0.5
    for name in ["ROC-AUC score", "AUPRC", "Average precision"]:
        assert actual[name] == 1


@pytest.mark.parametrize("label", [0, 1], ids=["all_negative", "all_positive"])
@pytest.mark.parametrize("constant_prediction", [False, True])
def test_single_class_split_reports_unavailable_metrics(label, constant_prediction):
    predictions = [label] * 3 if constant_prediction else [0, 1, 0]
    actual = values([label] * 3, [0.2, 0.8, 0.4], predictions)
    for name in ["Balanced accuracy", "MCC", "ROC-AUC score", "AUPRC", "Average precision"]:
        assert np.isnan(actual[name]), name
    assert np.isnan(actual["Recall" if label == 0 else "Specificity"])
    for name in ["Accuracy", "Precision", "F1 score", "Specificity" if label == 0 else "Recall"]:
        assert np.isfinite(actual[name]), name


@pytest.mark.parametrize("y_true,y_pred,probabilities", [
    ([], [], []),
    ([0, 1], [0], [0.1, 0.9]),
    ([[0, 1]], [[0, 1]], [[0.1, 0.9]]),
    ([0, 2], [0, 1], [0.1, 0.9]),
    ([0, 1], [0, 2], [0.1, 0.9]),
    ([0, 1], [0, 1], [0.1, np.nan]),
    ([0, 1], [0, 1], [0.1, np.inf]),
])
def test_invalid_evaluation_data_is_rejected(y_true, y_pred, probabilities):
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_pred, probabilities)


@pytest.mark.parametrize("actual,predicted", [([2.0], [3.0]), ([2.0, 2.0], [2.0, 2.0]),
                                           ([2.0, 2.0], [1.0, 3.0])])
def test_regression_export_marks_undefined_r2_as_missing(actual, predicted):
    rows = pd.DataFrame({"target": actual, "target (Prediction)": predicted})
    metrics = metrics_from_predictions("regression", rows, ["target"])
    assert list(metrics.columns) == ["Target", "Metric", "Value"]
    scores = metrics.set_index("Metric")["Value"]
    assert np.isnan(scores["R²"])
    assert np.isfinite(scores.drop("R²")).all()


def test_regression_export_retains_precision_and_original_units():
    rows = pd.DataFrame({"target": [100.001, 200.002], "target (Prediction)": [100, 200]})
    scores = metrics_from_predictions("regression", rows, ["target"]).set_index("Metric")["Value"]
    assert scores["MAE"] == pytest.approx(0.0015)
    assert scores["MSE"] == pytest.approx(0.0000025)
    assert scores["RMSE"] == pytest.approx(np.sqrt(0.0000025))


@pytest.mark.parametrize("y_true,probabilities", [
    ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
    ([0, 0, 0], [0.2, 0.8, 0.4]),
    ([1, 1, 1], [0.2, 0.8, 0.4]),
    ([0, 1], [0.1, 0.4]),
])
def test_2d_and_3d_tabs_report_identical_metrics(monkeypatch, y_true, probabilities):
    import binary_classification_hybrid as bc
    import binary_classification_3d as bc3

    logits = torch.logit(torch.tensor(probabilities)).reshape(-1, 1)
    targets = torch.tensor(y_true).reshape(-1, 1)
    smiles = ["CCO"] * len(y_true)
    monkeypatch.setattr(bc3, "test", lambda *args: (targets, logits, smiles))
    model = torch.nn.Linear(1, 1)
    dataset = SimpleNamespace(target_column_name="target")
    status, eval_df, metrics_3d, cm_3d, roc_3d, _, _ = bc3.on_evaluate(
        model, dataset, object(), "cpu", torch.float32,
    )
    assert "color: green" in status, status
    metrics_2d, cm_2d, roc_2d = bc.on_target_column_change(eval_df, "target")
    pd.testing.assert_frame_equal(metrics_2d, metrics_3d)
    assert all(figure is not None for figure in [cm_2d, cm_3d, roc_2d, roc_3d])


def test_2d_metrics_follow_the_selected_target():
    import binary_classification_hybrid as bc

    frame = pd.DataFrame({
        "target_a": [0, 0, 1, 1], "target_b": [1, 1, 0, 0],
        "target_a (Prediction)": [0, 0, 1, 1], "target_b (Prediction)": [0, 0, 1, 1],
        "target_a (Probability)": [0.1, 0.2, 0.8, 0.9],
        "target_b (Probability)": [0.1, 0.2, 0.8, 0.9],
    })
    first = bc.on_target_column_change(frame, "target_a")[0].set_index("Metric")["Value"]
    second = bc.on_target_column_change(frame, "target_b")[0].set_index("Metric")["Value"]
    assert first["MCC"] == 1 and second["MCC"] == -1
    assert first["AUPRC"] == 1 and second["AUPRC"] == 0.2917
