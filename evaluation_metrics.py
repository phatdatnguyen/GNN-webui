"""Shared evaluation metrics and all-target tables for all six workflows."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, probabilities):
    """Return the same metric table for a single target in any classification tab.

    Class 1 is positive. AUPRC integrates the precision-recall curve with the
    trapezoidal rule; average precision is its separate, non-interpolated summary.
    Class-balanced and ranking scores are unavailable without both true classes,
    including all-positive sets whose PR scores would otherwise misleadingly be 1.
    Recall/specificity are unavailable when their actual class is absent.
    Precision and F1 retain sklearn's zero_division=0 convention. MCC uses sklearn's
    value of 0 for constant predictions when both true classes are present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)
    if (y_true.ndim != 1 or y_pred.ndim != 1 or probabilities.ndim != 1
            or len(y_true) == 0 or len(y_true) != len(y_pred)
            or len(y_true) != len(probabilities)):
        raise ValueError("Metrics require nonempty, equally sized arrays for one target.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Classification labels and predictions must contain only 0 and 1.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities must be finite.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = (tp + fn > 0) and (tn + fp > 0)
    unavailable = float("nan")
    balanced_accuracy = roc_auc = auprc = average_precision = mcc = unavailable
    if both_classes:
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, probabilities)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities, pos_label=1)
        auprc = auc(recall_curve, precision_curve)
        average_precision = average_precision_score(y_true, probabilities, pos_label=1)
        mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0) if tp + fn > 0 else unavailable,
        "Specificity": tn / (tn + fp) if tn + fp > 0 else unavailable,
        "F1 score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": mcc,
        "ROC-AUC score": roc_auc,
        "AUPRC": auprc,
        "Average precision": average_precision,
    }
    return pd.DataFrame({
        "Metric": list(metrics),
        "Value": [round(float(value), 4) for value in metrics.values()],
    })


def metrics_from_predictions(task, evaluation_df, target_names):
    """Build metric tables from cached evaluation rows without calling an estimator."""
    if not isinstance(task, str) or task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    if not isinstance(evaluation_df, pd.DataFrame) or evaluation_df.empty or target_names is None:
        raise ValueError("Please evaluate a nonempty dataset before displaying metrics.")
    target_names = list(target_names)
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("Please specify unique evaluated target names.")
    tables = []
    for target in target_names:
        columns = [target, f"{target} (Prediction)"]
        if task == "classification":
            columns.append(f"{target} (Probability)")
        if not evaluation_df.columns.is_unique or any(column not in evaluation_df for column in columns):
            raise ValueError(f"Evaluation results are missing unique columns for target '{target}'.")
        try:
            values = evaluation_df.loc[:, columns].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Evaluation values must be finite numbers.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Evaluation values must be finite numbers.")
        actual, predictions = values[:, 0], values[:, 1]
        if task == "classification":
            probabilities = values[:, 2]
            if not ((0 <= probabilities) & (probabilities <= 1)).all():
                raise ValueError("Prediction probabilities must be between 0 and 1.")
            table = classification_metrics(actual, predictions, probabilities)
        else:
            mse = mean_squared_error(actual, predictions)
            table = pd.DataFrame({
                "Metric": ["MAE", "MSE", "RMSE", "R²"],
                "Value": [mean_absolute_error(actual, predictions), mse, np.sqrt(mse),
                          r2_score(actual, predictions) if len(actual) > 1 and np.ptp(actual) > 0
                          else float("nan")],
            })
        table.insert(0, "Target", target)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
