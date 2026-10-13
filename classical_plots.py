"""Evaluation figures made from cached classical-model predictions."""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


def make_evaluation_plots(task, evaluation_df, target):
    """Return confusion/ROC figures or an actual-versus-predicted regression figure.

    The returned figures remain renderable after being closed in pyplot, so a
    session can display them without leaving global matplotlib figures allocated.
    No estimator is consulted when switching the selected target.
    """
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    columns = [target, f"{target} (Prediction)"]
    if task == "classification":
        columns.append(f"{target} (Probability)")
    if (not isinstance(evaluation_df, pd.DataFrame) or evaluation_df.empty
            or not evaluation_df.columns.is_unique
            or any(column not in evaluation_df for column in columns)):
        raise ValueError("Please evaluate the model and select an available target first.")
    try:
        values = evaluation_df.loc[:, columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation values must be finite numbers.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Evaluation values must be finite numbers.")
    actual, predicted = values[:, 0], values[:, 1]
    if task == "classification":
        if (not np.isin(values[:, :2], [0, 1]).all()
                or not ((0 <= values[:, 2]) & (values[:, 2] <= 1)).all()):
            raise ValueError("Classification requires labels 0/1 and probabilities between 0 and 1.")

    figures = []
    try:
        if task == "classification":
            matrix_figure, matrix_axes = plt.subplots(figsize=(6, 6))
            figures.append(matrix_figure)
            display = ConfusionMatrixDisplay(confusion_matrix(actual, predicted, labels=[0, 1]),
                                             display_labels=[0, 1])
            display.plot(cmap="Blues", ax=matrix_axes)
            matrix_axes.set_title(f"Confusion matrix — {target}")
            matrix_figure.tight_layout()

            roc_figure, roc_axes = plt.subplots(figsize=(6, 6))
            figures.append(roc_figure)
            roc_axes.set_title(f"ROC curve — {target}")
            if np.unique(actual).size < 2:
                roc_axes.text(0.5, 0.5, "ROC curve unavailable:\nthe evaluated split contains a single class",
                              ha="center", va="center", transform=roc_axes.transAxes)
                roc_axes.set_axis_off()
            else:
                false_positive_rate, true_positive_rate, _ = roc_curve(actual, values[:, 2])
                area = auc(false_positive_rate, true_positive_rate)
                roc_axes.plot(false_positive_rate, true_positive_rate, color="darkorange", lw=2,
                              label=f"ROC curve (area = {area:.2f})")
                roc_axes.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
                roc_axes.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
                             ylabel="True positive rate")
                roc_axes.legend(loc="lower right")
            roc_figure.tight_layout()
            return matrix_figure, roc_figure

        scatter_figure, scatter_axes = plt.subplots(figsize=(7, 7))
        figures.append(scatter_figure)
        scatter_axes.scatter(actual, predicted, alpha=0.75)
        lower, upper = float(values.min()), float(values.max())
        if lower == upper:
            padding = max(abs(lower) * 0.05, 0.5)
            lower, upper = lower - padding, upper + padding
        scatter_axes.plot([lower, upper], [lower, upper], color="navy", lw=1,
                          linestyle="--", label="Perfect prediction")
        scatter_axes.set(xlabel="Actual value", ylabel="Predicted value",
                         title=f"Actual vs predicted — {target}")
        scatter_axes.grid(True, alpha=0.3)
        scatter_axes.legend()
        scatter_figure.tight_layout()
        return scatter_figure, None
    finally:
        for figure in figures:
            plt.close(figure)
