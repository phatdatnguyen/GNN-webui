"""Classical evaluation plots consume cached rows, including degenerate holdouts."""

from io import BytesIO

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest

from classical_plots import make_evaluation_plots


def classification_rows(actual, predicted, probability):
    return pd.DataFrame({"activity": actual, "activity (Prediction)": predicted,
                         "activity (Probability)": probability})


def assert_renderable_and_closed(figure):
    assert not plt.fignum_exists(figure.number)
    image = BytesIO()
    figure.savefig(image, format="png")
    assert image.tell() > 1000


def test_classification_plots_use_predictions_for_confusion_and_probabilities_for_roc():
    rows = classification_rows([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9])
    matrix, roc = make_evaluation_plots("classification", rows, "activity")
    np.testing.assert_array_equal(matrix.axes[0].images[0].get_array(), [[1, 1], [1, 1]])
    assert matrix.axes[0].get_xlabel() == "Predicted label"
    assert matrix.axes[0].get_ylabel() == "True label"
    assert "area = 1.00" in roc.axes[0].lines[0].get_label()
    assert_renderable_and_closed(matrix)
    assert_renderable_and_closed(roc)


@pytest.mark.parametrize("actual,predicted,expected", [
    ([0, 0], [0, 0], [[2, 0], [0, 0]]),
    ([1, 1], [0, 1], [[0, 0], [1, 1]]),
    ([1], [1], [[0, 0], [0, 1]]),
])
def test_single_class_holdouts_keep_two_class_matrix_and_explain_unavailable_roc(actual, predicted, expected):
    rows = classification_rows(actual, predicted, [0.7] * len(actual))
    matrix, roc = make_evaluation_plots("classification", rows, "activity")
    np.testing.assert_array_equal(matrix.axes[0].images[0].get_array(), expected)
    assert any("single class" in text.get_text() for text in roc.axes[0].texts)
    assert not roc.axes[0].lines
    assert_renderable_and_closed(matrix)
    assert_renderable_and_closed(roc)


@pytest.mark.parametrize("actual,predicted", [
    ([100, 200, 300], [110, 190, 290]),
    ([5, 5, 5], [5, 5, 5]),
    ([2], [4]),
    ([0], [0]),
])
def test_regression_scatter_keeps_original_units_and_a_finite_identity_line(actual, predicted):
    rows = pd.DataFrame({"energy": actual, "energy (Prediction)": predicted})
    scatter, extra = make_evaluation_plots("regression", rows, "energy")
    assert extra is None
    axes = scatter.axes[0]
    np.testing.assert_array_equal(axes.collections[0].get_offsets(), np.column_stack([actual, predicted]))
    identity = axes.lines[0]
    np.testing.assert_array_equal(identity.get_xdata(), identity.get_ydata())
    assert np.isfinite(identity.get_xdata()).all()
    assert identity.get_xdata()[0] < identity.get_xdata()[-1]
    assert identity.get_xdata()[0] <= min(actual + predicted)
    assert identity.get_xdata()[-1] >= max(actual + predicted)
    assert_renderable_and_closed(scatter)


def test_selected_target_switch_uses_the_corresponding_cached_columns():
    rows = pd.DataFrame({"first": [1, 2], "first (Prediction)": [3, 4],
                         "second": [10, 20], "second (Prediction)": [30, 40]})
    first, _ = make_evaluation_plots("regression", rows, "first")
    second, _ = make_evaluation_plots("regression", rows, "second")
    np.testing.assert_array_equal(first.axes[0].collections[0].get_offsets(), [[1, 3], [2, 4]])
    np.testing.assert_array_equal(second.axes[0].collections[0].get_offsets(), [[10, 30], [20, 40]])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "not a number"])
def test_nonfinite_or_nonnumeric_values_fail_before_creating_figures(bad_value):
    before = set(plt.get_fignums())
    rows = pd.DataFrame({"energy": [1], "energy (Prediction)": [bad_value]})
    with pytest.raises(ValueError, match="finite numbers"):
        make_evaluation_plots("regression", rows, "energy")
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("rows", [None, pd.DataFrame(), pd.DataFrame({"other": [1]})])
def test_missing_evaluation_or_target_is_a_readable_error(rows):
    with pytest.raises(ValueError, match="evaluate the model"):
        make_evaluation_plots("classification", rows, "activity")
