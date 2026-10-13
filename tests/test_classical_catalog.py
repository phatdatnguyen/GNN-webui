"""Model-specific requirements that must be checked before sklearn fitting."""

import numpy as np
import pytest

import classical_catalog as catalog


def _defaults(task, name):
    parameters = {key: spec["default"] for key, spec in catalog.MODEL_PARAMETERS[task][name].items()}
    if parameters.get("class_weight") == "None":
        parameters["class_weight"] = None
    if parameters.get("max_depth") == 0:
        parameters["max_depth"] = None
    return parameters


def test_qda_handles_more_fingerprint_bits_than_examples_per_class():
    rng = np.random.default_rng(11)
    X = rng.integers(0, 2, size=(12, 20)).astype(float)
    y = np.array([0, 1] * 6)
    parameters = _defaults("classification", "Quadratic Discriminant Analysis")
    estimator = catalog.make_pipeline("classification", "Quadratic Discriminant Analysis", parameters, 0)

    estimator.fit(X, y)
    probabilities = estimator.predict_proba(X)

    assert probabilities.shape == (12, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    catalog.validate_pipeline("classification", "Quadratic Discriminant Analysis", estimator)


@pytest.mark.parametrize("task,name", [("classification", "Dummy Classifier"), ("regression", "Dummy Regressor")])
def test_dummy_model_width_is_preserved_when_estimator_has_no_feature_metadata(task, name):
    """sklearn 1.4 dummy estimators did not expose n_features_in_."""
    pipeline = catalog.make_pipeline(task, name, _defaults(task, name), 0)
    X = np.array([[0, 1, 0], [1, 0, 1]], dtype=float)
    pipeline.fit(X, np.array([0, 1]))
    estimator = pipeline.named_steps["estimator"]
    if hasattr(estimator, "n_features_in_"):
        del estimator.n_features_in_

    assert pipeline.n_features_in_ == 3
    assert pipeline.predict(X).shape == (2,)
    catalog.validate_pipeline(task, name, pipeline)
    with pytest.raises(ValueError, match="features"):
        pipeline.predict(np.ones((2, 4)))
