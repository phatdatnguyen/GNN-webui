"""Classical models use aligned fingerprints and preserve fitted preprocessing/provenance."""

from copy import deepcopy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import classical_ml as ml
from evaluation_metrics import classification_metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


SCAFFOLDS = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "c1ccoc1", "CCO"]


@pytest.fixture
def fingerprint_workdir(tmp_path):
    smiles = [smiles for smiles in SCAFFOLDS for _ in range(4)]
    features = np.random.default_rng(17).integers(0, 2, (len(smiles), 8))
    frame = pd.DataFrame({"SMILES": smiles, "target_a": np.arange(24) % 2,
                          "target_b": (np.arange(24) // 2) % 2})
    frame.to_csv(tmp_path / "classification.csv", index=False)
    regression = frame.copy()
    regression["target_a"] = 100000.0 + np.arange(24) * 3.0
    regression["target_b"] = 200.0 - np.arange(24) * 2.0
    regression.to_csv(tmp_path / "regression.csv", index=False)
    pd.DataFrame({"SMILES": smiles}).to_csv(tmp_path / "prediction.csv", index=False)
    fingerprints = pd.DataFrame(features, columns=[f"Bit_{i}" for i in range(1, 9)])
    fingerprints.insert(0, "SMILES", smiles)
    (tmp_path / "FP_fps").mkdir()
    fingerprints.to_csv(tmp_path / "FP_fps" / "molecular_fingerprints.csv", index=False)
    (tmp_path / "FP_fps" / "fingerprint_config.json").write_text(json.dumps({
        "name": "Morgan fingerprint", "radius": 2, "number_of_bits": 8,
    }))
    return tmp_path


def processed(directory, task="classification", **kwargs):
    options = dict(test_ratio=0.25, random_seed=3)
    options.update(kwargs)
    return ml.process_data(directory, f"{task}.csv", "FP_fps", task, **options)


def quick_parameters(task, model_name):
    specs = ml.MODEL_PARAMETERS[task][model_name]
    parameters = {}
    if "n_estimators" in specs:
        parameters["n_estimators"] = 5
    if "max_depth" in specs:
        parameters["max_depth"] = 2
    if "n_neighbors" in specs:
        parameters["n_neighbors"] = 3
    if "hidden_layer_sizes" in specs:
        parameters["hidden_layer_sizes"] = specs["hidden_layer_sizes"]["choices"][0]
    return parameters


def fitted_pipeline(estimator):
    """Calibration ensemble=False refits the complete base pipeline on training rows."""
    if hasattr(estimator, "calibrated_classifiers_"):
        return estimator.calibrated_classifiers_[0].estimator
    return estimator


MODELS = [(task, model_name) for task, names in ml.MODEL_NAMES.items() for model_name in names]


@pytest.mark.parametrize("task,model_name", MODELS)
def test_every_model_fits_multi_target_evaluates_predicts_and_round_trips(fingerprint_workdir, task, model_name):
    split = processed(fingerprint_workdir, task)
    model = ml.fit_model(split, model_name, quick_parameters(task, model_name), random_seed=11)
    prediction_data = ml.load_fingerprint_data(fingerprint_workdir, "prediction.csv", "FP_fps", task,
                                              require_targets=False)
    predictions = ml.predict(model, prediction_data)
    # Reuse this fit for the reproducibility check instead of fitting the same
    # estimator twice again in a separate test for every catalogue entry.
    repeated = ml.fit_model(split, model_name, quick_parameters(task, model_name), random_seed=11)
    pd.testing.assert_frame_equal(ml.predict(repeated, prediction_data), predictions)
    assert predictions["SMILES"].tolist() == prediction_data.df["SMILES"].tolist()
    assert len(model.pipelines) == 2
    assert model.pipelines[0] is not model.pipelines[1]
    for target in model.target_names:
        assert np.isfinite(predictions[f"{target} (Prediction)"]).all()
        if task == "classification":
            assert predictions[f"{target} (Prediction)"].isin([0, 1]).all()
            assert predictions[f"{target} (Probability)"].between(0, 1).all()
        else:
            assert f"{target} (Probability)" not in predictions
    if task == "regression" and model_name in {"Random Forest", "Extra Trees", "Gradient Boosting", "Ridge", "SVR", "K-Nearest Neighbors"}:
        # Predictions retain original target units rather than normalized [0, 1] values.
        assert (predictions["target_a (Prediction)"] > 10000).all()
    for evaluation_split in ("train", "test"):
        metrics = ml.evaluate_model(model, split, evaluation_split)
        assert list(metrics) == ["Target", "Metric", "Value"]
        assert metrics["Target"].unique().tolist() == list(model.target_names)
        assert len(metrics) == (20 if task == "classification" else 8)
    checkpoint = fingerprint_workdir / "model.joblib"
    ml.save_model(model, checkpoint)
    restored = ml.load_model(checkpoint, task)
    pd.testing.assert_frame_equal(ml.predict(restored, prediction_data), predictions)
    pd.testing.assert_frame_equal(ml.evaluate_model(restored, split), ml.evaluate_model(model, split))


@pytest.mark.parametrize("task,model_name", [
    ("classification", "Logistic Regression"), ("classification", "SVC"),
    ("classification", "K-Nearest Neighbors"), ("regression", "Ridge"),
    ("regression", "SVR"), ("regression", "K-Nearest Neighbors"),
])
def test_scalers_and_estimators_never_fit_heldout_features(fingerprint_workdir, task, model_name):
    split = processed(fingerprint_workdir, task)
    first = ml.fit_model(split, model_name, quick_parameters(task, model_name), random_seed=11)
    changed = deepcopy(split)
    heldout_indices = changed.test_indices
    changed.data.X[heldout_indices] = 1e9
    if task == "regression":
        changed.data.y[heldout_indices] = 1e12
    else:
        changed.data.y[heldout_indices] = 1 - changed.data.y[heldout_indices]
    second = ml.fit_model(changed, model_name, quick_parameters(task, model_name), random_seed=11)
    for original_pipeline, changed_pipeline in zip(first.pipelines, second.pipelines):
        scaler = fitted_pipeline(changed_pipeline).named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, split.data.X[split.train_indices].mean(axis=0))
        assert scaler.n_samples_seen_ == len(split.train_indices)
        np.testing.assert_allclose(original_pipeline.predict(split.data.X), changed_pipeline.predict(split.data.X))
    assert first.training_identity != second.training_identity


def test_classification_metrics_match_shared_gnn_metrics(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5}, 11)
    prediction = ml.predict(model, split.data)
    table = ml.evaluate_model(model, split)
    for column, target in enumerate(model.target_names):
        expected = classification_metrics(
            split.data.y[split.test_indices, column],
            prediction.iloc[split.test_indices][f"{target} (Prediction)"].to_numpy(),
            prediction.iloc[split.test_indices][f"{target} (Probability)"].to_numpy(),
        )
        actual = table.loc[table["Target"] == target, ["Metric", "Value"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


def test_regression_metrics_use_original_target_units(fingerprint_workdir):
    split = processed(fingerprint_workdir, "regression")
    model = ml.fit_model(split, "Ridge")
    prediction = ml.predict(model, split.data)
    metrics = ml.evaluate_model(model, split).set_index(["Target", "Metric"])["Value"]
    residual = prediction.iloc[split.test_indices]["target_a (Prediction)"].to_numpy() - split.data.y[split.test_indices, 0]
    assert metrics["target_a", "MSE"] == pytest.approx(np.mean(residual**2))
    assert metrics["target_a", "RMSE"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert metrics["target_a", "MAE"] == pytest.approx(np.mean(np.abs(residual)))


def test_scaffold_processing_is_reproducible_and_keeps_groups_together(fingerprint_workdir):
    first = processed(fingerprint_workdir, split_method="Scaffold")
    repeated = processed(fingerprint_workdir, split_method="Scaffold")
    assert first.identity == repeated.identity
    groups = [set(first.data.df.iloc[indices]["SMILES"])
              for indices in (first.train_indices, first.test_indices)]
    assert groups[0].isdisjoint(groups[1])
    assert len(first.val_indices) == 0
    assert len(first.train_indices) + len(first.test_indices) == len(first.data)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_default_processing_reserves_twenty_percent_for_testing_only(fingerprint_workdir, task):
    split = ml.process_data(fingerprint_workdir, f"{task}.csv", "FP_fps", task)
    assert len(split.test_indices) == int(0.2 * len(split.data))
    assert len(split.val_indices) == 0
    assert len(split.train_indices) == len(split.data) - len(split.test_indices)
    np.testing.assert_array_equal(np.sort(np.concatenate([split.train_indices, split.test_indices])),
                                  np.arange(len(split.data)))


@pytest.mark.parametrize("split_name", ["validation", "val"])
def test_validation_evaluation_is_no_longer_offered(fingerprint_workdir, split_name):
    split = processed(fingerprint_workdir)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    with pytest.raises(ValueError, match="train or test"):
        ml.evaluate_model(model, split, split_name)


def test_processed_data_cannot_hide_rows_in_a_validation_partition(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    split.val_indices = split.train_indices[:1]
    split.train_indices = split.train_indices[1:]
    with pytest.raises(ValueError, match="training and test splits only"):
        ml.fit_model(split, "Random Forest")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cached_evaluation_rows_preserve_order_and_support_metrics_without_inference(fingerprint_workdir,
                                                                                    task, monkeypatch):
    split = processed(fingerprint_workdir, task)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    expected_metrics = ml.evaluate_model(model, split)
    rows = ml.evaluate_predictions(model, split)
    assert rows["SMILES"].tolist() == split.data.df.iloc[split.test_indices]["SMILES"].tolist()
    for index, target in enumerate(model.target_names):
        np.testing.assert_array_equal(rows[target], split.data.y[split.test_indices, index])

    def fail_inference(*args, **kwargs):
        raise AssertionError("Cached evaluation must not call the fitted estimator again")

    monkeypatch.setattr(ml, "_predict_arrays", fail_inference)
    pd.testing.assert_frame_equal(ml.metrics_from_predictions(task, rows, model.target_names), expected_metrics)
    for target in reversed(model.target_names):
        selected = ml.metrics_from_predictions(task, rows, (target,))
        expected = expected_metrics.loc[expected_metrics["Target"] == target].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, expected)


def test_evaluation_rejects_target_names_that_collide_with_generated_columns(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    split.data.target_names = ("target_a", "target_a (Prediction)")
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    with pytest.raises(ValueError, match="Target names overlap"):
        ml.evaluate_predictions(model, split)


@pytest.mark.parametrize("change", ["features", "labels", "split", "schema", "smiles"])
def test_evaluation_rejects_a_changed_dataset_or_partition(fingerprint_workdir, change):
    split = processed(fingerprint_workdir)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    changed = deepcopy(split)
    if change == "features":
        changed.data.X[0, 0] += 1
    elif change == "labels":
        changed.data.y[0, 0] = 1 - changed.data.y[0, 0]
    elif change == "split":
        changed.train_indices[0], changed.test_indices[0] = changed.test_indices[0], changed.train_indices[0]
    elif change == "schema":
        changed.data.target_names = ("renamed", "target_b")
    else:
        changed.data.df.loc[0, "SMILES"] = "CCN"
    with pytest.raises(ValueError, match="different data or split"):
        ml.evaluate_model(model, changed)


def test_prediction_allows_new_smiles_with_compatible_fingerprints(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    new_data = deepcopy(split.data)
    new_data.df["SMILES"] = "CCN"
    new_data.X[:] = 0
    assert ml.predict(model, new_data)["SMILES"].tolist() == ["CCN"] * len(new_data)


@pytest.mark.parametrize("config_change", ["radius", "name", "missing_prediction", "missing_model"])
def test_prediction_requires_consistent_fingerprint_settings(fingerprint_workdir, config_change):
    split = processed(fingerprint_workdir)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    data = deepcopy(split.data)
    if config_change == "radius":
        data.fingerprint_config["radius"] = 3
    elif config_change == "name":
        data.fingerprint_config = {"name": "RDKit fingerprint", "radius": None, "number_of_bits": 8}
    elif config_change == "missing_prediction":
        data.fingerprint_config = None
    else:
        model.fingerprint_config = None
    with pytest.raises(ValueError, match="re-extract"):
        ml.predict(model, data)


def test_legacy_metadata_free_fingerprints_remain_usable(fingerprint_workdir):
    (fingerprint_workdir / "FP_fps" / "fingerprint_config.json").unlink()
    split = processed(fingerprint_workdir)
    assert split.data.fingerprint_config is None
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    assert len(ml.predict(model, split.data)) == len(split.data)


@pytest.mark.parametrize("mutation,match", [
    ("reordered_smiles", "out of sync"), ("missing_row", "Number of molecular fingerprints"),
    ("reordered_bits", "feature columns"), ("extra_column", "feature columns"),
    ("nan_feature", "finite numeric"), ("infinite_feature", "finite numeric"),
    ("text_feature", "finite numeric"), ("nan_target", "finite numeric"),
    ("invalid_binary_target", "only 0 and 1"), ("missing_smiles", "SMILES"),
    ("duplicate_header", "unique column"), ("malformed_config", "configuration"),
    ("wrong_config_width", "configuration"),
])
def test_malformed_or_misaligned_input_is_rejected(fingerprint_workdir, mutation, match):
    fp_path = fingerprint_workdir / "FP_fps" / "molecular_fingerprints.csv"
    csv_path = fingerprint_workdir / "classification.csv"
    fingerprints = pd.read_csv(fp_path)
    dataset = pd.read_csv(csv_path)
    if mutation == "reordered_smiles":
        fingerprints.loc[0, "SMILES"] = "CCN"
    elif mutation == "missing_row":
        fingerprints = fingerprints.iloc[:-1]
    elif mutation == "reordered_bits":
        fingerprints = fingerprints[["SMILES", "Bit_2", "Bit_1", *[f"Bit_{i}" for i in range(3, 9)]]]
    elif mutation == "extra_column":
        fingerprints["unexpected"] = 1
    elif mutation in {"nan_feature", "infinite_feature", "text_feature"}:
        fingerprints["Bit_1"] = fingerprints["Bit_1"].astype(object)
        fingerprints.loc[0, "Bit_1"] = {"nan_feature": np.nan, "infinite_feature": np.inf, "text_feature": "bad"}[mutation]
    elif mutation == "nan_target":
        dataset.loc[0, "target_a"] = np.nan
    elif mutation == "invalid_binary_target":
        dataset.loc[0, "target_a"] = 2
    elif mutation == "missing_smiles":
        dataset = dataset.drop(columns="SMILES")
    elif mutation == "malformed_config":
        (fingerprint_workdir / "FP_fps" / "fingerprint_config.json").write_text("{bad")
    elif mutation == "wrong_config_width":
        (fingerprint_workdir / "FP_fps" / "fingerprint_config.json").write_text(json.dumps({
            "name": "Morgan fingerprint", "radius": 2, "number_of_bits": 16,
        }))
    fingerprints.to_csv(fp_path, index=False)
    dataset.to_csv(csv_path, index=False)
    if mutation == "duplicate_header":
        fp_path.write_text(fp_path.read_text().replace("Bit_2", "Bit_1", 1))
    with pytest.raises(ValueError, match=match):
        processed(fingerprint_workdir)


def test_unlabeled_prediction_data_is_allowed_but_not_training_data(fingerprint_workdir):
    data = ml.load_fingerprint_data(fingerprint_workdir, "prediction.csv", "FP_fps", "classification", False)
    assert data.y is None and not data.target_names
    with pytest.raises(ValueError, match="target column"):
        ml.load_fingerprint_data(fingerprint_workdir, "prediction.csv", "FP_fps", "classification")


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_empty_evaluation_split_is_a_readable_error(fingerprint_workdir, task):
    split = processed(fingerprint_workdir, task, test_ratio=0)
    model = ml.fit_model(split, "Random Forest", {"n_estimators": 5})
    with pytest.raises(ValueError, match="split is empty"):
        ml.evaluate_model(model, split, "test")


def test_training_requires_both_classes_per_target(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    split.data.y[split.train_indices, 1] = 1
    with pytest.raises(ValueError, match="target_b.*both classes"):
        ml.fit_model(split, "Random Forest")


def test_knn_neighbor_count_is_checked_before_fitting(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    with pytest.raises(ValueError, match="exceeds the training split size"):
        ml.fit_model(split, "K-Nearest Neighbors", {"n_neighbors": len(split.train_indices) + 1})


@pytest.mark.parametrize("model_name,parameters", [
    ("Random Forest", {"unexpected": 1}), ("Random Forest", {"n_estimators": 0}),
    ("Random Forest", {"max_depth": -1}), ("Random Forest", {"n_estimators": 1.5}),
    ("Random Forest", {"n_estimators": True}), ("Random Forest", {"n_estimators": "5"}),
    ("SVC", {"C": np.nan}), ("SVC", {"kernel": "invalid"}),
    ("Unknown", {}),
])
def test_invalid_model_parameters_are_readable_errors(fingerprint_workdir, model_name, parameters):
    with pytest.raises(ValueError):
        ml.fit_model(processed(fingerprint_workdir), model_name, parameters)


def test_target_task_and_checkpoint_schema_are_validated(fingerprint_workdir):
    model = ml.fit_model(processed(fingerprint_workdir), "Random Forest", {"n_estimators": 5})
    checkpoint = fingerprint_workdir / "model.joblib"
    ml.save_model(model, checkpoint)
    with pytest.raises(ValueError, match="classification model.*regression"):
        ml.load_model(checkpoint, "regression")
    model.feature_names = ("Bit_1",)
    joblib.dump({"format": "gnn-webui-classical", "version": 1, "model": model}, checkpoint)
    with pytest.raises(ValueError, match="configuration|schema"):
        ml.load_model(checkpoint, "classification")
    joblib.dump({"version": 99}, checkpoint)
    with pytest.raises(ValueError, match="supported classical model"):
        ml.load_model(checkpoint, "classification")


def test_failed_checkpoint_save_preserves_the_previous_file(fingerprint_workdir, monkeypatch):
    model = ml.fit_model(processed(fingerprint_workdir), "Random Forest", {"n_estimators": 5})
    path = fingerprint_workdir / "model.joblib"
    ml.save_model(model, path)
    previous = path.read_bytes()

    def fail_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ml.joblib, "dump", fail_dump)
    with pytest.raises(ValueError, match="disk full"):
        ml.save_model(model, path)
    assert path.read_bytes() == previous
    assert not list(fingerprint_workdir.glob(".classical-*"))


@pytest.mark.parametrize("attribute,value", [
    ("target_names", None), ("target_names", [["target_a"], "target_b"]),
    ("feature_names", 8), ("pipelines", None), ("training_identity", "z" * 64),
    ("parameters", None), ("random_seed", -1), ("task", []),
])
def test_malformed_checkpoint_metadata_is_a_readable_error(fingerprint_workdir, attribute, value):
    model = ml.fit_model(processed(fingerprint_workdir), "Random Forest", {"n_estimators": 5})
    setattr(model, attribute, value)
    checkpoint = fingerprint_workdir / "malformed.joblib"
    joblib.dump({"format": "gnn-webui-classical", "version": 1, "model": model}, checkpoint)
    with pytest.raises(ValueError):
        ml.load_model(checkpoint, "classification")


def test_overlapping_or_missing_partition_indices_are_rejected(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    split.test_indices[0] = split.train_indices[0]
    with pytest.raises(ValueError, match="without overlap"):
        ml.fit_model(split, "Random Forest")


def test_path_traversal_is_rejected(fingerprint_workdir):
    with pytest.raises(ValueError, match="dataset file name"):
        ml.load_fingerprint_data(fingerprint_workdir, "../classification.csv", "FP_fps", "classification")
    with pytest.raises(ValueError, match="fingerprint directory"):
        ml.load_fingerprint_data(fingerprint_workdir, "classification.csv", "../fps", "classification")


def test_parameter_metadata_returns_independent_copies():
    specs = ml.model_parameter_specs("classification")
    specs["Random Forest"]["n_estimators"]["default"] = 1
    assert ml.MODEL_PARAMETERS["classification"]["Random Forest"]["n_estimators"]["default"] == 100


@pytest.mark.parametrize("model_name", ["Random Forest", "Extra Trees", "Logistic Regression", "SVC"])
def test_supported_classifiers_accept_balanced_class_weights(fingerprint_workdir, model_name):
    split = processed(fingerprint_workdir)
    options = {**quick_parameters("classification", model_name), "class_weight": "balanced"}
    model = ml.fit_model(split, model_name, options, random_seed=11)
    assert all(fitted_pipeline(pipeline).named_steps["estimator"].class_weight == "balanced"
               for pipeline in model.pipelines)
    assert len(ml.predict(model, split.data)) == len(split.data)


def test_probability_calibration_fits_preprocessing_inside_training_folds(fingerprint_workdir, monkeypatch):
    split = processed(fingerprint_workdir)
    # Distinguishable rows reveal the exact subsets passed to each scaler fit.
    split.data.X[:, 0] = np.arange(len(split.data))
    observed_rows = []
    original_fit = StandardScaler.fit

    def record_fit(self, X, *args, **kwargs):
        observed_rows.append(set(np.asarray(X)[:, 0].astype(int)))
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    model = ml.fit_model(split, "SVC", random_seed=11)
    assert all(isinstance(estimator, CalibratedClassifierCV) for estimator in model.pipelines)
    training_rows = set(split.train_indices)
    assert observed_rows and all(rows <= training_rows for rows in observed_rows)
    assert any(rows < training_rows for rows in observed_rows)
    assert any(rows == training_rows for rows in observed_rows)
    assert len(ml.predict(model, split.data)) == len(split.data)


def test_calibration_reports_too_few_training_examples(fingerprint_workdir):
    split = processed(fingerprint_workdir)
    split.data.y[split.train_indices, 0] = 0
    split.data.y[split.train_indices[0], 0] = 1
    with pytest.raises(ValueError, match="(?i)two|2|calibrat"):
        ml.fit_model(split, "SVC")


@pytest.mark.parametrize("model_name,parameters,invalid_value,match", [
    ("Gamma Regression", {}, 0, "strictly positive"),
    ("Gamma Regression", {}, -1, "strictly positive"),
    ("Poisson Regression", {}, -1, "nonnegative"),
    ("Poisson Regression", {}, 0, "positive value"),
    ("Tweedie Regression", {"power": "1"}, -1, "nonnegative"),
    ("Tweedie Regression", {"power": "1"}, 0, "positive value"),
    ("Tweedie Regression", {"power": "1.5"}, -1, "nonnegative"),
    ("Tweedie Regression", {"power": "1.5"}, 0, "positive value"),
    ("Tweedie Regression", {"power": "2"}, 0, "strictly positive"),
    ("Tweedie Regression", {"power": "3"}, -1, "strictly positive"),
])
def test_glm_models_report_target_domain_errors(fingerprint_workdir, model_name, parameters,
                                               invalid_value, match):
    split = processed(fingerprint_workdir, "regression")
    split.data.y[split.train_indices, 1] = invalid_value
    with pytest.raises(ValueError, match=f"target_b.*{match}"):
        ml.fit_model(split, model_name, parameters)


def test_tweedie_power_zero_accepts_real_targets(fingerprint_workdir):
    split = processed(fingerprint_workdir, "regression")
    split.data.y[:, 0] -= 100050
    model = ml.fit_model(split, "Tweedie Regression", {"power": "0"})
    assert np.isfinite(ml.predict(model, split.data)["target_a (Prediction)"]).all()


@pytest.mark.parametrize("model_name", ["Poisson Regression", "Gamma Regression", "Tweedie Regression"])
def test_glm_training_domain_checks_only_training_targets(fingerprint_workdir, model_name):
    split = processed(fingerprint_workdir, "regression")
    split.data.y[split.test_indices] = -1
    model = ml.fit_model(split, model_name)
    assert len(ml.evaluate_model(model, split)) == 8


@pytest.mark.parametrize("model_name", ["Multinomial Naive Bayes", "Complement Naive Bayes"])
def test_count_naive_bayes_rejects_negative_training_features(fingerprint_workdir, model_name):
    split = processed(fingerprint_workdir)
    split.data.X[split.train_indices[0], 0] = -1
    with pytest.raises(ValueError, match="nonnegative fingerprint"):
        ml.fit_model(split, model_name)


@pytest.mark.parametrize("model_name", ["Multinomial Naive Bayes", "Complement Naive Bayes"])
def test_count_naive_bayes_rejects_negative_prediction_features(fingerprint_workdir, model_name):
    split = processed(fingerprint_workdir)
    model = ml.fit_model(split, model_name)
    prediction_data = deepcopy(split.data)
    prediction_data.X[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative fingerprint"):
        ml.predict(model, prediction_data)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_bagging_rejects_sampling_fraction_that_rounds_to_no_rows(fingerprint_workdir, task):
    with pytest.raises(ValueError, match="no training molecules"):
        ml.fit_model(processed(fingerprint_workdir, task), "Bagging", {"max_samples": 0.01})
