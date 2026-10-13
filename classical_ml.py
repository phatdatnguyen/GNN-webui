"""Classical molecular fingerprint models with fitted preprocessing and split provenance."""

from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from classical_catalog import (
    MODEL_NAMES, MODEL_PARAMETERS, MODEL_NOTES, make_pipeline,
    validate_training_data, validate_prediction_data, validate_pipeline,
)
from datasets import _check_fingerprints_match_dataset
from evaluation_metrics import metrics_from_predictions
from utils import FINGERPRINT_CONFIG, sanitize_path_component, split_molecular_dataset


_FORMAT = "gnn-webui-classical"
_VERSION = 1


def _validate_task(task):
    if not isinstance(task, str) or task not in MODEL_NAMES:
        raise ValueError("Task must be classification or regression.")


def model_parameter_specs(task):
    """Return editable UI specifications without exposing shared mutable defaults."""
    _validate_task(task)
    return deepcopy(MODEL_PARAMETERS[task])


@dataclass
class FingerprintData:
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray | None
    target_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    fingerprint_config: dict | None
    task: str

    def __len__(self):
        return len(self.df)


@dataclass
class ProcessedData:
    data: FingerprintData
    train_indices: np.ndarray
    val_indices: np.ndarray  # Always empty; retained in the persisted split identity format.
    test_indices: np.ndarray
    split_method: str = "Random"
    random_seed: int = 0

    @property
    def task(self):
        return self.data.task

    @property
    def identity(self):
        """Identify the exact data and partitions, including labels and feature values."""
        digest = hashlib.sha256()
        metadata = [self.task, self.data.df["SMILES"].tolist(), self.data.target_names,
                    self.data.feature_names, self.data.fingerprint_config]
        digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        for array in (self.data.X, self.data.y):
            if array is None:
                digest.update(b"none")
            else:
                values = np.ascontiguousarray(array, dtype="<f8")
                digest.update(str(values.shape).encode("ascii"))
                digest.update(values.tobytes())
        for indices in (self.train_indices, self.val_indices, self.test_indices):
            values = np.ascontiguousarray(indices, dtype="<i8")
            digest.update(str(values.shape).encode("ascii"))
            digest.update(values.tobytes())
        return digest.hexdigest()


@dataclass
class FittedModel:
    task: str
    model_name: str
    target_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    fingerprint_config: dict | None
    pipelines: tuple[object, ...]
    training_identity: str
    parameters: dict
    random_seed: int


def _read_csv(path, description):
    try:
        with open(path, newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), [])
        if not header or any(not name.strip() for name in header) or len(header) != len(set(header)):
            raise ValueError(f"{description} must have nonempty, unique column names.")
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read {description}: {exc}") from exc


def _validate_config(config, width):
    if config is None:
        return
    if (not isinstance(config, dict) or set(config) != {"name", "radius", "number_of_bits"}
            or not isinstance(config["name"], str) or not config["name"].strip()
            or type(config["number_of_bits"]) is not int or config["number_of_bits"] != width
            or (config["radius"] is not None and
                (type(config["radius"]) is not int or config["radius"] < 0))
            or (config["name"] == "Morgan fingerprint" and config["radius"] is None)):
        raise ValueError("Fingerprint configuration is invalid or does not match its columns. "
                         "Please re-extract molecular fingerprints.")


def _validate_data(data, require_targets=False):
    if not isinstance(data, FingerprintData):
        raise ValueError("Please load molecular fingerprints first.")
    _validate_task(data.task)
    if "SMILES" not in data.df or len(data) == 0:
        raise ValueError("The dataset must contain SMILES and at least one molecule.")
    if not all(isinstance(value, str) and value.strip() for value in data.df["SMILES"]):
        raise ValueError("Every dataset row must contain a nonempty SMILES string.")
    if (not isinstance(data.X, np.ndarray) or data.X.ndim != 2 or data.X.shape[0] != len(data)
            or data.X.shape[1] == 0 or len(data.feature_names) != data.X.shape[1]):
        raise ValueError("Fingerprint rows and feature columns do not match the dataset.")
    if tuple(data.feature_names) != tuple(f"Bit_{i}" for i in range(1, data.X.shape[1] + 1)):
        raise ValueError("Fingerprint feature columns must be Bit_1, Bit_2, ... in extraction order. "
                         "Please re-extract molecular fingerprints.")
    if not np.issubdtype(data.X.dtype, np.number) or not np.isfinite(data.X).all():
        raise ValueError("Molecular fingerprints must contain finite numeric values.")
    _validate_config(data.fingerprint_config, data.X.shape[1])
    if require_targets:
        if (data.y is None or not isinstance(data.y, np.ndarray) or data.y.ndim != 2
                or data.y.shape != (len(data), len(data.target_names)) or not data.target_names):
            raise ValueError("Please provide at least one target column with a value for every molecule.")
        if (len(set(data.target_names)) != len(data.target_names)
                or not all(isinstance(name, str) and name.strip() for name in data.target_names)):
            raise ValueError("Target column names must be nonempty and unique.")
        if not np.issubdtype(data.y.dtype, np.number) or not np.isfinite(data.y).all():
            raise ValueError("Target columns must contain finite numeric values for every molecule.")
        if data.task == "classification" and not np.isin(data.y, [0, 1]).all():
            raise ValueError("Binary classification targets must contain only 0 and 1.")


def load_fingerprint_data(workdir, dataset_name, fp_dir, task, require_targets=True):
    """Load aligned fingerprints; prediction mode ignores optional CSV target columns."""
    _validate_task(task)
    dataset_name = sanitize_path_component(dataset_name, "dataset file name")
    fp_dir = sanitize_path_component(fp_dir, "molecular fingerprint directory")
    if not workdir or not os.path.isdir(workdir):
        raise ValueError("Please open an existing working directory first.")
    df = _read_csv(os.path.join(workdir, dataset_name), "dataset CSV")
    fingerprint_directory = os.path.join(workdir, fp_dir)
    fingerprint_df = _read_csv(os.path.join(fingerprint_directory, "molecular_fingerprints.csv"),
                               "molecular fingerprint CSV")
    if "SMILES" not in df or "SMILES" not in fingerprint_df:
        raise ValueError("Dataset and fingerprint CSV files must both contain a SMILES column.")
    _check_fingerprints_match_dataset(fingerprint_df, df, fingerprint_directory)
    feature_names = tuple(column for column in fingerprint_df.columns if column != "SMILES")
    target_names = tuple(column for column in df.columns if column != "SMILES") if require_targets else ()
    try:
        X = fingerprint_df.loc[:, feature_names].to_numpy(dtype=np.float64)
        y = df.loc[:, target_names].to_numpy(dtype=np.float64) if target_names else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Fingerprint and target columns must contain finite numeric values.") from exc
    config_path = os.path.join(fingerprint_directory, FINGERPRINT_CONFIG)
    config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as stream:
                config = json.load(stream)
            if config is None:
                raise ValueError("Missing fingerprint settings.")
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot read fingerprint configuration. Please re-extract molecular fingerprints.") from exc
    data = FingerprintData(df, X, y, target_names, feature_names, config, task)
    _validate_data(data, require_targets=require_targets)
    return data


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32:
        raise ValueError("Random seed must be an integer from 0 to 4294967295.")
    return int(value)


def process_data(workdir, dataset_name, fp_dir, task, test_ratio=0.2,
                 random_seed=0, split_method="Random"):
    data = load_fingerprint_data(workdir, dataset_name, fp_dir, task)
    random_seed = _seed(random_seed)
    subsets = split_molecular_dataset(data, test_ratio, 0, random_seed, split_method)
    return ProcessedData(data, *(np.asarray(subset.indices, dtype=np.int64) for subset in subsets),
                         split_method=split_method, random_seed=random_seed)


def _validate_processed(processed):
    if not isinstance(processed, ProcessedData):
        raise ValueError("Please process the training data first.")
    _validate_data(processed.data, require_targets=True)
    partitions = [np.asarray(indices) for indices in
                  (processed.train_indices, processed.val_indices, processed.test_indices)]
    if any(indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) for indices in partitions):
        raise ValueError("Split indices must be one-dimensional integer arrays. Please process the data again.")
    if len(partitions[1]):
        raise ValueError("Classical models use training and test splits only. Please process the data again.")
    all_indices = np.concatenate(partitions)
    if (len(partitions[0]) == 0 or len(all_indices) != len(processed.data)
            or not np.array_equal(np.sort(all_indices), np.arange(len(processed.data)))):
        raise ValueError("Training and test splits must partition the dataset without overlap. "
                         "Please process the data again.")


def _parameters(task, model_name, supplied):
    if model_name not in MODEL_NAMES[task]:
        raise ValueError(f"Unknown {task} model '{model_name}'. Please choose a listed model.")
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("Model parameters must be supplied as a parameter dictionary.")
    specs = MODEL_PARAMETERS[task][model_name]
    unknown = set(supplied) - set(specs)
    if unknown:
        raise ValueError(f"Unknown parameters for {model_name}: {', '.join(sorted(unknown))}.")
    parameters = {}
    for name, spec in specs.items():
        value = supplied.get(name, spec["default"])
        if name == "max_depth" and value is None and spec["min"] == 0:
            value = 0
        if name == "class_weight" and value is None:
            value = "None"
        if spec["type"] == "choice":
            if value not in spec["choices"]:
                raise ValueError(f"Invalid {name}: choose one of {spec['choices']}.")
        else:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} must be a number.")
            if not np.isfinite(value) or not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{name} must be between {spec['min']} and {spec['max']}.")
            if spec["type"] == "int" and int(value) != value:
                raise ValueError(f"{name} must be an integer.")
            value = int(value) if spec["type"] == "int" else float(value)
        parameters[name] = None if ((name == "max_depth" and value == 0)
                                    or (name == "class_weight" and value == "None")) else value
    return parameters


def fit_model(processed, model_name, parameters=None, random_seed=0):
    _validate_processed(processed)
    random_seed = _seed(random_seed)
    parameters = _parameters(processed.task, model_name, parameters)
    X = processed.data.X[processed.train_indices]
    y = processed.data.y[processed.train_indices]
    if processed.task == "classification":
        for index, target in enumerate(processed.data.target_names):
            if not np.array_equal(np.unique(y[:, index]), [0, 1]):
                raise ValueError(f"Training target '{target}' must contain both classes 0 and 1. "
                                 "Choose another random seed or split, or add examples of the missing class.")
    pipelines = []
    for index, target in enumerate(processed.data.target_names):
        try:
            validate_training_data(processed.task, model_name, parameters, X, y[:, index])
            pipeline = make_pipeline(processed.task, model_name, parameters, random_seed,
                                     y_train=y[:, index])
            pipeline.fit(X, y[:, index])
        except Exception as exc:
            raise ValueError(f"Could not fit {model_name} for target '{target}': {exc}") from exc
        pipelines.append(pipeline)
    return FittedModel(processed.task, model_name, tuple(processed.data.target_names),
                       tuple(processed.data.feature_names), deepcopy(processed.data.fingerprint_config),
                       tuple(pipelines), processed.identity, parameters, random_seed)


def _validate_model(model, task=None):
    if not isinstance(model, FittedModel):
        raise ValueError("Please train or load a classical model first.")
    _validate_task(model.task)
    if task is not None and model.task != task:
        raise ValueError(f"This is a {model.task} model; please load a {task} model in this tab.")
    if (not isinstance(model.model_name, str) or model.model_name not in MODEL_NAMES[model.task]
            or not isinstance(model.target_names, (tuple, list)) or not model.target_names
            or not all(isinstance(name, str) and name.strip() for name in model.target_names)
            or len(set(model.target_names)) != len(model.target_names)
            or not isinstance(model.feature_names, (tuple, list)) or not model.feature_names
            or tuple(model.feature_names) != tuple(f"Bit_{i}" for i in range(1, len(model.feature_names) + 1))
            or not isinstance(model.pipelines, (tuple, list))
            or len(model.pipelines) != len(model.target_names)
            or not isinstance(model.training_identity, str) or len(model.training_identity) != 64
            or any(char not in "0123456789abcdef" for char in model.training_identity)
            or not isinstance(model.parameters, dict)):
        raise ValueError("The saved model has invalid target, feature, or training split metadata.")
    _seed(model.random_seed)
    _validate_config(model.fingerprint_config, len(model.feature_names))
    for pipeline in model.pipelines:
        try:
            validate_pipeline(model.task, model.model_name, pipeline)
            check_is_fitted(pipeline)
            if pipeline.n_features_in_ != len(model.feature_names):
                raise ValueError("Incorrect feature width.")
            if model.task == "classification" and not np.array_equal(pipeline.classes_, [0, 1]):
                raise ValueError("Incorrect classifier classes.")
        except Exception as exc:
            raise ValueError("The saved model pipeline is unfitted or does not match its feature schema.") from exc


def _compatible_data(model, data):
    _validate_model(model)
    _validate_data(data)
    if data.task != model.task:
        raise ValueError(f"Please load prediction data for {model.task}.")
    if tuple(data.feature_names) != tuple(model.feature_names):
        raise ValueError("Prediction fingerprint feature schema does not match the trained model. "
                         "Use the same fingerprint type and bit count, then re-extract molecular fingerprints.")
    if (model.fingerprint_config is None) != (data.fingerprint_config is None):
        raise ValueError("Fingerprint settings are missing for either the model or prediction data. "
                         "Please re-extract training and prediction fingerprints with matching settings and retrain.")
    if model.fingerprint_config != data.fingerprint_config:
        raise ValueError("Prediction fingerprint settings do not match the trained model. "
                         "Please re-extract molecular fingerprints with the same type, radius and bit count.")


def _predict_arrays(model, X):
    validate_prediction_data(model.task, model.model_name, X)
    predictions, probabilities = [], []
    for target, pipeline in zip(model.target_names, model.pipelines):
        try:
            prediction = np.asarray(pipeline.predict(X), dtype=np.float64)
            if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
                raise ValueError("Predictions are non-finite or have an incorrect shape.")
            predictions.append(prediction)
            if model.task == "classification":
                probability = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=np.float64)
                if (not np.isin(prediction, [0, 1]).all() or probability.shape != (len(X),)
                        or not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all()):
                    raise ValueError("The classifier returned invalid probabilities or labels.")
                probabilities.append(probability)
        except Exception as exc:
            raise ValueError(f"Could not predict target '{target}': {exc}") from exc
    return np.column_stack(predictions), np.column_stack(probabilities) if probabilities else None


def predict(model, data):
    _compatible_data(model, data)
    predictions, probabilities = _predict_arrays(model, data.X)
    result = pd.DataFrame({"SMILES": data.df["SMILES"].tolist()})
    for index, target in enumerate(model.target_names):
        result[f"{target} (Prediction)"] = (predictions[:, index].astype(int)
                                              if model.task == "classification" else predictions[:, index])
        if probabilities is not None:
            result[f"{target} (Probability)"] = probabilities[:, index]
    return result


def evaluate_predictions(model, processed, split="test"):
    """Infer a checked split once and return actual values alongside each prediction."""
    _validate_processed(processed)
    _compatible_data(model, processed.data)
    if model.training_identity != processed.identity:
        raise ValueError("This model was trained on different data or split assignments. "
                         "Restore its original processed split or train a new model before evaluation.")
    split_indices = {"train": processed.train_indices, "test": processed.test_indices}
    if split not in split_indices:
        raise ValueError("Evaluation split must be train or test.")
    indices = split_indices[split]
    if len(indices) == 0:
        raise ValueError(f"The {split} split is empty. Adjust the split sizes and train a new model.")
    predictions, probabilities = _predict_arrays(model, processed.data.X[indices])
    actual = processed.data.y[indices]
    columns = ["SMILES"]
    for target in model.target_names:
        columns.extend([target, f"{target} (Prediction)"])
        if probabilities is not None:
            columns.append(f"{target} (Probability)")
    if len(set(columns)) != len(columns):
        raise ValueError("Target names overlap generated evaluation columns. Rename the affected targets and retrain.")
    result = pd.DataFrame({"SMILES": processed.data.df.iloc[indices]["SMILES"].tolist()})
    for index, target in enumerate(model.target_names):
        result[target] = actual[:, index]
        result[f"{target} (Prediction)"] = (predictions[:, index].astype(int)
                                             if model.task == "classification" else predictions[:, index])
        if probabilities is not None:
            result[f"{target} (Probability)"] = probabilities[:, index]
    return result


def evaluate_model(model, processed, split="test"):
    """Return metrics; interactive callers can cache evaluate_predictions instead."""
    rows = evaluate_predictions(model, processed, split)
    return metrics_from_predictions(model.task, rows, model.target_names)


def save_model(model, path):
    """Atomically replace a joblib checkpoint containing preprocessing and provenance."""
    _validate_model(model)
    temporary_path = None
    try:
        path = os.path.abspath(os.fspath(path))
        with tempfile.NamedTemporaryFile(prefix=".classical-", suffix=".joblib",
                                         dir=os.path.dirname(path), delete=False) as stream:
            temporary_path = stream.name
        joblib.dump({"format": _FORMAT, "version": _VERSION, "model": model}, temporary_path, compress=3)
        os.replace(temporary_path, path)
    except Exception as exc:
        raise ValueError(f"Could not save the model: {exc}") from exc
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_model(path, task):
    _validate_task(task)
    try:
        checkpoint = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Could not load the model checkpoint: {exc}") from exc
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != _FORMAT
            or checkpoint.get("version") != _VERSION or "model" not in checkpoint):
        raise ValueError("This file is not a supported classical model checkpoint.")
    model = checkpoint["model"]
    _validate_model(model, task)
    return model
