"""Captured split/run identities and safe storage for reproducible experiments."""

from copy import deepcopy
import hashlib
import json
import numbers
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from utils import sanitize_path_component


MODEL_DIRECTORY_PREFIX = "MODEL_"


def validate_seed(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or int(value) != value or not 0 <= value < 2**32):
        raise ValueError("Random seed must be an integer from 0 to 4294967295.")
    return int(value)


def _json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _indices(values):
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError("Split row indices must be integers.")
        result.append(int(value))
    return result


def _split_identity(context):
    return {key: context[key] for key in (
        "dataset_file", "dataset_hash", "split_seed", "split_method", "train_indices",
        "test_indices", "validation_indices", "smiles", "target_columns")}


def make_split_context(working_directory, dataset_file, df, train_indices, test_indices,
                       split_seed, split_method, validation_indices=None):
    """Capture the data and assignments actually used, independent of later UI edits."""
    if not isinstance(df, pd.DataFrame) or df.empty or "SMILES" not in df or not df.columns.is_unique:
        raise ValueError("Please process a nonempty molecular dataset first.")
    if not working_directory:
        raise ValueError("Please open a working directory first.")
    context = {
        "format": "gnn-webui-split", "version": 1,
        "working_directory": os.path.abspath(os.fspath(working_directory)),
        "dataset_file": sanitize_path_component(dataset_file, "dataset file name"),
        "dataset_hash": hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest(),
        "split_seed": validate_seed(split_seed), "split_method": split_method,
        "train_indices": _indices(train_indices), "test_indices": _indices(test_indices),
        "validation_indices": _indices([] if validation_indices is None else validation_indices),
        "smiles": df["SMILES"].tolist(),
        "target_columns": [str(column) for column in df.columns if column != "SMILES"],
    }
    context["split_id"] = _hash(_split_identity(context))
    validate_split_context(context)
    return context


def validate_split_context(context):
    try:
        if context["format"] != "gnn-webui-split" or context["version"] != 1:
            raise ValueError("Unsupported split metadata.")
        sanitize_path_component(context["dataset_file"], "dataset file name")
        validate_seed(context["split_seed"])
        if context["split_method"] not in ("Random", "Scaffold"):
            raise ValueError("Unknown split method in the captured data.")
        smiles = context["smiles"]
        if not isinstance(smiles, list) or not smiles or any(not isinstance(s, str) or not s.strip() for s in smiles):
            raise ValueError("Captured split contains invalid SMILES rows.")
        partitions = [_indices(context[key]) for key in ("train_indices", "validation_indices", "test_indices")]
        if not partitions[0] or sorted(sum(partitions, [])) != list(range(len(smiles))):
            raise ValueError("Captured split assignments must partition the dataset without overlap.")
        if context["split_id"] != _hash(_split_identity(context)):
            raise ValueError("Captured split metadata has changed. Please process the data again.")
        if not isinstance(context["working_directory"], str) or not os.path.isabs(context["working_directory"]):
            raise ValueError("Captured split has no valid working directory.")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Please process the data to capture its split and random seed first.") from exc
    return context


def _run_identity(context):
    return {key: context[key] for key in (
        "model_name", "model_seed", "workflow", "model_class", "parameters", "split_seed")
    } | {"split_id": context["split_context"]["split_id"]}


def make_run_context(split_context, model_name, model_seed, workflow, model_class, parameters=None):
    validate_split_context(split_context)
    context = {
        "format": "gnn-webui-model-run", "version": 1,
        "working_directory": split_context["working_directory"],
        "model_name": sanitize_path_component(model_name, "model name"),
        "model_seed": validate_seed(model_seed), "workflow": str(workflow),
        "model_class": str(model_class), "parameters": deepcopy(parameters or {}),
        "split_seed": split_context["split_seed"], "split_context": deepcopy(split_context),
        "initialization_seed_verified": True,
    }
    context["run_id"] = _hash(_run_identity(context))
    validate_run_context(context)
    return context


def validate_run_context(context):
    try:
        if context["format"] != "gnn-webui-model-run" or context["version"] != 1:
            raise ValueError("Unsupported model-run metadata.")
        validate_split_context(context["split_context"])
        sanitize_path_component(context["model_name"], "model name")
        validate_seed(context["model_seed"])
        if (context["split_seed"] != context["split_context"]["split_seed"]
                or context["working_directory"] != context["split_context"]["working_directory"]):
            raise ValueError("Model and split metadata disagree.")
        if context["run_id"] != _hash(_run_identity(context)):
            raise ValueError("Model-run settings have changed. Create a new named model run.")
        _json_text(context)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Please create or fit a named model run before saving results.") from exc
    return context


def rebase_run_context(context, working_directory):
    """Keep recorded provenance when loading a copy inside another working directory."""
    validate_run_context(context)
    if not working_directory:
        raise ValueError("Please open a working directory first.")
    result = deepcopy(context)
    root = os.path.abspath(os.fspath(working_directory))
    result["working_directory"] = root
    result["split_context"]["working_directory"] = root
    return result


def run_directory(context):
    validate_run_context(context)
    name = context["model_name"]
    if not name.startswith(MODEL_DIRECTORY_PREFIX):
        name = MODEL_DIRECTORY_PREFIX + name
    return os.path.join(context["working_directory"], f"{name}_{context['model_seed']}")


def _root_directory(path):
    root = os.path.abspath(os.fspath(path))
    if not os.path.isdir(root) or os.path.islink(root):
        raise ValueError("Please open an existing working directory that is not a symbolic link.")
    return root


def _safe_child(parent, name, *, directory=False):
    name = sanitize_path_component(name, "run path component")
    path = os.path.join(parent, name)
    if os.path.islink(path):
        raise ValueError("Run files and directories cannot be symbolic links.")
    if directory:
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError(f"The run directory name '{name}' is already used by a file.")
        os.makedirs(path, exist_ok=True)
    return path


def _write_metadata(path, data):
    if os.path.islink(path):
        raise ValueError("Run metadata cannot be a symbolic link.")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as stream:
                previous = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ValueError("Existing run metadata cannot be read. Choose another model name.") from exc
        if previous != data:
            raise ValueError("This run directory belongs to different data or model settings. Choose another model name.")
        return
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(path),
                                         prefix=".metadata-", suffix=".json", delete=False) as stream:
            temporary_path = stream.name
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _portable(context):
    result = deepcopy(context)
    result.pop("working_directory", None)
    if "split_context" in result:
        result["split_context"].pop("working_directory", None)
    return result


def ensure_split_directory(context):
    validate_split_context(context)
    root = _root_directory(context["working_directory"])
    directory = _safe_child(root, f"random_seed_{context['split_seed']}", directory=True)
    # Different fractions/methods at the same seed are distinct captured splits.
    _write_metadata(_safe_child(directory, f"split_{context['split_id']}.json"), _portable(context))
    return directory


def ensure_run_directory(context):
    validate_run_context(context)
    ensure_split_directory(context["split_context"])
    root = _root_directory(context["working_directory"])
    directory = _safe_child(root, os.path.basename(run_directory(context)), directory=True)
    _write_metadata(_safe_child(directory, "run.json"), _portable(context))
    return directory


def list_saved_models(working_directory, suffix):
    """List checkpoints inside MODEL_ directories, excluding unrelated artifacts."""
    if not working_directory or not os.path.isdir(working_directory):
        return []
    root = _root_directory(working_directory)
    paths = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not name.startswith(MODEL_DIRECTORY_PREFIX) or os.path.islink(path):
            continue
        if os.path.isdir(path):
            for filename in sorted(os.listdir(path)):
                model_path = os.path.join(path, filename)
                if filename.endswith(suffix) and os.path.isfile(model_path) and not os.path.islink(model_path):
                    paths.append(os.path.join(name, filename))
    return paths


def resolve_saved_model(working_directory, relative_name, suffix):
    root = _root_directory(working_directory)
    if not isinstance(relative_name, str) or not relative_name or os.path.isabs(relative_name) or "\\" in relative_name:
        raise ValueError("Please select a saved model inside the working directory.")
    parts = relative_name.split("/")
    # Direct handler callers may still explicitly open a legacy root file. The UI
    # only lists MODEL_ folders; paths to any other subdirectory are refused.
    if len(parts) not in (1, 2) or (len(parts) == 2 and not parts[0].startswith(MODEL_DIRECTORY_PREFIX)):
        raise ValueError("Please select a checkpoint inside a MODEL_ subdirectory.")
    path = root
    for part in parts:
        path = _safe_child(path, part)
    if not path.endswith(suffix) or not os.path.isfile(path):
        raise ValueError("The selected saved model does not exist.")
    if os.path.commonpath([os.path.realpath(root), os.path.realpath(path)]) != os.path.realpath(root):
        raise ValueError("Saved model must remain inside its working directory.")
    return path


def read_run_context_for_file(path):
    """Read a portable MODEL_ run manifest; imported files may have no manifest."""
    file_path = Path(path).absolute()
    manifest = file_path.parent / "run.json"
    if not manifest.exists():
        return None
    if manifest.is_symlink() or file_path.is_symlink():
        raise ValueError("Saved model and metadata cannot be symbolic links.")
    try:
        context = json.loads(manifest.read_text(encoding="utf-8"))
        root = str(file_path.parent.parent)
        context["working_directory"] = root
        context["split_context"]["working_directory"] = root
        validate_run_context(context)
        if run_directory(context) != str(file_path.parent):
            raise ValueError("The saved model folder does not match its recorded run name and seeds.")
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read this model run's metadata.") from exc
    return context


def ensure_model_split(model, dataset):
    context = getattr(model, "_run_context", None)
    split = getattr(dataset, "split_context", None)
    if context is not None and split is not None:
        validate_run_context(context)
        validate_split_context(split)
        if context["split_context"]["split_id"] != split["split_id"]:
            raise ValueError("This model belongs to a different data split. Process its original split or create a new model.")
