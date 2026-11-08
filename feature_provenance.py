"""Capture feature identities once when preparing a GNN dataset."""

import hashlib
import json
import os

import pandas as pd

from utils import FINGERPRINT_CONFIG, GRAPH_MANIFEST, list_graph_files


_READ_SIZE = 1024 * 1024


def _file_signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _hash_file(path):
    """Stream one feature file and reject replacement during the snapshot."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        before = _file_signature(os.fstat(stream.fileno()))
        for chunk in iter(lambda: stream.read(_READ_SIZE), b""):
            digest.update(chunk)
        after = _file_signature(os.fstat(stream.fileno()))
    if before != after or before != _file_signature(os.stat(path)):
        raise ValueError("Feature files changed while processing. Please process the data again.")
    return digest.hexdigest()


class _DigestWriter:
    """A text sink lets pandas format bounded chunks without a second full CSV copy."""

    def __init__(self, digest):
        self.digest = digest

    def write(self, value):
        self.digest.update(value.encode("utf-8"))
        return len(value)


def _read_fingerprint_configuration(fingerprint_directory):
    configuration = None
    config_path = os.path.join(fingerprint_directory, FINGERPRINT_CONFIG)
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as stream:
                configuration = json.load(stream)
            if not isinstance(configuration, dict):
                raise ValueError("Fingerprint configuration must be an object.")
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot read fingerprint configuration. Please re-extract molecular fingerprints.") from exc
    try:
        json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fingerprint configuration contains invalid values. Please re-extract molecular fingerprints.") from exc
    return configuration


def capture_prediction_features(dataset, fingerprint_directory):
    """Capture extraction settings without equating new molecules with training rows."""
    dataset.feature_sources = {
        "fingerprint_config": (_read_fingerprint_configuration(fingerprint_directory)
                               if fingerprint_directory is not None else None),
    }


def ensure_prediction_features(model, prediction_dataset):
    """Reject incompatible inputs before evaluating a named hybrid model."""
    context = getattr(model, "_run_context", None)
    if context is None:
        return
    if not isinstance(context, dict) or not isinstance(context.get("parameters", {}), dict):
        raise ValueError("The model contains invalid feature metadata. Please reload a valid checkpoint.")
    parameters = context.get("parameters", {})
    recorded_mode = parameters.get("load_data")
    if recorded_mode is not None and recorded_mode != getattr(prediction_dataset, "load_data", None):
        raise ValueError("Prediction input mode differs from this model's graph/fingerprint inputs. "
                         "Please process prediction data with the original input mode.")
    if recorded_mode == "Graphs":
        return
    recorded_sources = parameters.get("feature_sources", {})
    if not isinstance(recorded_sources, dict):
        raise ValueError("The model contains invalid fingerprint metadata. Please reload a valid checkpoint.")
    recorded_config = recorded_sources.get("fingerprint_config")
    # Older models/extractions did not retain these settings. Their compatibility
    # remains governed by the existing model input shape checks.
    if recorded_config is None:
        return
    if not isinstance(recorded_config, dict):
        raise ValueError("The model contains invalid fingerprint metadata. Please reload a valid checkpoint.")
    prediction_sources = getattr(prediction_dataset, "feature_sources", {})
    prediction_config = (prediction_sources.get("fingerprint_config")
                         if isinstance(prediction_sources, dict) else None)
    if prediction_config is None:
        raise ValueError("Prediction fingerprints have no extraction metadata. Please re-extract molecular "
                         "fingerprints with this model's original type, radius and bit count.")
    if prediction_config != recorded_config:
        raise ValueError("Prediction fingerprint settings differ from this model's training fingerprints. "
                         "Please re-extract molecular fingerprints with the same type, radius and bit count.")


def capture_feature_sources(graph_directory=None, fingerprint_directory=None, fingerprint_df=None):
    """Return portable directory names and content hashes for the features actually loaded.

    Graph shards are streamed once. Fingerprints use the already loaded DataFrame
    when supplied, preserving its row order and feature schema. File timestamps and
    absolute working-directory paths are excluded so copied datasets retain identity.
    Consumers should retain this snapshot; training and evaluation compare snapshots
    without rescanning the feature files.
    """
    sources = {
        "graph_directory": os.path.basename(os.path.normpath(graph_directory)) if graph_directory else None,
        "fingerprint_directory": os.path.basename(os.path.normpath(fingerprint_directory)) if fingerprint_directory else None,
    }
    if graph_directory:
        indices = list_graph_files(graph_directory)
        if not indices or indices != list(range(len(indices))):
            raise ValueError("Graph features must contain consecutive files 0.pt, 1.pt, ... before processing.")
        graph_digest = hashlib.sha256(b"gnn-graph-features-v1\n")
        filenames = [f"{index}.pt" for index in indices]
        manifest = os.path.join(graph_directory, GRAPH_MANIFEST)
        if os.path.exists(manifest):
            filenames.append(GRAPH_MANIFEST)
        for filename in filenames:
            file_hash = _hash_file(os.path.join(graph_directory, filename))
            graph_digest.update(json.dumps([filename, file_hash], separators=(",", ":")).encode("utf-8"))
            graph_digest.update(b"\n")
        sources["graph_hash"] = graph_digest.hexdigest()

    if fingerprint_directory:
        if fingerprint_df is None:
            fingerprint_df = pd.read_csv(os.path.join(fingerprint_directory, "molecular_fingerprints.csv"))
        if (not isinstance(fingerprint_df, pd.DataFrame) or fingerprint_df.empty
                or "SMILES" not in fingerprint_df or not fingerprint_df.columns.is_unique):
            raise ValueError("Please load a nonempty fingerprint table with unique columns before processing.")
        configuration = _read_fingerprint_configuration(fingerprint_directory)
        config_text = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprint_digest = hashlib.sha256(b"gnn-fingerprint-features-v1\n")
        fingerprint_df.to_csv(_DigestWriter(fingerprint_digest), index=False, lineterminator="\n", chunksize=256)
        fingerprint_digest.update(b"\nconfiguration\n")
        fingerprint_digest.update(config_text.encode("utf-8"))
        sources["fingerprint_hash"] = fingerprint_digest.hexdigest()
        sources["fingerprint_config"] = configuration
    return sources
