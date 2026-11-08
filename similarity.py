"""Exact train–test fingerprint similarity without a pairwise similarity matrix.

Morgan bit fingerprints use the same salt stripping and explicit hydrogens as
feature extraction. Chirality is ignored, matching the existing Morgan extractor.
Validation molecules never participate in the comparison.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import SaltRemover, rdFingerprintGenerator


@dataclass
class SimilarityResult:
    context: dict
    settings: dict
    distribution: pd.DataFrame
    nearest: pd.DataFrame
    summary: dict


def _integer(value, name, minimum, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or int(value) != value or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    return int(value)


def _validate_context(context):
    if context is None:
        raise ValueError("Please process the data before calculating train–test similarity.")
    required = {"working_directory", "dataset_file", "split_seed", "split_method", "train_indices",
                "test_indices", "validation_indices", "smiles", "dataset_hash", "split_id"}
    if not isinstance(context, dict) or not required.issubset(context):
        raise ValueError("The captured split is incomplete. Please process the data again.")
    smiles = context["smiles"]
    if not isinstance(smiles, (list, tuple)) or not smiles:
        raise ValueError("The captured dataset has no molecules. Please process the data again.")
    for key in ("working_directory", "dataset_file", "split_method", "dataset_hash", "split_id"):
        if not isinstance(context[key], str) or not context[key]:
            raise ValueError("The captured split metadata is invalid. Please process the data again.")
    if not os.path.isabs(context["working_directory"]):
        raise ValueError("The captured working directory must be an absolute path.")
    _integer(context["split_seed"], "Data splitting seed", 0, 2**32 - 1)
    partitions = []
    for key in ("train_indices", "validation_indices", "test_indices"):
        indices = context[key]
        if (not isinstance(indices, (list, tuple))
                or any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                       or not 0 <= index < len(smiles) for index in indices)):
            raise ValueError("The captured split has invalid row indices. Please process the data again.")
        partitions.extend(indices)
    if len(partitions) != len(smiles) or len(set(partitions)) != len(smiles):
        raise ValueError("The captured training, validation and test rows must partition the dataset without overlap.")
    if not context["train_indices"]:
        raise ValueError("The training split is empty. Please process data with training molecules.")
    if not context["test_indices"]:
        raise ValueError("The test split is empty. Increase the test size and process the data again.")


def _fingerprint(smiles, row, generator, salt_remover):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("SMILES is empty.")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("SMILES could not be parsed.")
        molecule = salt_remover.StripMol(molecule, dontRemoveEverything=True)
        molecule = Chem.AddHs(molecule)
        return generator.GetFingerprint(molecule)
    except Exception as exc:
        raise ValueError(f"Cannot calculate similarity for dataset row {row}: invalid SMILES {smiles!r}.") from exc


def calculate_similarity(context, radius=2, number_of_bits=2048, *, bins=50, chunk_size=4096,
                         progress=None):
    """Compare every captured training/test pair and find each test row's nearest train row.

    Only the training fingerprints, histogram, and one result per test row are kept.
    Each temporary similarity vector contains at most ``chunk_size`` entries. Test
    rows preserve captured split order; ties use the lowest original training row ID.
    """
    _validate_context(context)
    radius = _integer(radius, "Morgan radius", 0, 16)
    number_of_bits = _integer(number_of_bits, "Fingerprint bit count", 64, 65536)
    bins = _integer(bins, "Histogram bin count", 1, 1000)
    chunk_size = _integer(chunk_size, "Similarity chunk size", 1, 65536)
    context = deepcopy(context)
    train_rows = sorted(context["train_indices"])
    test_rows = context["test_indices"]
    smiles = context["smiles"]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=number_of_bits,
                                                         includeChirality=False)
    salt_remover = SaltRemover.SaltRemover()

    def report(value, description):
        if progress is not None:
            progress(value, desc=description)

    report(0, "Generating training fingerprints")
    training_fingerprints = []
    for position, row in enumerate(train_rows):
        training_fingerprints.append(_fingerprint(smiles[row], row, generator, salt_remover))
        if position % 64 == 0 or position + 1 == len(train_rows):
            report(0.2 * (position + 1) / len(train_rows), "Generating training fingerprints")

    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    nearest_rows = []
    similarity_sum = 0.0
    pair_count = len(train_rows) * len(test_rows)
    completed_pairs = 0
    for test_row in test_rows:
        query = _fingerprint(smiles[test_row], test_row, generator, salt_remover)
        best_similarity, best_train_row = -1.0, None
        for start in range(0, len(training_fingerprints), chunk_size):
            chunk = training_fingerprints[start:start + chunk_size]
            values = np.asarray(DataStructs.BulkTanimotoSimilarity(query, chunk), dtype=np.float64)
            if (values.shape != (len(chunk),) or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("Fingerprint comparison returned invalid Tanimoto similarities.")
            counts += np.histogram(values, bins=edges)[0]
            similarity_sum += float(values.sum())
            best_in_chunk = int(values.argmax())
            # Strictly greater preserves the earliest (lowest original row) tie.
            if values[best_in_chunk] > best_similarity:
                best_similarity = float(values[best_in_chunk])
                best_train_row = train_rows[start + best_in_chunk]
            completed_pairs += len(chunk)
            report(0.2 + 0.8 * (completed_pairs / pair_count),
                   f"Comparing training–test pairs: {completed_pairs:,} / {pair_count:,}")
        nearest_rows.append((test_row, smiles[test_row], best_train_row,
                             smiles[best_train_row], best_similarity))

    distribution = pd.DataFrame({"Bin lower": edges[:-1], "Bin upper": edges[1:],
                                 "Pair count": counts, "Proportion": counts / pair_count})
    nearest = pd.DataFrame(nearest_rows, columns=["Test row", "Test SMILES", "Training row",
                                                "Training SMILES", "Tanimoto similarity"])
    nearest_values = nearest["Tanimoto similarity"].to_numpy()
    summary = {"training_molecules": len(train_rows), "test_molecules": len(test_rows),
               "validation_molecules_excluded": len(context["validation_indices"]),
               "pair_count": pair_count, "mean_pair_similarity": similarity_sum / pair_count,
               "mean_nearest_similarity": float(nearest_values.mean()),
               "median_nearest_similarity": float(np.median(nearest_values)),
               "minimum_nearest_similarity": float(nearest_values.min()),
               "maximum_nearest_similarity": float(nearest_values.max())}
    settings = {"fingerprint": "Morgan", "radius": radius, "number_of_bits": number_of_bits,
                "metric": "Tanimoto", "histogram_bins": bins, "include_chirality": False,
                "remove_salts": True, "dont_remove_everything": True, "explicit_hydrogens": True,
                "row_index_base": 0, "nearest_tie_break": "lowest original training row index",
                "histogram_interval": "left closed, right open; final bin includes 1",
                "rdkit_version": rdBase.rdkitVersion}
    return SimilarityResult(context, settings, distribution, nearest, summary)


def export_similarity(result, kind, file_name, *, current_context=None):
    """Export captured results beside a provenance/settings JSON file in the split folder."""
    from run_utils import ensure_split_directory
    from utils import export_result_csv

    if not isinstance(result, SimilarityResult):
        raise ValueError("Please calculate train–test similarity before exporting.")
    _validate_context(result.context)
    if current_context is not None:
        _validate_context(current_context)
        if current_context != result.context:
            raise ValueError("The processed split changed. Please calculate similarity again before exporting.")
    if kind not in {"distribution", "nearest"}:
        raise ValueError("Choose the similarity distribution or nearest training molecules to export.")
    directory = ensure_split_directory(result.context)
    frame = result.distribution if kind == "distribution" else result.nearest
    metadata = {"analysis": "train_test_similarity", "export": kind,
                "dataset_file": result.context["dataset_file"],
                "dataset_hash": result.context["dataset_hash"], "split_id": result.context["split_id"],
                "split_seed": result.context["split_seed"], "split_method": result.context["split_method"],
                "settings": result.settings, "summary": result.summary}
    # Prepare both files before replacing either existing result. Keep independent
    # backups until both replacements commit, including on interrupted publication.
    staging_directory = tempfile.mkdtemp(prefix=".similarity-", dir=directory)
    cleanup = True
    try:
        filename = export_result_csv(staging_directory, file_name, frame)
        csv_path = os.path.join(directory, filename)
        metadata_path = csv_path + ".metadata.json"
        staged_csv = os.path.join(staging_directory, filename)
        staged_metadata = staged_csv + ".metadata.json"
        destinations = [csv_path, metadata_path]
        for destination in destinations:
            if os.path.islink(destination):
                raise ValueError("Similarity export files cannot be symbolic links.")
            if os.path.exists(destination) and not os.path.isfile(destination):
                raise ValueError("Similarity exports must replace files, not directories.")
        with open(staged_metadata, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
        backups = []
        for index, destination in enumerate(destinations):
            backup = os.path.join(staging_directory, f"previous_{index}")
            existed = os.path.exists(destination)
            if existed:
                shutil.copy2(destination, backup)
            backups.append(backup if existed else None)
        try:
            os.replace(staged_csv, csv_path)
            os.replace(staged_metadata, metadata_path)
        except BaseException as publication_error:
            restore_errors = []
            for destination, backup in zip(destinations, backups):
                try:
                    if backup is not None:
                        os.replace(backup, destination)
                    elif os.path.lexists(destination):
                        os.unlink(destination)
                except BaseException as restore_error:
                    restore_errors.append(restore_error)
            if restore_errors:
                cleanup = False
                raise RuntimeError(
                    "Similarity export failed and its previous files could not all be restored. "
                    f"Recovery files are preserved at {staging_directory}. "
                    f"Restore error: {restore_errors[0]}") from publication_error
            raise
    finally:
        if cleanup:
            shutil.rmtree(staging_directory)
    return csv_path, metadata_path
