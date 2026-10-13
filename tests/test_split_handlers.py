"""Exercise split selection through every tab without generating 3D conformers."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import binary_classification_hybrid as bc
import binary_classification_3d as bc3
import regression_hybrid as rg
import regression_3d as rg3
from conftest import is_error, is_ok


MODULES = [bc, rg, bc3, rg3]
SCAFFOLD_GROUPS = [
    ["c1ccccc1", "Cc1ccccc1"],
    ["c1ccncc1", "Cc1ccncc1"],
    ["C1CCCCC1", "OC1CCCCC1"],
    ["C1CCCC1", "OC1CCCC1"],
    ["c1ccoc1", "Cc1ccoc1"],
    ["c1ccsc1", "Cc1ccsc1"],
    ["CCO", "CCCC"],  # Acyclic molecules share the empty Murcko scaffold.
]


@pytest.fixture
def split_workdir(tmp_path):
    smiles = [smiles for group in SCAFFOLD_GROUPS for smiles in group]
    pd.DataFrame({"SMILES": smiles, "target": np.arange(len(smiles)) % 2}).to_csv(
        tmp_path / "dataset.csv", index=False)

    # One-hot bits make training-only feature selection observable: a bit present
    # solely in a held-out molecule has zero variance in the training subset.
    fingerprint_df = pd.DataFrame(np.eye(len(smiles), dtype=int),
                                  columns=[f"Bit_{i}" for i in range(len(smiles))])
    fingerprint_df.insert(0, "SMILES", smiles)
    (tmp_path / "fps").mkdir()
    fingerprint_df.to_csv(tmp_path / "fps" / "molecular_fingerprints.csv", index=False)

    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    (graph_dir / "smiles.txt").write_text("\n".join(smiles))
    for index, smi in enumerate(smiles):
        # Split processing only needs aligned graph records, not optimized coordinates.
        torch.save(Data(z=torch.tensor([6]), pos=torch.zeros(1, 3), smiles=smi),
                   graph_dir / f"{index}.pt")
    return tmp_path


def process(module, workdir, method="Scaffold", seed=17, reduction=False, shuffle=False):
    if module in (bc, rg):
        args = [str(workdir), "dataset.csv", "Molecular fingerprints", None, "fps",
                torch.float32, reduction, 0.0, 2, 0.25, 0.25, 2, shuffle, seed]
    else:
        args = [str(workdir), "dataset.csv", "graphs", torch.float32,
                0.25, 0.25, 2, shuffle, seed]
    # None invokes the legacy positional signature, which must retain Random behavior.
    return module.on_process_data(*args, *([] if method is None else [method]))


def loaders(module, result):
    return result[4:7] if module in (bc, rg) else result[3:6]


def split_indices(module, result):
    return [list(loader.dataset.indices) for loader in loaders(module, result)]


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_scaffold_processing_keeps_groups_together_and_every_row(split_workdir, module):
    result = process(module, split_workdir)
    assert is_ok(result[0]), result[0]
    indices = split_indices(module, result)
    assert all(indices)
    assert sorted(index for subset in indices for index in subset) == list(range(14))
    for subset, label in zip(indices, ("training", "validation", "test")):
        assert f"{len(subset)} {label}" in result[0]
    # Fixture pairs have the same known scaffold, including the acyclic pair.
    group_sets = [{index // 2 for index in subset} for subset in indices]
    for first, second in combinations(group_sets, 2):
        assert first.isdisjoint(second)

    for loader in loaders(module, result):
        expected = loader.dataset.dataset.df.iloc[loader.dataset.indices]["SMILES"].tolist()
        assert [smiles for batch in loader for smiles in batch.smiles] == expected


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_scaffold_split_and_training_shuffle_are_reproducible(split_workdir, module):
    first = process(module, split_workdir, shuffle=True)
    second = process(module, split_workdir, shuffle=True)
    assert is_ok(first[0]), first[0]
    assert is_ok(second[0]), second[0]
    assert split_indices(module, first) == split_indices(module, second)
    for first_loader, second_loader in zip(loaders(module, first), loaders(module, second)):
        assert [batch.smiles for batch in first_loader] == [batch.smiles for batch in second_loader]


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_random_default_preserves_existing_seeded_splits(split_workdir, module):
    legacy = process(module, split_workdir, method=None)
    explicit = process(module, split_workdir, method="Random")
    repeated = process(module, split_workdir, method="Random")
    other_seed = process(module, split_workdir, method="Random", seed=18)
    for result in (legacy, explicit, repeated, other_seed):
        assert is_ok(result[0]), result[0]

    expected = torch.utils.data.random_split(
        list(range(14)), [8, 3, 3], generator=torch.Generator().manual_seed(17))
    assert split_indices(module, legacy) == [subset.indices for subset in expected]
    assert split_indices(module, legacy) == split_indices(module, explicit)
    assert split_indices(module, explicit) == split_indices(module, repeated)
    assert split_indices(module, explicit) != split_indices(module, other_seed)


@pytest.mark.parametrize("module", [bc, rg], ids=lambda module: module.__name__)
def test_scaffold_feature_transformers_fit_only_training_molecules(split_workdir, module):
    result = process(module, split_workdir, reduction=True)
    assert is_ok(result[0]), result[0]
    train_indices = split_indices(module, result)[0]
    selector, pca = result[7:9]
    fingerprints = pd.read_csv(split_workdir / "fps" / "molecular_fingerprints.csv").drop(columns="SMILES")
    train_features = fingerprints.iloc[train_indices].to_numpy()

    np.testing.assert_allclose(selector.variances_, train_features.var(axis=0))
    assert set(np.flatnonzero(selector.get_support())) == set(train_indices)
    assert pca.n_samples_ == len(train_indices)
    np.testing.assert_allclose(pca.mean_, selector.transform(train_features).mean(axis=0))


@pytest.mark.parametrize("module", [rg, rg3], ids=lambda module: module.__name__)
def test_scaffold_target_scaler_ignores_extreme_heldout_values(split_workdir, module):
    csv_path = split_workdir / "dataset.csv"
    df = pd.read_csv(csv_path)
    df["target"] = np.arange(len(df), dtype=float) + 1.0
    df.to_csv(csv_path, index=False)
    first = process(module, split_workdir)
    assert is_ok(first[0]), first[0]
    train_indices, validation_indices, test_indices = split_indices(module, first)
    heldout = validation_indices + test_indices
    df.loc[heldout, "target"] = [(-1 if index % 2 else 1) * 1e8 for index in range(len(heldout))]
    df.to_csv(csv_path, index=False)

    result = process(module, split_workdir)
    assert is_ok(result[0]), result[0]
    assert split_indices(module, first) == split_indices(module, result)
    scaler = result[-1]
    training_targets = df.iloc[train_indices][["target"]].to_numpy()
    np.testing.assert_allclose(scaler.data_min_, training_targets.min(axis=0))
    np.testing.assert_allclose(scaler.data_max_, training_targets.max(axis=0))
    assert scaler.n_samples_seen_ == len(train_indices)


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_unknown_split_method_is_reported_without_raising(split_workdir, module):
    result = process(module, split_workdir, method="Unknown")
    assert is_error(result[0]) and "split" in result[0].lower()
    assert result[1].get("interactive") is False
    assert all(loader is None for loader in loaders(module, result))


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__)
def test_too_few_scaffolds_reports_how_to_choose_another_split(split_workdir, module):
    # Distinct acyclic molecules still share one empty Murcko scaffold. Splitting
    # this fixture randomly would silently violate the requested scaffold isolation.
    smiles = ["CCO", "CCCC"] * 7
    dataset_path = split_workdir / "dataset.csv"
    df = pd.read_csv(dataset_path)
    df["SMILES"] = smiles
    df.to_csv(dataset_path, index=False)
    fingerprint_path = split_workdir / "fps" / "molecular_fingerprints.csv"
    fingerprints = pd.read_csv(fingerprint_path)
    fingerprints["SMILES"] = smiles
    fingerprints.to_csv(fingerprint_path, index=False)
    (split_workdir / "graphs" / "smiles.txt").write_text("\n".join(smiles))
    for index, smi in enumerate(smiles):
        torch.save(Data(z=torch.tensor([6]), pos=torch.zeros(1, 3), smiles=smi),
                   split_workdir / "graphs" / f"{index}.pt")

    result = process(module, split_workdir)

    assert is_error(result[0])
    assert "scaffold" in result[0].lower() and "random" in result[0].lower()
    assert result[1].get("interactive") is False
    assert all(loader is None for loader in loaders(module, result))
