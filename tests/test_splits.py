"""Split invariants independent of graph extraction or model training."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset, random_split

from utils import split_molecular_dataset


class SmilesDataset(Dataset):
    def __init__(self, smiles):
        self.df = pd.DataFrame({"SMILES": smiles})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]["SMILES"]


def _indices(subsets):
    return [subset.indices for subset in subsets]


def _assert_partition(subsets, dataset):
    rows = [index for subset in subsets for index in subset.indices]
    assert sorted(rows) == list(range(len(dataset)))
    assert len(set(rows)) == len(rows)
    assert all(subset.dataset is dataset for subset in subsets)
    assert len(subsets[0]) > 0


@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize("length,test_ratio,val_ratio", [(17, 0.2, 0.2), (3, 0, 0), (3, 0.5, 0.5)])
def test_random_default_preserves_existing_seeded_split(seed, length, test_ratio, val_ratio):
    dataset = SmilesDataset(["CCO"] * length)
    test_size, val_size = int(length * test_ratio), int(length * val_ratio)
    expected = random_split(dataset, [length - test_size - val_size, val_size, test_size],
                            generator=torch.Generator().manual_seed(seed))

    actual = split_molecular_dataset(dataset, test_ratio, val_ratio, seed)

    assert _indices(actual) == _indices(expected)
    _assert_partition(actual, dataset)


def test_scaffolds_keep_salts_substituents_duplicates_and_stereoisomers_together():
    groups = [
        ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1.[Na+]", "c1ccccc1"],
        ["c1ccncc1", "Cc1ccncc1", "Oc1ccncc1"],
        ["C1CCCCC1", "CC1CCCCC1"],
        ["C1C[C@H]2CCCC[C@@H]2C1", "C1C[C@@H]2CCCC[C@H]2C1"],
        ["CCO", "CCN", "CCCC", "CC(=O)O"],
    ]
    dataset = SmilesDataset([smiles for group in groups for smiles in group])
    original = dataset.df.copy(deep=True)

    subsets = split_molecular_dataset(dataset, 0.2, 0.2, 12, "Scaffold")

    _assert_partition(subsets, dataset)
    assert all(len(subset) > 0 for subset in subsets)
    membership = {index: split for split, subset in enumerate(subsets) for index in subset.indices}
    offset = 0
    for group in groups:
        assert len({membership[index] for index in range(offset, offset + len(group))}) == 1
        offset += len(group)
    assert all(subset.indices == sorted(subset.indices) for subset in subsets)
    pd.testing.assert_frame_equal(dataset.df, original)


def test_scaffold_ratios_are_exact_when_whole_groups_fit():
    # Equal groups make these targets attainable without dividing a scaffold.
    dataset = SmilesDataset([smiles for smiles in
                            ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "c1ccoc1"]
                            for _ in range(2)])

    subsets = split_molecular_dataset(dataset, 0.2, 0.2, 0, "Scaffold")

    _assert_partition(subsets, dataset)
    assert [len(subset) for subset in subsets] == [6, 2, 2]


def test_oversized_scaffold_stays_intact_and_requested_holdouts_stay_nonempty():
    dataset = SmilesDataset(["CCO"] * 20 + ["c1ccccc1", "C1CCCCC1"])

    subsets = split_molecular_dataset(dataset, 0.2, 0.2, 0, "Scaffold")

    _assert_partition(subsets, dataset)
    assert [len(subset) for subset in subsets] == [20, 1, 1]


def test_small_training_target_still_gets_a_scaffold():
    dataset = SmilesDataset(["CCO"] * 8 + ["c1ccccc1", "C1CCCCC1"])

    subsets = split_molecular_dataset(dataset, 0.8, 0.1, 0, "Scaffold")

    _assert_partition(subsets, dataset)
    assert [len(subset) for subset in subsets] == [1, 1, 8]


def test_final_groups_are_reserved_for_requested_holdouts():
    # The training deficit stays largest throughout a naive greedy allocation.
    # Reserving the final groups is necessary to populate both requested holdouts.
    dataset = SmilesDataset(["CCO"] * 34 + ["c1ccccc1"] * 33 + ["C1CCCCC1"] * 33)

    subsets = split_molecular_dataset(dataset, 0.01, 0.01, 0, "Scaffold")

    _assert_partition(subsets, dataset)
    assert [len(subset) for subset in subsets] == [34, 33, 33]


def test_scaffold_seed_breaks_equal_size_ties_reproducibly_without_global_rng_changes():
    dataset = SmilesDataset(["CCO", "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1", "c1ccoc1"])
    before = torch.random.get_rng_state().clone()

    first = split_molecular_dataset(dataset, 0.33, 0.33, 7, "Scaffold")
    repeated = split_molecular_dataset(dataset, 0.33, 0.33, 7, "Scaffold")
    alternatives = {
        tuple(tuple(subset.indices) for subset in
              split_molecular_dataset(dataset, 0.33, 0.33, seed, "Scaffold"))
        for seed in range(5)
    }

    assert _indices(first) == _indices(repeated)
    assert len(alternatives) > 1
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("test_ratio,val_ratio", [(0, 0), (0.1, 0.1)])
def test_zero_sized_holdouts_leave_acyclic_dataset_in_training(test_ratio, val_ratio):
    dataset = SmilesDataset(["CCO", "CCN", "CCC"])

    subsets = split_molecular_dataset(dataset, test_ratio, val_ratio, 0, "Scaffold")

    assert _indices(subsets) == [[0, 1, 2], [], []]


def test_zero_validation_target_stays_empty():
    dataset = SmilesDataset(["CCO", "CCN", "c1ccccc1", "C1CCCCC1"])

    subsets = split_molecular_dataset(dataset, 0.25, 0, 0, "Scaffold")

    _assert_partition(subsets, dataset)
    assert len(subsets[1]) == 0
    assert len(subsets[2]) > 0


@pytest.mark.parametrize("smiles", [["CCO", "CCN", "CCC", "CCCC"],
                                    ["CCO", "CCN", "c1ccccc1", "Cc1ccccc1"]])
def test_too_few_scaffold_groups_reports_actionable_error(smiles):
    with pytest.raises(ValueError, match="Reduce the holdout sizes or choose Random") as error:
        split_molecular_dataset(SmilesDataset(smiles), 0.25, 0.25, 0, "Scaffold")
    assert "All acyclic molecules share one scaffold" in str(error.value)


@pytest.mark.parametrize("invalid", ["NOT_A_SMILES", "", "   ", None, np.nan, 12])
def test_invalid_smiles_are_reported_without_silently_dropping_rows(invalid):
    dataset = SmilesDataset(["CCO", invalid, "c1ccccc1"])

    with pytest.raises(ValueError, match="Invalid SMILES at dataset row 1"):
        split_molecular_dataset(dataset, 0, 0, 0, "Scaffold")

    assert len(dataset) == 3


@pytest.mark.parametrize("split_method", ["Random", "Scaffold"])
@pytest.mark.parametrize("test_ratio,val_ratio", [(-0.01, 0), (0, -0.01), (1.1, 0),
                                                  (0.6, 0.6), (float("nan"), 0),
                                                  (0, float("inf")), (1, 0)])
def test_invalid_ratios_cannot_create_empty_or_invalid_training_split(split_method, test_ratio, val_ratio):
    with pytest.raises(ValueError, match="Invalid split sizes"):
        split_molecular_dataset(SmilesDataset(["CCO"] * 10), test_ratio, val_ratio, 0, split_method)


@pytest.mark.parametrize("split_method", ["Random", "Scaffold"])
def test_empty_dataset_is_rejected(split_method):
    with pytest.raises(ValueError, match="training split"):
        split_molecular_dataset(SmilesDataset([]), 0.2, 0.2, 0, split_method)


def test_unknown_split_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown split method"):
        split_molecular_dataset(SmilesDataset(["CCO"]), 0, 0, 0, "Typo")
