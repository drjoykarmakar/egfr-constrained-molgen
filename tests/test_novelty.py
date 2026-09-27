import numpy as np

from src.novelty import (
    fraction_at_or_above,
    nearest_reference,
    sampled_mean_pairwise_tanimoto,
)


def test_identical_smiles_has_nearest_similarity_one():
    refs = ["CCO", "c1ccccc1"]
    names, sims = nearest_reference(["CCO"], refs)
    assert names == ["CCO"]
    assert sims.shape == (1,)
    assert sims[0] == 1.0


def test_fraction_threshold_is_inclusive_for_reporting():
    values = np.array([0.49, 0.50, 0.70, 0.90])
    assert fraction_at_or_above(values, 0.5) == 0.75
    assert fraction_at_or_above(values, 0.7) == 0.5


def test_internal_pairwise_similarity_is_bounded():
    value = sampled_mean_pairwise_tanimoto(["CCO", "CCN", "c1ccccc1"], max_pairs=100)
    assert 0.0 <= value <= 1.0
