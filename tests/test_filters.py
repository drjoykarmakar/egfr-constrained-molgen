import numpy as np

import src.filters as filters
from src.filters import FilterConfig, is_interesting_similarity, passes_property_gates, run_filter_funnel


def test_property_gate_boundaries_are_inclusive_except_similarity():
    cfg = FilterConfig()
    assert passes_property_gates(mw=200.0, clogp=-1.0, qed=0.40, config=cfg)
    assert passes_property_gates(mw=600.0, clogp=5.0, qed=0.40, config=cfg)
    assert not passes_property_gates(mw=199.99, clogp=0.0, qed=0.8, config=cfg)
    assert is_interesting_similarity(0.499999, 0.5)
    assert not is_interesting_similarity(0.5, 0.5)


def test_funnel_drops_invalid_duplicates_benchmark_and_property_failures():
    # Caffeine is the benchmark/training reference. Ibuprofen passes the simple
    # property gate and is structurally distant enough to survive the 0.5 cut.
    caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    ibuprofen = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    raw = [
        "not_smiles",   # step 1 fail
        ibuprofen,
        ibuprofen,       # step 4 duplicate
        caffeine,        # step 5 benchmark exact match
        "CCO",           # step 6 MW fail
    ]
    result = run_filter_funnel(
        raw,
        project2_benchmark_smiles={caffeine},
        training_active_smiles=[caffeine],
        config=FilterConfig(pains_action="flag_only"),
    )

    assert result.counts["sampled_raw"] == 5
    assert result.counts["step1_valid_parse_sanitize"] == 4
    assert result.counts["step4_unique_samples"] == 3
    assert result.counts["step5_not_exact_project2_benchmark"] == 2
    assert result.counts["step6_property_pass"] == 1
    assert result.counts["step9_tanimoto_lt_cutoff"] == 1
    assert result.survivors.iloc[0]["standardized_smiles"] == ibuprofen
    assert result.survivors.iloc[0]["nearest_training_active_tanimoto"] < 0.5
    assert "nearest_training_active_tanimoto" in result.valid_unique.columns
    assert "exact_training_active" in result.valid_unique.columns


def test_pains_flag_only_does_not_silently_delete(monkeypatch):
    ibuprofen = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    monkeypatch.setattr(filters, "pains_flag", lambda mol: (True, "synthetic-test-alert"))

    result = run_filter_funnel(
        [ibuprofen],
        project2_benchmark_smiles=set(),
        training_active_smiles=[caffeine],
        config=FilterConfig(pains_action="flag_only"),
    )
    assert result.counts["pains_flagged_at_step7"] == 1
    assert result.counts["step7_after_pains_policy"] == 1
    assert bool(result.survivors.iloc[0]["pains"]) is True


def test_optional_scorer_adds_scores_without_changing_membership():
    ibuprofen = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    result = run_filter_funnel(
        [ibuprofen],
        project2_benchmark_smiles=set(),
        training_active_smiles=[caffeine],
        scorer=lambda smiles: np.full(len(smiles), 0.42),
    )
    assert len(result.survivors) == 1
    assert result.counts["step10_optional_activity_scored"] == 1
    assert result.survivors.iloc[0]["project2_active_probability"] == 0.42
