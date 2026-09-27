"""Summary metrics for generated-SMILES quality and novelty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.novelty import fraction_at_or_above, sampled_mean_pairwise_tanimoto


def generation_metrics(
    *,
    counts: dict[str, int],
    valid_unique: pd.DataFrame,
    survivors: pd.DataFrame,
    seed: int = 2026,
) -> dict:
    sampled = int(counts.get("sampled_raw", 0))
    valid_canonical = int(counts.get("step3_canonical_nonempty", 0))
    valid_unique_n = int(counts.get("step4_unique_samples", 0))

    if valid_unique.empty:
        exact_novelty_pct = None
        pains_rate_pct = None
        frac_05 = None
        frac_07 = None
        median_similarity = None
    else:
        exact_novelty_pct = 100.0 * float((~valid_unique["exact_training_active"].astype(bool)).mean())
        pains_rate_pct = 100.0 * float(valid_unique["pains"].astype(bool).mean())
        similarities = valid_unique["nearest_training_active_tanimoto"].to_numpy(dtype=float)
        frac_05 = fraction_at_or_above(similarities, 0.5)
        frac_07 = fraction_at_or_above(similarities, 0.7)
        median_similarity = float(np.median(similarities))

    internal_similarity = None
    if len(survivors) >= 2:
        value = sampled_mean_pairwise_tanimoto(
            survivors["standardized_smiles"].tolist(),
            max_pairs=20000,
            seed=int(seed),
        )
        if np.isfinite(value):
            internal_similarity = float(value)

    return {
        "generation_quality": {
            "validity_pct": 100.0 * valid_canonical / sampled if sampled else None,
            "uniqueness_pct_among_valid": (
                100.0 * valid_unique_n / valid_canonical if valid_canonical else None
            ),
            "exact_novelty_pct_vs_training": exact_novelty_pct,
            "survivor_mean_pairwise_tanimoto": internal_similarity,
        },
        "chemistry": {
            "pains_rate_pct_among_valid_unique": pains_rate_pct,
        },
        "novelty_vs_training_actives": {
            "fraction_tanimoto_ge_0_5": frac_05,
            "fraction_tanimoto_ge_0_7": frac_07,
            "median_nearest_tanimoto": median_similarity,
            "n_property_filtered_tanimoto_lt_0_5": int(
                counts.get("step9_tanimoto_lt_cutoff", 0)
            ),
        },
    }
