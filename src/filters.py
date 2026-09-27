"""Ordered chemistry rejection funnel for generated SMILES.

The mutation order is fixed and mirrors the project specification:

1. RDKit parse + sanitization.
2. Parent fragment / salt strip / neutralization.
3. Canonical SMILES integrity check; drop empty/broken structures.
4. Exact-SMILES uniqueness among generated samples.
5. Remove exact match to any Project 2 EGFR benchmark SMILES.
6. MW / cLogP / QED property gates.
7. Record PAINS; default policy is flag-only.
8. Nearest scaffold-training-active Morgan radius-2 Tanimoto.
9. Keep molecules with Tanimoto strictly < 0.5.
10. Optionally add a frozen Project 2 activity-model score.

The funnel does not relax thresholds after seeing the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.data import (
    canonicalize_checked,
    compute_basic_properties,
    pains_flag,
    parent_fragment_and_neutralize,
    parse_and_sanitize_smiles,
)


ScoreFunction = Callable[[Sequence[str]], Sequence[float] | np.ndarray]


@dataclass(frozen=True)
class FilterConfig:
    mw_min: float = 200.0
    mw_max: float = 600.0
    clogp_min: float = -1.0
    clogp_max: float = 5.0
    qed_min: float = 0.40
    pains_action: str = "flag_only"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    use_chirality: bool = True
    interesting_max_tanimoto: float = 0.50

    def __post_init__(self) -> None:
        if self.pains_action not in {"flag_only", "drop"}:
            raise ValueError("pains_action must be 'flag_only' or 'drop'.")
        if not 0.0 <= self.interesting_max_tanimoto <= 1.0:
            raise ValueError("interesting_max_tanimoto must be between 0 and 1.")


@dataclass
class FunnelResult:
    counts: dict[str, int]
    valid_unique: pd.DataFrame
    property_pass: pd.DataFrame
    survivors: pd.DataFrame

    def counts_frame(self) -> pd.DataFrame:
        ordered = [
            "sampled_raw",
            "step1_valid_parse_sanitize",
            "step2_parent_fragment_neutralized",
            "step3_canonical_nonempty",
            "step4_unique_samples",
            "step5_not_exact_project2_benchmark",
            "step6_property_pass",
            "step7_after_pains_policy",
            "step8_similarity_scored",
            "step9_tanimoto_lt_cutoff",
            "step10_optional_activity_scored",
        ]
        return pd.DataFrame(
            [{"stage": stage, "count": int(self.counts.get(stage, 0))} for stage in ordered]
        )


def passes_property_gates(
    *,
    mw: float,
    clogp: float,
    qed: float,
    config: FilterConfig,
) -> bool:
    """Return whether a molecule passes the fixed Project 3 property window."""

    return (
        config.mw_min <= float(mw) <= config.mw_max
        and config.clogp_min <= float(clogp) <= config.clogp_max
        and float(qed) >= config.qed_min
    )


def is_interesting_similarity(value: float, cutoff: float = 0.50) -> bool:
    """The novelty criterion is strict: equality to the cutoff does not survive."""

    return float(value) < float(cutoff)


def _morgan_generator(config: FilterConfig):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(config.fingerprint_radius),
        fpSize=int(config.fingerprint_bits),
        includeChirality=bool(config.use_chirality),
    )


def _training_fingerprints(
    training_active_smiles: Iterable[str],
    config: FilterConfig,
) -> tuple[list[str], list[DataStructs.ExplicitBitVect]]:
    generator = _morgan_generator(config)
    names: list[str] = []
    fps: list[DataStructs.ExplicitBitVect] = []
    for smiles in training_active_smiles:
        mol = parse_and_sanitize_smiles(smiles)
        if mol is None:
            raise ValueError(f"Training-active reference contains invalid SMILES: {smiles!r}")
        names.append(str(smiles))
        fps.append(generator.GetFingerprint(mol))
    if not fps:
        raise ValueError("At least one training-active SMILES is required for novelty filtering.")
    return names, fps


def _nearest_training_active(
    smiles: str,
    training_names: list[str],
    training_fps: list[DataStructs.ExplicitBitVect],
    config: FilterConfig,
) -> tuple[str, float]:
    mol = parse_and_sanitize_smiles(smiles)
    if mol is None:
        raise ValueError(f"Internal error: standardized candidate is invalid: {smiles!r}")
    fp = _morgan_generator(config).GetFingerprint(mol)
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, training_fps), dtype=float)
    index = int(np.argmax(sims))
    return training_names[index], float(sims[index])


def run_filter_funnel(
    raw_smiles: Sequence[str],
    *,
    project2_benchmark_smiles: Iterable[str],
    training_active_smiles: Iterable[str],
    config: FilterConfig | None = None,
    scorer: ScoreFunction | None = None,
) -> FunnelResult:
    """Apply the fixed rejection funnel and return stage counts plus survivors."""

    cfg = config or FilterConfig()
    counts: dict[str, int] = {"sampled_raw": int(len(raw_smiles))}

    # 1. RDKit parse + sanitization.
    parsed_records: list[dict] = []
    for sample_id, raw in enumerate(raw_smiles):
        mol = parse_and_sanitize_smiles(raw)
        if mol is not None:
            parsed_records.append({"sample_id": sample_id, "raw_smiles": str(raw), "_mol": mol})
    counts["step1_valid_parse_sanitize"] = len(parsed_records)

    # 2. Parent fragment / salt strip / neutralization.
    standardized_records: list[dict] = []
    for record in parsed_records:
        mol = parent_fragment_and_neutralize(record["_mol"])
        if mol is not None:
            standardized_records.append({**record, "_standardized_mol": mol})
    counts["step2_parent_fragment_neutralized"] = len(standardized_records)

    # 3. Canonical SMILES and final integrity check.
    canonical_records: list[dict] = []
    for record in standardized_records:
        canonical = canonicalize_checked(record["_standardized_mol"])
        if canonical is None:
            continue
        smiles, mol = canonical
        if not smiles:
            continue
        canonical_records.append(
            {
                "sample_id": record["sample_id"],
                "raw_smiles": record["raw_smiles"],
                "standardized_smiles": smiles,
                "_mol": mol,
            }
        )
    counts["step3_canonical_nonempty"] = len(canonical_records)

    canonical_frame = pd.DataFrame.from_records(canonical_records)
    if canonical_frame.empty:
        empty = pd.DataFrame(
            columns=[
                "sample_id",
                "raw_smiles",
                "standardized_smiles",
                "mw",
                "clogp",
                "qed",
                "pains",
                "pains_description",
                "exact_project2_benchmark",
                "exact_training_active",
                "nearest_training_active_smiles",
                "nearest_training_active_tanimoto",
            ]
        )
        for key in [
            "step4_unique_samples",
            "step5_not_exact_project2_benchmark",
            "step6_property_pass",
            "step7_after_pains_policy",
            "step8_similarity_scored",
            "step9_tanimoto_lt_cutoff",
            "step10_optional_activity_scored",
            "pains_flagged_at_step7",
        ]:
            counts[key] = 0
        return FunnelResult(counts=counts, valid_unique=empty, property_pass=empty, survivors=empty)

    # 4. Exact-SMILES uniqueness among samples. Keep first occurrence deterministically.
    unique = canonical_frame.drop_duplicates(subset=["standardized_smiles"], keep="first").reset_index(drop=True)
    counts["step4_unique_samples"] = len(unique)

    # Diagnostic-only annotations for the valid unique population. These do not
    # alter membership or the fixed rejection order below. They exist because
    # the report must quantify exact novelty, PAINS rate, property distributions,
    # and nearest-training-active similarity among valid unique samples.
    benchmark = {str(x) for x in project2_benchmark_smiles}
    training_values = [str(x) for x in training_active_smiles]
    training_set = set(training_values)
    training_names, training_fps = _training_fingerprints(training_values, cfg)
    diagnostic_rows: list[dict] = []
    for _, row in unique.iterrows():
        mol = row["_mol"]
        props = compute_basic_properties(mol)
        pains, pains_description = pains_flag(mol)
        nearest_name, nearest_similarity = _nearest_training_active(
            row["standardized_smiles"], training_names, training_fps, cfg
        )
        diagnostic_rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "raw_smiles": row["raw_smiles"],
                "standardized_smiles": row["standardized_smiles"],
                "mw": props.mw,
                "clogp": props.clogp,
                "qed": props.qed,
                "pains": bool(pains),
                "pains_description": pains_description,
                "exact_project2_benchmark": row["standardized_smiles"] in benchmark,
                "exact_training_active": row["standardized_smiles"] in training_set,
                "nearest_training_active_smiles": nearest_name,
                "nearest_training_active_tanimoto": nearest_similarity,
            }
        )
    valid_unique = pd.DataFrame.from_records(diagnostic_rows)

    # 5. Drop exact Project 2 benchmark matches.
    nonbenchmark = unique.loc[~unique["standardized_smiles"].isin(benchmark)].copy().reset_index(drop=True)
    counts["step5_not_exact_project2_benchmark"] = len(nonbenchmark)

    # 6. Fixed property gates.
    property_rows: list[dict] = []
    for _, row in nonbenchmark.iterrows():
        mol = row["_mol"]
        props = compute_basic_properties(mol)
        passes = passes_property_gates(mw=props.mw, clogp=props.clogp, qed=props.qed, config=cfg)
        if passes:
            property_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "raw_smiles": row["raw_smiles"],
                    "standardized_smiles": row["standardized_smiles"],
                    "mw": props.mw,
                    "clogp": props.clogp,
                    "qed": props.qed,
                    "_mol": mol,
                }
            )
    property_frame = pd.DataFrame.from_records(property_rows)
    counts["step6_property_pass"] = len(property_frame)

    if property_frame.empty:
        empty = property_frame.drop(columns=["_mol"], errors="ignore")
        for key in [
            "step7_after_pains_policy",
            "step8_similarity_scored",
            "step9_tanimoto_lt_cutoff",
            "step10_optional_activity_scored",
            "pains_flagged_at_step7",
        ]:
            counts[key] = 0
        return FunnelResult(counts=counts, valid_unique=valid_unique, property_pass=empty, survivors=empty)

    # 7. Record PAINS and optionally apply an explicit drop policy.
    pains_values: list[bool] = []
    pains_descriptions: list[str | None] = []
    for mol in property_frame["_mol"]:
        flag, description = pains_flag(mol)
        pains_values.append(flag)
        pains_descriptions.append(description)
    property_frame["pains"] = pains_values
    property_frame["pains_description"] = pains_descriptions
    counts["pains_flagged_at_step7"] = int(np.sum(pains_values))
    if cfg.pains_action == "drop":
        after_pains = property_frame.loc[~property_frame["pains"]].copy().reset_index(drop=True)
    else:
        after_pains = property_frame.copy().reset_index(drop=True)
    counts["step7_after_pains_policy"] = len(after_pains)

    property_pass_public = property_frame.drop(columns=["_mol"], errors="ignore").copy()
    if after_pains.empty:
        counts["step8_similarity_scored"] = 0
        counts["step9_tanimoto_lt_cutoff"] = 0
        counts["step10_optional_activity_scored"] = 0
        return FunnelResult(
            counts=counts,
            valid_unique=valid_unique,
            property_pass=property_pass_public,
            survivors=after_pains.drop(columns=["_mol"], errors="ignore"),
        )

    # 8. Nearest scaffold-training-active Tanimoto. Reuse the diagnostic
    # similarities computed for valid unique molecules; no threshold has been
    # applied to them yet.
    nearest_lookup = valid_unique.set_index("standardized_smiles")[
        ["nearest_training_active_smiles", "nearest_training_active_tanimoto"]
    ]
    after_pains = after_pains.join(
        nearest_lookup, on="standardized_smiles", how="left", validate="many_to_one"
    )
    if after_pains["nearest_training_active_tanimoto"].isna().any():
        raise RuntimeError("Internal error: missing valid-unique similarity diagnostics.")
    counts["step8_similarity_scored"] = len(after_pains)

    # 9. Strict novelty cut; equality to 0.5 is rejected at the default threshold.
    survivors = after_pains.loc[
        after_pains["nearest_training_active_tanimoto"].map(
            lambda value: is_interesting_similarity(value, cfg.interesting_max_tanimoto)
        )
    ].copy().reset_index(drop=True)
    counts["step9_tanimoto_lt_cutoff"] = len(survivors)

    # 10. Optional frozen Project 2 model score. This never changes membership.
    if scorer is not None and not survivors.empty:
        scores = np.asarray(list(scorer(survivors["standardized_smiles"].tolist())), dtype=float)
        if len(scores) != len(survivors):
            raise ValueError("Optional scorer returned a different number of scores than input SMILES.")
        survivors["project2_active_probability"] = scores
        counts["step10_optional_activity_scored"] = len(survivors)
    else:
        counts["step10_optional_activity_scored"] = 0

    return FunnelResult(
        counts=counts,
        valid_unique=valid_unique,
        property_pass=property_pass_public,
        survivors=survivors.drop(columns=["_mol"], errors="ignore"),
    )
