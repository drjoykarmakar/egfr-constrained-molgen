"""Project 3 vs Project 2 comparison utilities.

The comparison is descriptive. It does not re-rank or retune the generator.
Generated survivors are compared with Project 2's novelty-aware top 20,
property-filtered top 50, and a deterministic random sample from the frozen
external screening library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from src.data import parse_and_sanitize_smiles
from src.filters import compute_basic_properties, parent_fragment_and_neutralize, canonicalize_checked
from src.novelty import FingerprintConfig, nearest_reference, sampled_mean_pairwise_tanimoto


@dataclass(frozen=True)
class ComparisonConfig:
    random_library_sample_size: int = 1000
    seed: int = 2026
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True

    @property
    def fingerprint(self) -> FingerprintConfig:
        return FingerprintConfig(
            radius=self.radius,
            n_bits=self.n_bits,
            use_chirality=self.use_chirality,
        )


def read_smi(path: str | Path) -> pd.DataFrame:
    """Read a whitespace/tab-delimited SMI file with optional comment header."""

    rows: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        rows.append(
            {
                "raw_smiles": fields[0],
                "library_id": fields[1] if len(fields) > 1 else "",
            }
        )
    return pd.DataFrame.from_records(rows)


def standardize_smiles(smiles: str) -> str | None:
    mol = parse_and_sanitize_smiles(str(smiles))
    if mol is None:
        return None
    mol = parent_fragment_and_neutralize(mol)
    if mol is None:
        return None
    checked = canonicalize_checked(mol)
    return None if checked is None else checked[0]


def prepare_random_library_sample(
    library_path: str | Path,
    *,
    benchmark_smiles: Iterable[str],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Build a deterministic target-agnostic random comparator.

    The source is the frozen Project 2 external library. Structures are
    standardized with the Project 3 cleaning rules, exact Project 2 benchmark
    overlaps are removed, duplicates are collapsed, then rows are sampled with
    a fixed seed. No EGFR score or activity is used to choose the sample.
    """

    raw = read_smi(library_path)
    if raw.empty:
        raise ValueError("Project 2 external library is empty.")
    benchmark = {str(x) for x in benchmark_smiles}
    records: list[dict] = []
    for _, row in raw.iterrows():
        standardized = standardize_smiles(row["raw_smiles"])
        if not standardized or standardized in benchmark:
            continue
        mol = Chem.MolFromSmiles(standardized)
        if mol is None:
            continue
        props = compute_basic_properties(mol)
        records.append(
            {
                "library_id": row["library_id"],
                "standardized_smiles": standardized,
                "mw": props.mw,
                "clogp": props.clogp,
                "qed": props.qed,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame = frame.drop_duplicates("standardized_smiles", keep="first").reset_index(drop=True)
    if len(frame) < int(sample_size):
        raise ValueError(
            f"Only {len(frame)} standardized non-benchmark library molecules are available; "
            f"requested {sample_size}."
        )
    return frame.sample(n=int(sample_size), random_state=int(seed), replace=False).reset_index(drop=True)


def ensure_basic_properties(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure standardized SMILES + MW/cLogP/QED are available."""

    if "standardized_smiles" not in frame.columns:
        raise KeyError("Comparison frame requires 'standardized_smiles'.")
    out = frame.copy()
    missing_props = [name for name in ("mw", "clogp", "qed") if name not in out.columns]
    if not missing_props:
        return out
    records = []
    for smiles in out["standardized_smiles"].astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid standardized SMILES in comparison set: {smiles!r}")
        props = compute_basic_properties(mol)
        records.append({"mw": props.mw, "clogp": props.clogp, "qed": props.qed})
    props_frame = pd.DataFrame.from_records(records)
    for name in missing_props:
        out[name] = props_frame[name].to_numpy()
    return out


def add_training_novelty(
    frame: pd.DataFrame,
    training_active_smiles: Sequence[str],
    *,
    fingerprint: FingerprintConfig,
) -> pd.DataFrame:
    out = ensure_basic_properties(frame)
    if out.empty:
        out["nearest_training_active_smiles"] = pd.Series(dtype=str)
        out["nearest_training_active_tanimoto"] = pd.Series(dtype=float)
        return out
    names, sims = nearest_reference(
        out["standardized_smiles"].astype(str).tolist(),
        list(training_active_smiles),
        config=fingerprint,
    )
    out["nearest_training_active_smiles"] = names
    out["nearest_training_active_tanimoto"] = sims
    return out


def summarize_group(
    name: str,
    frame: pd.DataFrame,
    *,
    seed: int,
    fingerprint: FingerprintConfig,
) -> dict[str, float | int | str | None]:
    if frame.empty:
        return {"group": name, "n": 0}
    values = frame["nearest_training_active_tanimoto"].astype(float)
    internal = None
    if len(frame) >= 2:
        internal = sampled_mean_pairwise_tanimoto(
            frame["standardized_smiles"].astype(str).tolist(),
            seed=seed,
            config=fingerprint,
        )
    return {
        "group": name,
        "n": int(len(frame)),
        "mw_median": float(frame["mw"].median()),
        "clogp_median": float(frame["clogp"].median()),
        "qed_median": float(frame["qed"].median()),
        "nearest_training_tanimoto_median": float(values.median()),
        "fraction_training_tanimoto_lt_0_5": float((values < 0.5).mean()),
        "fraction_training_tanimoto_ge_0_7": float((values >= 0.7).mean()),
        "mean_pairwise_tanimoto": None if internal is None or not np.isfinite(internal) else float(internal),
    }


def cross_set_summary(
    query_name: str,
    query: pd.DataFrame,
    reference_name: str,
    reference: pd.DataFrame,
    *,
    fingerprint: FingerprintConfig,
) -> dict[str, float | int | str | None]:
    if query.empty or reference.empty:
        return {
            "query_group": query_name,
            "reference_group": reference_name,
            "n_query": int(len(query)),
            "n_reference": int(len(reference)),
        }
    q = query["standardized_smiles"].astype(str).tolist()
    r = reference["standardized_smiles"].astype(str).tolist()
    _, sims = nearest_reference(q, r, config=fingerprint)
    exact_overlap = len(set(q) & set(r))
    return {
        "query_group": query_name,
        "reference_group": reference_name,
        "n_query": int(len(query)),
        "n_reference": int(len(reference)),
        "exact_smiles_overlap": int(exact_overlap),
        "nearest_cross_set_tanimoto_median": float(np.median(sims)),
        "fraction_cross_set_tanimoto_ge_0_5": float(np.mean(sims >= 0.5)),
        "fraction_cross_set_tanimoto_ge_0_7": float(np.mean(sims >= 0.7)),
        "max_cross_set_tanimoto": float(np.max(sims)),
    }
