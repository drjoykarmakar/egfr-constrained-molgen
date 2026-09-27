"""Fingerprint-based novelty and diversity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.data import parse_and_sanitize_smiles


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


def _generator(config: FingerprintConfig):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(config.radius),
        fpSize=int(config.n_bits),
        includeChirality=bool(config.use_chirality),
    )


def fingerprints(smiles: Iterable[str], config: FingerprintConfig | None = None):
    cfg = config or FingerprintConfig()
    gen = _generator(cfg)
    names: list[str] = []
    fps = []
    for value in smiles:
        text = str(value)
        mol = parse_and_sanitize_smiles(text)
        if mol is None:
            raise ValueError(f"Invalid SMILES in novelty reference/evaluation set: {text!r}")
        names.append(text)
        fps.append(gen.GetFingerprint(mol))
    if not fps:
        raise ValueError("At least one valid SMILES is required.")
    return names, fps


def nearest_reference(
    query_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    *,
    config: FingerprintConfig | None = None,
) -> tuple[list[str], np.ndarray]:
    cfg = config or FingerprintConfig()
    ref_names, ref_fps = fingerprints(reference_smiles, cfg)
    gen = _generator(cfg)
    nearest_names: list[str] = []
    nearest_values: list[float] = []
    for query in query_smiles:
        mol = parse_and_sanitize_smiles(query)
        if mol is None:
            raise ValueError(f"Invalid query SMILES for novelty calculation: {query!r}")
        fp = gen.GetFingerprint(mol)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=float)
        idx = int(np.argmax(sims))
        nearest_names.append(ref_names[idx])
        nearest_values.append(float(sims[idx]))
    return nearest_names, np.asarray(nearest_values, dtype=float)


def fraction_at_or_above(values: Sequence[float] | np.ndarray, threshold: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr >= float(threshold)))


def sampled_mean_pairwise_tanimoto(
    smiles: Sequence[str],
    *,
    max_pairs: int = 20000,
    seed: int = 2026,
    config: FingerprintConfig | None = None,
) -> float:
    """Estimate internal similarity without an O(n^2) materialized matrix."""

    values = list(smiles)
    if len(values) < 2:
        return float("nan")
    _, fps = fingerprints(values, config or FingerprintConfig())
    n = len(fps)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= int(max_pairs):
        sims: list[float] = []
        for i in range(n - 1):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
        return float(np.mean(sims)) if sims else float("nan")

    rng = np.random.default_rng(int(seed))
    sims = []
    while len(sims) < int(max_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        sims.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    return float(np.mean(sims))
