"""Project 2 provenance import and shared molecule standardization.

This module intentionally reuses the same chemical standardization order as
``drjoykarmakar/egfr-virtual-screen``:

1. RDKit SMILES parse + sanitization.
2. RDKit MolStandardize cleanup and parent-fragment selection.
3. Charge neutralization with ``Uncharger``.
4. Canonical isomeric SMILES.

Project 3 does not silently query ChEMBL. The generation source is frozen from
Project 2's processed classification table and scaffold split assignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, QED
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class BasicProperties:
    mw: float
    clogp: float
    qed: float


@dataclass(frozen=True)
class FrozenSourceSummary:
    n_benchmark_smiles: int
    n_training_actives: int
    n_heldout_test_actives: int
    chembl_release: str
    chembl_release_date: str | None
    training_file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_and_sanitize_smiles(smiles: Any) -> Chem.Mol | None:
    """Parse a SMILES string with RDKit sanitization, returning ``None`` on failure."""

    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text, sanitize=True)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        # Explicit call makes the funnel's first step auditable even though
        # MolFromSmiles(sanitize=True) already sanitizes by default.
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def parent_fragment_and_neutralize(mol: Chem.Mol) -> Chem.Mol | None:
    """Apply Project 2 parent-fragment selection and simple neutralization."""

    try:
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(cleaned)
        if parent is None or parent.GetNumAtoms() == 0:
            return None
        neutral = rdMolStandardize.Uncharger().uncharge(parent)
        if neutral is None or neutral.GetNumAtoms() == 0:
            return None
        return neutral
    except Exception:
        return None


def canonicalize_checked(mol: Chem.Mol) -> tuple[str, Chem.Mol] | None:
    """Write canonical isomeric SMILES and reparse it as a final integrity check."""

    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True).strip()
    except Exception:
        return None
    if not smiles:
        return None
    parsed = parse_and_sanitize_smiles(smiles)
    if parsed is None:
        return None
    return smiles, parsed


def standardize_smiles(smiles: Any) -> str | None:
    """Convenience wrapper for the Project 2 standardization sequence."""

    mol = parse_and_sanitize_smiles(smiles)
    if mol is None:
        return None
    mol = parent_fragment_and_neutralize(mol)
    if mol is None:
        return None
    canonical = canonicalize_checked(mol)
    return None if canonical is None else canonical[0]


def compute_basic_properties(mol: Chem.Mol) -> BasicProperties:
    """Compute the three properties used by the hard Project 3 property gate."""

    return BasicProperties(
        mw=float(Descriptors.MolWt(mol)),
        clogp=float(Crippen.MolLogP(mol)),
        qed=float(QED.qed(mol)),
    )


def _build_pains_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


_PAINS_CATALOG = _build_pains_catalog()


def pains_flag(mol: Chem.Mol) -> tuple[bool, str | None]:
    """Return the first RDKit PAINS alert, if any; this is a flag, not a verdict."""

    match = _PAINS_CATALOG.GetFirstMatch(mol)
    if match is None:
        return False, None
    return True, str(match.GetDescription())


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def freeze_project2_generation_source(
    *,
    project2_root: str | Path,
    processed_relpath: str,
    split_relpath: str,
    chembl_manifest_relpath: str,
    output_training_path: str | Path,
    output_test_path: str | Path,
    output_benchmark_path: str | Path,
    output_manifest_path: str | Path,
    project2_repo_url: str,
    project2_release_tag: str,
    expected_chembl_release: str = "ChEMBL_37",
    active_label: int = 1,
    active_pactivity_cutoff: float = 6.0,
) -> FrozenSourceSummary:
    """Freeze Project 2 scaffold-training actives for generator training.

    The held-out scaffold-test actives are written separately for evaluation
    only. The complete Project 2 benchmark SMILES are frozen for the exact-match
    rejection step. No network access or new ChEMBL query occurs here.
    """

    p2 = Path(project2_root).expanduser().resolve()
    processed_path = p2 / processed_relpath
    split_path = p2 / split_relpath
    chembl_manifest_path = p2 / chembl_manifest_relpath
    for path in (processed_path, split_path, chembl_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Project 2 artifact not found: {path}")

    processed = pd.read_csv(processed_path)
    split = pd.read_csv(split_path)
    _require_columns(processed, {"standardized_smiles", "activity_label"}, "Project 2 table")
    _require_columns(
        split,
        {"standardized_smiles", "activity_label", "split"},
        "Project 2 scaffold assignments",
    )

    if processed["standardized_smiles"].duplicated().any():
        raise ValueError("Project 2 processed table unexpectedly contains duplicate standardized SMILES.")
    if split["standardized_smiles"].duplicated().any():
        raise ValueError("Project 2 scaffold assignment unexpectedly contains duplicate standardized SMILES.")

    merged = processed[["standardized_smiles", "activity_label"]].merge(
        split[["standardized_smiles", "activity_label", "split"]],
        on="standardized_smiles",
        suffixes=("_processed", "_split"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(processed) or len(merged) != len(split):
        raise ValueError("Project 2 processed table and scaffold assignment do not contain the same molecules.")
    if not (merged["activity_label_processed"].astype(int) == merged["activity_label_split"].astype(int)).all():
        raise ValueError("Activity labels disagree between Project 2 source artifacts.")

    merged["activity_label"] = merged["activity_label_processed"].astype(int)
    training = merged.loc[
        (merged["split"].astype(str) == "train") & (merged["activity_label"] == int(active_label)),
        ["standardized_smiles"],
    ].drop_duplicates().sort_values("standardized_smiles").reset_index(drop=True)
    heldout_test = merged.loc[
        (merged["split"].astype(str) == "test") & (merged["activity_label"] == int(active_label)),
        ["standardized_smiles"],
    ].drop_duplicates().sort_values("standardized_smiles").reset_index(drop=True)
    benchmark = merged[["standardized_smiles"]].drop_duplicates().sort_values("standardized_smiles").reset_index(drop=True)

    if training.empty:
        raise ValueError("No scaffold-training actives were found in Project 2 artifacts.")
    if heldout_test.empty:
        raise ValueError("No held-out scaffold-test actives were found in Project 2 artifacts.")

    output_training = Path(output_training_path)
    output_test = Path(output_test_path)
    output_benchmark = Path(output_benchmark_path)
    output_manifest = Path(output_manifest_path)
    for path in (output_training, output_test, output_benchmark, output_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)

    training.to_csv(output_training, index=False)
    heldout_test.to_csv(output_test, index=False)
    benchmark.to_csv(output_benchmark, index=False)

    with chembl_manifest_path.open("r", encoding="utf-8") as handle:
        upstream = json.load(handle)
    observed_release = str(upstream.get("observed_chembl_release", ""))
    if observed_release != expected_chembl_release:
        raise ValueError(
            f"Project 2 ChEMBL release mismatch: expected {expected_chembl_release}, got {observed_release}."
        )

    training_sha = sha256_file(output_training)
    manifest = {
        "source_project": "egfr-virtual-screen",
        "source_repo": project2_repo_url,
        "source_release_tag": project2_release_tag,
        "source_processed_table": str(processed_path),
        "source_processed_table_sha256": sha256_file(processed_path),
        "source_scaffold_assignments": str(split_path),
        "source_scaffold_assignments_sha256": sha256_file(split_path),
        "source_chembl_manifest": str(chembl_manifest_path),
        "chembl_release": observed_release,
        "chembl_release_date": upstream.get("observed_chembl_release_date"),
        "chembl_retrieved_at_utc": upstream.get("retrieved_at_utc"),
        "target_chembl_id": upstream.get("query", {}).get("target_chembl_id", "CHEMBL203"),
        "activity_cutoff_pactivity": float(active_pactivity_cutoff),
        "training_partition": "scaffold train",
        "training_label": int(active_label),
        "n_project2_benchmark_smiles": int(len(benchmark)),
        "n_training_smiles": int(len(training)),
        "n_heldout_scaffold_test_actives": int(len(heldout_test)),
        "training_file": str(output_training),
        "training_file_sha256": training_sha,
        "heldout_test_file": str(output_test),
        "heldout_test_file_sha256": sha256_file(output_test),
        "benchmark_file": str(output_benchmark),
        "benchmark_file_sha256": sha256_file(output_benchmark),
    }
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return FrozenSourceSummary(
        n_benchmark_smiles=len(benchmark),
        n_training_actives=len(training),
        n_heldout_test_actives=len(heldout_test),
        chembl_release=observed_release,
        chembl_release_date=upstream.get("observed_chembl_release_date"),
        training_file_sha256=training_sha,
    )
