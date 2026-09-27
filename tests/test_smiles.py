import json
from pathlib import Path

import pandas as pd

from src.data import (
    freeze_project2_generation_source,
    parse_and_sanitize_smiles,
    sha256_file,
    standardize_smiles,
)


def test_invalid_smiles_returns_none():
    assert parse_and_sanitize_smiles("not a smiles") is None
    assert standardize_smiles("") is None


def test_standardization_strips_salt_and_neutralizes():
    # Sodium acetate -> acetic acid under the same parent/Uncharger logic used in Project 2.
    assert standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"


def test_freeze_project2_source_uses_scaffold_train_actives_only(tmp_path: Path):
    p2 = tmp_path / "egfr-virtual-screen"
    (p2 / "data/processed").mkdir(parents=True)
    (p2 / "results/tables").mkdir(parents=True)
    (p2 / "data/raw").mkdir(parents=True)

    processed = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
            "activity_label": [1, 0, 1, 1],
        }
    )
    split = pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
            "activity_label": [1, 0, 1, 1],
            "split": ["train", "train", "test", "validation"],
        }
    )
    processed.to_csv(p2 / "data/processed/egfr_activity_classification.csv", index=False)
    split.to_csv(p2 / "results/tables/scaffold_split_assignments.csv", index=False)
    with (p2 / "data/raw/egfr_chembl37_activities.manifest.json").open("w") as handle:
        json.dump(
            {
                "observed_chembl_release": "ChEMBL_37",
                "observed_chembl_release_date": "2026-05-01",
                "retrieved_at_utc": "2026-09-27T00:09:22+00:00",
                "query": {"target_chembl_id": "CHEMBL203"},
            },
            handle,
        )

    out = tmp_path / "out"
    summary = freeze_project2_generation_source(
        project2_root=p2,
        processed_relpath="data/processed/egfr_activity_classification.csv",
        split_relpath="results/tables/scaffold_split_assignments.csv",
        chembl_manifest_relpath="data/raw/egfr_chembl37_activities.manifest.json",
        output_training_path=out / "training.csv",
        output_test_path=out / "test.csv",
        output_benchmark_path=out / "benchmark.csv",
        output_manifest_path=out / "manifest.json",
        project2_repo_url="https://github.com/drjoykarmakar/egfr-virtual-screen",
        project2_release_tag="v1.0.0",
    )

    training = pd.read_csv(out / "training.csv")
    test = pd.read_csv(out / "test.csv")
    assert training["standardized_smiles"].tolist() == ["CCO"]
    assert test["standardized_smiles"].tolist() == ["c1ccccc1"]
    assert summary.n_training_actives == 1
    assert summary.n_heldout_test_actives == 1

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source_release_tag"] == "v1.0.0"
    assert manifest["training_file_sha256"] == sha256_file(out / "training.csv")
