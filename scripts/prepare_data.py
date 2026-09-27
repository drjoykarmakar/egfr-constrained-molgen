"""Freeze generator training data from the local Project 2 checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import freeze_project2_generation_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--project2-root",
        default=None,
        help="Override provenance.project2.local_root from the config.",
    )
    return parser.parse_args()


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    p2 = config["provenance"]["project2"]
    td = config["training_data"]
    project2_root = args.project2_root or p2["local_root"]

    summary = freeze_project2_generation_source(
        project2_root=resolve(root, project2_root),
        processed_relpath=p2["processed_table"],
        split_relpath=p2["scaffold_assignments"],
        chembl_manifest_relpath=p2["chembl_manifest"],
        output_training_path=resolve(root, td["output_smiles"]),
        output_test_path=resolve(root, td["heldout_test_actives"]),
        output_benchmark_path=resolve(root, td["benchmark_smiles"]),
        output_manifest_path=resolve(root, td["manifest"]),
        project2_repo_url=p2["repo_url"],
        project2_release_tag=p2["release_tag"],
        expected_chembl_release=p2["expected_chembl_release"],
        active_label=int(td["active_label"]),
        active_pactivity_cutoff=float(p2["expected_activity_cutoff_pactivity"]),
    )
    print(f"Frozen {summary.n_training_actives} scaffold-training actives")
    print(f"Held out {summary.n_heldout_test_actives} scaffold-test actives")
    print(f"Project 2 benchmark SMILES: {summary.n_benchmark_smiles}")
    print(f"Training SHA-256: {summary.training_file_sha256}")


if __name__ == "__main__":
    main()
