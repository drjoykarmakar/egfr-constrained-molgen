"""Run the fixed chemistry rejection funnel on a saved generator sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import sha256_file
from src.filters import FilterConfig, run_filter_funnel
from src.evaluate import generation_metrics
from src.train import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _load_smiles_csv(path: Path, column: str) -> list[str]:
    frame = pd.read_csv(path, keep_default_na=False)
    if column not in frame.columns:
        raise ValueError(f"{path} is missing required column {column!r}.")
    return [str(value) for value in frame[column].tolist()]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    scfg = config["sampling"]
    sample_path = resolve(root, scfg["demo_output_csv"] if args.demo else scfg["output_csv"])
    if not sample_path.exists():
        raise FileNotFoundError(f"Generated sample file not found: {sample_path}")

    td = config["training_data"]
    training_path = resolve(root, td["output_smiles"])
    benchmark_path = resolve(root, td["benchmark_smiles"])
    if args.demo and (not training_path.exists() or not benchmark_path.exists()):
        # Demo mode is a code smoke test, so it can use a tiny chemistry reference
        # when Project 2 has not been frozen locally.
        training_active = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
        benchmark = set(training_active)
        reference_note = "demo fallback reference; not a reported EGFR result"
    else:
        if not training_path.exists() or not benchmark_path.exists():
            raise FileNotFoundError("Frozen Project 2 training/benchmark files are missing. Run prepare_data.py.")
        training_active = _load_smiles_csv(training_path, "standardized_smiles")
        benchmark = set(_load_smiles_csv(benchmark_path, "standardized_smiles"))
        reference_note = "Project 2 scaffold-training actives and full benchmark"

    raw_smiles = _load_smiles_csv(sample_path, "sampled_smiles")
    fcfg = config["filters"]
    ncfg = fcfg["nearest_training_active"]
    filter_config = FilterConfig(
        mw_min=float(fcfg["mw_min"]),
        mw_max=float(fcfg["mw_max"]),
        clogp_min=float(fcfg["clogp_min"]),
        clogp_max=float(fcfg["clogp_max"]),
        qed_min=float(fcfg["qed_min"]),
        pains_action=str(fcfg["pains_action"]),
        fingerprint_radius=int(ncfg["radius"]),
        fingerprint_bits=int(ncfg["n_bits"]),
        use_chirality=bool(ncfg.get("use_chirality", True)),
        interesting_max_tanimoto=float(ncfg["interesting_max_tanimoto"]),
    )
    result = run_filter_funnel(
        raw_smiles,
        project2_benchmark_smiles=benchmark,
        training_active_smiles=training_active,
        config=filter_config,
        scorer=None,
    )

    tables_dir = resolve(root, config["paths"]["results_tables"])
    lists_dir = resolve(root, config["paths"]["results_lists"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    lists_dir.mkdir(parents=True, exist_ok=True)
    prefix = "demo_" if args.demo else ""

    counts_path = tables_dir / f"{prefix}funnel_counts.csv"
    metrics_path = tables_dir / f"{prefix}generation_metrics.json"
    survivors_path = lists_dir / f"{prefix}survivors.csv"
    valid_unique_path = lists_dir / f"{prefix}valid_unique_diagnostics.csv"
    result.counts_frame().to_csv(counts_path, index=False)
    result.survivors.to_csv(survivors_path, index=False)
    result.valid_unique.to_csv(valid_unique_path, index=False)

    counts = result.counts
    summary_metrics = generation_metrics(
        counts=counts,
        valid_unique=result.valid_unique,
        survivors=result.survivors,
        seed=int(config["project"]["seed"]),
    )

    metrics = {
        "demo_mode": bool(args.demo),
        "reference_note": reference_note,
        "sample_file": str(sample_path),
        "sample_file_sha256": sha256_file(sample_path),
        "n_training_active_references": len(training_active),
        **summary_metrics,
        "funnel_counts": {key: int(value) for key, value in counts.items()},
        "outputs": {
            "funnel_counts_csv": str(counts_path),
            "survivors_csv": str(survivors_path),
            "valid_unique_diagnostics_csv": str(valid_unique_path),
        },
    }
    write_json(metrics_path, metrics)
    print(f"Raw samples: {counts['sampled_raw']}")
    print(f"Valid canonical: {counts['step3_canonical_nonempty']}")
    print(f"Unique: {counts['step4_unique_samples']}")
    print(f"Property pass: {counts['step6_property_pass']}")
    print(f"Strict Tanimoto < {filter_config.interesting_max_tanimoto}: {counts['step9_tanimoto_lt_cutoff']}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
