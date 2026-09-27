"""Compare frozen Project 3 survivors with Project 2 screening references."""

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

from src.compare import (
    ComparisonConfig,
    add_training_novelty,
    cross_set_summary,
    prepare_random_library_sample,
    summarize_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def read_csv_required(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required comparison file not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    training_path = resolve(root, config["training_data"]["output_smiles"])
    benchmark_path = resolve(root, config["training_data"]["benchmark_smiles"])
    survivors_path = resolve(root, config["paths"]["results_lists"]) / "survivors.csv"
    training = read_csv_required(training_path, {"standardized_smiles"})
    benchmark = read_csv_required(benchmark_path, {"standardized_smiles"})
    survivors = read_csv_required(survivors_path, {"standardized_smiles"})

    ccfg = config["comparison"]
    fcfg = config["filters"]["nearest_training_active"]
    comparison_cfg = ComparisonConfig(
        random_library_sample_size=int(ccfg.get("random_library_sample_size", 1000)),
        seed=int(ccfg.get("seed", config["project"]["seed"])),
        radius=int(fcfg["radius"]),
        n_bits=int(fcfg["n_bits"]),
        use_chirality=bool(fcfg.get("use_chirality", True)),
    )

    p2_top20 = read_csv_required(
        resolve(root, ccfg["project2_interesting_top20"]), {"standardized_smiles"}
    )
    p2_top50 = read_csv_required(
        resolve(root, ccfg["project2_filtered_top50"]), {"standardized_smiles"}
    )
    random_library = prepare_random_library_sample(
        resolve(root, ccfg["project2_external_library"]),
        benchmark_smiles=benchmark["standardized_smiles"].astype(str),
        sample_size=comparison_cfg.random_library_sample_size,
        seed=comparison_cfg.seed,
    )

    training_smiles = training["standardized_smiles"].astype(str).tolist()
    fp_cfg = comparison_cfg.fingerprint
    groups = {
        "generated_survivors": add_training_novelty(survivors, training_smiles, fingerprint=fp_cfg),
        "project2_interesting_top20": add_training_novelty(p2_top20, training_smiles, fingerprint=fp_cfg),
        "project2_filtered_top50": add_training_novelty(p2_top50, training_smiles, fingerprint=fp_cfg),
        "project2_random_library": add_training_novelty(random_library, training_smiles, fingerprint=fp_cfg),
    }

    summaries = [
        summarize_group(name, frame, seed=comparison_cfg.seed, fingerprint=fp_cfg)
        for name, frame in groups.items()
    ]
    generated = groups["generated_survivors"]
    cross = [
        cross_set_summary(
            "generated_survivors",
            generated,
            name,
            frame,
            fingerprint=fp_cfg,
        )
        for name, frame in groups.items()
        if name != "generated_survivors"
    ]

    tables_dir = resolve(root, config["paths"]["results_tables"])
    lists_dir = resolve(root, config["paths"]["results_lists"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    lists_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tables_dir / "project2_comparison.csv"
    cross_path = tables_dir / "project2_cross_set_similarity.csv"
    json_path = tables_dir / "project2_comparison.json"
    random_path = lists_dir / "project2_random_library_comparator.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(cross).to_csv(cross_path, index=False)
    groups["project2_random_library"].to_csv(random_path, index=False)

    payload = {
        "selection_note": (
            "Project 3 generated survivors are frozen before this comparison. "
            "The Project 2 random comparator is sampled deterministically from the frozen external library "
            "after Project 2 benchmark exact-match removal; EGFR activity/model scores are not used to choose it."
        ),
        "group_summary": summaries,
        "generated_vs_project2": cross,
        "inputs": {
            "generated_survivors": str(survivors_path),
            "project2_interesting_top20": str(resolve(root, ccfg["project2_interesting_top20"])),
            "project2_filtered_top50": str(resolve(root, ccfg["project2_filtered_top50"])),
            "project2_external_library": str(resolve(root, ccfg["project2_external_library"])),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nGenerated survivors vs Project 2 reference sets:")
    print(pd.DataFrame(cross).to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {cross_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
