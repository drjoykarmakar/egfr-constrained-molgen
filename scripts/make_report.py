"""Generate the required funnel, novelty, property, and survivor figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import compute_basic_properties, parse_and_sanitize_smiles
from src.plots import (
    plot_funnel,
    plot_nearest_training_similarity,
    plot_property_distributions,
    plot_survivor_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _training_properties(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "standardized_smiles" not in frame.columns:
        raise ValueError("Training active file is missing standardized_smiles.")
    rows = []
    for smiles in frame["standardized_smiles"]:
        mol = parse_and_sanitize_smiles(smiles)
        if mol is None:
            raise ValueError(f"Frozen training active is invalid: {smiles!r}")
        props = compute_basic_properties(mol)
        rows.append({"standardized_smiles": smiles, "mw": props.mw, "clogp": props.clogp, "qed": props.qed})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tables_dir = resolve(root, config["paths"]["results_tables"])
    figures_dir = resolve(root, config["paths"]["results_figures"])
    lists_dir = resolve(root, config["paths"]["results_lists"])
    prefix = "demo_" if args.demo else ""

    counts_path = tables_dir / f"{prefix}funnel_counts.csv"
    diagnostics_path = lists_dir / f"{prefix}valid_unique_diagnostics.csv"
    survivors_path = lists_dir / f"{prefix}survivors.csv"
    required = [counts_path, diagnostics_path, survivors_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Filter outputs are incomplete. Run scripts/run_filters.py first. Missing: " + ", ".join(missing))

    counts = pd.read_csv(counts_path)
    diagnostics = pd.read_csv(diagnostics_path)
    survivors = pd.read_csv(survivors_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = "demo_" if args.demo else ""
    plot_funnel(counts, figures_dir / f"{suffix}funnel.png")
    plot_nearest_training_similarity(diagnostics, figures_dir / f"{suffix}nearest_training_active_tanimoto.png")
    plot_survivor_grid(survivors, figures_dir / f"{suffix}survivor_grid.png")

    training_path = resolve(root, config["training_data"]["output_smiles"])
    if training_path.exists():
        training_props = _training_properties(training_path)
        plot_property_distributions(
            survivors,
            training_props,
            figures_dir / f"{suffix}property_distributions.png",
        )
    elif not args.demo:
        raise FileNotFoundError(f"Frozen training active file is missing: {training_path}")
    else:
        print("Demo mode: skipped property comparison because frozen Project 2 training data is absent.")

    print(f"Wrote report figures under {figures_dir}")


if __name__ == "__main__":
    main()
