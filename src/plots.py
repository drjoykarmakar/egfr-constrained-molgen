"""Required Project 3 report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def plot_funnel(counts: pd.DataFrame, output_path: str | Path) -> None:
    required = {"stage", "count"}
    if missing := required.difference(counts.columns):
        raise ValueError(f"Funnel table is missing columns: {sorted(missing)}")
    frame = counts.copy()
    # Optional scoring is not a rejection stage and is excluded from the bar chart.
    frame = frame.loc[~frame["stage"].eq("step10_optional_activity_scored")].copy()
    labels = [
        "Sampled",
        "Valid parse",
        "Parent + neutralized",
        "Canonical non-empty",
        "Unique",
        "Not Project 2 exact",
        "Property pass",
        "After PAINS policy",
        "Similarity scored",
        "Tanimoto < 0.5",
    ]
    if len(frame) != len(labels):
        raise ValueError("Unexpected funnel stage count; expected the fixed 10-stage rejection funnel.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(frame))
    ax.bar(x, frame["count"].astype(int).to_numpy())
    ax.set_xticks(x, labels, rotation=42, ha="right")
    ax.set_ylabel("Molecules remaining")
    ax.set_title("Generated-SMILES rejection funnel")
    for idx, value in enumerate(frame["count"].astype(int)):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_nearest_training_similarity(valid_unique: pd.DataFrame, output_path: str | Path) -> None:
    column = "nearest_training_active_tanimoto"
    if column not in valid_unique.columns:
        raise ValueError(f"Valid-unique diagnostics are missing {column!r}.")
    values = pd.to_numeric(valid_unique[column], errors="coerce").dropna().to_numpy(dtype=float)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if len(values):
        ax.hist(values, bins=20, range=(0.0, 1.0), edgecolor="black")
    else:
        ax.text(0.5, 0.5, "No valid unique molecules", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(0.5, linestyle="--", linewidth=1.2, label="strict novelty cutoff = 0.5")
    ax.axvline(0.7, linestyle=":", linewidth=1.2, label="analog diagnostic = 0.7")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Tanimoto to nearest scaffold-training active")
    ax.set_ylabel("Valid unique generated molecules")
    ax.set_title("Nearest-training-active similarity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_property_distributions(
    survivors: pd.DataFrame,
    training_actives: pd.DataFrame,
    output_path: str | Path,
) -> None:
    for name, frame in (("survivors", survivors), ("training actives", training_actives)):
        missing = {"mw", "clogp", "qed"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} are missing property columns: {sorted(missing)}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, column, label in zip(axes, ["mw", "clogp", "qed"], ["MW", "cLogP", "QED"], strict=True):
        train_values = pd.to_numeric(training_actives[column], errors="coerce").dropna()
        survivor_values = pd.to_numeric(survivors[column], errors="coerce").dropna()
        if len(train_values):
            ax.hist(train_values, bins=25, alpha=0.55, density=True, label="Training actives")
        if len(survivor_values):
            ax.hist(survivor_values, bins=20, alpha=0.55, density=True, label="Survivors")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.suptitle("Property distributions: generated survivors vs training actives")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_survivor_grid(
    survivors: pd.DataFrame,
    output_path: str | Path,
    *,
    max_molecules: int = 20,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if survivors.empty:
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No generated molecules survived the fixed property + Tanimoto < 0.5 funnel.",
            ha="center",
            va="center",
            wrap=True,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return
    if "standardized_smiles" not in survivors.columns:
        raise ValueError("Survivor table is missing standardized_smiles.")
    subset = survivors.head(int(max_molecules))
    mols = [Chem.MolFromSmiles(str(value)) for value in subset["standardized_smiles"]]
    if any(mol is None for mol in mols):
        raise ValueError("Survivor grid contains an invalid standardized SMILES.")
    legends = []
    for idx, row in subset.reset_index(drop=True).iterrows():
        sim = row.get("nearest_training_active_tanimoto", float("nan"))
        legends.append(f"#{idx + 1} Tanimoto={float(sim):.2f}")
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(260, 210),
        legends=legends,
        useSVG=False,
    )
    image.save(str(output))
