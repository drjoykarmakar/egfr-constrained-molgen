"""Sample raw SMILES from a trained generator checkpoint.

Sampling does not repair invalid strings. Raw outputs, including empty/invalid
ones, are persisted for the rejection funnel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import sha256_file
from src.sample import generate_smiles
from src.train import load_checkpoint, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None)
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parents[1]
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    gcfg = config["generator"]
    scfg = config["sampling"]

    checkpoint = resolve(
        root,
        gcfg["demo_checkpoint_path"] if args.demo else gcfg["checkpoint_path"],
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")
    model, vocab, payload = load_checkpoint(checkpoint, device=str(gcfg["device"]))

    default_n = int(scfg["demo_n_samples"] if args.demo else scfg["n_samples"])
    n_samples = int(args.n_samples) if args.n_samples is not None else default_n
    samples = generate_smiles(
        model,
        vocab,
        n_samples=n_samples,
        max_length=int(scfg["max_length"]),
        temperature=float(scfg["temperature"]),
        top_k=int(scfg["top_k"]),
        top_p=float(scfg["top_p"]),
        seed=int(scfg["seed"]),
        batch_size=int(scfg.get("batch_size", 256)),
        device=str(gcfg["device"]),
    )

    output = resolve(root, scfg["demo_output_csv"] if args.demo else scfg["output_csv"])
    manifest = resolve(root, scfg["demo_manifest"] if args.demo else scfg["manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": range(len(samples)), "sampled_smiles": samples}).to_csv(
        output, index=False
    )
    write_json(
        manifest,
        {
            "demo_mode": bool(args.demo),
            "n_samples": len(samples),
            "output_csv": str(output),
            "output_sha256": sha256_file(output),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_training_metadata": payload.get("metadata", {}),
            "sampling": {
                "max_length": int(scfg["max_length"]),
                "temperature": float(scfg["temperature"]),
                "top_k": int(scfg["top_k"]),
                "top_p": float(scfg["top_p"]),
                "seed": int(scfg["seed"]),
            },
        },
    )
    print(f"Wrote {len(samples)} raw samples to {output}")
    print(f"Sampling manifest: {manifest}")


if __name__ == "__main__":
    main()
