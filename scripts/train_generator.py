"""Train the small character-level SMILES LSTM.

Full mode requires ``scripts/prepare_data.py`` to have frozen Project 2
scaffold-training actives. ``--demo`` uses a tiny built-in toy set and is only
a smoke test; demo outputs are never scientific results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import sha256_file
from src.dataset import (
    DEMO_SMILES,
    SmilesLanguageModelDataset,
    load_smiles_csv,
    make_collate_fn,
    split_smiles,
)
from src.model import SmilesLSTM
from src.sample import generate_smiles
from src.smiles_vocab import SmilesVocab, SpecialTokens
from src.train import fit_generator, save_checkpoint, set_global_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--demo", action="store_true", help="Run a tiny toy-data smoke train and sample.")
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

    seed = int(config["generator"]["seed"])
    set_global_seed(seed)

    if args.demo:
        all_smiles = list(DEMO_SMILES)
        source_metadata = {"demo_mode": True, "source": "built-in toy SMILES smoke set"}
    else:
        training_path = resolve(root, config["training_data"]["output_smiles"])
        manifest_path = resolve(root, config["training_data"]["manifest"])
        if not training_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                "Frozen generation source is missing. Run scripts/prepare_data.py first."
            )
        all_smiles = load_smiles_csv(training_path)
        source_metadata = {
            "demo_mode": False,
            "training_file": str(training_path),
            "training_file_sha256": sha256_file(training_path),
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": sha256_file(manifest_path),
        }

    train_smiles, val_smiles = split_smiles(
        all_smiles,
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    vocab_cfg = config["vocab"]
    specials = SpecialTokens(
        pad=str(vocab_cfg["pad_token"]),
        bos=str(vocab_cfg["bos_token"]),
        eos=str(vocab_cfg["eos_token"]),
        unk=str(vocab_cfg["unk_token"]),
    )
    vocab = SmilesVocab.build(train_smiles, special_tokens=specials)

    train_ds = SmilesLanguageModelDataset(train_smiles, vocab)
    val_ds = SmilesLanguageModelDataset(val_smiles, vocab)
    gcfg = config["generator"]
    batch_size = 8 if args.demo else int(gcfg["batch_size"])
    loader_rng = torch.Generator().manual_seed(seed)
    collate = make_collate_fn(vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_rng,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = SmilesLSTM(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        embedding_dim=48 if args.demo else int(gcfg["embedding_dim"]),
        hidden_dim=96 if args.demo else int(gcfg["hidden_dim"]),
        num_layers=1 if args.demo else int(gcfg["num_layers"]),
        dropout=0.0 if args.demo else float(gcfg["dropout"]),
    )
    started = time.time()
    result = fit_generator(
        model,
        train_loader,
        val_loader,
        pad_id=vocab.pad_id,
        learning_rate=float(gcfg["learning_rate"]),
        weight_decay=float(gcfg["weight_decay"]),
        max_epochs=2 if args.demo else int(gcfg["max_epochs"]),
        early_stopping_patience=1 if args.demo else int(gcfg["early_stopping_patience"]),
        gradient_clip_norm=float(gcfg["gradient_clip_norm"]),
        device=str(gcfg["device"]),
    )
    elapsed = time.time() - started

    if args.demo:
        checkpoint_path = resolve(root, gcfg["demo_checkpoint_path"])
        history_path = resolve(root, gcfg["demo_training_history"])
        summary_path = resolve(root, gcfg["demo_training_summary"])
    else:
        checkpoint_path = resolve(root, gcfg["checkpoint_path"])
        history_path = resolve(root, gcfg["training_history"])
        summary_path = resolve(root, gcfg["training_summary"])
        vocab.save(resolve(root, gcfg["vocab_path"]))

    metadata = {
        **source_metadata,
        "seed": seed,
        "n_total_smiles": len(all_smiles),
        "n_train_smiles": len(train_smiles),
        "n_validation_smiles": len(val_smiles),
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        vocab=vocab,
        training_result=result,
        metadata=metadata,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.history).to_csv(history_path, index=False)
    write_json(
        summary_path,
        {
            **metadata,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "vocab_size": len(vocab),
            "best_epoch": result.best_epoch,
            "best_validation_token_loss": result.best_validation_loss,
            "epochs_ran": result.epochs_ran,
            "elapsed_seconds": elapsed,
        },
    )
    print(
        f"Training complete: best epoch {result.best_epoch}, "
        f"validation token loss {result.best_validation_loss:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")

    # One-command train+sample smoke test for CI/developer use. Demo samples are
    # intentionally not interpreted as chemistry results.
    if args.demo:
        scfg = config["sampling"]
        samples = generate_smiles(
            model,
            vocab,
            n_samples=int(scfg["demo_n_samples"]),
            max_length=min(80, int(scfg["max_length"])),
            temperature=float(scfg["temperature"]),
            top_k=int(scfg["top_k"]),
            top_p=float(scfg["top_p"]),
            seed=int(scfg["seed"]),
            batch_size=min(64, int(scfg.get("batch_size", 256))),
            device=str(gcfg["device"]),
        )
        output = resolve(root, scfg["demo_output_csv"])
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": range(len(samples)), "sampled_smiles": samples}).to_csv(
            output, index=False
        )
        write_json(
            resolve(root, scfg["demo_manifest"]),
            {
                "demo_mode": True,
                "n_samples": len(samples),
                "output_csv": str(output),
                "output_sha256": sha256_file(output),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "sampling": {
                    "max_length": min(80, int(scfg["max_length"])),
                    "temperature": float(scfg["temperature"]),
                    "top_k": int(scfg["top_k"]),
                    "top_p": float(scfg["top_p"]),
                    "seed": int(scfg["seed"]),
                },
            },
        )
        print(f"Demo sampled {len(samples)} raw strings to {output}")


if __name__ == "__main__":
    main()
