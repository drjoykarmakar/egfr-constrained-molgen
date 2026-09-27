"""Training utilities for the small SMILES LSTM."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import SmilesLSTM
from src.smiles_vocab import SmilesVocab


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int
    history: list[dict[str, float | int]]


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _token_loss_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
) -> tuple[torch.Tensor, int]:
    criterion = nn.CrossEntropyLoss(ignore_index=int(pad_id), reduction="sum")
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    n_tokens = int(targets.ne(int(pad_id)).sum().item())
    return loss, n_tokens


def evaluate_token_loss(
    model: SmilesLSTM,
    loader: DataLoader,
    *,
    pad_id: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            logits, _ = model(inputs)
            loss, n_tokens = _token_loss_sum(logits, targets, pad_id=pad_id)
            total_loss += float(loss.item())
            total_tokens += n_tokens
    if total_tokens == 0:
        raise ValueError("Validation loader contained zero non-padding target tokens.")
    return total_loss / total_tokens


def fit_generator(
    model: SmilesLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    pad_id: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    early_stopping_patience: int = 5,
    gradient_clip_norm: float = 1.0,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train with teacher forcing and early stopping on validation token loss."""

    dev = torch.device(device)
    model.to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_used = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in train_loader:
            inputs = batch["input_ids"].to(dev)
            targets = batch["target_ids"].to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss_sum, n_tokens = _token_loss_sum(logits, targets, pad_id=pad_id)
            if n_tokens == 0:
                continue
            loss = loss_sum / n_tokens
            loss.backward()
            if gradient_clip_norm and float(gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
            total_loss += float(loss_sum.item())
            total_tokens += n_tokens

        if total_tokens == 0:
            raise ValueError("Training loader contained zero non-padding target tokens.")
        train_loss = total_loss / total_tokens
        val_loss = evaluate_token_loss(model, validation_loader, pad_id=pad_id, device=dev)
        history.append({"epoch": epoch, "train_token_loss": train_loss, "validation_token_loss": val_loss})

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= int(early_stopping_patience):
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=int(best_epoch),
        best_validation_loss=float(best_loss),
        epochs_ran=len(history),
        history=history,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: SmilesLSTM,
    vocab: SmilesVocab,
    training_result: TrainingResult,
    metadata: dict | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_class": "SmilesLSTM",
        "model_kwargs": model.model_kwargs,
        "model_state_dict": model.state_dict(),
        "vocab": vocab.to_dict(),
        "training": {
            "best_epoch": training_result.best_epoch,
            "best_validation_loss": training_result.best_validation_loss,
            "epochs_ran": training_result.epochs_ran,
        },
        "metadata": metadata or {},
    }
    torch.save(payload, output)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SmilesLSTM, SmilesVocab, dict]:
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
    if payload.get("model_class") != "SmilesLSTM":
        raise ValueError("Unsupported generator checkpoint model class.")
    vocab = SmilesVocab.from_dict(payload["vocab"])
    model = SmilesLSTM(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, vocab, payload


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
