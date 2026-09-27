"""Dataset helpers for next-character SMILES language modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.smiles_vocab import SmilesVocab


# Used only for --demo smoke tests. These strings are not part of any reported
# EGFR experiment and are never mixed into a full Project 3 run.
DEMO_SMILES = [
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1CCC[C@H]1c1cccnc1",
    "CCN(CC)CCOc1ccc2nc3ccccc3cc2c1",
    "COc1ccc(CCN(C)C)cc1OC",
    "CC(C)NCC(O)c1ccc(O)c(O)c1",
    "COc1ccc2[nH]cc(CCNC(C)C)c2c1",
    "CCOC(=O)c1ccc(N)cc1",
    "CC(C)c1ccc(cc1)C(C)C(=O)O",
    "COc1ccccc1OCCN(C)C",
    "CN(C)CCc1c[nH]c2ccccc12",
    "CCN1C(=O)NC(c2ccccc2)C1=O",
    "COc1ccc(C(=O)NCCN2CCOCC2)cc1",
    "CCOC(=O)N1CCC(CC1)c1ccccc1",
    "CN1CCN(CC1)C(=O)c1ccccc1",
]


def load_smiles_csv(path: str | Path, column: str = "standardized_smiles") -> list[str]:
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ValueError(f"SMILES file is missing required column {column!r}.")
    values = [str(x).strip() for x in frame[column].tolist() if str(x).strip()]
    if not values:
        raise ValueError(f"No non-empty SMILES were found in {path}.")
    return values


def split_smiles(
    smiles: Sequence[str],
    *,
    validation_fraction: float = 0.10,
    seed: int = 2026,
) -> tuple[list[str], list[str]]:
    """Return deterministic molecule-level train/validation splits."""

    values = list(smiles)
    if len(values) < 2:
        raise ValueError("At least two SMILES are required for a train/validation split.")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(values))
    n_val = max(1, int(round(len(values) * float(validation_fraction))))
    n_val = min(n_val, len(values) - 1)
    val_idx = set(int(i) for i in order[:n_val])
    train = [value for i, value in enumerate(values) if i not in val_idx]
    validation = [value for i, value in enumerate(values) if i in val_idx]
    return train, validation


class SmilesLanguageModelDataset(Dataset):
    """Teacher-forced next-token dataset.

    A sequence ``<bos> C C O <eos>`` becomes input ``<bos> C C O`` and target
    ``C C O <eos>``. Padding is applied only in the collate function.
    """

    def __init__(self, smiles: Iterable[str], vocab: SmilesVocab):
        self.smiles = [str(value) for value in smiles]
        if not self.smiles:
            raise ValueError("SMILES dataset cannot be empty.")
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocab.encode(self.smiles[index], add_bos=True, add_eos=True)
        if len(ids) < 2:
            raise RuntimeError("Encoded sequence is unexpectedly shorter than two tokens.")
        full = torch.tensor(ids, dtype=torch.long)
        return full[:-1], full[1:]


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty batch.")
        inputs, targets = zip(*batch, strict=True)
        lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
        return {
            "input_ids": pad_sequence(inputs, batch_first=True, padding_value=int(pad_id)),
            "target_ids": pad_sequence(targets, batch_first=True, padding_value=int(pad_id)),
            "lengths": lengths,
        }

    return collate
