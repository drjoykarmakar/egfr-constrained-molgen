"""Small character-level LSTM used as the default SMILES generator."""

from __future__ import annotations

import torch
from torch import nn


class SmilesLSTM(nn.Module):
    """A compact next-token model: embedding -> LSTM -> vocabulary logits."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if vocab_size <= 4:
            raise ValueError("vocab_size must include chemistry tokens in addition to special tokens.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.model_kwargs = {
            "vocab_size": int(vocab_size),
            "pad_id": int(pad_id),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        }
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input_ids)
        encoded, hidden_out = self.lstm(embedded, hidden)
        return self.output(encoded), hidden_out
