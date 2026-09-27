"""Autoregressive sampling for the character-level SMILES LSTM."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from src.model import SmilesLSTM
from src.smiles_vocab import SmilesVocab


def _filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    filtered = logits.clone()
    vocab_size = filtered.size(-1)

    if top_k is not None and int(top_k) > 0 and int(top_k) < vocab_size:
        threshold = torch.topk(filtered, k=int(top_k), dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None:
        p = float(top_p)
        if not 0.0 < p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            restored = torch.full_like(filtered, float("-inf"))
            restored.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            filtered = restored
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and > 0.")
    scaled = logits / float(temperature)
    scaled = _filter_logits(scaled, top_k=top_k, top_p=top_p)
    probs = torch.softmax(scaled, dim=-1)
    if not torch.isfinite(probs).all() or torch.any(probs.sum(dim=-1) <= 0):
        raise ValueError("Sampling distribution became invalid after top-k/top-p filtering.")
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def generate_smiles(
    model: SmilesLSTM,
    vocab: SmilesVocab,
    *,
    n_samples: int,
    max_length: int = 160,
    temperature: float = 1.0,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    seed: int = 2026,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> list[str]:
    """Generate exactly ``n_samples`` raw SMILES strings.

    Invalid and empty strings are returned as sampled. They are not repaired here;
    rejection belongs to the chemistry funnel.
    """

    if int(n_samples) < 1:
        raise ValueError("n_samples must be >= 1.")
    if int(max_length) < 1:
        raise ValueError("max_length must be >= 1.")
    dev = torch.device(device)
    if dev.type != "cpu":
        raise ValueError("The reference sampler is intentionally CPU-only for reproducibility.")
    model = model.to(dev)
    model.eval()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    forbidden = [vocab.pad_id, vocab.bos_id, vocab.unk_id]
    all_outputs: list[str] = []
    remaining = int(n_samples)

    with torch.no_grad():
        while remaining > 0:
            current_batch = min(int(batch_size), remaining)
            token = torch.full((current_batch, 1), vocab.bos_id, dtype=torch.long, device=dev)
            hidden = None
            finished = torch.zeros(current_batch, dtype=torch.bool, device=dev)
            sequences: list[list[int]] = [[] for _ in range(current_batch)]

            for _ in range(int(max_length)):
                logits, hidden = model(token, hidden)
                next_logits = logits[:, -1, :].clone()
                next_logits[:, forbidden] = float("-inf")
                next_ids = sample_next_token(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generator=rng,
                )
                next_ids = torch.where(finished, torch.full_like(next_ids, vocab.eos_id), next_ids)

                for row, idx in enumerate(next_ids.tolist()):
                    if not bool(finished[row].item()):
                        sequences[row].append(int(idx))
                finished = finished | next_ids.eq(vocab.eos_id)
                token = next_ids.unsqueeze(1)
                if bool(finished.all().item()):
                    break

            all_outputs.extend(vocab.decode(seq, stop_at_eos=True, skip_special=True) for seq in sequences)
            remaining -= current_batch

    return all_outputs
