from pathlib import Path

import torch

from src.dataset import SmilesLanguageModelDataset, make_collate_fn, split_smiles
from src.model import SmilesLSTM
from src.sample import generate_smiles
from src.smiles_vocab import SmilesVocab


def test_vocab_round_trip_and_special_tokens(tmp_path: Path):
    vocab = SmilesVocab.build(["CCO", "N#N", "c1ccccc1"])
    ids = vocab.encode("CCO")
    assert ids[0] == vocab.bos_id
    assert ids[-1] == vocab.eos_id
    assert vocab.decode(ids) == "CCO"

    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = SmilesVocab.load(path)
    assert loaded.tokens == vocab.tokens
    assert loaded.decode(loaded.encode("N#N")) == "N#N"


def test_unknown_character_uses_unk_id():
    vocab = SmilesVocab.build(["CCO"])
    ids = vocab.encode("C?O")
    assert vocab.unk_id in ids


def test_dataset_teacher_forcing_and_padding():
    vocab = SmilesVocab.build(["CCO", "N"])
    ds = SmilesLanguageModelDataset(["CCO", "N"], vocab)
    x0, y0 = ds[0]
    assert x0[0].item() == vocab.bos_id
    assert y0[-1].item() == vocab.eos_id
    batch = make_collate_fn(vocab.pad_id)([ds[0], ds[1]])
    assert batch["input_ids"].shape == batch["target_ids"].shape
    assert batch["input_ids"].shape[0] == 2
    assert batch["lengths"].tolist() == [len(x0), len(ds[1][0])]


def test_split_is_deterministic_and_disjoint():
    smiles = [f"C{'C' * i}O" for i in range(1, 12)]
    a_train, a_val = split_smiles(smiles, validation_fraction=0.2, seed=7)
    b_train, b_val = split_smiles(smiles, validation_fraction=0.2, seed=7)
    assert a_train == b_train
    assert a_val == b_val
    assert set(a_train).isdisjoint(a_val)
    assert sorted(a_train + a_val) == sorted(smiles)


def test_lstm_forward_shape_and_sampling_exact_count():
    vocab = SmilesVocab.build(["CCO", "CCN"])
    model = SmilesLSTM(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        embedding_dim=8,
        hidden_dim=12,
        num_layers=1,
        dropout=0.0,
    )
    x = torch.tensor([[vocab.bos_id, vocab.token_to_id["C"]]], dtype=torch.long)
    logits, _ = model(x)
    assert logits.shape == (1, 2, len(vocab))

    # Force EOS to be selected under top_k=1. Empty strings are legitimate raw
    # generator outputs and must later be rejected by the chemistry funnel.
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-10.0)
        model.output.bias[vocab.eos_id] = 10.0
    samples = generate_smiles(
        model,
        vocab,
        n_samples=7,
        max_length=5,
        top_k=1,
        top_p=1.0,
        seed=3,
        batch_size=4,
    )
    assert samples == [""] * 7
