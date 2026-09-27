"""Character-level SMILES vocabulary with explicit special tokens.

The generator is deliberately simple: canonical SMILES are represented as
individual characters. Special tokens are kept separate from chemistry
characters so sequence boundaries and padding are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"

    def as_list(self) -> list[str]:
        values = [self.pad, self.bos, self.eos, self.unk]
        if len(set(values)) != len(values):
            raise ValueError("Special tokens must be unique.")
        return values


class SmilesVocab:
    """Deterministic character vocabulary for canonical SMILES."""

    def __init__(self, tokens: Sequence[str], special_tokens: SpecialTokens | None = None):
        self.special_tokens = special_tokens or SpecialTokens()
        required = self.special_tokens.as_list()
        token_list = [str(token) for token in tokens]
        if len(token_list) != len(set(token_list)):
            raise ValueError("Vocabulary tokens must be unique.")
        if token_list[: len(required)] != required:
            raise ValueError("Vocabulary must start with pad, bos, eos, and unk special tokens.")
        self.tokens = token_list
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}

    @classmethod
    def build(
        cls,
        smiles: Iterable[str],
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> "SmilesVocab":
        special = special_tokens or SpecialTokens()
        specials = special.as_list()
        chars: set[str] = set()
        n_strings = 0
        for value in smiles:
            text = str(value)
            if not text:
                continue
            n_strings += 1
            chars.update(text)
        if n_strings == 0:
            raise ValueError("Cannot build a vocabulary from an empty SMILES collection.")
        collisions = sorted(chars.intersection(specials))
        if collisions:
            raise ValueError(f"SMILES characters collide with special tokens: {collisions}")
        return cls([*specials, *sorted(chars)], special_tokens=special)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens.pad]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.special_tokens.bos]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.special_tokens.eos]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special_tokens.unk]

    def encode(self, smiles: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(char, self.unk_id) for char in str(smiles))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        stop_at_eos: bool = True,
        skip_special: bool = True,
    ) -> str:
        chars: list[str] = []
        special_set = set(self.special_tokens.as_list())
        for raw_id in token_ids:
            idx = int(raw_id)
            if idx < 0 or idx >= len(self.tokens):
                raise ValueError(f"Token id out of range: {idx}")
            token = self.tokens[idx]
            if token == self.special_tokens.eos and stop_at_eos:
                break
            if skip_special and token in special_set:
                continue
            chars.append(token)
        return "".join(chars)

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "special_tokens": {
                "pad": self.special_tokens.pad,
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
                "unk": self.special_tokens.unk,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SmilesVocab":
        st = payload.get("special_tokens", {})
        special = SpecialTokens(
            pad=str(st.get("pad", "<pad>")),
            bos=str(st.get("bos", "<bos>")),
            eos=str(st.get("eos", "<eos>")),
            unk=str(st.get("unk", "<unk>")),
        )
        return cls(payload["tokens"], special_tokens=special)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SmilesVocab":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
