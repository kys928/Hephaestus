"""Measured compatibility with the original immutable tokenizer file.

No tokenizer training, vocabulary substitution, text deletion or threshold
relaxation. Recheck bytes at use time so cached identity claims cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.utils.hashing import hash_file


@dataclass(frozen=True, slots=True)
class FrozenTokenizerCompatibilityChecker:
    tokenizer_path: Path
    tokenizer_ref: str
    expected_file_sha256: str
    unknown_token: str = "<|unk|>"

    @property
    def tokenizer_file_sha256(self):
        digest = hash_file(self.tokenizer_path)
        if digest != self.expected_file_sha256:
            raise ValueError("original tokenizer file bytes changed")
        return digest

    @property
    def checker_id(self):
        return "frozen-tokenizer-exact-counts-one-percent-v1:" + self.tokenizer_file_sha256

    def check(self, texts):
        from tokenizers import Tokenizer
        digest = self.tokenizer_file_sha256
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        unknown_id = tokenizer.token_to_id(self.unknown_token)
        if unknown_id is None:
            raise ValueError("original tokenizer has no explicitly configured unknown token")
        total, unknown = 0, 0
        for text in texts:
            ids = tokenizer.encode(text).ids
            total += len(ids)
            unknown += sum(token == unknown_id for token in ids)
        # Detect a source file changed during loading/measurement too.
        if self.tokenizer_file_sha256 != digest:
            raise ValueError("original tokenizer changed during recheck")
        return total > 0 and unknown * 100 <= total, {
            "tokenizer_file_sha256": digest, "tokenizer_ref": self.tokenizer_ref,
            "total_tokens": total, "unknown_tokens": unknown,
            "unknown_fraction": unknown / total if total else None,
            "maximum_unknown_fraction": 0.01, "all_processed_text_checked": True,
        }
