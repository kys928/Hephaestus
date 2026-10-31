"""Strict reader for already reviewed frozen-tokenizer continuation windows.

Artifact tokenizer identity and HF loader location are intentionally distinct.
No decoding, re-encoding, EOS insertion, sampling or truncation is permitted.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

FORMAT = "trainable-token-ids.v1"
KIND = "frozen_token_continuation"


def continuation_contract(evidence: dict):
    from hephaestus.schemas.continuation_subset import ContinuationSubsetContract

    if evidence.get("data_format") != FORMAT:
        raise ValueError("explicit frozen token data format is required")
    contract = ContinuationSubsetContract(**evidence["continuation_contract"])
    contract.validate()
    return contract


def verified_original_tokenizer(evidence: dict):
    from tokenizers import Tokenizer

    contract = continuation_contract(evidence)
    path = Path(evidence["original_tokenizer_path"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != contract.tokenizer_file_sha256:
        raise ValueError("original frozen tokenizer bytes differ from reviewed SHA")
    original = Tokenizer.from_str(raw.decode("utf-8"))
    checked = evidence.get("tokenizer_compatibility", {})
    if (checked.get("status") != "checked" or checked.get("compatible") is not True
            or checked.get("tokenizer_ref") != contract.tokenizer_ref
            or checked.get("tokenizer_file_sha256") != contract.tokenizer_file_sha256):
        raise ValueError("frozen tokenizer artifact evidence differs from subset contract")
    return original


def validate_tokenizer_location(loader_path: str, evidence: dict) -> None:
    """Bind a separately pinned local HF tokenizer location to reviewed bytes."""
    from tokenizers import Tokenizer

    original = verified_original_tokenizer(evidence)
    local = Path(loader_path) / "tokenizer.json"
    if local.is_symlink() or not local.is_file():
        raise ValueError("frozen continuation requires a verified local tokenizer backend")
    loaded = Tokenizer.from_str(local.read_text(encoding="utf-8"))
    if json.loads(loaded.to_str()) != json.loads(original.to_str()):
        raise ValueError("selected tokenizer location differs from original frozen backend")


def validate_token_records(records: list[dict], evidence: dict, *, vocabulary_size: int) -> int:
    """Check all positions, masked targets, counts and selection commitment."""
    contract = continuation_contract(evidence)
    width = contract.prefix_tokens + contract.target_tokens
    if type(vocabulary_size) is not int or vocabulary_size < 1:
        raise ValueError("valid vocabulary size required for frozen tokens")
    if len(records) != contract.windows:
        raise ValueError("frozen continuation window count differs from reviewed contract")
    digest = sha256()
    source_rows = 0
    previous_row = -1
    previous_start = -contract.stride_tokens
    previous_hash = None
    previous_ids = None
    fields = {"record_kind", "source_row_index", "source_content_sha256",
              "window_start_token", "token_count", "input_ids", "labels"}
    for record in records:
        if not isinstance(record, dict) or set(record) != fields or record["record_kind"] != KIND:
            raise ValueError("frozen token record has unexpected fields or kind")
        row = record["source_row_index"]
        start = record["window_start_token"]
        content_hash = record["source_content_sha256"]
        if (type(row) is not int or not 0 <= row < contract.source_rows
                or type(start) is not int or start < 0
                or not isinstance(content_hash, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", content_hash)):
            raise ValueError("invalid frozen source row identity or offset")
        ids, labels = record["input_ids"], record["labels"]
        if (type(record["token_count"]) is not int or record["token_count"] != width
                or not isinstance(ids, list) or len(ids) != width
                or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids)
                or not isinstance(labels, list)
                or any(type(token) is not int for token in labels)
                or labels != [-100] * contract.prefix_tokens + ids[contract.prefix_tokens:]):
            raise ValueError("frozen continuation tokens or exact prompt mask differ")
        if row != previous_row:
            if row <= previous_row or start != 0:
                raise ValueError("frozen continuation source rows must be ordered and start at zero")
            digest.update(f"{row}\t{content_hash.removeprefix('sha256:')}\n".encode())
            source_rows += 1
        else:
            if content_hash != previous_hash or start != previous_start + contract.stride_tokens:
                raise ValueError("frozen continuation row hash or stride differs")
            overlap = width - contract.stride_tokens
            if overlap > 0 and previous_ids[-overlap:] != ids[:overlap]:
                raise ValueError("frozen continuation windows disagree on overlapping token IDs")
        previous_row, previous_start, previous_hash, previous_ids = row, start, content_hash, ids
    if (source_rows != contract.accepted_rows
            or "sha256:" + digest.hexdigest() != contract.selection_commitment_sha256):
        raise ValueError("frozen continuation source selection commitment differs")
    return len(records) * width


def encode_frozen_records(tokenizer, records: list[dict], config: dict, evidence: dict):
    contract = continuation_contract(evidence)
    original = verified_original_tokenizer(evidence)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or json.loads(backend.to_str()) != json.loads(original.to_str()):
        raise ValueError("loaded tokenizer backend differs from original frozen tokenizer")
    total = validate_token_records(records, evidence, vocabulary_size=int(config["vocabulary_size"]))
    if (int(config["context_length"]) < contract.prefix_tokens + contract.target_tokens
            or total > int(config["max_total_tokens"])
            or config.get("ignored_label_token") != -100
            or config.get("eos_handling") != "no_insertion"
            or config.get("truncation") != "forbidden"):
        raise ValueError("frozen continuation recipe would alter or truncate reviewed windows")
    return [{"input_ids": list(row["input_ids"]), "labels": list(row["labels"])} for row in records], {
        "source_samples": len(records), "encoded_samples": len(records),
        "dropped_samples": 0, "truncated_samples": 0,
        "prompt_masked_samples": len(records), "total_tokens": total,
    }
