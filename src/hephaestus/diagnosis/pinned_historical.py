"""Read-only, byte-pinned historical evidence intake for model-repair diagnosis.

A file hash establishes the identity of the *observed bytes*, not the truth of
claims inside them. These records remain historical/exposed; the adapter never
grants evaluation integrity, independent certification, or training authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hephaestus.schemas.diagnosis_contract import DiagnosisRequest


_DIGEST = re.compile(r"sha256:([0-9a-f]{64})\Z")
_MAX_BYTES = 8 * 1024 * 1024


class PinnedEvidenceError(ValueError):
    """Evidence was missing, mutable, malformed, or not independently pinned."""


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise PinnedEvidenceError("duplicate JSON object key")
        obj[key] = value
    return obj


def _reject_nonfinite(value: str) -> object:
    raise PinnedEvidenceError("non-finite JSON constant")


def _finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _parse_json(raw: str) -> object:
    try:
        value = json.loads(
            raw, object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PinnedEvidenceError("invalid UTF-8 or JSON evidence") from exc
    if not _finite_tree(value):
        raise PinnedEvidenceError("non-finite decoded JSON value")
    return value


@dataclass(slots=True)
class PinnedHistoricalEvidenceAdapter:
    """Load flat JSON/JSONL files against hashes retained outside those files.

    Exact file names and hashes must be supplied by the trusted caller. Files
    and their directory are opened without following symbolic links. The bytes
    are bounded, checked before parsing, and all references are read before
    any records are returned (no partial admission on failure).

    The adapter never interprets the payload's diagnostic claims as verified.
    A downstream diagnosis must independently verify evaluation and scorecards.
    """

    root: Path
    expected_sha256: Mapping[str, str]
    max_bytes: int = _MAX_BYTES

    def load(self, request: DiagnosisRequest) -> list[dict[str, object]]:
        if not isinstance(request, DiagnosisRequest):
            raise PinnedEvidenceError("expected DiagnosisRequest")
        if not isinstance(self.max_bytes, int) or not 0 < self.max_bytes <= _MAX_BYTES:
            raise PinnedEvidenceError("invalid evidence byte limit")

        references = list(request.evidence_refs)
        if len(references) != len(set(references)):
            raise PinnedEvidenceError("duplicate evidence file reference")
        records: list[dict[str, object]] = []
        if not references:
            return records

        directory = os.open(
            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            for name in references:
                if not isinstance(name, str) or not name or Path(name).name != name:
                    raise PinnedEvidenceError("evidence must be a flat relative filename")
                if name.startswith(".") or Path(name).suffix not in {".json", ".jsonl"}:
                    raise PinnedEvidenceError("unsupported historical evidence filename")
                expected = self.expected_sha256.get(name)
                if not isinstance(expected, str) or not _DIGEST.fullmatch(expected):
                    raise PinnedEvidenceError("missing trusted SHA-256 for evidence file")
                handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    info = os.fstat(handle)
                    if not stat.S_ISREG(info.st_mode):
                        raise PinnedEvidenceError("evidence is not a regular file")
                    if info.st_size > self.max_bytes:
                        raise PinnedEvidenceError("evidence exceeds byte limit")
                    chunks: list[bytes] = []
                    remaining = self.max_bytes + 1
                    while remaining:
                        chunk = os.read(handle, min(65536, remaining))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    payload = b"".join(chunks)
                    if len(payload) > self.max_bytes:
                        raise PinnedEvidenceError("evidence exceeds byte limit")
                finally:
                    os.close(handle)
                digest = hashlib.sha256(payload).hexdigest()
                if expected != "sha256:" + digest:
                    raise PinnedEvidenceError("historical evidence SHA-256 mismatch")
                try:
                    decoded = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise PinnedEvidenceError("invalid UTF-8 evidence") from exc
                if name.endswith(".jsonl"):
                    items = [_parse_json(line) for line in decoded.splitlines() if line.strip()]
                else:
                    parsed = _parse_json(decoded)
                    items = parsed if isinstance(parsed, list) else [parsed]
                for index, item in enumerate(items):
                    if not isinstance(item, dict):
                        raise PinnedEvidenceError("evidence record must be a JSON object")
                    # An old report can *claim* eval_integrity_verified or
                    # arbitrary diagnosis signals. Hashing its bytes does not
                    # validate those scientific claims. Never pass its nested
                    # keys through the signal-extracting normalizer.
                    summary = item.get("summary")
                    records.append({
                        "evidence_kind": "historical_observation",
                        "source_ref": f"sha256:{digest}#{index}",
                        "summary": summary[:500] if isinstance(summary, str)
                                   else "Pinned historical evidence available for independent review",
                        "pinned_source_file": name,
                        "original_evidence_kind": str(item.get("evidence_kind") or
                                                     item.get("kind") or "unknown")[:80],
                        "source_hash_verified": True,
                        "historical_exposed": True,
                    })
        finally:
            os.close(directory)
        return records
