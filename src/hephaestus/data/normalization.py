from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from hephaestus.schemas.discovery_contract import DatasetCandidate
from hephaestus.utils.hashing import hash_json


_INSTRUCTION_PAIR_KEYS: tuple[tuple[str, str], ...] = (
    ("prompt", "target"),
    ("input", "output"),
    ("instruction", "response"),
    ("question", "answer"),
)


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({str(value).strip().casefold() for value in values if str(value).strip()})


def _stable_object(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _stable_object(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set)):
        normalized = [_stable_object(item) for item in value]
        return sorted(normalized, key=lambda item: hash_json(item))
    return value


def normalize_dataset_candidate(candidate: DatasetCandidate, *, provider_id: str | None = None) -> DatasetCandidate:
    """Return a deterministic canonical candidate without changing shared contracts."""

    normalized_provider = (provider_id or candidate.provider_id).strip().casefold()
    dataset_id = candidate.dataset_id.strip()
    revision = candidate.revision.strip() if candidate.revision and candidate.revision.strip() else None
    license_name = candidate.license.strip().casefold() if candidate.license and candidate.license.strip() else None
    provenance = dict(_stable_object(candidate.provenance))
    compatibility = dict(_stable_object(candidate.compatibility))
    metadata = dict(_stable_object(candidate.metadata))
    missing = set(_sorted_unique(candidate.missing_metadata))
    if not license_name:
        missing.add("license")
    if not provenance:
        missing.add("provenance")
    if not revision:
        missing.add("revision")
    seed = {"provider_id": normalized_provider, "dataset_id": dataset_id, "revision": revision}
    candidate_id = candidate.candidate_id.strip() or f"dataset-{hash_json(seed)[:16]}"
    return DatasetCandidate(
        candidate_id=candidate_id,
        provider_id=normalized_provider,
        dataset_id=dataset_id,
        revision=revision,
        splits=_sorted_unique(candidate.splits),
        task_types=_sorted_unique(candidate.task_types),
        languages=_sorted_unique(candidate.languages),
        domains=_sorted_unique(candidate.domains),
        format_profile=dict(_stable_object(candidate.format_profile)),
        estimated_rows=max(0, candidate.estimated_rows) if candidate.estimated_rows is not None else None,
        estimated_bytes=max(0, candidate.estimated_bytes) if candidate.estimated_bytes is not None else None,
        license=license_name,
        provenance=provenance,
        trust_level=candidate.trust_level.strip().casefold() or "unknown",
        compatibility=compatibility,
        risk_signals=_sorted_unique(candidate.risk_signals),
        artifact_ref=candidate.artifact_ref.strip() if candidate.artifact_ref else None,
        evidence_refs=sorted({str(ref).strip() for ref in candidate.evidence_refs if str(ref).strip()}),
        missing_metadata=sorted(missing),
        score_components={str(key): float(value) for key, value in sorted(candidate.score_components.items())},
        metadata=metadata,
        contract_version=candidate.contract_version,
    )


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")).strip()


def normalize_record(record: Mapping[str, Any]) -> dict[str, object] | None:
    """Normalize common text and instruction-pair records conservatively.

    Pair aliases are normalized into the existing ``prompt``/``target`` shape so
    downstream wrapper construction and TrainableDataContract behavior remain
    unchanged. A partially present pair is treated as malformed rather than
    silently falling through to another schema.
    """

    for prompt_key, target_key in _INSTRUCTION_PAIR_KEYS:
        if prompt_key in record or target_key in record:
            prompt = normalize_text(record.get(prompt_key, ""))
            target = normalize_text(record.get(target_key, ""))
            if not prompt or not target:
                return None
            return {"prompt": prompt, "target": target}
    for key in ("text", "content", "completion"):
        if key in record:
            text = normalize_text(record.get(key, ""))
            return {"text": text} if text else None
    return None
