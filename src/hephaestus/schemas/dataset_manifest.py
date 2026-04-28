from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._base import JsonSchema

_DATASET_ENTRY_DEFAULTS: dict[str, object | None] = {
    "dataset_id": None,
    "source": None,
    "version": None,
    "split": None,
    "row_count": None,
    "byte_size": None,
    "content_hash": None,
    "hash_type": None,
    "license": None,
    "trust_level": None,
    "domain": None,
    "notes": None,
}


def _as_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _as_dict(value: object | None) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _as_list_of_dicts(value: object | None) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append({str(k): v for k, v in item.items()})
    return normalized


def _normalize_dataset_entry(entry: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = dict(_DATASET_ENTRY_DEFAULTS)
    dataset_id = _as_str(entry.get("dataset_id") or entry.get("id") or entry.get("name"))
    normalized["dataset_id"] = dataset_id
    normalized["source"] = _as_str(entry.get("source") or entry.get("source_identity"))
    normalized["version"] = _as_str(entry.get("version"))
    normalized["split"] = _as_str(entry.get("split"))
    normalized["row_count"] = _as_int(entry.get("row_count") or entry.get("total_examples"))
    normalized["byte_size"] = _as_int(entry.get("byte_size"))
    normalized["content_hash"] = _as_str(entry.get("content_hash") or entry.get("hash"))
    normalized["hash_type"] = _as_str(entry.get("hash_type"))
    normalized["license"] = _as_str(entry.get("license"))
    normalized["trust_level"] = _as_str(entry.get("trust_level"))
    normalized["domain"] = _as_str(entry.get("domain"))
    normalized["notes"] = _as_str(entry.get("notes"))
    return normalized


def compute_manifest_completeness(payload: dict[str, object]) -> tuple[float, list[str], list[str], str]:
    missing_fields: list[str] = []
    warnings: list[str] = []

    def mark_missing(path: str) -> None:
        missing_fields.append(path)

    required_roots = ("manifest_id", "run_id", "lineage_id", "datasets")
    for key in required_roots:
        if payload.get(key) in (None, "", []):
            mark_missing(key)

    datasets = _as_list_of_dicts(payload.get("datasets"))
    if not datasets:
        warnings.append("manifest includes no dataset entries")
        mark_missing("datasets[0].dataset_id")
        mark_missing("datasets[0].row_count")
        mark_missing("datasets[0].version_or_content_hash")

    mixture_weights = _as_dict(payload.get("mixture_weights"))
    if not mixture_weights:
        mark_missing("mixture_weights")
        warnings.append("mixture_weights missing or empty")

    complete_datasets = True
    has_any_dataset_identity = False
    for index, dataset in enumerate(datasets):
        dataset_id = _as_str(dataset.get("dataset_id"))
        row_count = _as_int(dataset.get("row_count"))
        version = _as_str(dataset.get("version"))
        content_hash = _as_str(dataset.get("content_hash"))

        if dataset_id:
            has_any_dataset_identity = True
        else:
            complete_datasets = False
            mark_missing(f"datasets[{index}].dataset_id")
            warnings.append(f"dataset entry {index} missing dataset_id")

        if row_count is None:
            complete_datasets = False
            mark_missing(f"datasets[{index}].row_count")
            warnings.append(f"dataset {dataset_id or index} missing row_count")

        if not version and not content_hash:
            complete_datasets = False
            mark_missing(f"datasets[{index}].version_or_content_hash")
            warnings.append(f"dataset {dataset_id or index} missing version/content_hash")

        if dataset_id and mixture_weights and dataset_id not in mixture_weights:
            complete_datasets = False
            mark_missing(f"mixture_weights.{dataset_id}")
            warnings.append(f"missing mixture weight for dataset {dataset_id}")

    denominator = 8 + max(len(datasets), 1) * 3
    present = denominator - len(set(missing_fields))
    score = 0.0 if denominator <= 0 else max(0.0, min(1.0, present / denominator))

    if datasets and complete_datasets and bool(mixture_weights):
        integrity_level = "complete"
    elif has_any_dataset_identity and score >= 0.45:
        integrity_level = "partial"
    elif has_any_dataset_identity or bool(payload.get("artifact_ref")) or bool(datasets):
        integrity_level = "reference_only"
    else:
        integrity_level = "insufficient"

    if integrity_level != "complete" and _as_str(payload.get("artifact_ref")):
        warnings.append("manifest is not fully reproducible despite artifact_ref presence")

    return round(score, 4), sorted(set(missing_fields)), sorted(set(warnings)), integrity_level


def normalize_dataset_manifest(payload: dict[str, object]) -> dict[str, object]:
    run_id = _as_str(payload.get("run_id")) or ""
    normalized: dict[str, object] = {
        "manifest_id": _as_str(payload.get("manifest_id")) or (f"manifest-{run_id}" if run_id else "manifest-unknown"),
        "run_id": run_id,
        "lineage_id": _as_str(payload.get("lineage_id")) or "",
        "stage_name": _as_str(payload.get("stage_name")),
        "created_at": _as_str(payload.get("created_at")) or datetime.now(timezone.utc).isoformat(),
        "artifact_ref": _as_str(payload.get("artifact_ref")),
        "datasets": [],
        "mixture_weights": _as_dict(payload.get("mixture_weights")),
        "sampling_policy": _as_dict(payload.get("sampling_policy")),
        "stage_data_policy_ref": _as_str(payload.get("stage_data_policy_ref") or payload.get("data_policy_ref")),
        "filtering_profile": _as_dict(payload.get("filtering_profile")),
        "preprocessing_profile": _as_dict(payload.get("preprocessing_profile")),
        "deduplication_profile": _as_dict(payload.get("deduplication_profile")),
        "contamination_checks": _as_dict(payload.get("contamination_checks")),
        "chunking_policy": _as_dict(payload.get("chunking_policy")),
        "wrapper_policy": _as_dict(payload.get("wrapper_policy")),
        "prompt_target_boundary_policy": _as_dict(payload.get("prompt_target_boundary_policy")),
        "tokenizer_ref": _as_str(payload.get("tokenizer_ref")),
        "tokenizer_compatibility": _as_dict(payload.get("tokenizer_compatibility")),
        "uses_synthetic_data": bool(payload.get("uses_synthetic_data", False)),
        "synthetic_data_profile": _as_dict(payload.get("synthetic_data_profile")),
        "uses_hard_negatives": bool(payload.get("uses_hard_negatives", False)),
        "hard_negative_profile": _as_dict(payload.get("hard_negative_profile")),
        "uses_support_sets": bool(payload.get("uses_support_sets", False)),
        "support_set_profile": _as_dict(payload.get("support_set_profile")),
        "manifest_integrity_level": "insufficient",
        "completeness_score": 0.0,
        "missing_fields": [],
        "warnings": [],
        "metadata": _as_dict(payload.get("metadata")),
    }

    datasets = _as_list_of_dicts(payload.get("datasets"))
    if not datasets:
        dataset_from_legacy = {
            "dataset_id": payload.get("dataset_id"),
            "source": payload.get("source_identity"),
            "row_count": payload.get("total_examples"),
            "license": payload.get("license"),
        }
        if payload.get("source_ids"):
            dataset_from_legacy["notes"] = f"source_ids={payload.get('source_ids')}"
        if any(value is not None for value in dataset_from_legacy.values()):
            datasets = [dataset_from_legacy]

    normalized_datasets = [_normalize_dataset_entry(item) for item in datasets]
    normalized["datasets"] = normalized_datasets

    if not normalized["mixture_weights"]:
        fallback_weights: dict[str, float] = {}
        if len(normalized_datasets) == 1:
            dataset_id = _as_str(normalized_datasets[0].get("dataset_id"))
            if dataset_id:
                fallback_weights[dataset_id] = 1.0
        normalized["mixture_weights"] = fallback_weights

    if not normalized["uses_synthetic_data"]:
        synthetic_hints = [
            _as_str(dataset.get("source")) or ""
            for dataset in normalized_datasets
        ]
        normalized["uses_synthetic_data"] = any("synthetic" in hint.lower() for hint in synthetic_hints)

    score, missing_fields, warnings, integrity = compute_manifest_completeness(normalized)
    normalized["completeness_score"] = score
    normalized["missing_fields"] = missing_fields
    normalized["warnings"] = warnings
    normalized["manifest_integrity_level"] = integrity
    return normalized


@dataclass(slots=True)
class DatasetManifest(JsonSchema):
    manifest_id: str
    run_id: str
    lineage_id: str
    stage_name: str | None = None
    created_at: str | None = None
    artifact_ref: str | None = None
    datasets: list[dict[str, object]] = field(default_factory=list)
    mixture_weights: dict[str, float] = field(default_factory=dict)
    sampling_policy: dict[str, object] = field(default_factory=dict)
    stage_data_policy_ref: str | None = None
    filtering_profile: dict[str, object] = field(default_factory=dict)
    preprocessing_profile: dict[str, object] = field(default_factory=dict)
    deduplication_profile: dict[str, object] = field(default_factory=dict)
    contamination_checks: dict[str, object] = field(default_factory=dict)
    chunking_policy: dict[str, object] = field(default_factory=dict)
    wrapper_policy: dict[str, object] = field(default_factory=dict)
    prompt_target_boundary_policy: dict[str, object] = field(default_factory=dict)
    tokenizer_ref: str | None = None
    tokenizer_compatibility: dict[str, object] = field(default_factory=dict)
    uses_synthetic_data: bool = False
    synthetic_data_profile: dict[str, object] = field(default_factory=dict)
    uses_hard_negatives: bool = False
    hard_negative_profile: dict[str, object] = field(default_factory=dict)
    uses_support_sets: bool = False
    support_set_profile: dict[str, object] = field(default_factory=dict)
    manifest_integrity_level: str = "insufficient"
    completeness_score: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(**normalize_dataset_manifest(payload))
