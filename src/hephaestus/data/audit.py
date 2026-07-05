"""Schema-backed audit helpers for acquired datasets."""

from __future__ import annotations

from hephaestus.data.registry import as_ref, as_str_list
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.dataset_profile import DatasetProfile


def build_dataset_profile(acquired: dict[str, object]) -> DatasetProfile:
    risks = as_str_list(acquired.get("risks"))
    try:
        quality_score = float(acquired.get("quality_score", 0.0))
    except (TypeError, ValueError):
        quality_score = 0.0
    return DatasetProfile(
        dataset_id=as_ref(acquired.get("dataset_id")) or "unknown-dataset",
        source_identity=as_ref(acquired.get("source_identity") or acquired.get("source")) or "unknown-source",
        license=as_ref(acquired.get("license")) or "unknown",
        quality_score=max(0.0, min(1.0, quality_score)),
        risks=risks,
    )


def audit_manifest(manifest: DatasetManifest) -> DatasetManifest:
    """Return a normalized manifest with schema-computed completeness fields."""
    return DatasetManifest.from_dict(manifest.to_dict())
