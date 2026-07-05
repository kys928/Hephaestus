"""Dataset acquisition boundary for the Data Acquisition & Audit role."""

from __future__ import annotations

from hephaestus.backends.base import ExecutionBackend
from hephaestus.data.audit import audit_manifest, build_dataset_profile
from hephaestus.data.manifest_builder import build_dataset_manifest
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.dataset_profile import DatasetProfile


def acquire_and_audit_dataset(
    *,
    backend: ExecutionBackend,
    run_id: str,
    lineage_id: str,
    stage_name: str | None = None,
) -> tuple[DatasetProfile, DatasetManifest]:
    """Acquire backend metadata and emit explicit schema-backed outputs."""
    acquired = dict(backend.acquire_dataset(run_id))
    profile = build_dataset_profile(acquired)
    manifest = build_dataset_manifest(run_id=run_id, lineage_id=lineage_id, stage_name=stage_name, acquired=acquired)
    return profile, audit_manifest(manifest)
