from __future__ import annotations

from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.safety_guard import SafetyGuardResult


def evaluate_dataset_manifest_guard(manifest: DatasetManifest | dict[str, object]) -> SafetyGuardResult:
    record = manifest.to_dict() if isinstance(manifest, DatasetManifest) else dict(manifest)
    reasons: list[str] = []
    datasets = record.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        reasons.append("missing_datasets")
    if record.get("manifest_integrity_level") == "insufficient":
        reasons.append("manifest_integrity_insufficient")
    if any("license" not in row or not row.get("license") for row in datasets if isinstance(row, dict)):
        reasons.append("dataset_license_missing")
    return SafetyGuardResult(
        guard_id="dataset_manifest_guard",
        passed=not reasons,
        severity="error" if reasons else "info",
        reasons=reasons,
        evidence_refs=[str(record.get("artifact_ref"))] if record.get("artifact_ref") else [],
        metadata={"manifest_id": record.get("manifest_id")},
    )
