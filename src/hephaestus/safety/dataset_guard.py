"""Dataset safety checks for acquisition and preprocessing boundaries."""

from __future__ import annotations

from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.safety._helpers import mapping, result, text


def check_dataset_manifest(inp: SafetyGuardInput) -> SafetyGuardResult:
    manifest = DatasetManifest.from_dict(inp.payload)
    reasons: list[str] = []
    warnings = list(manifest.warnings)
    if manifest.run_id != inp.run_id:
        reasons.append("run_id_mismatch")
    if manifest.lineage_id != inp.lineage_id:
        reasons.append("lineage_id_mismatch")
    if manifest.manifest_integrity_level in {"insufficient", "reference_only"}:
        reasons.append(f"manifest_integrity_{manifest.manifest_integrity_level}")
    if manifest.uses_synthetic_data and not mapping(manifest.synthetic_data_profile):
        warnings.append("synthetic_data_profile_missing")
    if not manifest.stage_data_policy_ref:
        warnings.append("stage_data_policy_ref_missing")
    return result(inp, reasons, warnings, {"manifest_id": manifest.manifest_id, "integrity_level": manifest.manifest_integrity_level})


def check_trainable_data_contract(inp: SafetyGuardInput) -> SafetyGuardResult:
    contract = TrainableDataContract.from_dict(inp.payload)
    reasons: list[str] = []
    if contract.run_id != inp.run_id:
        reasons.append("run_id_mismatch")
    if not text(contract.manifest_id):
        reasons.append("manifest_id_missing")
    if not text(contract.processed_dataset_ref):
        reasons.append("processed_dataset_ref_missing")
    if int(contract.min_tokens) <= 0:
        reasons.append("min_tokens_non_positive")
    return result(inp, reasons, metadata={"contract_id": contract.contract_id, "min_tokens": contract.min_tokens})
