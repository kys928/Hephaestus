"""Launch safety checks for runtime job preparation."""

from __future__ import annotations

from hephaestus.schemas.launch_config import LaunchConfig
from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult
from hephaestus.schemas.training_plan import TrainingPlan
from hephaestus.safety._helpers import mapping, result, text


def check_launch_request(inp: SafetyGuardInput) -> SafetyGuardResult:
    launch = LaunchConfig.from_dict(mapping(inp.payload.get("launch_config")) or inp.payload)
    plan_payload = mapping(inp.payload.get("training_plan"))
    plan = TrainingPlan.from_dict(plan_payload) if plan_payload else None
    data_contract = mapping(inp.payload.get("data_contract"))
    reasons: list[str] = []
    warnings: list[str] = []
    if launch.run_id != inp.run_id:
        reasons.append("launch_run_id_mismatch")
    if not text(launch.artifact_root):
        reasons.append("artifact_root_missing")
    if not launch.dry_run and not text(launch.parameters.get("backend_profile")):
        reasons.append("backend_profile_missing_for_live_launch")
    if not text(launch.parameters.get("processed_dataset_ref")):
        reasons.append("processed_dataset_ref_missing")
    if plan:
        if plan.run_id != inp.run_id:
            reasons.append("training_plan_run_id_mismatch")
        if plan.max_steps <= 0 or plan.eval_every_steps <= 0 or plan.checkpoint_every_steps <= 0:
            reasons.append("non_positive_training_cadence")
        if plan.eval_every_steps > plan.max_steps:
            warnings.append("eval_cadence_exceeds_max_steps")
        if plan.checkpoint_every_steps > plan.max_steps:
            warnings.append("checkpoint_cadence_exceeds_max_steps")
    if data_contract and text(data_contract.get("processed_dataset_ref")) != text(launch.parameters.get("processed_dataset_ref")):
        reasons.append("launch_dataset_ref_mismatch")
    return result(inp, reasons, warnings, {"launch_id": launch.launch_id, "backend": launch.backend, "dry_run": launch.dry_run})
