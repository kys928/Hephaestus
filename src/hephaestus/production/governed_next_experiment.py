"""Approved branch/restart/rollback -> verified, lineage-bound next-run handoff.

The previous Judge cannot manufacture this receipt. The normal independent
GovernedActionExecutor must first admit the action using its durable external
approval and retained lineage/checkpoint facts. The handoff is a new science
identity, not a second application of the lineage mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json

_ACTIONS=frozenset({"branch_new_experiment","rollback_to_checkpoint","restart_lineage"})
_DIGEST=re.compile(r"sha256:[0-9a-f]{64}\Z")
_SLUG=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")


@dataclass(frozen=True, slots=True)
class GovernedNextExperiment:
    """Fail-closed materialization of the next scientific run ID and baseline."""

    maximum_transitions: int = 3

    def preflight(self, runtime, state, result) -> str:
        if result.judge_action not in _ACTIONS:
            raise ValueError("not a governed next-experiment action")
        if (type(result.run_id) is not str or not _SLUG.fullmatch(result.run_id)
                or type(result.experiment_id) is not str or not result.experiment_id
                or state.cycle_index>=self.maximum_transitions
                or result.approval_status not in {"approved","override_approved"}
                or not result.approval_ref):
            raise PermissionError("next experiment lacks exact run/approval/budget scope")
        checkpoint=result.checkpoint_ref
        if not checkpoint or not Path(checkpoint).is_absolute():
            raise PermissionError("next experiment needs an exact absolute checkpoint")
        from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
        valid,manifest_hash=validate_checkpoint_manifest(Path(checkpoint))
        if not valid:
            raise PermissionError("next experiment checkpoint manifest is not valid")
        # A source hash on disk alone is not trust-on-first-use.
        retained=CheckpointIdentityStore(runtime.state_repository).verify(
            state.lineage_id,checkpoint,manifest_hash)
        lineage=LineageStore(runtime.settings.state_root).get_current(state.lineage_id)
        if not isinstance(lineage,dict):
            raise PermissionError("no retained source lineage for approved transition")
        if checkpoint not in {
            lineage.get("best_checkpoint_ref"),
            lineage.get("last_stable_checkpoint_ref"),
            lineage.get("certified_stable_checkpoint_ref"),
            lineage.get("origin_checkpoint_ref"),
            lineage.get("branch_origin_checkpoint_ref"),
        }:
            raise PermissionError("next experiment checkpoint is not a retained lineage baseline")
        return retained

    def after_action(self, runtime, state, result, action_record,
                     *, checkpoint_manifest_sha256: str) -> dict[str,object]:
        if result.judge_action not in _ACTIONS:
            raise ValueError("only three governed transitions can create new experiments")
        if (action_record.get("status")!="applied"
                or action_record.get("effective_action")!=result.judge_action
                or action_record.get("lineage_id")!=state.lineage_id
                or action_record.get("run_id")!=result.run_id
                or action_record.get("approval_ref")!=result.approval_ref
                or action_record.get("checkpoint_ref")!=result.checkpoint_ref
                or action_record.get("approval_status") not in {"approved","override_approved"}
                or not action_record.get("authorization")
                or action_record.get("governance",{}).get("promotion_gate_applied") is not False
                or not _DIGEST.fullmatch(checkpoint_manifest_sha256)):
            raise PermissionError("no independently approved, applied lineage transition")
        before=action_record.get("before")
        after=action_record.get("after")
        if (not isinstance(before,dict) or not isinstance(after,dict)
                or before.get("lineage_id")!=state.lineage_id
                or after.get("lineage_id")!=state.lineage_id):
            raise PermissionError("approved transition lacks matching before/after lineage")
        action=result.judge_action
        target=state.lineage_id
        if action=="branch_new_experiment":
            child=action_record.get("child_after")
            if (not isinstance(child,dict) or
                    child.get("parent_lineage_id")!=state.lineage_id or
                    child.get("origin_run_id")!=result.run_id or
                    child.get("origin_checkpoint_ref")!=result.checkpoint_ref):
                raise PermissionError("committed child lineage does not match approved checkpoint")
            target=str(child.get("lineage_id") or "")
            stored=LineageStore(runtime.settings.state_root).get_current(target)
            if not _SLUG.fullmatch(target) or stored!=child:
                raise PermissionError("committed branch child cannot be independently reopened")
        elif after.get("best_checkpoint_ref")!=result.checkpoint_ref:
            raise PermissionError("rollback/restart did not bind the selected baseline")
        digest="sha256:"+sha256(canonical_bytes({
            "source_action_execution_id":action_record["execution_id"],
            "source_run_id":result.run_id,
            "source_experiment_id":result.experiment_id,
            "target_lineage_id":target,
            "baseline_ref":result.checkpoint_ref,
            "baseline_manifest_sha256":checkpoint_manifest_sha256,
            "approval_ref":result.approval_ref,
            "next_cycle_index":state.cycle_index+1,
        })).hexdigest()
        plan={
            "schema":"governed-next-experiment-handoff.v1",
            "source_action_execution_id":action_record["execution_id"],
            "source_run_id":result.run_id,
            "source_experiment_id":result.experiment_id,
            "source_cycle_id":result.cycle_id,
            "parent_lineage_id":state.lineage_id,
            "target_lineage_id":target,
            "stage_name":state.stage_name,
            "run_id":"next-run-"+digest[7:31],
            "baseline_checkpoint_ref":result.checkpoint_ref,
            "baseline_manifest_sha256":checkpoint_manifest_sha256,
            "approval_ref":result.approval_ref,
            "previous_action":action,
            "next_cycle_index":state.cycle_index+1,
            "training_authorized_by_handoff":False,
            "independent_scientific_certification":False,
        }
        receipt=runtime.artifact_store.put_bytes(canonical_bytes(plan),media_type="application/json")
        return {**plan,"handoff_ref":receipt.artifact_ref}

    def verify_resume(self, runtime, state, plan: dict, *, cycle_index: int) -> None:
        ref=plan.get("handoff_ref")
        if not isinstance(ref,str) or not _DIGEST.fullmatch(ref):
            raise PermissionError("continued scientific run lacks immutable handoff")
        committed=read_verified_json(runtime.artifact_store,ref)
        if committed!={k:v for k,v in plan.items() if k!="handoff_ref"}:
            raise PermissionError("the next experiment handoff was modified")
        if (committed["schema"]!="governed-next-experiment-handoff.v1"
                or committed["next_cycle_index"]!=cycle_index
                or committed["target_lineage_id"]!=state.lineage_id
                or committed["stage_name"]!=state.stage_name
                or committed["training_authorized_by_handoff"] is not False
                or not _SLUG.fullmatch(committed["run_id"])):
            raise PermissionError("resumed next experiment changed approved scientific identity")
        record=runtime.state_repository.get_latest(
            "production_action_executions","execution_id",
            committed["source_action_execution_id"])
        if (record is None or record.get("status")!="applied"
                or record.get("run_id")!=committed["source_run_id"]
                or record.get("effective_action")!=committed["previous_action"]
                or record.get("approval_ref")!=committed["approval_ref"]):
            raise PermissionError("next experiment lost its authorized source action")
