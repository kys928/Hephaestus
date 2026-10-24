"""Synthetic authoritative approval/evaluation records for CPU integration tests."""
from datetime import datetime, timedelta, timezone
import json

from hephaestus.policy.action_authorization import build_action_approval_request
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.approval_decision import ApprovalDecision
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.run_store import RunStore
from hephaestus.storage.filesystem import FileSystemArtifactStore


def grant_action(root, run, action, checkpoint, ref, *, artifacts=None, child=None):
    """Explicit operator test double, never used by production code."""
    return grant_for_lineage(root, "lineage-main", run, action, checkpoint, ref,
                             artifacts=artifacts, child=child)


def grant_for_lineage(root, lineage, run, action, checkpoint, ref, *, artifacts=None, child=None):
    current = LineageStore(root).get_current(lineage)
    if current is None:
        current = LineageState(lineage_id=lineage, stage_name="smoke_test").to_dict()
        LineageStore(root).set_current(current)
    evaluation_ref = manifest_ref = None
    if action == "promote_checkpoint":
        assert artifacts is not None
        report = EvalReport(eval_id="eval-"+run, run_id=run, stage_name=current["stage_name"],
            pack_name="test-synthetic", checkpoint_resolution={"selected_checkpoint_ref": checkpoint},
            confidence=.95, evidence_completeness=1., stability_confidence=.95,
            certification_readiness="certification_passed", promotion_bundle_passed=True,
            observed_consistent_runs=3, repeated_eval_count=3, repeatability_ready=True,
            repeatability_sufficient=True, variance_risk="low", eval_pack_id="test-pack-v1",
            eval_pack_version="1", eval_pack_integrity_level="content_hash_verified",
            deterministic_scorecard={"passed": True}, deterministic_passed=True,
            scorecard_integrity_level="content_hash_verified")
        evaluation_ref = artifacts.put_bytes(json.dumps(report.to_dict()).encode()).artifact_ref
        manifest_ref = artifacts.put_bytes(json.dumps({"manifest_id":"test-synthetic",
            "manifest_integrity_level":"content_hash_verified", "completeness_score":1.}).encode()).artifact_ref
        RunStore(root).append({"run_id":run, "lineage_id":lineage, "checkpoint_ref":checkpoint})
    request = build_action_approval_request(request_id="req-"+ref, current=current, run_id=run,
        action=action, checkpoint_ref=checkpoint, child_lineage_id=child,
        expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        eval_report_ref=evaluation_ref, data_manifest_ref=manifest_ref, policy=ApprovalPolicy())
    decisions = DecisionStore(root)
    decisions.append_approval_request(request.to_dict())
    decisions.append_approval_decision(ApprovalDecision(ref, request.request_id, lineage, run,
        "synthetic-test-operator", "approved", "approved", "Explicit synthetic operator grant",
        "execute_requested_action", datetime.now(timezone.utc).isoformat()).to_dict())
    return request
