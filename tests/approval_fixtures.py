"""Synthetic authoritative approval/evaluation records for CPU integration tests."""
from datetime import datetime, timedelta, timezone
import json
import hashlib
import shutil
from pathlib import Path
from hephaestus.schemas.scorecard import Scorecard
from hephaestus.evaluation.pack_loader import load_eval_pack

from hephaestus.policy.action_authorization import build_action_approval_request, build_scoped_approval_decision
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.approval_decision import ApprovalDecision
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.run_store import RunStore
from hephaestus.storage.filesystem import FileSystemArtifactStore


def grant_action(root, run, action, checkpoint, ref, *, artifacts=None, child=None, policy=None):
    """Explicit operator test double, never used by production code."""
    return grant_for_lineage(root, "lineage-main", run, action, checkpoint, ref,
                             artifacts=artifacts, child=child, policy=policy)


def grant_for_lineage(root, lineage, run, action, checkpoint, ref, *, artifacts=None, child=None, policy=None):
    current = LineageStore(root).get_current(lineage)
    if current is None:
        current = LineageState(lineage_id=lineage, stage_name="smoke_test").to_dict()
        LineageStore(root).set_current(current)
    policy = policy or ApprovalPolicy()
    evaluation_ref = manifest_ref = None
    if action == "promote_checkpoint":
        assert artifacts is not None
        # A test-only approved profile uses the existing cryptographically frozen
        # semantic pack. Production stage/eval policy files are never edited.
        test_config = root / "test-approved-config"
        shutil.copytree("configs", test_config, dirs_exist_ok=True)
        profile_path = test_config / "stage_profiles/stabilization.yaml"
        profile = json.loads(profile_path.read_text())
        profile["eval_pack"] = "semantic_behavior_v1"
        profile_path.write_text(json.dumps(profile))
        policy.config_dir = test_config
        current["stage_name"] = "stabilization"
        current["status"] = "active"
        LineageStore(root).set_current(current)
        pack = load_eval_pack("semantic_behavior_v1", config_dir=test_config)
        scorecard = Scorecard("scorecard-"+run, run, pack["eval_pack_id"], pack["eval_pack_version"], checkpoint,
            deterministic_passed=True, metrics={"probe_score":1.0,"toxicity":0.0},
            thresholds=profile["deterministic_gates"],
            gate_results={"probe_score_gate":{"passed":True},"toxicity_gate":{"passed":True}})
        report = EvalReport(eval_id="eval-"+run, run_id=run, stage_name=current["stage_name"],
            pack_name="semantic_behavior_v1", checkpoint_resolution={"selected_checkpoint_ref": checkpoint},
            confidence=.99, evidence_completeness=1., stability_confidence=.99,
            certification_readiness="certification_passed", promotion_bundle_passed=True,
            observed_consistent_runs=3, repeated_eval_count=3, consistency_score=1., repeatability_ready=True,
            repeatability_sufficient=True, variance_risk="low", eval_pack_id=pack["eval_pack_id"],
            eval_pack_version=pack["eval_pack_version"], eval_pack_integrity_level="content_hash_verified",
            deterministic_scorecard=scorecard.to_dict(), deterministic_passed=True,
            scorecard_integrity_level="content_hash_verified")
        evaluation_ref = artifacts.put_bytes(json.dumps(report.to_dict()).encode()).artifact_ref
        manifest_ref = artifacts.put_bytes(json.dumps({"manifest_id":"test-synthetic",
            "manifest_integrity_level":"content_hash_verified", "completeness_score":1.}).encode()).artifact_ref
        RunStore(root).append({"run_id":run, "lineage_id":lineage, "checkpoint_ref":checkpoint})
    request = build_action_approval_request(request_id="req-"+ref, current=current, run_id=run,
        action=action, checkpoint_ref=checkpoint, child_lineage_id=child,
        expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        eval_report_ref=evaluation_ref, data_manifest_ref=manifest_ref, policy=policy)
    decisions = DecisionStore(root)
    decisions.append_approval_request(request.to_dict())
    decisions.append_approval_decision(build_scoped_approval_decision(request, decision_event_id=ref,
        operator_id="synthetic-test-operator", outcome="approved", note="Explicit synthetic operator grant",
        policy=policy).to_dict())
    return request


def synthetic_checkpoint(root, name):
    """A real manifest around explicitly synthetic bytes, never trained weights."""
    checkpoint = root / "test-checkpoints" / name
    if (checkpoint / "checkpoint_manifest.json").exists():
        return str(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    raw = b"synthetic-checkpoint-for-CPU-governance-tests"
    (checkpoint / "model.safetensors").write_bytes(raw)
    components = {"model.safetensors":"sha256:"+hashlib.sha256(raw).hexdigest()}
    manifest_hash = "sha256:"+hashlib.sha256(json.dumps(components,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"components":components,
        "manifest_hash":manifest_hash,"partial_write":False}))
    return str(checkpoint)
