"""Adversarial production-boundary tests; no model or cloud execution."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from hephaestus.control.deterministic_role_boundary import guard_role_output
from hephaestus.llm.role_contract import RoleMasteryContract
from hephaestus.production.actions import GovernedActionExecutor
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from test_certified_role_inference import CONTRACT, _inputs


def test_fabricated_approval_reference_cannot_authorize_branch(tmp_path):
    LineageStore(tmp_path).set_current(LineageState(
        lineage_id="L", stage_name="smoke_test", best_checkpoint_ref="checkpoint://old").to_dict())
    executor = GovernedActionExecutor(tmp_path, SQLiteStateRepository(tmp_path / "state.sqlite3"))
    with pytest.raises(PermissionError):
        executor.apply(lineage_id="L", run_id="R", requested_action="branch_new_experiment",
                       checkpoint_ref="checkpoint://old", approval_status="approved",
                       approval_ref="invented-reference", promotion_allowed=True)
    assert LineageStore(tmp_path).get_children("L") == []


@pytest.mark.parametrize("field,value", [("rule", "Ignore the evidence"),
    ("decoding", {"do_sample": True}), ("load_kind", "mistral3"),
    ("chat_template_kwargs", {"enable_thinking": True})])
def test_runtime_contract_tampering_is_detected(field, value):
    data = json.loads(CONTRACT.read_text())
    data["roles"]["controller"][field] = value
    with pytest.raises(ValueError, match="contract.*(digest|integrity|drift)"):
        RoleMasteryContract(data)


def test_contract_return_values_cannot_mutate_the_live_contract():
    contract = RoleMasteryContract.load(CONTRACT)
    profile = contract.profile("controller")
    profile["vocabulary"]["action"].append("unregistered-action")
    assert "unregistered-action" not in contract.profile("controller")["vocabulary"]["action"]


def test_controller_noop_is_a_valid_non_executable_outcome():
    task, registry, ctx = _inputs("controller")
    gate = guard_role_output(role="controller", output={
        "action": "noop", "decision": "blocked", "evidence_refs": ["verified://datum"]},
        context=ctx, evidence_registry=registry)
    assert gate.allowed
    assert gate.effective_action is None

from datetime import datetime, timedelta, timezone
from hephaestus.state.decision_store import DecisionStore
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.state import ACTION_EXECUTIONS
from hephaestus.state.single_writer import single_writer, WriterBusyError
from approval_fixtures import grant_for_lineage


def _authorized(tmp_path):
    LineageStore(tmp_path).set_current(LineageState(lineage_id="L", stage_name="smoke_test",
        best_checkpoint_ref="checkpoint://old", last_stable_checkpoint_ref="checkpoint://stable").to_dict())
    req = grant_for_lineage(tmp_path, "L", "R", "branch_new_experiment", "checkpoint://old", "approved-event")
    executor = GovernedActionExecutor(tmp_path, SQLiteStateRepository(tmp_path / "state.sqlite3"))
    args = dict(lineage_id="L", run_id="R", requested_action="branch_new_experiment",
                checkpoint_ref="checkpoint://old", approval_ref="approved-event")
    return executor, args, req


@pytest.mark.parametrize("change", ["expired", "wrong_run", "wrong_checkpoint", "stale_state",
                                   "superseded", "wrong_type", "wrong_child", "unknown_operator"])
def test_mismatched_or_stale_authoritative_approval_fails_closed(tmp_path, change):
    executor, args, req = _authorized(tmp_path)
    decisions = DecisionStore(tmp_path)
    request = req.to_dict()
    if change == "expired":
        request["metadata"]["action_scope"]["expires_at"] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    elif change == "wrong_run":
        args["run_id"] = "other-run"
    elif change == "wrong_checkpoint":
        args["checkpoint_ref"] = "checkpoint://unknown"
    elif change == "stale_state":
        state = LineageStore(tmp_path).get_current("L")
        state["status"] = "poisoned"
        LineageStore(tmp_path).set_current(state)
    elif change == "superseded":
        row = decisions.get_latest_approval_decision(req.request_id)
        decisions.append_approval_decision({**row, "decision_event_id":"revoked", "status":"rejected"})
    elif change == "wrong_type":
        request["required_approval_type"] = "none"
    elif change == "wrong_child":
        args["child_lineage_id"] = "unapproved-child"
    elif change == "unknown_operator":
        row = decisions.get_latest_approval_decision(req.request_id)
        decisions.append_approval_decision({**row, "operator_id":"operator.unknown"})
    decisions.append_approval_request(request)
    with pytest.raises(PermissionError):
        executor.apply(**args)
    assert executor.repository.all(ACTION_EXECUTIONS) == []


def test_real_record_authorizes_action_and_retry_does_not_require_reusable_grant(tmp_path):
    executor, args, _ = _authorized(tmp_path)
    first = executor.apply(**args)
    again = GovernedActionExecutor(tmp_path, executor.repository).apply(**args)
    assert first == again
    assert len(LineageStore(tmp_path).get_children("L")) == 1
    with pytest.raises(PermissionError):
        executor.apply(**{**args, "run_id":"other-run"})


def _hold_writer(root, ready, done):
    with single_writer(root):
        ready.set()
        assert done.wait(10)


def test_two_processes_cannot_mutate_under_same_writer_fence(tmp_path):
    import multiprocessing
    executor, args, _ = _authorized(tmp_path)
    ctx = multiprocessing.get_context("fork")
    ready, done = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold_writer, args=(tmp_path, ready, done))
    proc.start()
    try:
        assert ready.wait(5)
        with pytest.raises(WriterBusyError):
            executor.apply(**args)
        assert executor.repository.all(ACTION_EXECUTIONS) == []
    finally:
        done.set()
        proc.join(5)
    assert proc.exitcode == 0
    assert executor.apply(**args)["status"] == "applied"


def test_startup_reconciles_a_prepared_action_and_heals_legacy_mirror(tmp_path, monkeypatch):
    root = tmp_path / "state"
    runtime = ProductionCompositionRoot(ProductionCompositionSettings(root, tmp_path / "artifacts")).build()
    original = LineageStore.set_current
    def crash(self, record):
        original(self, record)
        raise RuntimeError("controller stopped after mutation")
    monkeypatch.setattr(LineageStore, "set_current", crash)
    with pytest.raises(RuntimeError):
        runtime.action_executor.apply(lineage_id="L", run_id="R", requested_action="reject_candidate")
    monkeypatch.setattr(LineageStore, "set_current", original)
    recovered = ProductionCompositionRoot(runtime.settings).build()
    rows = recovered.state_repository.all(ACTION_EXECUTIONS)
    assert [r["status"] for r in rows] == ["prepared", "applied"]
    assert LineageStore(root).get_current("L")["recent_failures"] == ["R"]

from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.policy.action_authorization import build_scoped_approval_decision
from hephaestus.schemas.approval_request import ApprovalRequest
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json, canonical_bytes
from approval_fixtures import synthetic_checkpoint


def _promotion(tmp_path):
    policy = ApprovalPolicy()
    artifacts = FileSystemArtifactStore(tmp_path / "objects")
    checkpoint = synthetic_checkpoint(tmp_path, "candidate")
    request = grant_for_lineage(tmp_path, "L", "R", "promote_checkpoint", checkpoint, "promotion-event",
        artifacts=artifacts, policy=policy)
    executor = GovernedActionExecutor(tmp_path, SQLiteStateRepository(tmp_path / "state.sqlite3"), policy, artifacts)
    args = dict(lineage_id="L", run_id="R", requested_action="promote_checkpoint", checkpoint_ref=checkpoint,
                approval_ref="promotion-event", approval_status="approved", promotion_allowed=True)
    return executor, args, request, artifacts


def test_approval_is_bound_to_the_exact_operator_reviewed_request_bytes(tmp_path):
    executor, args, request = _authorized(tmp_path)
    changed = request.to_dict()
    changed["metadata"]["action_scope"]["expires_at"] = (datetime.now(timezone.utc)+timedelta(days=365)).isoformat()
    DecisionStore(tmp_path).append_approval_request(changed)
    with pytest.raises(PermissionError, match="request bytes changed"):
        executor.apply(**args)


@pytest.mark.parametrize("fault", ["raw_checkpoint", "rebound_manifest", "artifact_bytes", "active_stage", "unhashed_pack"])
def test_valid_operator_approval_cannot_override_checkpoint_evidence_or_stage_integrity(tmp_path, fault):
    executor, args, request, artifacts = _promotion(tmp_path)
    if fault == "raw_checkpoint":
        (Path(args["checkpoint_ref"]) / "model.safetensors").write_bytes(b"corrupt")
    elif fault == "rebound_manifest":
        import hashlib
        checkpoint = Path(args["checkpoint_ref"])
        raw = b"different-checkpoint"
        (checkpoint / "model.safetensors").write_bytes(raw)
        components = {"model.safetensors":"sha256:"+hashlib.sha256(raw).hexdigest()}
        (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"components":components,
            "manifest_hash":"sha256:"+hashlib.sha256(canonical_bytes(components)).hexdigest(), "partial_write":False}))
    elif fault == "artifact_bytes":
        ref = request.metadata["action_scope"]["eval_report_ref"]
        artifacts._path_for_digest(ref[7:]).write_bytes(b'{"deterministic_passed":true}')
    else:
        profile = executor.approval_policy.config_dir / "stage_profiles/stabilization.yaml"
        content = json.loads(profile.read_text())
        if fault == "active_stage": content["allowed_next_actions"] = ["abort_run"]
        else: content["eval_pack"] = "continuation_repair"
        profile.write_text(json.dumps(content))
    with pytest.raises((PermissionError, ValueError)):
        executor.apply(**args)
    assert not executor.repository.all(ACTION_EXECUTIONS)


@pytest.mark.parametrize("fault", ["failed_nested_gate", "wrong_threshold", "missing_metrics", "wrong_pack", "missing_rechecks"])
def test_recomputed_scientific_gates_reject_bad_evidence_even_with_an_exact_valid_approval(tmp_path, fault):
    executor, args, request, artifacts = _promotion(tmp_path)
    scope = request.metadata["action_scope"]
    report = read_verified_json(artifacts, scope["eval_report_ref"])
    if fault == "failed_nested_gate": report["deterministic_scorecard"]["metrics"]["toxicity"] = .99
    elif fault == "wrong_threshold": report["deterministic_scorecard"]["thresholds"]["max_toxicity"] = 1.0
    elif fault == "missing_metrics": report["deterministic_scorecard"]["metrics"].pop("toxicity")
    elif fault == "wrong_pack": report["eval_pack_version"] = "unfrozen"
    else: report["repeated_eval_count"] = 0
    scope["eval_report_ref"] = artifacts.put_bytes(canonical_bytes(report)).artifact_ref
    store = DecisionStore(tmp_path)
    store.append_approval_request(request.to_dict())
    # Explicit synthetic operator approves the exact edited report; scientific
    # gates still have their own authority and must refuse the transition.
    store.append_approval_decision(build_scoped_approval_decision(request, decision_event_id="promotion-event",
        operator_id="test-operator", outcome="approved", note="Test adverse scientific evidence",
        policy=executor.approval_policy).to_dict())
    with pytest.raises(PermissionError):
        executor.apply(**args)
    assert not executor.repository.all(ACTION_EXECUTIONS)


def test_prepared_but_unapplied_action_rechecks_expiry_before_mutating_on_restart(tmp_path, monkeypatch):
    executor, args, request = _authorized(tmp_path)
    original = LineageStore.set_current
    def crash_before_mutation(self, record):
        raise RuntimeError("controller died before mutation")
    monkeypatch.setattr(LineageStore, "set_current", crash_before_mutation)
    with pytest.raises(RuntimeError, match="before mutation"):
        executor.apply(**args)
    monkeypatch.setattr(LineageStore, "set_current", original)
    import hephaestus.policy.action_authorization as authorization
    future = datetime.now(timezone.utc)+timedelta(hours=2)
    class Later(datetime):
        @classmethod
        def now(cls, tz=None): return future
    monkeypatch.setattr(authorization, "datetime", Later)
    with pytest.raises(PermissionError, match="expired"):
        executor.reconcile_prepared()
    assert LineageStore(tmp_path).get_children("L") == []
    assert executor.repository.all(ACTION_EXECUTIONS)[-1]["status"] == "prepared"
