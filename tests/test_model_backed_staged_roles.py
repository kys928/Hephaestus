"""Staged integration calls all five role-model pathways with explicit CPU test doubles.

This exercises the real staged orchestrator, verified S3 adapter and contract,
and deterministic boundary; it is NOT a claim that the real 4B-8B weights
have run in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.staged_state import StagedAutonomousDependencies, WORKFLOW_RECORD_COLLECTION
from hephaestus.llm.role_contract import RoleTask, RoleEvidence, RoleMasteryContract
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner, VerifiedRoleAdapters
from hephaestus.production.model_roles import with_model_backed_roles
from hephaestus.storage.filesystem import JsonLineStateRepository

from test_certified_role_inference import _archive, _roles, FakeS3
from test_staged_autonomous_orchestration import FakeOperationRouter, _services


ROOT = Path(__file__).resolve().parents[1]


class BoundForRun:
    def bind(self, *, role, request):
        task = RoleTask(
            role=role,
            situation="Assess the phase evidence; do not invent machine facts.",
            evidence=(RoleEvidence("verified://run-evidence", "The deterministic service completed its stage."),),
            run_id=request.run_id, lineage_id=request.lineage_id,
            stage_name=request.stage_name,
        )
        registry = EvidenceRegistry({
            "verified://run-evidence": EvidenceRecord(
                ref="verified://run-evidence", kind="completed-stage",
                run_id=request.run_id, lineage_id=request.lineage_id,
                sha256="a"*64, observed_sha256="a"*64,
            )
        })
        context = DeterministicRoleContext(
            run_id=request.run_id, lineage_id=request.lineage_id,
            stage_name=request.stage_name, approval_status="not_required",
            stage_allowed_actions=("continue_lineage_best", "request_recheck",
                                   "reject_candidate"),
        )
        return task, registry, context


class ModelProposals:
    def __init__(self, *, judge_action="continue_lineage_best", corrupt_role=None):
        self.calls = []
        self.judge_action = judge_action
        self.corrupt_role = corrupt_role

    def generate(self, *, spec, profile, adapter_path, prompt, max_new_tokens, seed):
        assert (adapter_path / "adapter_model.safetensors").is_file()
        assert profile["rule"] in prompt
        self.calls.append(spec.role)
        if spec.role == self.corrupt_role:
            return "the model answered without a JSON decision"
        v = profile["vocabulary"]
        action = ("request_recheck" if spec.role == "controller" else
                  self.judge_action if spec.role == "judge" else v["action"][0])
        return json.dumps({
            "decision": v["decision"][0],
            "action": action,
            "primary_variable": v["primary_variable"][0],
            "confidence": 0.7,
            "evidence_refs": ["verified://run-evidence"],
            "uncertainties": [],
            "rationale": "The validated evidence supports this review only.",
        })


def _run(tmp_path, model):
    router = FakeOperationRouter(action="continue_lineage_best")
    blob = _archive()
    s3 = FakeS3(blob)
    reasoner = CertifiedRoleReasoner(
        _roles(blob),
        RoleMasteryContract.load(ROOT / "configs/models/hephaestus_role_mastery_runtime_v1.json"),
        VerifiedRoleAdapters(s3, "read-only-source", tmp_path / "cache"),
        model,
    )
    services = with_model_backed_roles(_services(router), reasoner, BoundForRun(), enabled=True)
    repository = JsonLineStateRepository(tmp_path / "state")
    deps = StagedAutonomousDependencies(services=services, state_repository=repository)
    orchestrator = build_orchestrator(
        state_root=tmp_path / "legacy-unused", run_id="role-model-1",
        lineage_id="model-lineage", stage_name="smoke_test",
        mode="governed_autonomous", staged_dependencies=deps,
    )
    return orchestrator.run("role-model-1"), repository, model, s3


def test_real_staged_orchestrator_invokes_all_five_explicit_model_roles(tmp_path):
    state, repo, model, s3 = _run(tmp_path, ModelProposals())
    assert state.status == "completed"
    assert model.calls == ["controller", "diagnosis", "planner", "evaluator", "judge"]
    assert len(s3.gets) == 5
    rows = [r for r in repo.all(WORKFLOW_RECORD_COLLECTION) if r.get("kind") == "model_role_proposal"]
    assert len(rows) == 5
    assert {r["payload"]["role"] for r in rows} == set(model.calls)
    assert all(r["payload"]["advisory_only"] is True for r in rows)


def test_model_judge_disagreement_blocks_before_staged_action(tmp_path):
    state, repo, model, s3 = _run(tmp_path, ModelProposals(judge_action="reject_checkpoint"))
    assert state.status == "blocked"
    assert "model_judge_disagreed_with_governed_verdict" in state.blocking_issues
    assert not any(
        r.get("substep") == "action_application" and r.get("kind") == "action_decision"
        for r in repo.all(WORKFLOW_RECORD_COLLECTION)
    )


def test_malformed_model_response_blocks_and_does_not_fallback(tmp_path):
    state, repo, model, s3 = _run(tmp_path, ModelProposals(corrupt_role="planner"))
    assert state.status == "terminal_failure"
    assert model.calls == ["controller", "diagnosis", "planner"]
    assert not any(
        r.get("kind") == "action_decision"
        for r in repo.all(WORKFLOW_RECORD_COLLECTION)
    )


def test_explicit_enable_required(tmp_path):
    router = FakeOperationRouter()
    blob = _archive()
    reasoner = CertifiedRoleReasoner(
        _roles(blob),
        RoleMasteryContract.load(ROOT / "configs/models/hephaestus_role_mastery_runtime_v1.json"),
        VerifiedRoleAdapters(FakeS3(blob), "volume", tmp_path), ModelProposals(),
    )
    with pytest.raises(PermissionError, match="explicit"):
        with_model_backed_roles(_services(router), reasoner, BoundForRun())
