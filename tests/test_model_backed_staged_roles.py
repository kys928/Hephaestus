"""Staged integration calls all five role-model pathways with explicit CPU test doubles.

This exercises the real staged orchestrator, verified S3 adapter and contract,
and deterministic boundary; it is NOT a claim that the real 4B-8B weights
have run in CI.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from hephaestus.llm.inference_receipts import RoleGeneration
from hephaestus.storage.filesystem import FileSystemArtifactStore

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.control.staged_state import StagedAutonomousDependencies, WORKFLOW_RECORD_COLLECTION
from hephaestus.llm.role_contract import RoleTask, RoleEvidence, RoleMasteryContract
from hephaestus.llm.certified_runtime import CertifiedRoleReasoner, VerifiedRoleAdapters
from hephaestus.production.model_roles import with_model_backed_roles
from hephaestus.production.evidence import ArtifactModelEvidenceBinder
from hephaestus.state.evidence_store import ScientificEvidenceStore
from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
from hephaestus.schemas.diagnosis_contract import DiagnosticHypothesis
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.storage.verified import canonical_bytes
from hephaestus.control.staged_state import StagedOutputRecord
from test_closed_loop_planner import _diagnosis
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
            return RoleGeneration("the model answered without a JSON decision", metadata={"test_double":True})
        v = profile["vocabulary"]
        action = ("request_recheck" if spec.role == "controller" else
                  self.judge_action if spec.role == "judge" else v["action"][0])
        decision = v["decision"][0]
        primary = v["primary_variable"][0]
        if spec.role == "diagnosis":
            decision, action, primary = "evaluation_integrity", "repair_evaluation", "evaluation_protocol"
        refs = re.findall(r"^- (sha256:[0-9a-f]{64}):", prompt, re.M)
        return RoleGeneration(json.dumps({
            "decision": decision,
            "action": action,
            "primary_variable": primary,
            "confidence": 0.7,
            "evidence_refs": refs[:1],
            "uncertainties": [],
            "rationale": "The validated evidence supports this review only.",
        }), metadata={"test_double":True})


def _run(tmp_path, model):
    repository = JsonLineStateRepository(tmp_path / "state")
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    initial = artifacts.put_bytes(canonical_bytes({"run_id":"role-model-1", "lineage_id":"model-lineage",
        "stage_name":"smoke_test", "kind":"eval_report", "payload":{"status":"failed", "eval_integrity_verified":True}}))
    evidence_store = ScientificEvidenceStore(repository, artifacts)
    evidence_store.register(ScientificEvidenceBinding("initial", "role-model-1", "role-model-1", "model-lineage",
        "smoke_test", "eval_report", initial.artifact_ref))

    class ScientificRouter(FakeOperationRouter):
        def execute(self, request):
            result = super().execute(request)
            if request.substep == "evidence_based_diagnosis":
                report = _diagnosis(hypotheses=[DiagnosticHypothesis("hyp-eval", "evaluation_integrity",
                    "Inspect the completed frozen comparison.", supporting_evidence_refs=[initial.artifact_ref],
                    recommended_intervention_kinds=["collect_more_evidence", "repair_evaluation"], confidence=.8)])
                report.run_id, report.lineage_id, report.stage_name = request.run_id, request.lineage_id, request.stage_name
                return replace(result, records=(StagedOutputRecord("diagnosis_report", report.to_dict()),))
            return result
    router = ScientificRouter(action="continue_lineage_best")
    blob = _archive()
    s3 = FakeS3(blob)
    reasoner = CertifiedRoleReasoner(_roles(blob),
        RoleMasteryContract.load(ROOT / "configs/models/hephaestus_role_mastery_runtime_v1.json"),
        VerifiedRoleAdapters(s3, "read-only-source", tmp_path / "cache"), model, artifacts)
    services = with_model_backed_roles(_services(router), reasoner, ArtifactModelEvidenceBinder(evidence_store),
                                      enabled=True, planner=ClosedLoopExperimentPlanner())
    deps = StagedAutonomousDependencies(services=services, state_repository=repository, artifact_store=artifacts)
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
    assert {r["payload"]["role"] for r in rows if r["payload"]["advisory_only"] is False} == {"planner", "diagnosis"}
    proposals = [r["payload"] for r in repo.all(WORKFLOW_RECORD_COLLECTION) if r["kind"] == "experiment_proposal"]
    assert proposals[0]["metadata"]["model_originated"] is True
    assert proposals[0]["primary_variable"] == "diagnostic_measurement"


def test_model_judge_disagreement_blocks_before_staged_action(tmp_path):
    state, repo, model, s3 = _run(tmp_path, ModelProposals(judge_action="rerun_same_config"))
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
        FileSystemArtifactStore(tmp_path / "receipts"),
    )
    with pytest.raises(PermissionError, match="explicit"):
        with_model_backed_roles(_services(router), reasoner, BoundForRun())
