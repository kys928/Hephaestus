"""Production-owned early-stage smoke integration; no invented science workers."""
from __future__ import annotations

import pytest

from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import (
    StagedAutonomousDependencies, StagedOperationRequest, WORKFLOW_RECORD_COLLECTION,
)
from hephaestus.production.composition import (
    ProductionCompositionRoot, ProductionCompositionSettings,
)
from hephaestus.production.staged_core import make_staged_core_services
from hephaestus.storage.verified import canonical_bytes, read_verified_json


def _runtime(tmp_path):
    return ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path / "state", artifact_root=tmp_path / "artifacts")).build()


def _source(rt, *, complete=True, wrong_run=False):
    rows = [
        {"evidence_kind": "eval_report",
         "eval_pack_integrity_level": "content_hash_verified",
         "scorecard_integrity_level": "content_hash_verified",
         "deterministic_scorecard": {"gate_results": {"probe": {"passed": True}}}},
        {"evidence_kind": "run_record", "status": "completed"},
        {"evidence_kind": "replay_verification", "status": "reproducible",
         "confidence": 0.95},
        {"evidence_kind": "dataset_manifest",
         "manifest_integrity_level": "complete",
         "tokenizer_compatibility": {"status": "compatible"},
         "wrapper_policy": {"status": "compatible"}},
        {"evidence_kind": "checkpoint_evidence", "checkpoint_verified": True},
        {"evidence_kind": "training_metrics", "optimizer_stable": True,
         "numerically_stable": True, "training_sufficient": True},
        {"evidence_kind": "data_audit", "data_quality_failed": True},
    ]
    if not complete:
        rows = rows[:1]
    references = [
        rt.artifact_store.put_bytes(canonical_bytes({
            **row, "run_id": "real-early-1", "lineage_id": "lineage-1"
        }), media_type="application/json").artifact_ref
        for row in rows
    ]
    return rt.artifact_store.put_bytes(canonical_bytes({
        "schema": "staged-diagnosis-input.v1",
        "run_id": "foreign-run" if wrong_run else "real-early-1",
        "lineage_id": "lineage-1", "stage_name": "smoke_test",
        "evidence_refs": references,
    }), media_type="application/json").artifact_ref


def _request(substep, prior=None):
    return StagedOperationRequest(
        workflow_id="workflow-real-early-1", run_id="real-early-1",
        lineage_id="lineage-1", stage_name="smoke_test",
        phase="judge_entry", substep=substep,
        operation_id="operation-" + substep, attempt=1,
        input_refs=(), prior_outputs=prior or {},
    )


def test_real_diagnosis_and_planner_handoff_execute_from_sha_bound_sources(tmp_path):
    rt = _runtime(tmp_path)
    source_ref = _source(rt)
    svc = make_staged_core_services(rt, diagnosis_request_ref=source_ref)
    # The Controller is the FIRST operation. Its model binder must see an
    # immutable verified source before any staged records exist.
    task, registry, context = rt.model_evidence.bind(
        role="controller", request=_request("judge_entry_decision"))
    assert task.evidence and registry.validate(
        [item.ref for item in task.evidence], run_id=task.run_id,
        lineage_id=task.lineage_id).passed
    assert any("staged_diagnosis_source_manifest" in item.fact for item in task.evidence)
    deps = StagedAutonomousDependencies(
        services=svc, state_repository=rt.state_repository,
        artifact_store=rt.artifact_store,
    )
    orchestrator = build_staged_autonomous_orchestrator(
        workflow_id="workflow-real-early-1", run_id="real-early-1",
        lineage_id="lineage-1", stage_name="smoke_test", dependencies=deps)
    workflow = orchestrator.run()
    # Real immutable evidence was loaded and the actual diagnosis service ran.
    # Later stages correctly block without a separately configured real reasoner.
    for substep in ("judge_entry_decision", "evidence_collection",
                    "evidence_based_diagnosis", "readiness_to_plan"):
        assert workflow.steps["judge_entry." + substep].completion_marker
    assert workflow.steps["planner.diagnosis_handoff"].completion_marker
    assert workflow.current_substep == "intervention_ranking"
    assert not workflow.completion_marker
    rows = [row for row in rt.state_repository.all(WORKFLOW_RECORD_COLLECTION)
            if row.get("substep") == "evidence_based_diagnosis"
            and row.get("kind") == "diagnosis_report"]
    assert len(rows) == 1
    payload = read_verified_json(rt.artifact_store, rows[0]["artifact_ref"])
    report = payload["payload"]
    assert report["status"] == "completed"
    assert report["missing_evidence"] == []
    assert report["run_id"] == "real-early-1"
    assert report["metadata"]["deterministic_layer_authoritative"]


@pytest.mark.parametrize("variant", ["wrong_run", "incomplete"])
def test_bad_evidence_never_generates_an_approved_plan(tmp_path, variant):
    rt = _runtime(tmp_path)
    if variant == "wrong_run":
        # The stronger Controller bootstrap refuses cross-run evidence at
        # composition time, before the first operation is ever admitted.
        with pytest.raises(ValueError, match="different run"):
            make_staged_core_services(
                rt, diagnosis_request_ref=_source(rt, wrong_run=True))
        return
    svc = make_staged_core_services(
        rt, diagnosis_request_ref=_source(rt, complete=False))
    deps = StagedAutonomousDependencies(
        services=svc, state_repository=rt.state_repository,
        artifact_store=rt.artifact_store,
    )
    flow = build_staged_autonomous_orchestrator(
        workflow_id="workflow-real-early-1", run_id="real-early-1",
        lineage_id="lineage-1", stage_name="smoke_test",
        dependencies=deps).run()
    assert not flow.completion_marker
    assert "planner.intervention_ranking" not in flow.steps
    if variant == "wrong_run":
        assert flow.current_substep == "judge_entry_decision"
    else:
        assert flow.current_substep == "readiness_to_plan"


@pytest.mark.parametrize("field", ["runtime_monitor", "recovery", "judge_exit"])
def test_downstream_services_refuse_missing_verified_workflow_receipts(tmp_path, field):
    rt = _runtime(tmp_path)
    svc = make_staged_core_services(rt, diagnosis_request_ref=_source(rt))
    selection = {
        "runtime_monitor": "runtime_observation",
        "recovery": "bounded_recovery_advice",
        "judge_exit": "governed_verdict",
    }
    result = getattr(svc, field).execute(_request(selection[field]))
    assert result.status == "blocked"
    assert not result.records
    assert result.blocking_issues


class RoleGeneratorFixture:
    def generate(self, **kwargs):
        raise AssertionError("must never run")


def test_production_composition_blocks_disguised_role_inference_fixture(tmp_path):
    from hephaestus.llm.certified_runtime import CertifiedRoleReasoner
    from hephaestus.production.staged_composition import compose_verified_staged_dependencies
    rt = _runtime(tmp_path)
    reasoner = CertifiedRoleReasoner(
        stack=object(), contract=object(), adapters=object(),
        generator=RoleGeneratorFixture(), artifact_store=rt.artifact_store,
    )
    with pytest.raises(PermissionError, match="frozen CUDA"):
        compose_verified_staged_dependencies(
            rt, core_services=make_staged_core_services(
                rt, diagnosis_request_ref=_source(rt)),
            approval_service=type("NoGrantApprovals", (), {
                "decision_for": lambda self, req: None,
            })(),
            reasoner=reasoner, generation_backend=object(),
            baseline_run_artifact_ref="sha256:" + "a" * 64,
            approved_recipe_refs=("sha256:" + "b" * 64,),
            approved_baseline_refs=("frozen-parent",),
        )


def test_production_composition_blocks_unverified_generation_backend(tmp_path):
    from hephaestus.llm.certified_runtime import (
        CertifiedRoleReasoner, PinnedTransformersRoleGenerator,
    )
    from hephaestus.production.staged_composition import compose_verified_staged_dependencies
    rt = _runtime(tmp_path)
    reasoner = CertifiedRoleReasoner(
        stack=object(), contract=object(), adapters=object(),
        generator=PinnedTransformersRoleGenerator(hf_cache_dir=tmp_path / "cached"),
        artifact_store=rt.artifact_store,
    )
    with pytest.raises(PermissionError, match="real Transformers"):
        compose_verified_staged_dependencies(
            rt, core_services=make_staged_core_services(
                rt, diagnosis_request_ref=_source(rt)),
            approval_service=type("NoGrantApprovals", (), {
                "decision_for": lambda self, req: None,
            })(),
            reasoner=reasoner, generation_backend=object(),
            baseline_run_artifact_ref="sha256:" + "a" * 64,
            approved_recipe_refs=("sha256:" + "b" * 64,),
            approved_baseline_refs=("frozen-parent",),
        )
