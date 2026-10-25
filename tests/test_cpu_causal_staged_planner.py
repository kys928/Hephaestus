"""End-to-end genuine CPU causal-LM ranking at real staged Planner boundary.

Uses the actual persisted diagnosis, admission policy, scored pretrained CPU
causal language model, immutable proposal receipt and real governed state
orchestrator. Later services remain unconfigured and MUST block; this is not
a bogus 45-stage completion assertion.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import (
    StagedAutonomousDependencies, WORKFLOW_RECORD_COLLECTION,
    PHASE_SUBSTEPS,
)
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.llm.cpu_causal_research import CpuCausalProposalScorer, SCOPE
from hephaestus.production.cpu_causal_planner import CpuResearchCausalPlanner
from hephaestus.production.staged_core import make_staged_core_services
from hephaestus.storage.verified import read_verified_json, canonical_bytes

from test_staged_real_core_science import _runtime, _source


@pytest.mark.real_cpu
def test_real_staged_cpu_planner_scientifically_admits_causal_model_choice(tmp_path):
    rt = _runtime(tmp_path)
    source_ref = _source(rt)
    source = read_verified_json(rt.artifact_store, source_ref)
    source["bounded_research_limits"] = {"max_training_steps": 2}
    source_ref = rt.artifact_store.put_bytes(
        canonical_bytes(source), media_type="application/json").artifact_ref
    scorer = CpuCausalProposalScorer.load(
        rt.artifact_store, cache_dir=tmp_path / "hf-cache")
    core = make_staged_core_services(rt, diagnosis_request_ref=source_ref)
    core = replace(core, planner=CpuResearchCausalPlanner(
        core.planner, scorer, rt.model_evidence, rt.planner))
    dependencies = StagedAutonomousDependencies(
        services=core, state_repository=rt.state_repository,
        artifact_store=rt.artifact_store,
    )
    workflow = build_staged_autonomous_orchestrator(
        workflow_id="workflow-real-early-1", run_id="real-early-1",
        lineage_id="lineage-1", stage_name="smoke_test",
        dependencies=dependencies).run()
    assert not workflow.completion_marker
    assert all(workflow.steps[key].completion_marker for key in (
        "judge_entry.judge_entry_decision",
        "judge_entry.evidence_collection",
        "judge_entry.evidence_based_diagnosis",
        "judge_entry.readiness_to_plan",
        "planner.diagnosis_handoff"))
    # The causal LM may only select a *real* domain-planner-admitted choice.
    # If no such option exists, this is a correct scientific refusal.
    sixth = workflow.steps.get("planner.intervention_ranking")
    assert sixth is not None
    print("REAL_CPU_SCIENTIFIC_PLANNER_STATUS", json.dumps({
        "status": sixth.status,
        "issues": list(sixth.blocking_issues),
        "output": sixth.output,
        "completed": sixth.completion_marker,
    }, default=str))
    assert sixth.completion_marker, (
        "Real causal Planner did not admit an intervention: " +
        str(sixth.blocking_issues))
    if sixth.completion_marker:
        assert sixth.status == "completed"
        ref = sixth.output["cpu_causal_model_inference_ref"]
        lm = read_verified_json(rt.artifact_store, ref)
        assert lm["schema"] == SCOPE
        assert lm["role"] == "planner"
        assert lm["advisory_only"]
        assert not lm["action_executed"]
        assert lm["selected_candidate"] in lm["candidates"]
        model_decision_rows = [
            row for row in rt.state_repository.all(WORKFLOW_RECORD_COLLECTION)
            if row.get("kind") == "intervention_proposal"
            and row.get("operation_id") == sixth.operation_id
        ]
        assert len(model_decision_rows) == 1
        source = read_verified_json(rt.artifact_store, model_decision_rows[0]["artifact_ref"])
        assert source["payload"]["metadata"]["model_originated"]
    else:
        assert sixth.status in {"blocked", "terminal_failure"}
        assert sixth.blocking_issues
    failure_detail = None
    if sixth.status == "terminal_failure":
        from hephaestus.control.staged_state import StagedOperationRequest
        audit_request = StagedOperationRequest(
            workflow_id=workflow.workflow_id, run_id=workflow.run_id,
            lineage_id=workflow.lineage_id,
            stage_name=workflow.stage_name,
            phase="planner", substep="intervention_ranking",
            operation_id=sixth.operation_id, attempt=1,
            input_refs=tuple(sixth.input_refs),
            prior_outputs={key: value.output for key, value in workflow.steps.items()},
        )
        try:
            core.planner.execute(audit_request)
            failure_detail = "unexpected_manual_success_after_recorded_failure"
        except Exception as exc:
            failure_detail = type(exc).__name__ + ":" + str(exc)[:500]
        print("CPU_CAUSAL_STAGE_REPLAY_DIAGNOSTIC", failure_detail)
    report = {
        "schema":"hephaestus-cpu-real-causal-staged-planner.v1",
        "actual_pinned_cpu_causal_lm_loaded": True,
        "actual_immutable_diagnosis_and_domain_policy": True,
        "first_five_stages_completed": True,
        "sixth_stage_complete": bool(sixth.completion_marker),
        "sixth_stage_status": sixth.status,
        "sixth_stage_blocking": list(sixth.blocking_issues),
        "failure_detail": failure_detail,
        "next_substep": workflow.current_substep,
        "full_45_scientific_campaign": False,
        "production_certified": False,
        "operation_statuses": {
            phase.value+"."+step: (
                workflow.steps[phase.value+"."+step].status
                if phase.value+"."+step in workflow.steps
                else "not_executed")
            for phase in SPINE_ORDER for step in PHASE_SUBSTEPS[phase]
        },
    }
    assert len(report["operation_statuses"]) == 45
    if os.environ.get("HEPHAESTUS_CPU_STAGED_CAUSAL_REPORT"):
        p=Path(os.environ["HEPHAESTUS_CPU_STAGED_CAUSAL_REPORT"])
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")


def test_cpu_causal_planner_never_infers_scientific_approval():
    from hephaestus.llm.cpu_causal_research import CANDIDATES
    from hephaestus.production.cpu_causal_planner import CpuResearchCausalPlanner
    assert CANDIDATES["planner"]
    assert all(row[1] != "promote_checkpoint" for row in CANDIDATES["planner"])
    assert "promotion" not in CpuResearchCausalPlanner.__doc__.casefold()
