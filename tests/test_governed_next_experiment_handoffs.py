"""Authentic local checkpoint commitments and governed transition replay.

The synthetic checkpoint and synthetic test operator grant are explicitly
test-only. These tests prove the production trust boundary, not an actual
trained model, independent scientific review or authorized GPU spend.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from approval_fixtures import grant_for_lineage, synthetic_checkpoint

from hephaestus.control.staged_state import StagedOperationRequest
from hephaestus.production.composition import (
    ProductionCompositionRoot, ProductionCompositionSettings,
)
from hephaestus.production.governed_next_experiment import GovernedNextExperiment
from hephaestus.production.loop import ProductionCycleResult, ProductionLoopRunner
from hephaestus.production.staged_driver import StagedActionIntentService
from hephaestus.production.state import ProductionLoopState
from hephaestus.schemas.checkpoint_identity import CheckpointIdentity
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest


def _runtime(tmp_path):
    return ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=tmp_path/"state",artifact_root=tmp_path/"artifacts")).build()


def _eligible(tmp_path, action):
    runtime=_runtime(tmp_path)
    root=runtime.settings.state_root
    checkpoint=synthetic_checkpoint(root,"trusted-parent")
    ok,manifest=validate_checkpoint_manifest(Path(checkpoint))
    assert ok is True
    LineageStore(root).set_current(LineageState(
        lineage_id="original-lineage",stage_name="smoke_test",status="active",
        best_checkpoint_ref=checkpoint,last_stable_checkpoint_ref=checkpoint,
    ).to_dict())
    CheckpointIdentityStore(runtime.state_repository).retain(CheckpointIdentity(
        "original-lineage",checkpoint,manifest,"producer://synthetic-test-only"))
    grant_for_lineage(root,"original-lineage","source-run",action,
                      checkpoint,"approval://governed-next",
                      artifacts=runtime.artifact_store)
    state=ProductionLoopState("program","original-lineage","smoke_test",cycle_index=0)
    result=ProductionCycleResult(
        cycle_id="cycle-1",run_id="source-run",experiment_id="experiment-1",
        status="completed",judge_action=action,checkpoint_ref=checkpoint,
        approval_status="approved",approval_ref="approval://governed-next",
        comparison_ref="comparison-1",
        phase_evidence={"judge_entry":["test-receipt"]},
    )
    return runtime,state,result,manifest


@pytest.mark.parametrize("action",[
    "branch_new_experiment","rollback_to_checkpoint","restart_lineage",
])
def test_governed_transition_creates_actual_verified_new_run_identity(tmp_path,action):
    runtime,state,result,manifest=_eligible(tmp_path,action)
    engine=GovernedNextExperiment()
    assert engine.preflight(runtime,state,result)==manifest
    committed=runtime.action_executor.apply(
        lineage_id=state.lineage_id,run_id=result.run_id,
        requested_action=action,checkpoint_ref=result.checkpoint_ref,
        approval_status=result.approval_status,approval_ref=result.approval_ref)
    state.cycle_index=1
    next_run=engine.after_action(runtime,state,result,committed,
                                  checkpoint_manifest_sha256=manifest)
    assert next_run["schema"]=="governed-next-experiment-handoff.v1"
    assert next_run["run_id"].startswith("next-run-")
    assert next_run["training_authorized_by_handoff"] is False
    assert next_run["baseline_checkpoint_ref"]==result.checkpoint_ref
    if action=="branch_new_experiment":
        assert next_run["target_lineage_id"]!=state.lineage_id
    else:
        assert next_run["target_lineage_id"]==state.lineage_id
    state.lineage_id=next_run["target_lineage_id"]
    engine.verify_resume(runtime,state,next_run,cycle_index=2)
    with pytest.raises(PermissionError):
        engine.verify_resume(runtime,state,{**next_run,"run_id":"attacker"},cycle_index=2)
    with pytest.raises(PermissionError):
        engine.verify_resume(runtime,state,next_run,cycle_index=3)


def test_unretained_checkpoint_is_never_admitted_for_branch(tmp_path):
    runtime,state,result,manifest=_eligible(tmp_path,"branch_new_experiment")
    # The producer/custodian commitment is not a generic path or a model claim.
    other=synthetic_checkpoint(runtime.settings.state_root,"outside-producer")
    with pytest.raises(PermissionError):
        GovernedNextExperiment().preflight(runtime,state,
            replace(result,checkpoint_ref=other))


def test_missing_action_approval_blocks_before_production_lineage_mutation(tmp_path):
    runtime,state,result,manifest=_eligible(tmp_path,"restart_lineage")
    engine=GovernedNextExperiment()
    with pytest.raises(PermissionError):
        engine.preflight(runtime,state,replace(result,approval_ref=None))
    assert LineageStore(runtime.settings.state_root).get_current(
        state.lineage_id)["status"]=="active"


def test_staged_judge_intent_defaults_to_blocked_without_bound_handoff():
    request=StagedOperationRequest(
        workflow_id="workflow",run_id="source-run",
        lineage_id="original-lineage",stage_name="smoke_test",
        phase="judge_exit",substep="action_application",
        operation_id="op1",attempt=1,input_refs=(),prior_outputs={
            "judge_exit.action_boundary":{"proposed_action":"branch_new_experiment"},
            "judge_exit.action_approval_gate":{"approval_status":"approved"},
            "judge_exit.promotion_gate":{},
        })
    assert StagedActionIntentService().execute(request).status=="blocked"
    enabled=StagedActionIntentService(governed_next_experiments=True).execute(request)
    assert enabled.status=="completed"
    assert enabled.metadata["action_applied"] is False
    bad=replace(request,prior_outputs={
        **request.prior_outputs,
        "judge_exit.action_approval_gate":{"approval_status":"not_required"},
    })
    assert StagedActionIntentService(governed_next_experiments=True).execute(bad).status=="blocked"


def test_runner_cannot_commit_unmaterialized_high_impact_transition(tmp_path):
    runtime,state,result,manifest=_eligible(tmp_path,"branch_new_experiment")
    from hephaestus.control.spine import SPINE_ORDER
    result.phase_evidence={phase.value:["synthetic-control-receipt"] for phase in SPINE_ORDER}
    class Driver:
        def execute_cycle(self,*,runtime,state,cycle_index):
            return result
    final=ProductionLoopRunner(runtime,Driver(),maximum_cycles=2).run(
        program_id="no-unverified-branch",lineage_id=state.lineage_id,stage_name=state.stage_name)
    assert final.status=="blocked"
    assert final.stop_reason=="governed_next_experiment_not_configured"
    assert runtime.state_repository.all("production_action_executions")==[]



@pytest.mark.parametrize("action",[
    "branch_new_experiment","rollback_to_checkpoint","restart_lineage",
])
def test_full_persistent_loop_executes_next_cycle_with_real_governed_lineage(action,tmp_path):
    runtime,initial,result,manifest=_eligible(tmp_path,action)
    from hephaestus.control.spine import SPINE_ORDER
    phases={phase.value:["test-only-phase-receipt"] for phase in SPINE_ORDER}
    result.phase_evidence=phases
    class Driver:
        second_seen=None
        def execute_cycle(self,*,runtime,state,cycle_index):
            if cycle_index==1:
                return result
            assert cycle_index==2
            self.second_seen=(state.lineage_id,dict(state.metadata["next_cycle"]))
            assert state.metadata["next_cycle"]["baseline_manifest_sha256"]==manifest
            return ProductionCycleResult(
                cycle_id="cycle-2",run_id=state.metadata["next_cycle"]["run_id"],
                experiment_id="experiment-2",status="completed",
                judge_action="reject_candidate",phase_evidence=phases,
                comparison_ref="comparison-2")
    driver=Driver()
    runner=ProductionLoopRunner(runtime,driver,maximum_cycles=2,
                               governed_next_experiments=GovernedNextExperiment())
    final=runner.run(program_id="program",lineage_id="original-lineage",
                     stage_name="smoke_test")
    assert final.status=="stopped" and final.stop_reason=="maximum_cycles_reached"
    assert final.cycle_index==2
    assert driver.second_seen is not None
    child,plan=driver.second_seen
    assert plan["source_run_id"]=="source-run"
    if action=="branch_new_experiment":
        assert child!="original-lineage"
        assert LineageStore(runtime.settings.state_root).get_parent(child)=="original-lineage"
    else:
        assert child=="original-lineage"
    assert runtime.loop_state.get("program").lineage_id==child
    assert runner.run(program_id="program",lineage_id="original-lineage",
                      stage_name="smoke_test").cycle_index==2
    applied=[x for x in runtime.state_repository.all("production_action_executions")
             if x["status"]=="applied"]
    assert len(applied)==2
