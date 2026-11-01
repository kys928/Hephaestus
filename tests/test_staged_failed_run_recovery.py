"""Real retry boundary; receipt tests and one real bounded CPU bootstrap fault."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from hephaestus.control.staged_state import StagedAutonomousDependencies, StagedOutputRecord, WORKFLOW_RECORD_COLLECTION
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionLoopRunner
from hephaestus.production.staged_driver import StagedProductionCycleDriver
from hephaestus.production.staged_retry import prepare_staged_retry
from hephaestus.production.state import ProductionLoopState
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.schemas.staged_retry import StagedRetry
from hephaestus.state.staged_retry_store import StagedRetryStore
from hephaestus.storage.verified import canonical_bytes


def runtime(root):
    rt = ProductionCompositionRoot(ProductionCompositionSettings(state_root=root / "state", artifact_root=root / "artifacts")).build()
    # A unit/research test of a controlled local failure needs no provider backoff.
    rt.recovery.policy.backoff_bases["training"] = 0
    rt.recovery.policy.deterministic_jitter_max_seconds = 0
    return rt


def seal(rt, *, workflow, run, substep, kind, payload):
    row = {"workflow_id": workflow, "run_id": run, "lineage_id": "L", "stage_name": "smoke_test",
           "substep": substep, "kind": kind, "payload": payload}
    row["artifact_ref"] = rt.artifact_store.put_bytes(canonical_bytes(row)).artifact_ref
    rt.state_repository.append(WORKFLOW_RECORD_COLLECTION, row)


@pytest.mark.parametrize("mutation", [None, "missing", "changed_hash", "after_update", "in_flight", "oom", "checkpoint", "resumable"])
def test_only_proven_zero_update_terminal_process_crashes_get_new_attempt(tmp_path, mutation):
    rt = runtime(tmp_path)
    progress_file = tmp_path / "optimizer_progress.json"
    proof = {"schema": "hf-optimizer-progress.v1", "run_id": "p-cycle-1", "config_fingerprint": "fp",
             "optimizer_steps": 0, "optimizer_update_started": False, "phase": "pre_optimizer"}
    if mutation == "after_update":
        proof.update(optimizer_steps=1, optimizer_update_started=True, phase="updated")
    if mutation == "in_flight":
        proof.update(optimizer_update_started=True, phase="update_in_flight")
    progress_file.write_bytes(canonical_bytes(proof))
    h = "sha256:" + sha256(progress_file.read_bytes()).hexdigest()
    handle = TrainingRunHandle("p-cycle-1", "e", "transformers_causal_lm", "failed",
        issues=[ContractIssue("out_of_memory" if mutation == "oom" else "non_zero_exit", "runtime_failure", "observed process failure")],
        checkpoint_refs=["some-checkpoint"] if mutation == "checkpoint" else [],
        metadata={"optimizer_progress_ref": str(progress_file), "optimizer_progress_hash": h,
                  "config_fingerprint": "fp"})
    seal(rt, workflow="w", run=handle.run_id, substep="training_status_poll", kind="training_run_handle", payload=handle.to_dict())
    proposal = ExperimentProposal("e", handle.run_id, "L", "smoke_test", "d", "i", "learning_rate", training_constraints={"learning_rate": .01})
    seal(rt, workflow="w", run=handle.run_id, substep="training_input_binding", kind="experiment_training_binding", payload=proposal.to_dict())
    rt.training_lifecycle.delegate = SimpleNamespace(status=lambda _: handle)
    if mutation == "missing":
        progress_file.unlink()
    if mutation == "changed_hash":
        progress_file.write_bytes(progress_file.read_bytes()+b" ")
    state = ProductionLoopState("p", "L", "smoke_test")
    workflow = SimpleNamespace(workflow_id="w", run_id=handle.run_id, current_substep="training_status_poll",
                               resumable=mutation == "resumable", status="terminal_failure")
    first = StagedRetry("p:cycle:1", 1, handle.run_id, 2)
    StagedRetryStore(rt.state_repository).save(first)
    retry = prepare_staged_retry(rt, state, workflow, first)
    if mutation:
        assert retry is None
        assert StagedRetryStore(rt.state_repository).current(first.cycle_key).attempt == 1
    else:
        assert retry.attempt == 2 and retry.predecessor_run_id == first.run_id
        assert retry.failure_ref.startswith("sha256:") and retry.decision_ref.startswith("sha256:")
        assert prepare_staged_retry(rt, state, workflow, retry) is None


def test_actual_cpu_failed_workflow_is_preserved_then_repaired_on_restart(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("transformers"); pytest.importorskip("peft")
    from hephaestus.development.prospective_rehearsal import create_tiny_model, prepare_cpu_proposal
    from hephaestus.data.generated_factory import generate_original_splits
    from hephaestus.schemas.generated_data import GeneratedDataSpec
    from test_staged_real_training_handoff import PlannedWorker
    from test_staged_autonomous_orchestration import _services
    from hephaestus.control.staged_state import WORKFLOW_STATE_COLLECTION
    model = tmp_path / "tiny-model"
    tokenizer, count = create_tiny_model(model)
    rows = generate_original_splits(GeneratedDataSpec(20261028))[0]["train"][:4]
    base = prepare_cpu_proposal(tmp_path / "prepared", run_id="common-fixture-inputs", seed=1117,
        model_path=model, tokenizer=tokenizer, parameter_count=count, rows=rows, steps=2)
    base.training_constraints.update(data_contract_ref=base.training_constraints["trainable_data_contract_ref"],
                                     training_recipe_ref="bounded-research-fixed-recipe-v1")

    class SourceWorker(PlannedWorker):
        def execute(self, req):
            result = super().execute(req)
            if req.substep == "experiment_proposal":
                proposal = replace(base, run_id=req.run_id, experiment_id="experiment-"+req.run_id,
                                   lineage_id=req.lineage_id, stage_name=req.stage_name,
                                   training_constraints={**base.training_constraints, "fixture_bootstrap_failure": "attempt-" not in req.run_id})
                return replace(result, records=(StagedOutputRecord("experiment_proposal", proposal.to_dict()),),
                               metadata={"experiment_id": proposal.experiment_id})
            return result

    def factory(rt, state, cycle):
        services = _services(SourceWorker())
        services.action_executor = None
        return StagedAutonomousDependencies(rt.with_training_services(services, execute_enabled=True), rt.state_repository,
                                            artifact_store=rt.artifact_store)

    rt = runtime(tmp_path)
    runner = ProductionLoopRunner(rt, StagedProductionCycleDriver(factory), maximum_cycles=1)
    deadline = time.monotonic()+90
    while True:
        result = runner.run(program_id="actual-repair", lineage_id="L", stage_name="bounded_cpu_research")
        if result.status != "waiting" or result.recovery_attempts:
            break
        assert time.monotonic() < deadline
        time.sleep(.05)
    assert result.status == "waiting" and result.recovery_attempts == 1 and result.cycle_index == 0
    failed = rt.state_repository.get_latest(WORKFLOW_STATE_COLLECTION, "workflow_id", "workflow-actual-repair-cycle-1")
    assert failed["status"] == "terminal_failure" and not failed["completion_marker"]
    # New composition, no retained trainer/controller Python objects.
    rt = runtime(tmp_path)
    runner = ProductionLoopRunner(rt, StagedProductionCycleDriver(factory), maximum_cycles=1)
    while True:
        result = runner.run(program_id="actual-repair", lineage_id="L", stage_name="bounded_cpu_research")
        if result.status != "waiting":
            break
        assert time.monotonic() < deadline
        time.sleep(.05)
    assert result.cycle_index == 1 and result.recovery_attempts == 1
    assert result.latest_run_id == "actual-repair-cycle-1-attempt-2"
    assert rt.state_repository.get_latest(WORKFLOW_STATE_COLLECTION, "workflow_id", failed["workflow_id"]) == failed
    assert json.loads((rt.settings.artifact_root / "training-runs" / result.latest_run_id / "metrics_summary.json").read_text())["steps"] == 2


def test_retry_cannot_change_learning_rate_before_launch(tmp_path):
    from hephaestus.production.staged_retry import RetryInvariantTrainingService
    from hephaestus.policy.staged_retry import scientific_fingerprint
    from hephaestus.control.staged_state import StagedOperationRequest
    rt = runtime(tmp_path)
    original = ExperimentProposal("e", "r", "L", "smoke_test", "d", "i", "learning_rate",
                                  training_constraints={"learning_rate": .01, "seed": 1117})
    expected = scientific_fingerprint(original)
    original.training_constraints["learning_rate"] = .02
    seal(rt, workflow="w", run="r", substep="training_input_binding", kind="experiment_training_binding", payload=original.to_dict())
    launches = []
    service = RetryInvariantTrainingService(SimpleNamespace(execute=lambda req: launches.append(req)), rt, expected, "p:cycle:1")
    req = StagedOperationRequest("w", "r", "L", "smoke_test", "training_engineer", "lifecycle_launch", "op", 1, (), {})
    result = service.execute(req)
    assert result.status == "blocked" and launches == []


@pytest.mark.parametrize("action",["branch_new_experiment","restart_lineage","rollback_to_checkpoint","change_stage",
                                  "modify_data_policy","modify_training_recipe","modify_eval_policy"])
def test_state_only_transitions_cannot_claim_a_next_executable_staged_experiment(action):
    from hephaestus.production.staged_driver import StagedActionIntentService
    from hephaestus.control.staged_state import StagedOperationRequest
    req=StagedOperationRequest("w","r","L","smoke_test","judge_exit","action_application","op",1,(),{
        "judge_exit.action_boundary":{"proposed_action":action},
        "judge_exit.action_approval_gate":{"approval_status":"approved"}})
    result=StagedActionIntentService().execute(req)
    assert result.status=="blocked" and "next_experiment_transition_not_implemented" in result.blocking_issues[0]


def test_optimizer_progress_fence_flushes_file_and_directory_before_return(tmp_path,monkeypatch):
    from hephaestus.training.hf_worker import _fence_optimizer_progress
    from hephaestus.schemas.optimizer_progress import OptimizerProgress
    import os
    import stat
    calls=[]
    original=os.fsync
    def flush(fd):
        calls.append((stat.S_ISDIR(os.fstat(fd).st_mode),os.readlink(f"/proc/self/fd/{fd}"))); original(fd)
    monkeypatch.setattr(os,"fsync",flush)
    proof=OptimizerProgress("r","fp",0,False,"pre_optimizer")
    _fence_optimizer_progress(tmp_path,proof.to_dict())
    assert calls[0][0] is False
    assert (True,str(tmp_path)) in calls
    assert all(directory for directory,_ in calls[1:])
    assert OptimizerProgress.from_dict(json.loads((tmp_path/"optimizer_progress.json").read_text()))==proof
    with pytest.raises(ValueError):
        OptimizerProgress("r","fp",0,False,"update_in_flight")
