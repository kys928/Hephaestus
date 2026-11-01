"""Actual three-cycle CPU research with a separately reconstructed cold verifier."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hephaestus.policy.cpu_autonomy import learning_rate,accept_candidate
from hephaestus.schemas.cpu_autonomy import CpuAutonomyPlan
from hephaestus.evaluation.cpu_autonomy import compare_policies


def test_finite_adaptive_policy_uses_previous_immutable_outcome_and_stays_bounded():
    plan=CpuAutonomyPlan()
    memory=lambda outcome,rate:[{"metadata":{"outcome":outcome,"controlled_variables":{"selected_learning_rate":rate}}}]
    assert learning_rate(plan,"adaptive",1,[])[0]==.02
    assert learning_rate(plan,"adaptive",2,memory("improved",.02))[0]==.03
    assert learning_rate(plan,"adaptive",2,memory("regressed",.02))[0]==.01
    assert learning_rate(plan,"adaptive",3,memory("regressed",.005))[0]==.005
    assert learning_rate(plan,"heuristic",2,memory("improved",.04))[0]==.01
    assert learning_rate(plan,"fixed",3,memory("regressed",.005))[0]==.01
    with pytest.raises(ValueError):
        learning_rate(plan,"adaptive",2,memory("improved",1.))
    with pytest.raises(ValueError):
        CpuAutonomyPlan(cycles=4)


def test_lower_nll_cannot_accept_regressed_exact_answers():
    before={"mean_target_nll":5.,"exact_correct":1}
    assert not accept_candidate(before,{"mean_target_nll":4.,"exact_correct":0},1e-6)
    assert not accept_candidate(before,{"mean_target_nll":5.,"exact_correct":1},1e-6)
    assert accept_candidate(before,{"mean_target_nll":4.9,"exact_correct":1},1e-6)


def test_three_seed_policy_comparison_reports_inconclusive_even_all_signs_favor_adaptive():
    plan=CpuAutonomyPlan()
    results=[{"policy":p,"seed":s,"test_mean_target_nll":1. if p=="adaptive" else 2.}
             for s in plan.seeds for p in plan.policies]
    report=compare_policies(results,plan)
    assert not report["adaptive_superiority_established"]
    assert report["comparisons"]["fixed"]["exact_two_sided_training_seed_sign_p"]==.25
    assert report["comparisons"]["fixed"]["holm_adjusted_p"]==.5
    with pytest.raises(ValueError): compare_policies(results[:-1],plan)


@pytest.mark.real_cpu
def test_three_authentic_native_cycles_failure_repair_lineage_memory_and_full_cold_replay(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("transformers"); pytest.importorskip("peft")
    from hephaestus.development.cpu_autonomy_benchmark import prepare,run_program,load
    from hephaestus.state.cpu_autonomy_store import CpuAutonomyStore
    from hephaestus.storage.verified import canonical_bytes
    root=tmp_path/"cpu-autonomy"
    prepare(root)
    result=run_program(root,"adaptive",1117)
    assert result["actual_optimizer_updates"]==12 and result["native_completed_operations"]==135
    assert result["repair_count"]==1 and result["injected_failure_diagnoses"]==["process_crash"]
    assert result["restart_after_completed_cycle"]
    assert result["lineage"]["stage_name"]=="bounded_cpu_research"
    assert not result["external_certification"] and not result["production_promotion"]
    rates=[c["proposal"]["training_constraints"]["learning_rate"] for c in result["cycles"]]
    assert len(set(rates))>1
    assert run_program(root,"adaptive",1117)==result  # final evidence is cached, never reconsulted
    cold=subprocess.run([sys.executable,"-m","hephaestus.development.cpu_autonomy_benchmark",
                         "cold-program",str(root),"--policy","adaptive","--seed","1117"],
                        capture_output=True,text=True,check=True,timeout=180)
    verified=json.loads(cold.stdout)
    assert verified["all_fixed_models_reexecuted"] and verified["all_optimizer_tensors_reproduced"]
    assert verified["failed_predecessor_preserved"]
    assert not verified["external_independent_certification"]
    # No reset of a consumed but incomplete final session, even after restart.
    rt,_,_,_=load(root)
    store=CpuAutonomyStore(rt.state_repository,rt.artifact_store)
    with pytest.raises(PermissionError): store.consume_final(result["program_id"],"replacement")
    # A changed frozen input must fail cold verification before inference.
    from hephaestus.evaluation.cpu_autonomy_verification import verify_program
    source=root/"public-source.json"
    source.write_bytes(source.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="input or method"):
        verify_program(root,result["program_id"],reexecute_models=False,reexecute_training=False)
