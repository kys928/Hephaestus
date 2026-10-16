"""Unit guards for the single paid D factor-only inference launcher; no network."""
import copy,json
from pathlib import Path

import pytest
from evaluator_semantic_factor_probe_protocol import (
    panel,parse_factor_json,D_CHECKPOINT_SHA256,
)
import launch_evaluator_factor_probe as runner
import run_evaluator_factor_probe as remote


@pytest.fixture()
def cfg():
    return json.loads(runner.CONFIG.read_text())


@pytest.fixture()
def approved_sandbox(tmp_path,monkeypatch):
    """Simulate historical approval for unit tests; current live record is spent."""
    original=json.loads(runner.APPROVAL.read_text())
    assert original["status"]=="completed_spent_and_disabled"
    copyrecord=dict(original,status="approved_for_exactly_one_inference_only_run")
    temp=tmp_path/"mock-previous-approval.json"
    temp.write_text(json.dumps(copyrecord))
    monkeypatch.setattr(runner,"APPROVAL",temp)
    monkeypatch.setattr(remote,"APPROVAL",temp)
    return copyrecord


def test_consumed_real_approval_refuses_any_new_pod(cfg,monkeypatch):
    monkeypatch.setenv("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED","ONE_APPROVED_FACTOR_INFERENCE_ONLY")
    with pytest.raises(PermissionError,match="approved inference-only"):
        runner.validate_approval(cfg)


def test_one_exact_D_bootstrap_and_no_legacy_evaluator_training(cfg):
    body=runner.pod_request(cfg,"a"*40,"factor-only-D-dryrun",real_env=False)
    raw=json.dumps(body)
    assert "scripts/run_evaluator_factor_probe.py --execute" in raw
    assert "scripts/run_evaluator_semantic_repair_v2.py --mode evaluate" not in raw
    assert "scripts/run_evaluator_semantic_repair_v2.py --mode train" not in raw
    assert "HEPHAESTUS_EVALUATOR_V2_AUTHORIZED" not in body["env"]
    assert body["env"]["HEPHAESTUS_FACTOR_PROBE_RUN_ID"]=="factor-only-D-dryrun"
    assert body["env"]["HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY"].startswith(runner.PREFIX)
    assert "RUNPOD_API_KEY" not in raw


def test_expected_approval_without_training_scope(cfg,monkeypatch,approved_sandbox):
    monkeypatch.setenv("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED","ONE_APPROVED_FACTOR_INFERENCE_ONLY")
    x=runner.validate_approval(cfg)
    assert not x["gpu_training_authorized"]
    assert not x["repeat_inference_authorized"]
    assert cfg["initial_estimated_ceiling_usd"]==0.75
    assert cfg["exceptional_hard_ceiling_usd"]==1.50
    monkeypatch.delenv("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED")
    with pytest.raises(PermissionError):
        runner.validate_approval(cfg)


def test_initial_ceiling_stops_when_no_generation_progress(cfg):
    with pytest.raises(TimeoutError):
        runner.capacity_and_cost_guard(cfg,None,1060,2.49,False)


def test_ordinary_progress_never_raises_contingency(cfg):
    progress={"stage":"factor_probe_inference","completed":16,
              "total_generation_seconds":32.0}
    guard=runner.capacity_and_cost_guard(cfg,progress,320,2.49,False)
    assert guard["contingency_activated"] is False
    assert guard["active_ceiling_usd"]==0.75


def test_measured_need_can_allow_single_exceptional_continuation(cfg):
    progress={"stage":"factor_probe_inference","completed":24,
              "total_generation_seconds":180}
    guard=runner.capacity_and_cost_guard(cfg,progress,650,2.49,False)
    assert guard["contingency_activated"] is True
    assert guard["active_ceiling_usd"]==1.50
    assert guard["projected_full_cost_usd"]>0.75
    assert guard["projected_full_cost_usd"]<=1.50


def test_contingency_never_exceeds_absolute_ceiling(cfg):
    progress={"stage":"factor_probe_inference","completed":24,
              "total_generation_seconds":500}
    with pytest.raises(TimeoutError):
        runner.capacity_and_cost_guard(cfg,progress,2100,2.49,False)


def test_bad_or_overpriced_gpu_rejected(cfg):
    for price in (None,0,-1,2.50,float("nan")):
        with pytest.raises((TypeError,ValueError)):
            runner.capacity_and_cost_guard(cfg,None,30,price,False)


def test_runner_approval_excludes_training_and_wrong_checkpoint(cfg,monkeypatch,approved_sandbox):
    monkeypatch.setenv("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED","ONE_APPROVED_FACTOR_INFERENCE_ONLY")
    remote.authorize(cfg,"factor-only-D-dryrun","a"*40)
    wrong=copy.deepcopy(cfg)
    wrong["exceptional_hard_ceiling_usd"]=1.51
    with pytest.raises(PermissionError):
        remote.authorize(wrong,"factor-only-D-dryrun","a"*40)


def test_report_json_contract_and_stop_logic_without_torch():
    reports=panel()
    valid=reports[0]["reference_factors"].to_dict()
    assert parse_factor_json(json.dumps(valid)) is not None
    class DummyTokenizer:
        def decode(self,tokens,skip_special_tokens=True):
            return json.dumps(valid)
    stop=remote.StopOnComplete(DummyTokenizer(),0,float("inf"))
    assert stop([list(range(8))],None) is True
    assert stop.complete is True
    assert stop.deadline_hit is False


def test_no_replay_check_happens_before_create_pod(cfg,monkeypatch,approved_sandbox):
    monkeypatch.setenv("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED","ONE_APPROVED_FACTOR_INFERENCE_ONLY")
    class Client:
        def __init__(self):self.writes=[]
        def put_object(self,**x):self.writes.append(x)
    class NoPods:
        def _create_pod(self,*a,**k):
            pytest.fail("should not allocate")
    monkeypatch.setattr(runner.infrastructure,"maybe_json",lambda *a,**k:{"status":"factor_probe_complete"})
    with pytest.raises(RuntimeError,match="already exists"):
        runner.execute(cfg,Client(),NoPods(),"a"*40,"factor-only-D-test")
