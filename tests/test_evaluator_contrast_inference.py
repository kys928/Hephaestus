"""No-GPU tests for one-run D contrasts and paired scoring."""
import json
import pytest
import evaluator_contrast_inference_protocol as p
import launch_evaluator_contrast_probe as launch

def test_panel_dedup_and_invariants():
    unique,pairs=p.panel()
    assert len(pairs)==60 and 10<len(unique)<120
    assert len({x["prompt_sha256"] for x in unique})==len(unique)
    assert sum(x["scientific_change_expected"] for x in pairs)==48
    assert p.manifest()["case_appearances"]==120
    assert all('"expected"' not in x["prompt"] for x in unique)

def test_secondary_measurement_constant_in_all_pairs():
    from evaluator_effect_polarity_contrasts import contrast_panel
    for row in contrast_panel():
        a=row["baseline_evidence"];b=row["modified_evidence"]
        assert sum(a[k]!=b[k] for k in a)==1
        if row["throughput_distractor"]:
            assert a["secondary_throughput_delta"]==b["secondary_throughput_delta"]

def test_strict_and_exploratory_labels_separated():
    direct=p.parse_decision('{"decision":"equivalent"}')
    assert direct["strict"]==direct["exploratory"]=="equivalent"
    fence=chr(96)*3
    wrapped=p.parse_decision(fence+'json\n[{"decision":"recheck_required"}]\n'+fence)
    assert wrapped["strict"] is None and wrapped["exploratory"]=="recheck_required"
    assert p.parse_decision('{"decision":"equivalent","decision":"regressed"}')["exploratory"] is None
    assert p.parse_decision('{"decision":"unknown"}')["exploratory"] is None

def test_perfect_oracle_decisions_score_all_60_pairs():
    unique,pairs=p.panel()
    gold={}
    for row in pairs:
        for side in ("baseline","modified"):
            key=row[side+"_prompt_sha256"];state=row[side+"_state"]
            if key in gold:assert gold[key]==state
            gold[key]=state
    outputs=[dict(prompt_sha256=x["prompt_sha256"],
                  checkpoint_sha256="5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1",
                  raw=json.dumps({"decision":gold[x["prompt_sha256"]]})) for x in unique]
    result=p.assess(unique,pairs,outputs)
    assert result["both_ends_correct"]==60 and result["pair_transition_consistency"]==60
    assert result["strict_format_valid_unique"]==len(unique)
    with pytest.raises(ValueError):p.assess(unique,pairs,outputs[:-1])
    with pytest.raises(ValueError):p.assess(unique,pairs,outputs[:-1]+[outputs[0]])

def test_single_paid_bootstrap_route_and_budget():
    cfg=json.loads(launch.CONFIG.read_text())
    assert cfg["max_total_estimated_gpu_usd"]==0.75
    assert cfg["max_hourly_usd"]<=2.49
    body=launch.pod_request(cfg,"a"*40,"D-contrast-dryrun",real_env=False)
    raw=json.dumps(body)
    assert "scripts/run_evaluator_contrast_probe.py" in raw
    assert "scripts/run_evaluator_factor_probe.py" not in raw
    assert '"$PY" scripts/run_evaluator_semantic_repair_v2.py' not in raw
    assert body["env"]["HEPHAESTUS_CONTRAST_AUTHORIZED"]=="SINGLE_D_CONTRAST_PROBE"
    assert "HEPHAESTUS_FACTOR_PROBE_AUTHORIZED" not in body["env"]

def test_cost_guard():
    cfg=json.loads(launch.CONFIG.read_text())
    for bad in (0,-1,2.50,float("nan")):
        with pytest.raises(ValueError):launch.guard(cfg,10,bad)
    with pytest.raises(TimeoutError):launch.guard(cfg,980,2.49)
    assert launch.guard(cfg,0,2.49)>0

def test_requires_separate_explicit_authorization(monkeypatch):
    cfg=json.loads(launch.CONFIG.read_text())
    monkeypatch.delenv("HEPHAESTUS_CONTRAST_AUTHORIZED",raising=False)
    with pytest.raises(PermissionError):launch.authorize(cfg)
    monkeypatch.setenv("HEPHAESTUS_CONTRAST_AUTHORIZED","SINGLE_D_CONTRAST_PROBE")
    # Previous one-shot S3 claim was used. The instrumented next run has
    # never been authorized and must remain impossible even with the env key.
    with pytest.raises(PermissionError,match="not yet approved"):
        launch.authorize(cfg)
    record=json.loads(launch.APPROVAL.read_text())
    assert record["status"]=="pending_explicit_user_launch_command"
    assert record["additional_allocations_authorized"] is False
    assert record["prior_capacity_blocker"]["consumed_approval_id"]=="heph-D-contrasts-20261019-capacity-retry-1"
    assert record["max_total_estimated_gpu_usd"]==0.75
    assert cfg["runpod_launch_enabled"] is False
