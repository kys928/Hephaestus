"""CPU-only contract tests for startup S3 receipts and safe Blackwell retry."""
import json
import pytest
import evaluator_D_bootstrap_instrumentation as instrument
import launch_evaluator_contrast_probe as launch

def test_stage_bootstrap_persisted_without_launch():
    cfg=json.loads(launch.CONFIG.read_text())
    r=launch.pod_request(cfg,"f"*40,"D-contrast-dryrun",real_env=False)
    assert r["gpuTypeIds"]==["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]
    assert r["dataCenterIds"]==["EU-CZ-1"]
    assert r["cloudType"]=="SECURE"
    assert r["imageName"]=="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    assert r["containerDiskInGb"]==100
    assert r["networkVolumeId"]=="<volume>" and r["volumeMountPath"]=="/workspace"
    shell=r["dockerStartCmd"][2]
    assert "persist_bootstrap container_started" in shell
    assert "persist_bootstrap runner_started" in shell
    assert "HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY" in shell
    assert "RUNPOD_S3_ENDPOINT_URL" in shell
    assert "D_CONTRAST_S3_BOOTSTRAP" in shell
    assert "D-contrast-WK-cu128-torch280-r1" in shell
    assert "interface-repair-v2-py312-cu130-r2" not in shell
    assert "persist_bootstrap bootstrap_failed" in shell
    assert "bootstrap_failed" in instrument.BEGIN
    assert "events/" in instrument.BEGIN
    assert 'Key=os.environ[\'HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY\']' in shell

def test_bad_shell_rejected_without_s3_or_pod_calls():
    for broken in ([],["bash","-c","echo hi"],["bash","-lc","echo hi"]):
        with pytest.raises(ValueError):
            instrument.instrument(broken)

def test_after_claim_cannot_reuse_pod_or_raise_budget(monkeypatch):
    cfg=json.loads(launch.CONFIG.read_text())
    assert cfg["max_total_estimated_gpu_usd"]==1.80
    assert cfg["max_hourly_usd"]==2.59
    assert cfg["bootstrap_heartbeat_max_wait_seconds"]==600
    assert cfg["runpod_launch_enabled"] is True
    assert cfg["allow_replay_after_create_attempt"] is False
    a=json.loads(launch.APPROVAL.read_text())
    assert a["approval_id"]==cfg["approval_id"]
    assert a["prior_A40_consumed_id"]!=a["approval_id"]
    assert a["status"]=="authorized_once"
    assert a["max_pods"]==a["max_launch_attempts"]==1
    assert a["source_D_step96_sha256"]=="5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1"

def test_current_one_shot_cpu_runner_accepts_approved_budget(monkeypatch):
    import run_evaluator_contrast_probe as runner
    monkeypatch.setenv("HEPHAESTUS_CONTRAST_AUTHORIZED","SINGLE_D_CONTRAST_PROBE")
    cfg=json.loads(launch.CONFIG.read_text())
    assert cfg["required_gpu_datacenter"]=="EU-CZ-1"
    assert cfg["gpu_type_id"]=="NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    assert cfg["max_total_estimated_gpu_usd"]==1.80
    a=runner.check(cfg,"D-contrast-offline-check","f"*40)
    assert a["approval_id"]==cfg["approval_id"]
    assert a["prior_WK_consumed_id"]!=a["approval_id"]
