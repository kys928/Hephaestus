"""CPU-only tests for durable frozen-D lifecycle checkpoints and launch blockade."""
import json
from pathlib import Path
import pytest
import evaluator_D_durable_timing as trace
import launch_evaluator_contrast_probe as launch
import evaluator_contrast_inference_protocol as protocol


def test_trace_persists_distinct_stages_and_moving_pointer(monkeypatch):
    entries={}
    def put(_client,key,data):
        entries[key]=dict(data)
    monkeypatch.setattr(trace.runtime,"put_json",put)
    tick=iter([100.0,100.1,103.2])
    obj=trace.DurableTrace(object(),"opaque/run/evaluation","D-contrast-unit","b"*40,
                           monotonic=lambda: next(tick),
                           wall_clock=lambda:"2026-10-22T18:00:00+00:00")
    x=obj("hf_ephemeral_download_start",cache_source="ephemeral")
    y=obj("hf_ephemeral_download_done",cache_source="ephemeral")
    assert x["sequence"]==1 and y["sequence"]==2
    assert y["elapsed_monotonic_seconds"]>x["elapsed_monotonic_seconds"]
    assert entries["opaque/run/evaluation/timing/events/0001.json"]["stage"]=="hf_ephemeral_download_start"
    assert entries["opaque/run/evaluation/timing/events/0002.json"]["stage"]=="hf_ephemeral_download_done"
    assert entries["opaque/run/evaluation/timing/latest.json"]["sequence"]==2
    assert "runpod" not in json.dumps(entries)


def test_unknown_stage_and_unapproved_metadata_are_rejected(monkeypatch):
    writes=[]
    monkeypatch.setattr(trace.runtime,"put_json",lambda *args: writes.append(args))
    obj=trace.DurableTrace(object(),"x","D-contrast-unit","a"*40)
    with pytest.raises(ValueError,match="unknown trace stage"):
        obj("secret_key_dump")
    with pytest.raises(ValueError,match="unsupported trace metadata"):
        obj("runner_enter",api_key="SECRET")
    with pytest.raises(ValueError,match="invalid trace metadata"):
        obj("runner_failed",error_type="x"*81)
    assert writes==[]


def test_persistent_trace_fail_closed_when_s3_write_fails(monkeypatch):
    def blocked(*args):
        raise IOError("simulated S3 outage")
    monkeypatch.setattr(trace.runtime,"put_json",blocked)
    obj=trace.DurableTrace(object(),"x","D-contrast-unit","a"*40)
    with pytest.raises(IOError):
        obj("runner_enter")


def test_load_and_materialize_have_inner_stage_checkpoints():
    root=Path(__file__).resolve().parents[1]
    materializer=(root/"scripts/interface_repair_v2_bootstrap.py").read_text()
    loader=(root/"scripts/run_interface_repair_v1.py").read_text()
    runner=(root/"scripts/run_evaluator_contrast_probe.py").read_text()
    assert 'observer("hf_ephemeral_download_start"' in materializer
    assert 'observer("adapter_archive_verify_start"' in materializer
    assert 'observer("adapter_extract_start"' in materializer
    assert 'observer("base_weights_load_start"' in loader
    assert 'observer("base_weights_load_done"' in loader
    assert 'observer("lora_attach_start"' in loader
    assert 'observer=timing' in runner
    assert 'timing("inference_start"' in runner
    assert 'timing("sample_shard_persisted"' in runner


def test_new_run_stays_launch_disabled_while_science_is_frozen(monkeypatch):
    cfg=json.loads(launch.CONFIG.read_text())
    approval=json.loads(launch.APPROVAL.read_text())
    assert cfg["runpod_launch_enabled"] is False
    assert approval["additional_allocations_authorized"] is False
    assert approval["status"]=="pending_explicit_user_launch_command"
    assert approval["approval_id"]==cfg["approval_id"]
    assert approval["source_D_step96_sha256"]==protocol.D_CHECKPOINT_SHA256
    assert protocol.manifest()["paired_challenges"]==60
    assert protocol.manifest()["unique_prompts"]==56
    monkeypatch.setenv("HEPHAESTUS_CONTRAST_AUTHORIZED","SINGLE_D_CONTRAST_PROBE")
    with pytest.raises(PermissionError):
        launch.authorize(cfg)
