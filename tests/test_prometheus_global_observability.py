"""Test metadata-only Prometheus Actions progress logs and post-run summary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hephaestus.development.prometheus_global_observability import (
    event, summary_file,
)


def test_structured_events_stream_stdout_and_persist_local_metadata(tmp_path, monkeypatch, capsys):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("PROMETHEUS_ACTION_LOG", str(log))
    # Only /tmp paths are admitted. A pytest fixture may not use /tmp on all CI.
    if not str(tmp_path).startswith("/tmp/"):
        pytest.skip("Run under admitted temporary root")
    result = event("launcher", "GPU_ALLOCATED", "pass", pod_id="pod123",
                   gpu="RTX_PRO_6000", lease_seconds=2700)
    visible = capsys.readouterr().out
    assert visible.startswith("PROMETHEUS_EVENT=")
    assert result["details"]["lease_seconds"] == 2700
    assert result["details"]["pod_id"] == "pod123"
    assert json.loads(log.read_text().strip()) == result
    recap = summary_file(log, job="launcher")
    assert "GPU_ALLOCATED" in recap and "pod123" not in recap


@pytest.mark.parametrize("details", [
    {"token": "secret"},
    {"reason_code": "private response from model"},
    {"reason_code": "token=abc\nPRIVATE"},
    {"receipt_count": -1.0 / 0.0} if False else {"receipt_count": float("inf")},
])
def test_public_event_refuses_unreviewed_or_unsafe_fields(details):
    with pytest.raises(ValueError):
        event("worker", "INFERENCE_PROGRESS", "progress", **details)


def test_bad_phase_and_component_refused():
    with pytest.raises(ValueError):
        event("launcher", "RAW_MODEL_OUTPUT", "progress", payload="x")
    with pytest.raises(ValueError):
        event("other", "START", "pass")


def test_missing_log_has_conservative_summary(tmp_path):
    assert "No structured event log" in summary_file(tmp_path / "missing", job="watchdog")


def test_log_summary_rejects_untrusted_category(tmp_path):
    log = tmp_path / "bad.jsonl"
    log.write_text(json.dumps({"schema": "prometheus-action-progress.v1",
                               "component": "worker", "phase": "leak",
                               "status": "pass"}) + "\n")
    with pytest.raises(ValueError):
        summary_file(log, job="launcher")
