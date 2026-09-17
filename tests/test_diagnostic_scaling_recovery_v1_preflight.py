from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_diagnostic_scaling_recovery_v1.py"
spec = importlib.util.spec_from_file_location("diagnostic_scaling_recovery_preflight", SCRIPT)
assert spec and spec.loader
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def contract_raw() -> bytes:
    return (ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json").read_bytes()


def test_repository_preflight_reports_current_authorization_state(monkeypatch):
    payload = json.loads(contract_raw())
    monkeypatch.setattr(preflight.validator, "validate_remote", lambda contract: [
        {
            "model_id": row["model_id"],
            "revision": row["revision"],
            "model_type": row["model_type"],
            "remote_revision_verified": True,
            "fixed_non_thinking_eligible": bool(row["fixed_non_thinking"]["eligible"]),
            "enable_thinking_template_support": True if row["fixed_non_thinking"]["eligible"] else None,
        }
        for row in contract["candidates"]
    ])
    summary, objects = preflight.build_preflight(repo_sha="1" * 40, verify_s3=False)
    paid = bool(payload["governance"]["paid_launch_allowed_now"])
    assert summary["status"] == ("recovery_inputs_validated_launch_authorized" if paid else "recovery_inputs_validated_not_launchable")
    assert summary["paid_launch_allowed_now"] is paid
    assert summary["paid_launch_admission_persisted"] is False
    assert "admission.json" not in objects


def test_persist_launch_admission_is_blocked_when_contract_is_not_authorized(monkeypatch):
    payload = json.loads(contract_raw())
    payload["governance"]["paid_launch_allowed_now"] = False
    protocol_raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    summary = {
        "candidate_revisions": {row["model_id"]: row["revision"] for row in payload["candidates"]},
        "fixed_non_thinking_eligible": [row["model_id"] for row in payload["candidates"] if row["fixed_non_thinking"]["eligible"]],
        "reasoning_aware_eligible": [row["model_id"] for row in payload["candidates"]],
        "authorization_state": "blocked_pre_user_go",
    }
    objects = {
        "protocol.json": protocol_raw,
        "training.jsonl": b"{}\n",
        "contamination_report.json": b'{"passed":true}\n',
    }
    monkeypatch.setenv(preflight.AUTH_ENV, "YES")
    with pytest.raises(RuntimeError, match="paid_launch_allowed_now is false"):
        preflight.persist_launch_admission(repo_sha="1" * 40, summary=summary, objects=objects)


def test_launchable_admission_requires_second_authorization_lock(monkeypatch):
    payload = json.loads(contract_raw())
    payload["governance"]["paid_launch_allowed_now"] = True
    objects = {
        "protocol.json": (json.dumps(payload, sort_keys=True) + "\n").encode(),
        "training.jsonl": b"{}\n",
        "contamination_report.json": b'{"passed":true}\n',
    }
    summary = {
        "candidate_revisions": {row["model_id"]: row["revision"] for row in payload["candidates"]},
        "fixed_non_thinking_eligible": [row["model_id"] for row in payload["candidates"] if row["fixed_non_thinking"]["eligible"]],
        "reasoning_aware_eligible": [row["model_id"] for row in payload["candidates"]],
        "authorization_state": "authorized_by_user",
    }
    monkeypatch.delenv(preflight.AUTH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="cannot persist launch admission without"):
        preflight.persist_launch_admission(repo_sha="1" * 40, summary=summary, objects=objects)
