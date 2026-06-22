from __future__ import annotations

import json

from hephaestus.cli.create_demo_state import create_demo_state
from hephaestus.cli.doctor import main

_EXPECTED = {"status", "python_version", "imports", "state_root", "state_root_provided", "state_root_exists", "run_count", "latest_run_id", "latest_replay_status", "warnings", "errors"}


def test_doctor_import_only_missing_empty_and_demo(tmp_path, capsys):
    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == _EXPECTED
    assert payload["status"] == "ok"
    assert payload["state_root_provided"] is False

    missing = tmp_path / "missing"
    assert main(["--state-root", str(missing), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert "state_root_missing" in payload["warnings"]
    assert not missing.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    before = {p.relative_to(empty): p.read_bytes() for p in empty.rglob("*") if p.is_file()}
    assert main(["--state-root", str(empty), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert "state_root_empty" in payload["warnings"]
    after = {p.relative_to(empty): p.read_bytes() for p in empty.rglob("*") if p.is_file()}
    assert after == before

    root = tmp_path / "state"
    create_demo_state(root, "demo-run")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["--state-root", str(root), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == _EXPECTED
    assert payload["run_count"] >= 1
    assert payload["latest_run_id"] == "demo-run"
    assert payload["latest_replay_status"]
    assert main(["--state-root", str(root), "--format", "text"]) == 0
    assert "status:" in capsys.readouterr().out
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before
