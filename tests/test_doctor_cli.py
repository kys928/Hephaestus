from __future__ import annotations

import json

from hephaestus.cli.create_demo_state import create_demo_state
from hephaestus.cli.doctor import main
from hephaestus.state.run_store import RunStore

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


def test_doctor_accepts_reproducible_replay_and_warns_on_insufficient(tmp_path, capsys):
    root = tmp_path / "state"
    create_demo_state(root, "demo-run")
    store = RunStore(root)
    reproducible = dict(store.get("demo-run") or {})
    reproducible["replay_metadata"] = {
        **dict(reproducible.get("replay_metadata") or {}),
        "checkpoint_content_hash": "sha256:demo",
        "content_hash_available": True,
        "requires_content_hash_match": True,
    }
    store.append(reproducible)

    assert main(["--state-root", str(root), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["latest_replay_status"] == "reproducible"
    assert "latest_replay_insufficient" not in payload["warnings"]

    insufficient = dict(reproducible)
    insufficient["replay_metadata"] = {}
    store.append(insufficient)

    assert main(["--state-root", str(root), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["latest_replay_status"] == "insufficient"
    assert "latest_replay_insufficient" in payload["warnings"]
