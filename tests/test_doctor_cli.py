from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from hephaestus.cli.create_demo_state import main as create_demo_main
from hephaestus.cli.doctor import STABLE_KEYS, main


def _snapshot(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_doctor_no_state_root_does_not_create_files(tmp_path: Path) -> None:
    before = _snapshot(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["state_root_provided"] is False
    assert _snapshot(tmp_path) == before


def test_doctor_missing_state_root_does_not_create_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-state"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(missing), "--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "warning"
    assert payload["state_root_exists"] is False
    assert not missing.exists()


def test_doctor_empty_existing_state_root_reports_warning(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(state_root), "--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "warning"
    assert payload["run_count"] == 0


def test_doctor_after_create_demo_state_reports_runs_and_stable_keys(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    with redirect_stdout(io.StringIO()):
        assert create_demo_main(["--state-root", str(state_root), "--run-id", "doctor-run"]) == 0
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(state_root), "--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert set(payload) == set(STABLE_KEYS)
    assert payload["run_count"] >= 1
    assert payload["latest_run_id"] == "doctor-run"


def test_doctor_text_contains_status(tmp_path: Path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(tmp_path), "--format", "text"])
    assert code == 0
    assert "status:" in buf.getvalue()


def test_doctor_is_read_only_for_existing_state_root(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    with redirect_stdout(io.StringIO()):
        assert create_demo_main(["--state-root", str(state_root), "--run-id", "readonly-run"]) == 0
    before = _snapshot(state_root)
    with redirect_stdout(io.StringIO()):
        assert main(["--state-root", str(state_root), "--format", "json"]) == 0
    assert _snapshot(state_root) == before
