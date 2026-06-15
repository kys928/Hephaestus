from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from hephaestus.cli.create_demo_state import main
from hephaestus.state.run_store import RunStore


def _snapshot(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_create_demo_state_json_creates_dry_run_state(tmp_path: Path) -> None:
    state_root = tmp_path / "demo-state"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(state_root), "--run-id", "demo-a", "--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["run_id"] == "demo-a"
    assert payload["replay_status"] in {"reproducible", "partial", "insufficient"}
    assert "next_commands" in payload
    assert "inspect_run" in payload["next_commands"]
    assert "verify_replay" in payload["next_commands"]
    assert "operator_console" in payload["next_commands"]
    assert RunStore(state_root).get("demo-a") is not None


def test_create_demo_state_text_contains_run_id_and_next_commands(tmp_path: Path) -> None:
    state_root = tmp_path / "demo-state"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(state_root), "--run-id", "text-run", "--format", "text"])
    assert code == 0
    out = buf.getvalue()
    assert "run_id: text-run" in out
    assert "next_commands:" in out
    assert "inspect_run" in out
    assert "verify_replay" in out
    assert "operator_console" in out


def test_create_demo_state_default_run_id_advances_deterministically(tmp_path: Path) -> None:
    state_root = tmp_path / "demo-state"
    for expected in ["demo-run", "demo-run-2"]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert main(["--state-root", str(state_root), "--format", "json"]) == 0
        assert json.loads(buf.getvalue())["run_id"] == expected


def test_create_demo_state_does_not_write_outside_state_root(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    keep = parent / "keep.txt"
    keep.write_text("unchanged")
    state_root = parent / "chosen-state"
    before = _snapshot(parent)

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["--state-root", str(state_root), "--run-id", "safe-run", "--format", "json"]) == 0

    after = _snapshot(parent)
    for rel, content in before.items():
        assert after[rel] == content
    changed = {rel for rel in after if rel not in before or after[rel] != before.get(rel)}
    assert changed
    assert all(rel == "keep.txt" or rel.startswith("chosen-state/") for rel in after)
    assert all(rel.startswith("chosen-state/") for rel in changed)
