from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.cli.inspect_run import main
from hephaestus.control.orchestrator import build_orchestrator


def test_inspect_run_json_and_text(tmp_path: Path) -> None:
    run_id = "inspect-run"
    orch = build_orchestrator(state_root=tmp_path, run_id=run_id, backend=DryRunBackend())
    orch.run(run_id)

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(tmp_path), "--run-id", run_id, "--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())
    for key in ["run", "lineage", "manifest", "eval", "decision", "replay", "memory", "artifacts", "warnings"]:
        assert key in payload
    assert payload["run"].get("run_id") == run_id
    assert payload["run"].get("data_manifest_id")
    assert "replay_scope" in payload["replay"]
    assert payload["decision"].get("effective_action")

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(tmp_path), "--run-id", run_id, "--no-color"])
    assert code == 0
    out = buf.getvalue()
    assert "HEPHAESTUS RUN INSPECTION" in out
    assert "run_id" in out
    for section in ["Lineage", "Data Manifest", "Decision / Gates", "Replay"]:
        assert section in out
    assert not re.search(r"\x1b\[[0-9;]*m", out)


def test_inspect_run_missing_run(tmp_path: Path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["--state-root", str(tmp_path), "--run-id", "missing"])
    assert code != 0
    assert "not found" in buf.getvalue()
