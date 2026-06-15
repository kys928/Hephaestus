from __future__ import annotations

import json

from hephaestus.cli.create_demo_state import main
from hephaestus.state.run_store import RunStore


def test_create_demo_state_writes_under_state_root(tmp_path, capsys):
    root = tmp_path / "state"
    assert main(["--state-root", str(root), "--run-id", "demo-run", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state_root"] == str(root)
    assert RunStore(root).get("demo-run") is not None
    assert (root / "run_records.jsonl").exists()
