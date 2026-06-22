from __future__ import annotations

import json

from hephaestus.cli.create_demo_state import main
from hephaestus.state.run_store import RunStore


def test_create_demo_state_orchestrator_json_and_text(tmp_path, capsys):
    root = tmp_path / "state"
    assert main(["--state-root", str(root), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "demo-run"
    assert payload["lineage_id"] == "lineage-demo"
    assert payload["stage_name"] == "early_pretraining"
    assert payload["phase_count"] == 8
    assert payload["replay_status"]
    assert "next_commands" in payload
    assert RunStore(root).get("demo-run") is not None

    assert main(["--state-root", str(root), "--format", "text"]) == 0
    text = capsys.readouterr().out
    assert "run_id: demo-run-2" in text
    assert "next_commands:" in text
    assert RunStore(root).get("demo-run-2") is not None
    assert not (tmp_path / "run_records.jsonl").exists()
