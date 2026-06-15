from __future__ import annotations

import json

from hephaestus.cli.doctor import main


def test_doctor_does_not_create_missing_state_root(tmp_path, capsys):
    missing = tmp_path / "missing"
    assert main(["--state-root", str(missing), "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "missing"
    assert payload["created"] is False
    assert not missing.exists()
