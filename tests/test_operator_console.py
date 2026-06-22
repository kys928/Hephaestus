from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hephaestus.app.console import make_handler
from hephaestus.cli.create_demo_state import create_demo_state
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore


def _server(root: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def _get(base: str, path: str) -> str:
    with urlopen(f"{base}{path}") as response:
        return response.read().decode("utf-8")


def test_operator_console_routes_are_read_only_and_complete(tmp_path):
    root = tmp_path / "state"
    create_demo_state(root, "demo-run")
    CodeEditProposalStore(root).append({"proposal_id": "prop-1", "run_id": "demo-run", "lineage_id": "lineage-demo", "purpose": "inspect only", "rollback_plan": "discard proposal", "target_files": ["README.md"]})
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    server, thread, base = _server(root)
    try:
        assert "Hephaestus Operator Console" in _get(base, "/")
        assert "demo-run" in _get(base, "/runs")
        assert "Replay Verification" in _get(base, "/runs/demo-run")
        assert "lineage-demo" in _get(base, "/lineages")
        assert "prop-1" in _get(base, "/code-edits")
        runs = json.loads(_get(base, "/api/runs"))
        assert runs["read_only"] is True and runs["runs"][0]["run_id"] == "demo-run"
        replay = json.loads(_get(base, "/api/runs/demo-run/replay"))
        assert replay["read_only"] is True and replay["replay"]["status"]
        compat = json.loads(_get(base, "/api/run?run_id=demo-run"))
        assert compat["run"]["run"]["run_id"] == "demo-run"
        assert "Replay Verification" in _get(base, "/run?run_id=demo-run")
        try:
            _get(base, "/runs/missing")
        except HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover
            raise AssertionError("missing run did not 404")
        for method in ["POST", "PUT", "PATCH", "DELETE"]:
            try:
                urlopen(Request(f"{base}/api/runs", data=b"{}", method=method))
            except HTTPError as exc:
                assert exc.code == 405
                assert "read_only" in exc.read().decode("utf-8")
            else:  # pragma: no cover
                raise AssertionError(f"{method} unexpectedly succeeded")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before
