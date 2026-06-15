from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from hephaestus.app.console import ConsoleConfig, ConsoleServer
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.code_edit_workflow import CodeEditProposalWorkflow
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore


def _server(state_root: Path):
    server = ConsoleServer(("127.0.0.1", 0), ConsoleConfig(state_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urlopen(url, timeout=5) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def _demo_state(state_root: Path, run_id: str = "console-run") -> str:
    orch = build_orchestrator(state_root=state_root, run_id=run_id, backend=DryRunBackend())
    orch.run(run_id)
    return run_id


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_home_renders_with_empty_existing_state_root(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    server, base = _server(tmp_path)
    try:
        status, content_type, body = _get(base + "/")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert "text/html" in content_type
    text = body.decode()
    assert "Hephaestus Operator Console" in text
    assert "Run count" in text
    assert str(tmp_path) in text


def test_runs_includes_dry_run_orchestrator_run(tmp_path: Path) -> None:
    run_id = _demo_state(tmp_path)
    server, base = _server(tmp_path)
    try:
        status, _, body = _get(base + "/runs")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert run_id in body.decode()


def test_run_detail_includes_replay_verification(tmp_path: Path) -> None:
    run_id = _demo_state(tmp_path)
    server, base = _server(tmp_path)
    try:
        status, _, body = _get(base + f"/runs/{run_id}")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert "Replay Verification" in body.decode()


def test_api_runs_returns_json_list(tmp_path: Path) -> None:
    run_id = _demo_state(tmp_path)
    server, base = _server(tmp_path)
    try:
        status, content_type, body = _get(base + "/api/runs")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert "application/json" in content_type
    payload = json.loads(body)
    assert isinstance(payload, list)
    assert any(row.get("run_id") == run_id for row in payload)


def test_api_run_replay_returns_replay_status(tmp_path: Path) -> None:
    run_id = _demo_state(tmp_path)
    server, base = _server(tmp_path)
    try:
        status, content_type, body = _get(base + f"/api/runs/{run_id}/replay")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 200
    assert "application/json" in content_type
    payload = json.loads(body)
    assert payload["run_id"] == run_id
    assert payload["status"] in {"reproducible", "partial", "insufficient", "missing"}


def test_code_edits_renders_existing_proposals(tmp_path: Path) -> None:
    workflow = CodeEditProposalWorkflow(CodeEditProposalStore(tmp_path))
    proposal = workflow.create_proposal(
        run_id="console-code-run",
        lineage_id="console-lineage",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/operator_console.md"],
        rollback_plan="Revert docs change",
        test_plan=["pytest tests/test_operator_console.py -q"],
    )
    server, base = _server(tmp_path)
    try:
        status, _, body = _get(base + "/code-edits")
    finally:
        server.shutdown()
        server.server_close()

    text = body.decode()
    assert status == 200
    assert proposal.proposal_id in text
    assert "docs/operator_console.md" in text
    assert "approval_required" in text
    assert "Approve" not in text
    assert "Reject" not in text


def test_invalid_run_page_returns_404_with_useful_message(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    server, base = _server(tmp_path)
    try:
        status, _, body = _get(base + "/runs/no-such-run")
    finally:
        server.shutdown()
        server.server_close()

    assert status == 404
    text = body.decode()
    assert "Run not found" in text
    assert "no-such-run" in text


def test_console_get_requests_do_not_change_existing_files(tmp_path: Path) -> None:
    run_id = _demo_state(tmp_path)
    workflow = CodeEditProposalWorkflow(CodeEditProposalStore(tmp_path))
    workflow.create_proposal(
        run_id=run_id,
        lineage_id="lineage-console",
        requested_by="operator",
        purpose="Update docs",
        target_files=["docs/operator_console.md"],
        rollback_plan="Revert docs change",
        test_plan=["pytest tests/test_operator_console.py -q"],
    )
    before = _snapshot(tmp_path)
    server, base = _server(tmp_path)
    try:
        for route in ["/", "/runs", f"/runs/{run_id}", "/lineages", "/code-edits", "/api/runs", f"/api/runs/{run_id}", f"/api/runs/{run_id}/replay", "/api/code-edits"]:
            status, _, _ = _get(base + route)
            assert status == 200
    finally:
        server.shutdown()
        server.server_close()

    assert _snapshot(tmp_path) == before
