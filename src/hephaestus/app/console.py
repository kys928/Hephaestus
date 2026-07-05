from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from hephaestus.cli.inspect_run import _load as load_run
from hephaestus.control.replay_verification import verify_run_replay
from hephaestus.policy.operator_console_policy import OperatorConsolePolicy
from hephaestus.schemas.approval_decision import ApprovalDecision
from hephaestus.schemas.operator_action import OperatorActionRecord, OperatorActionRequest
from hephaestus.schemas.operator_console import OperatorConsolePayload
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.operator_action_store import OperatorActionStore
from hephaestus.state.run_store import RunStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, status: int, body_text: str) -> None:
    body = body_text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pre(value: object) -> str:
    return f"<pre>{_e(json.dumps(value, indent=2, sort_keys=True))}</pre>"


def _page(title: str, body: str) -> str:
    nav = " | ".join([
        "<a href='/'>Home</a>", "<a href='/runs'>Runs</a>", "<a href='/lineages'>Lineages</a>", "<a href='/code-edits'>Code Edits</a>", "<a href='/healthz'>Health</a>",
    ])
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{_e(title)}</title></head><body><nav>{nav}</nav>{body}</body></html>"


def _run_href(run_id: object) -> str:
    rid = str(run_id or "")
    return f"/runs/{quote(rid)}"


def _runs_html(root: Path) -> str:
    rows = []
    for run in RunStore(root).all():
        run_id = str(run.get("run_id", ""))
        rows.append(f"<tr><td><a href='{_e(_run_href(run_id))}'>{_e(run_id)}</a></td><td>{_e(run.get('lineage_id'))}</td><td>{_e(run.get('stage_name'))}</td><td>{_e(run.get('status'))}</td><td>{_e(run.get('judge_action'))}</td></tr>")
    body_rows = "".join(rows) or "<tr><td colspan='5'>No run records found.</td></tr>"
    return _page("Runs", f"<h1>Runs</h1><table><thead><tr><th>Run</th><th>Lineage</th><th>Stage</th><th>Status</th><th>Judge Action</th></tr></thead><tbody>{body_rows}</tbody></table>")


def _index_html(state_root: Path) -> str:
    return _page("Hephaestus Operator Console", f"<h1>Hephaestus Operator Console</h1><p><strong>Read-only:</strong> this console performs inspection only and exposes no approval, rejection, execution, or training-launch actions.</p><p>State root: <code>{_e(state_root)}</code></p>" + _runs_html(state_root).split("<h1>Runs</h1>", 1)[1].split("</body>", 1)[0])


def _lineages_html(root: Path) -> str:
    rows = []
    for lineage_id, lineage in LineageStore(root).all().items():
        rows.append(f"<tr><td>{_e(lineage_id)}</td><td>{_e(lineage.get('stage_name'))}</td><td>{_e(lineage.get('status'))}</td><td>{_e(lineage.get('latest_run_id'))}</td><td>{_e(lineage.get('last_decision'))}</td></tr>")
    return _page("Lineages", f"<h1>Lineages</h1><table><tbody>{''.join(rows) or '<tr><td>No lineages found.</td></tr>'}</tbody></table>")


def _code_edits_html(root: Path) -> str:
    rows = []
    for proposal in CodeEditProposalStore(root).list_all():
        rows.append(f"<tr><td>{_e(proposal.get('proposal_id'))}</td><td>{_e(proposal.get('run_id'))}</td><td>{_e(proposal.get('lineage_id'))}</td><td>{_e(proposal.get('status'))}</td><td>{_e(proposal.get('risk_level'))}</td></tr>")
    return _page("Code Edits", f"<h1>Code Edits</h1><table><tbody>{''.join(rows) or '<tr><td>No code edit proposals found.</td></tr>'}</tbody></table>")


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("json_object_required")
    return payload


def _lineage_for_run(root: Path, run: dict[str, object] | None) -> dict[str, object]:
    lineage_id = str((run or {}).get("lineage_id") or "")
    lineage = LineageStore(root).get_current(lineage_id) if lineage_id else None
    return lineage or {}


def _record_operator_action(root: Path, record: OperatorActionRecord) -> dict[str, object]:
    payload = record.to_dict()
    OperatorActionStore(root).append(payload)
    return payload


def _run_detail(root: Path, run_id: str) -> dict[str, object] | None:
    try:
        payload = load_run(root, run_id)
    except ValueError:
        return None
    payload["replay_verification"] = verify_run_replay(root, run_id).to_dict()
    return payload


def _run_html(root: Path, run_id: str) -> str | None:
    data = _run_detail(root, run_id)
    if data is None:
        return None
    sections = [("Run", "run"), ("Lineage", "lineage"), ("Manifest", "manifest"), ("Evaluation", "eval"), ("Judge Decision / Gates", "decision"), ("Replay Verification", "replay_verification"), ("Artifacts", "artifacts"), ("Memory", "memory"), ("Warnings", "warnings")]
    body = f"<h1>Run {_e(run_id)}</h1>"
    for title, key in sections:
        value = data.get(key)
        body += f"<h2>{_e(title)}</h2>" + ("<p>Missing.</p>" if value in (None, [], {}) else _pre(value))
    return _page(f"Run {run_id}", body)


def make_handler(state_root: Path) -> type[BaseHTTPRequestHandler]:
    root = state_root

    class ConsoleHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                _html_response(self, 200, _index_html(root)); return
            if path == "/runs":
                _html_response(self, 200, _runs_html(root)); return
            if path.startswith("/runs/"):
                run_id = unquote(path.removeprefix("/runs/"))
                page = _run_html(root, run_id)
                if page is None: _json_response(self, 404, OperatorConsolePayload(status="error", error="run_not_found", read_only=True).to_dict())
                else: _html_response(self, 200, page)
                return
            if path == "/lineages":
                _html_response(self, 200, _lineages_html(root)); return
            if path == "/code-edits":
                _html_response(self, 200, _code_edits_html(root)); return
            if path == "/healthz":
                _json_response(self, 200, OperatorConsolePayload(status="ok", read_only=True).to_dict()); return
            if path == "/api/runs":
                _json_response(self, 200, OperatorConsolePayload(runs=RunStore(root).all(), read_only=True).to_dict()); return
            if path.startswith("/api/runs/"):
                suffix = path.removeprefix("/api/runs/")
                if suffix.endswith("/replay"):
                    run_id = unquote(suffix[: -len("/replay")])
                    report = verify_run_replay(root, run_id).to_dict()
                    status = 404 if report.get("status") == "missing" else 200
                    _json_response(self, status, {"read_only": True, "replay": report}); return
                data = _run_detail(root, unquote(suffix))
                if data is None: _json_response(self, 404, OperatorConsolePayload(status="error", error="run_not_found", read_only=True).to_dict())
                else: _json_response(self, 200, OperatorConsolePayload(run=data, read_only=True).to_dict())
                return
            if path == "/api/code-edits":
                _json_response(self, 200, {"read_only": True, "code_edits": CodeEditProposalStore(root).list_all()}); return
            if path == "/api/approvals/pending":
                pending = [r for r in DecisionStore(root).all_approval_requests() if r.get("status") == "pending" and not DecisionStore(root).get_latest_approval_decision(str(r.get("request_id", "")))]
                _json_response(self, 200, {"read_only": True, "approval_requests": pending}); return
            if path == "/api/operator-actions":
                _json_response(self, 200, {"read_only": True, "operator_actions": OperatorActionStore(root).all()}); return
            if path in {"/api/run", "/run"}:
                run_id = (parse_qs(parsed.query).get("run_id") or [""])[0]
                if not run_id:
                    _json_response(self, 400, OperatorConsolePayload(status="error", error="run_id_required", read_only=True).to_dict()); return
                data = _run_detail(root, run_id)
                if data is None:
                    _json_response(self, 404, OperatorConsolePayload(status="error", error="run_not_found", read_only=True).to_dict()); return
                if path == "/api/run": _json_response(self, 200, OperatorConsolePayload(run=data, read_only=True).to_dict())
                else: _html_response(self, 200, _run_html(root, run_id) or "")
                return
            _json_response(self, 404, OperatorConsolePayload(status="error", error="not_found", read_only=True).to_dict())

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                payload = _read_json_body(self)
                if path.startswith("/api/approvals/") and path.endswith("/decisions"):
                    request_id = unquote(path.removeprefix("/api/approvals/")[: -len("/decisions")])
                    request = DecisionStore(root).get_approval_request(request_id)
                    if request is None:
                        _json_response(self, 404, {"read_only": False, "status": "error", "error": "approval_request_not_found"}); return
                    action_request = OperatorActionRequest(
                        action=str(request.get("proposed_action", "")),
                        operator_id=str(payload.get("operator_id", "operator.unknown")),
                        run_id=str(request.get("run_id", "")),
                        lineage_id=str(request.get("lineage_id", "")),
                        request_id=request_id,
                        outcome=str(payload.get("outcome", "rejected")),
                        note=str(payload.get("note", "")),
                        metadata={"source": "operator_console"},
                    )
                    override_allowed = str(request.get("required_approval_type")) != "operator_approval_no_override"
                    resolution = OperatorConsolePolicy().approval_policy.resolve_operator_outcome(action_request.outcome or "rejected", override_allowed=override_allowed)
                    decision = ApprovalDecision(
                        decision_event_id=f"apd-console-{request_id}-{len(DecisionStore(root).all_approval_decisions()) + 1}",
                        request_id=request_id,
                        lineage_id=action_request.lineage_id or "",
                        run_id=action_request.run_id or "",
                        operator_id=action_request.operator_id,
                        outcome=resolution.outcome,
                        status=resolution.status,
                        note=action_request.note,
                        effect_on_action=resolution.effect_on_action,
                        created_at=_now(),
                        metadata={"resolution_reason": resolution.reason, "override_blocked": resolution.override_blocked, "source": "operator_console"},
                    )
                    DecisionStore(root).append_approval_decision(decision.to_dict())
                    action = _record_operator_action(root, OperatorActionRecord(
                        action_event_id=f"opa-{decision.decision_event_id}", action_kind="approval_decision", action=action_request.action,
                        operator_id=action_request.operator_id, status=resolution.status, created_at=decision.created_at,
                        run_id=action_request.run_id, lineage_id=action_request.lineage_id, request_id=request_id,
                        note=action_request.note, policy_decision=decision.metadata, metadata={"approval_decision": decision.to_dict()},
                    ))
                    _json_response(self, 201, {"read_only": False, "approval_decision": decision.to_dict(), "operator_action": action}); return
                if path.startswith("/api/runs/") and path.endswith("/replay-requests"):
                    run_id = unquote(path.removeprefix("/api/runs/")[: -len("/replay-requests")])
                    run = RunStore(root).get(run_id)
                    if run is None:
                        _json_response(self, 404, {"read_only": False, "status": "error", "error": "run_not_found"}); return
                    report = verify_run_replay(root, run_id).to_dict()
                    action = _record_operator_action(root, OperatorActionRecord(
                        action_event_id=f"opr-{run_id}-{len(OperatorActionStore(root).all()) + 1}", action_kind="replay_verification_request",
                        action="request_recheck", operator_id=str(payload.get("operator_id", "operator.unknown")), status="recorded",
                        created_at=_now(), run_id=run_id, lineage_id=str(run.get("lineage_id", "")), note=str(payload.get("note", "")),
                        reason=str(payload.get("reason", "operator_requested_replay_verification")), metadata={"replay_verification": report},
                    ))
                    _json_response(self, 201, {"read_only": False, "operator_action": action, "replay": report}); return
                if path.startswith("/api/runs/") and path.endswith("/commands"):
                    run_id = unquote(path.removeprefix("/api/runs/")[: -len("/commands")])
                    run = RunStore(root).get(run_id)
                    if run is None:
                        _json_response(self, 404, {"read_only": False, "status": "error", "error": "run_not_found"}); return
                    command = str(payload.get("command", ""))
                    lineage = _lineage_for_run(root, run)
                    policy = OperatorConsolePolicy().decide_run_command(command=command, stage_name=str(run.get("stage_name", "")), trust_level=str(lineage.get("trust_level", "unknown")))
                    status = "accepted" if bool(policy.get("allowed")) else "rejected"
                    action = _record_operator_action(root, OperatorActionRecord(
                        action_event_id=f"opc-{run_id}-{len(OperatorActionStore(root).all()) + 1}", action_kind="run_command", action=command,
                        operator_id=str(payload.get("operator_id", "operator.unknown")), status=status, created_at=_now(), run_id=run_id,
                        lineage_id=str(run.get("lineage_id", "")), note=str(payload.get("note", "")), reason=str(payload.get("reason", "")),
                        policy_decision=policy, metadata={"command_record_only": True},
                    ))
                    _json_response(self, 201 if status == "accepted" else 403, {"read_only": False, "operator_action": action}); return
            except (json.JSONDecodeError, ValueError) as exc:
                _json_response(self, 400, {"read_only": False, "status": "error", "error": str(exc)}); return
            _json_response(self, 405, OperatorConsolePayload(status="error", error="method_not_allowed", read_only=True).to_dict())

        def do_PUT(self) -> None:  # noqa: N802
            _json_response(self, 405, OperatorConsolePayload(status="error", error="method_not_allowed", read_only=True).to_dict())

        do_DELETE = do_PATCH = do_PUT

        def log_message(self, format: str, *args: object) -> None:
            return

    return ConsoleHandler


def serve(state_root: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state_root))
    server.serve_forever()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only stdlib Hephaestus Operator Console.")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(Path(args.state_root), args.host, args.port)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
