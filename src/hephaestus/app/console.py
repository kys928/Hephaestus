from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from hephaestus.cli.inspect_run import _load as load_run
from hephaestus.control.replay_verification import verify_run_replay
from hephaestus.schemas.operator_console import OperatorConsolePayload
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.run_store import RunStore


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
            _json_response(self, 405, OperatorConsolePayload(status="error", error="method_not_allowed", read_only=True).to_dict())

        do_PUT = do_DELETE = do_PATCH = do_POST

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
