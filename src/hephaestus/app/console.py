from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hephaestus.cli.inspect_run import _load as load_run
from hephaestus.schemas.operator_console import OperatorConsolePayload
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


def _index_html(state_root: Path) -> str:
    runs = RunStore(state_root).all()
    items = "".join(
        f'<li><a href="/run?run_id={html.escape(str(run.get("run_id", "")))}">{html.escape(str(run.get("run_id", "")))}</a> — {html.escape(str(run.get("status", "")))}</li>'
        for run in runs
    ) or "<li>No run records found.</li>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Hephaestus Operator Console</title></head><body><h1>Hephaestus Operator Console</h1><p><strong>Read-only:</strong> this console performs inspection only and exposes no approval, rejection, execution, or training-launch actions.</p><p>State root: <code>{html.escape(str(state_root))}</code></p><h2>Runs</h2><ul>{items}</ul></body></html>"""


def make_handler(state_root: Path) -> type[BaseHTTPRequestHandler]:
    root = state_root

    class ConsoleHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                _html_response(self, 200, _index_html(root))
                return
            if parsed.path == "/healthz":
                _json_response(self, 200, OperatorConsolePayload(status="ok", read_only=True).to_dict())
                return
            if parsed.path == "/api/runs":
                _json_response(self, 200, OperatorConsolePayload(runs=RunStore(root).all(), read_only=True).to_dict())
                return
            if parsed.path in {"/api/run", "/run"}:
                run_id = (parse_qs(parsed.query).get("run_id") or [""])[0]
                if not run_id:
                    _json_response(self, 400, OperatorConsolePayload(status="error", error="run_id_required", read_only=True).to_dict())
                    return
                try:
                    payload = load_run(root, run_id)
                except ValueError as exc:
                    _json_response(self, 404, OperatorConsolePayload(status="error", error=str(exc), read_only=True).to_dict())
                    return
                if parsed.path == "/api/run":
                    _json_response(self, 200, OperatorConsolePayload(run=payload, read_only=True).to_dict())
                else:
                    escaped = html.escape(json.dumps(payload, indent=2, sort_keys=True))
                    _html_response(self, 200, f"<!doctype html><html><body><p><a href='/'>Runs</a></p><h1>Run {html.escape(run_id)}</h1><pre>{escaped}</pre></body></html>")
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
