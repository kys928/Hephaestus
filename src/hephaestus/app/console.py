from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from hephaestus.control.replay_verification import verify_run_replay
from hephaestus.state.artifact_index import ArtifactIndex
from hephaestus.state.code_edit_proposal_store import CodeEditProposalStore
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.memory_store import MemoryStore
from hephaestus.state.report_store import ReportStore
from hephaestus.state.run_store import RunStore


@dataclass(frozen=True, slots=True)
class ConsoleConfig:
    state_root: Path


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pre(value: Any) -> str:
    return f"<pre>{_esc(_json(value))}</pre>"


def _page(title: str, state_root: Path, body: str, status: HTTPStatus = HTTPStatus.OK) -> tuple[int, str, bytes]:
    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>{_esc(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #172033; background: #f7f8fb; }}
a {{ color: #174ea6; }}
nav a {{ margin-right: 1rem; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #d8dee9; padding: .45rem; text-align: left; vertical-align: top; }}
th {{ background: #edf2f7; }}
.card {{ background: white; border: 1px solid #d8dee9; border-radius: .5rem; padding: 1rem; margin: 1rem 0; }}
.badge {{ display: inline-block; border-radius: 999px; padding: .1rem .5rem; background: #e2e8f0; }}
pre {{ white-space: pre-wrap; overflow-x: auto; background: #111827; color: #f9fafb; padding: .75rem; border-radius: .35rem; }}
.empty {{ color: #667085; font-style: italic; }}
</style>
</head>
<body>
<nav><a href=\"/\">Home</a><a href=\"/runs\">Runs</a><a href=\"/lineages\">Lineages</a><a href=\"/code-edits\">Code edits</a></nav>
<h1>{_esc(title)}</h1>
<p><strong>state_root:</strong> <code>{_esc(state_root)}</code></p>
{body}
</body>
</html>"""
    return int(status), "text/html; charset=utf-8", doc.encode("utf-8")


def _run_link(run_id: Any) -> str:
    text = str(run_id or "")
    return f'<a href="/runs/{quote(text, safe="")}">{_esc(text)}</a>' if text else ""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return '<p class="empty">No records found.</p>'
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _find_eval_report(reports: list[dict[str, Any]], run_id: str, eval_report_id: str | None) -> dict[str, Any] | None:
    if eval_report_id:
        for report in reversed(reports):
            if str(report.get("kind") or "") == "eval_report" and str(report.get("eval_id") or "") == eval_report_id:
                return report
    for report in reversed(reports):
        if str(report.get("kind") or "") == "eval_report" and str(report.get("run_id") or "") == run_id:
            return report
    return None


def _run_detail(root: Path, run_id: str) -> dict[str, Any] | None:
    run = RunStore(root).get(run_id)
    if not run:
        return None
    lineage_id = str(run.get("lineage_id") or "")
    manifest_id = str(run.get("data_manifest_id") or "")
    reports = ReportStore(root).all()
    return {
        "run": run,
        "lineage": LineageStore(root).get_current(lineage_id) if lineage_id else None,
        "manifest": ManifestStore(root).get(manifest_id) if manifest_id else None,
        "evaluation": _find_eval_report(reports, run_id, str(run.get("eval_report_id") or "") or None),
        "decision": DecisionStore(root).get(f"dec-{run_id}-exit"),
        "replay": verify_run_replay(root, run_id).to_dict(),
        "artifacts": [row for row in ArtifactIndex(root).all() if str(row.get("run_id") or "") == run_id],
        "memory": MemoryStore(root).list_for_run(run_id),
        "warnings": [],
    }


def render_home(root: Path) -> tuple[int, str, bytes]:
    runs = RunStore(root).all()
    latest = runs[-10:]
    rows = [[_run_link(r.get("run_id")), _esc(r.get("lineage_id")), _esc(r.get("stage_name")), _esc(r.get("status"))] for r in latest]
    body = f'<div class="card"><h2>Hephaestus Operator Console</h2><p>Run count: <span class="badge">{len(runs)}</span></p></div><h2>Latest runs</h2>{_table(["run_id", "lineage_id", "stage_name", "status"], rows)}<p><a href="/runs">Runs</a> · <a href="/lineages">Lineages</a> · <a href="/code-edits">Code edits</a></p>'
    return _page("Hephaestus Operator Console", root, body)


def render_runs(root: Path) -> tuple[int, str, bytes]:
    rows = []
    for r in RunStore(root).all():
        rows.append([_run_link(r.get("run_id")), _esc(r.get("lineage_id")), _esc(r.get("stage_name")), _esc(r.get("status")), _esc(r.get("judge_action") or r.get("effective_action")), _esc(r.get("checkpoint_ref"))])
    return _page("Runs", root, _table(["run_id", "lineage_id", "stage_name", "status", "judge_action", "checkpoint_ref"], rows))


def _section_value(value: Any) -> Any:
    return "Missing or empty" if value in (None, [], {}) else value


def render_run(root: Path, run_id: str) -> tuple[int, str, bytes]:
    detail = _run_detail(root, run_id)
    if detail is None:
        return _page("Run not found", root, f'<p>Run not found: <code>{_esc(run_id)}</code></p>', HTTPStatus.NOT_FOUND)
    labels = [("Run", "run"), ("Lineage", "lineage"), ("Manifest", "manifest"), ("Evaluation", "evaluation"), ("Judge Decision / Gates", "decision"), ("Replay Verification", "replay"), ("Artifacts", "artifacts"), ("Memory", "memory"), ("Warnings", "warnings")]
    body = "".join(f'<section class="card"><h2>{_esc(label)}</h2>{_pre(_section_value(detail.get(key)))}</section>' for label, key in labels)
    return _page(f"Run {run_id}", root, body)


def render_lineages(root: Path) -> tuple[int, str, bytes]:
    items = LineageStore(root).list_lineages()
    rows = []
    for lid in sorted(items):
        row = items[lid]
        rows.append([_esc(lid), _esc(row.get("latest_run_id")), _esc(row.get("status")), _esc(row.get("trust_level")), _esc(row.get("last_effective_action")), _esc(row.get("best_checkpoint_ref"))])
    return _page("Lineages", root, _table(["lineage_id", "latest_run_id", "status", "trust_level", "last_effective_action", "best_checkpoint_ref"], rows))


def _targets(value: Any) -> str:
    if isinstance(value, list):
        return _esc(", ".join(str(v) for v in value))
    return _esc(value)


def render_code_edits(root: Path) -> tuple[int, str, bytes]:
    rows = []
    for p in CodeEditProposalStore(root).list_all():
        rows.append([_esc(p.get("proposal_id")), _esc(p.get("run_id")), _esc(p.get("lineage_id")), _esc(p.get("status")), _esc(p.get("risk_level")), _targets(p.get("target_files"))])
    return _page("Code Edit Proposals", root, _table(["proposal_id", "run_id", "lineage_id", "status", "risk_level", "target_files"], rows))


def _api(payload: Any, status: HTTPStatus = HTTPStatus.OK) -> tuple[int, str, bytes]:
    return int(status), "application/json; charset=utf-8", _json(payload).encode("utf-8")


class ConsoleHandler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        root = self.server.config.state_root  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path == "/":
            response = render_home(root)
        elif path == "/runs":
            response = render_runs(root)
        elif path.startswith("/runs/"):
            response = render_run(root, unquote(path.removeprefix("/runs/")))
        elif path == "/lineages":
            response = render_lineages(root)
        elif path == "/code-edits":
            response = render_code_edits(root)
        elif path == "/api/runs":
            response = _api(RunStore(root).all())
        elif path.startswith("/api/runs/") and path.endswith("/replay"):
            run_id = unquote(path.removeprefix("/api/runs/").removesuffix("/replay"))
            response = _api(verify_run_replay(root, run_id).to_dict())
        elif path.startswith("/api/runs/"):
            run_id = unquote(path.removeprefix("/api/runs/"))
            detail = _run_detail(root, run_id)
            response = _api(detail if detail is not None else {"error": "run not found", "run_id": run_id}, HTTPStatus.OK if detail is not None else HTTPStatus.NOT_FOUND)
        elif path == "/api/code-edits":
            response = _api(CodeEditProposalStore(root).list_all())
        else:
            response = _page("Not found", root, f"<p>Unknown route: <code>{_esc(path)}</code></p>", HTTPStatus.NOT_FOUND)
        self.send_response(response[0])
        self.send_header("Content-Type", response[1])
        self.send_header("Content-Length", str(len(response[2])))
        self.end_headers()
        self.wfile.write(response[2])

    def log_message(self, format: str, *args: Any) -> None:
        return


class ConsoleServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: ConsoleConfig):
        super().__init__(server_address, ConsoleHandler)
        self.config = config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only Hephaestus Operator Console.")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ConsoleServer((args.host, args.port), ConsoleConfig(Path(args.state_root)))
    print(f"Hephaestus Operator Console serving {args.state_root} at http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
