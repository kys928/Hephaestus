"""Safe, structured observability for the Prometheus Unbound research campaign.

Public Actions stream is metadata only. Raw model generations, credentials,
provider payloads and protected evidence remain in private SHA-checked storage.
Observability must never become an authorization source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

COMPONENTS = frozenset({"launcher", "watchdog", "worker"})
PHASES = frozenset({
    "START", "PREFLIGHT", "STORAGE_CHECK", "BUDGET_CHECK", "WATCHDOG_READY",
    "SOURCE_VERIFIED", "SOURCE_STAGED", "GPU_REQUESTED", "GPU_ALLOCATED",
    "GPU_POLLING", "BOOT", "SCIENTIFIC_INFERENCE", "INFERENCE_PROGRESS",
    "DECISION_RECORDED", "EVIDENCE_UPLOAD", "EVIDENCE_VERIFIED",
    "CLEANUP_REQUESTED", "CLEANUP_CONFIRMED", "NO_ALLOCATION", "COMPLETE",
    "BLOCKED", "FAILED", "DEADLINE",
})
STATUSES = frozenset({"start", "progress", "pass", "fail", "blocked"})
ALLOWED_DETAILS = frozenset({
    "pod_id", "datacenter", "gpu", "volume_id", "report",
    "receipt_count", "evidence_files", "elapsed_seconds", "lease_seconds",
    "manifest_sha256", "scientific_action", "exit_code", "reason_code",
    "source_sha", "campaign", "heartbeat_age_seconds",
})
TOKEN = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")


def event(component: str, phase: str, status: str = "progress",
          **details: object) -> dict:
    if component not in COMPONENTS or phase not in PHASES or status not in STATUSES:
        raise ValueError("unreviewed observability category")
    info = {}
    for key, value in details.items():
        if key not in ALLOWED_DETAILS or value is None:
            raise ValueError("unreviewed observability detail")
        if type(value) in (int, float, bool):
            if type(value) is float and not __import__("math").isfinite(value):
                raise ValueError("nonfinite observability metric")
            info[key] = value
        elif isinstance(value, str) and TOKEN.fullmatch(value):
            info[key] = value
        else:
            raise ValueError("unsafe or unbounded observability value")
    doc = {
        "schema": "prometheus-action-progress.v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "phase": phase,
        "status": status,
        "details": info,
    }
    # One machine-readable line per checkpoint, also visible immediately in UI.
    line = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print("PROMETHEUS_EVENT=" + line, file=sys.stdout, flush=True)
    local = os.environ.get("PROMETHEUS_ACTION_LOG")
    if local:
        path = Path(local)
        # An Actions runner uses $RUNNER_TEMP, commonly outside /tmp.
        # Enforce a local temporary root and disallow symlink traversal.
        roots = [Path("/tmp")]
        if os.environ.get("RUNNER_TEMP"):
            roots.append(Path(os.environ["RUNNER_TEMP"]))
        target = path.resolve()
        if (not path.is_absolute() or path.is_symlink()
                or not any(root.is_absolute() and target.is_relative_to(root.resolve())
                           for root in roots)):
            raise PermissionError("public action log must stay in a local temporary root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return doc


def summary_file(path: Path, *, job: str, max_events: int = 1000) -> str:
    if not path.is_file():
        return f"### Prometheus Unbound — {job}\n\nNo structured event log was created.\n"
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= max_events:
                raise ValueError("oversized structured action log")
            doc = json.loads(line)
            if (doc.get("schema") != "prometheus-action-progress.v1"
                    or doc.get("component") not in COMPONENTS
                    or doc.get("phase") not in PHASES
                    or doc.get("status") not in STATUSES):
                raise ValueError("untrusted structured action log row")
            rows.append(doc)
    last = rows[-1] if rows else None
    header = f"### Prometheus Unbound — {job}\n\n"
    if not last:
        return header + "No progress events were recorded.\n"
    table = "\n| UTC timestamp | Component | Phase | Status |\n|---|---|---|---|\n"
    for doc in rows[-30:]:
        table += f"| {doc['at']} | {doc['component']} | {doc['phase']} | {doc['status']} |\n"
    return (header + f"Recorded checkpoints: **{len(rows)}**. Last known phase: "
            f"**{last['phase']}** ({last['status']}).\n" + table
            + "\nA workflow or Pod exit is not scientific certification. "
            "Check independent source, receipt and Pod-cleanup gates separately.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("summary",))
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--job", choices=("launcher", "watchdog"), required=True)
    args = parser.parse_args()
    print(summary_file(args.path, job=args.job))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
