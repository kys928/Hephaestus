"""Tiny dependency-free model trainer used for bounded local lifecycle tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

_STOP_REASON: str | None = None


def _request_interrupt(_signum: int, _frame: object) -> None:
    global _STOP_REASON
    _STOP_REASON = "interrupted"


def _request_cancel(_signum: int, _frame: object) -> None:
    global _STOP_REASON
    _STOP_REASON = "cancelled"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_points(dataset_ref: Path) -> list[tuple[float, float]]:
    text = dataset_ref.read_text(encoding="utf-8")
    values = [ord(char) / 255.0 for char in text if not char.isspace()]
    if len(values) < 2:
        raise ValueError("fixture dataset must contain at least two non-whitespace characters")
    return list(pairwise(values))[:256]


def _event(events_ref: Path, run_id: str, step: int, status: str, message: str) -> None:
    _append_jsonl(events_ref, {
        "run_id": run_id,
        "step": step,
        "category": "status",
        "status": status,
        "message": message,
        "created_at_unix": time.time(),
    })


def _save_checkpoint(job: dict[str, Any], artifact_root: Path, step: int, weight: float, bias: float, loss: float) -> tuple[Path, Path]:
    checkpoint = artifact_root / f"checkpoint_step_{step}.json"
    checkpoint_payload = {
        "run_id": job["run_id"],
        "experiment_id": job["experiment_id"],
        "step": step,
        "epoch": None,
        "model": {"weight": weight, "bias": bias},
        "metric": {"loss": loss},
        "compatibility": job["compatibility"],
        "config_fingerprint": job["config_fingerprint"],
    }
    _write_json(checkpoint, checkpoint_payload)
    content_hash = _hash_file(checkpoint)
    record = artifact_root / "checkpoint_record.json"
    _write_json(record, {
        "run_id": job["run_id"],
        "experiment_id": job["experiment_id"],
        "checkpoint_ref": str(checkpoint),
        "step": step,
        "epoch": None,
        "metric_evidence": {"loss": loss, "metrics_ref": str(artifact_root / "metrics.jsonl")},
        "content_hash": content_hash,
        "hash_type": "sha256",
        "integrity_level": "content_hash_verified",
        "trainer_ref": "hephaestus.training.fixture_worker:v1",
        "config_ref": str(artifact_root / "prepared_job.json"),
        "tokenizer_ref": job["compatibility"]["tokenizer_ref"],
        "model_revision": job["compatibility"]["model_revision"],
        "resume_compatibility": dict(job["compatibility"]),
        "partial_write": False,
        "failure_status": None,
    })
    _write_json(artifact_root / "resume_token.json", {
        "run_id": job["run_id"],
        "checkpoint_ref": str(checkpoint),
        "checkpoint_hash": content_hash,
        "hash_type": "sha256",
        "config_fingerprint": job["config_fingerprint"],
        "compatibility": dict(job["compatibility"]),
    })
    return checkpoint, record


def run(job_path: Path, resume_token_path: Path | None = None) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    artifact_root = Path(job["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    events_ref = artifact_root / "events.jsonl"
    metrics_ref = artifact_root / "metrics.jsonl"
    result_ref = artifact_root / "runtime_result.json"
    points = _training_points(Path(job["data_contract_ref"]))

    start_step = 0
    weight = 0.0
    bias = 0.0
    if resume_token_path is not None:
        token = json.loads(resume_token_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(Path(token["checkpoint_ref"]).read_text(encoding="utf-8"))
        start_step = int(checkpoint["step"])
        weight = float(checkpoint["model"]["weight"])
        bias = float(checkpoint["model"]["bias"])
        _event(events_ref, job["run_id"], start_step, "resuming", "checkpoint state loaded")

    maximum_steps = int(job["max_steps"])
    learning_rate = float(job["learning_rate"])
    delay = float(job.get("step_delay_seconds", 0.0))
    checkpoint: Path | None = None
    record: Path | None = None
    last_loss = math.inf
    _event(events_ref, job["run_id"], start_step, "running", "bounded fixture optimization started")

    for step in range(start_step + 1, maximum_steps + 1):
        grad_weight = 0.0
        grad_bias = 0.0
        loss = 0.0
        for input_value, target in points:
            error = weight * input_value + bias - target
            loss += error * error
            grad_weight += 2.0 * error * input_value
            grad_bias += 2.0 * error
        scale = 1.0 / len(points)
        weight -= learning_rate * grad_weight * scale
        bias -= learning_rate * grad_bias * scale
        last_loss = loss * scale
        _append_jsonl(metrics_ref, {"run_id": job["run_id"], "step": step, "loss": last_loss})
        if delay:
            time.sleep(delay)
        if _STOP_REASON:
            checkpoint, record = _save_checkpoint(job, artifact_root, step, weight, bias, last_loss)
            _event(events_ref, job["run_id"], step, _STOP_REASON, f"graceful {_STOP_REASON}")
            _write_json(result_ref, {
                "status": _STOP_REASON,
                "step": step,
                "checkpoint_ref": str(checkpoint),
                "checkpoint_record_ref": str(record),
            })
            return 130 if _STOP_REASON == "interrupted" else 143

    omitted = {str(item) for item in job.get("omit_artifacts", [])}
    if "checkpoint" not in omitted:
        checkpoint, record = _save_checkpoint(job, artifact_root, maximum_steps, weight, bias, last_loss)
    metrics_summary = artifact_root / "metrics_summary.json"
    if "metrics" not in omitted:
        _write_json(metrics_summary, {
            "run_id": job["run_id"],
            "steps": maximum_steps,
            "final_loss": last_loss,
            "finite": math.isfinite(last_loss),
        })
    status = "completed"
    _event(events_ref, job["run_id"], maximum_steps, status, "bounded fixture optimization completed")
    _write_json(result_ref, {
        "status": status,
        "step": maximum_steps,
        "checkpoint_ref": str(checkpoint) if checkpoint else None,
        "checkpoint_record_ref": str(record) if record else None,
        "metrics_ref": str(metrics_summary) if metrics_summary.exists() else None,
    })
    return int(job.get("force_exit_code", 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--resume-token", type=Path)
    arguments = parser.parse_args()
    signal.signal(signal.SIGINT, _request_interrupt)
    signal.signal(signal.SIGTERM, _request_cancel)
    try:
        return run(arguments.job, arguments.resume_token)
    except Exception as exc:  # noqa: BLE001 - process boundary must persist unknown failures
        try:
            job = json.loads(arguments.job.read_text(encoding="utf-8"))
            root = Path(job["artifact_root"])
            root.mkdir(parents=True, exist_ok=True)
            _write_json(root / "runtime_result.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            _event(root / "events.jsonl", str(job.get("run_id", "unknown")), 0, "failed", "fixture worker failed")
        finally:
            print(f"fixture worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
