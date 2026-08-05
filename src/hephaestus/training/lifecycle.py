"""Honest local subprocess implementation of the autonomous training lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import (
    ExperimentProposal,
    TrainingControlRequest,
    TrainingRunHandle,
)

_ACTIVE_STATUSES = {"preparing", "queued", "running", "interrupting", "resuming"}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_SUPPORTED_ACTIONS = {"interrupt", "resume", "cancel", "status"}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COMPATIBILITY_FIELDS = (
    "backend_id",
    "model_id",
    "model_revision",
    "architecture_family",
    "tokenizer_ref",
    "training_recipe_ref",
    "data_contract_ref",
    "data_contract_hash",
)


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _issue(code: str, category: str, message: str, *, evidence_refs: list[str] | None = None, retryable: bool = False) -> ContractIssue:
    return ContractIssue(
        code=code,
        category=category,
        message=message,
        retryable=retryable,
        blocking=True,
        evidence_refs=evidence_refs or [],
    )


@dataclass(slots=True)
class _ManagedRun:
    handle: TrainingRunHandle
    job_ref: Path
    process: subprocess.Popen[bytes] | None = None
    log_stream: IO[bytes] | None = None
    requested_action: str | None = None
    compatibility: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LocalTrainingLifecycleService:
    """Launch a tiny real training process and persist every decision-critical ref."""

    artifact_root: Path
    python_executable: str = sys.executable
    maximum_allowed_steps: int = 10_000
    backend_id: str = "local_fixture"
    _runs: dict[str, _ManagedRun] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        with self._lock:
            if not _RUN_ID_PATTERN.fullmatch(proposal.run_id):
                return self._failure_handle(proposal, _issue(
                    "invalid_run_id",
                    "invalid_request",
                    "run_id must use 1-128 ASCII letters, digits, dots, underscores, or hyphens.",
                ))
            existing = self._runs.get(proposal.run_id)
            if existing and existing.handle.status in _ACTIVE_STATUSES:
                return self._with_issue(existing.handle, _issue(
                    "duplicate_active_run", "policy_blocked", "An active run already exists for this run_id."
                ))
            run_root = self.artifact_root / proposal.run_id
            if run_root.exists():
                return self._failure_handle(proposal, _issue(
                    "existing_run_evidence", "policy_blocked", "Existing run evidence prevents an implicit overwrite.",
                    evidence_refs=[str(run_root)],
                ))

            job, issues = self._prepare_job(proposal, run_root)
            if issues:
                handle = self._failure_handle(proposal, *issues)
                self._persist_handle(handle)
                return handle

            run_root.mkdir(parents=True, exist_ok=False)
            job_ref = run_root / "prepared_job.json"
            _write_json(job_ref, job)
            events_ref = run_root / "events.jsonl"
            _append_jsonl(events_ref, {
                "run_id": proposal.run_id,
                "status": "preparing",
                "category": "status",
                "message": "prepared job validated",
                "created_at_unix": time.time(),
            })
            handle = TrainingRunHandle(
                run_id=proposal.run_id,
                experiment_id=proposal.experiment_id,
                backend_id=self.backend_id,
                status="preparing",
                event_stream_ref=str(events_ref),
                metadata={
                    "artifact_root": str(run_root),
                    "prepared_job_ref": str(job_ref),
                    "log_ref": str(run_root / "runtime.log"),
                    "checkpoint_record_ref": str(run_root / "checkpoint_record.json"),
                    "incidents_ref": str(run_root / "incidents.jsonl"),
                },
            )
            managed = _ManagedRun(handle=handle, job_ref=job_ref, compatibility=dict(job["compatibility"]))
            self._runs[proposal.run_id] = managed
            try:
                self._start(managed)
            except OSError as exc:
                if managed.log_stream is not None:
                    managed.log_stream.close()
                    managed.log_stream = None
                managed.handle.status = "failed"
                self._record_incident(
                    managed,
                    "process_launch_failure",
                    f"Training subprocess could not start: {type(exc).__name__}: {exc}",
                    [str(job_ref)],
                )
                self._persist_handle(managed.handle)
                return TrainingRunHandle.from_dict(managed.handle.to_dict())
            managed.handle.status = "running"
            managed.handle.metadata["pid"] = managed.process.pid if managed.process else None
            self._persist_handle(managed.handle)
            return TrainingRunHandle.from_dict(managed.handle.to_dict())

    def status(self, run_id: str) -> TrainingRunHandle:
        with self._lock:
            if not _RUN_ID_PATTERN.fullmatch(run_id):
                return TrainingRunHandle(
                    run_id=run_id,
                    experiment_id="unknown",
                    backend_id=self.backend_id,
                    status="failed",
                    issues=[_issue("invalid_run_id", "invalid_request", "run_id has an unsafe or unsupported format.")],
                )
            managed = self._runs.get(run_id)
            if managed is None:
                restored = self._restore(run_id)
                if restored is None:
                    return TrainingRunHandle(
                        run_id=run_id,
                        experiment_id="unknown",
                        backend_id=self.backend_id,
                        status="failed",
                        issues=[_issue("run_not_found", "invalid_request", f"No lifecycle evidence exists for run '{run_id}'.")],
                    )
                if restored.status in _ACTIVE_STATUSES:
                    pid = restored.metadata.get("pid")
                    if isinstance(pid, int) and self._pid_alive(pid):
                        restored.metadata["process_observation"] = "pid_alive_unattached"
                        return restored
                    restored.status = "failed"
                    restored.issues.append(_issue(
                        "process_exit_evidence_missing",
                        "missing_evidence",
                        "The service restarted and cannot verify the active process exit status.",
                        evidence_refs=[str(self.artifact_root / run_id / "handle.json")],
                        retryable=True,
                    ))
                    self._persist_handle(restored)
                return restored
            if managed.process is not None and managed.process.poll() is None:
                if managed.handle.status == "resuming":
                    managed.handle.status = "running"
                    self._persist_handle(managed.handle)
                return TrainingRunHandle.from_dict(managed.handle.to_dict())
            if managed.handle.status not in _TERMINAL_STATUSES and managed.handle.status != "interrupted":
                self._finalize(managed)
            return TrainingRunHandle.from_dict(managed.handle.to_dict())

    def control(self, request: TrainingControlRequest) -> TrainingRunHandle:
        if request.action not in _SUPPORTED_ACTIONS:
            current = self.status(request.run_id)
            return self._with_issue(current, _issue(
                "unsupported_control_action", "invalid_request", f"Unsupported training control action: {request.action}"
            ))
        if request.action == "status":
            return self.status(request.run_id)
        with self._lock:
            managed = self._runs.get(request.run_id)
            if managed is None:
                restored = self._restore(request.run_id)
                if restored is None or request.action != "resume":
                    return self.status(request.run_id)
                job_ref = Path(str(restored.metadata.get("prepared_job_ref", "")))
                job = json.loads(job_ref.read_text(encoding="utf-8"))
                managed = _ManagedRun(restored, job_ref, compatibility=dict(job["compatibility"]))
                self._runs[request.run_id] = managed

            self.status(request.run_id)
            if request.action in {"interrupt", "cancel"}:
                return self._stop(managed, request.action)
            return self._resume(managed, request)

    def _prepare_job(self, proposal: ExperimentProposal, run_root: Path) -> tuple[dict[str, Any], list[ContractIssue]]:
        constraints = dict(proposal.training_constraints)
        issues: list[ContractIssue] = []
        if proposal.status not in {"ready", "approved"}:
            issues.append(_issue("experiment_not_ready", "policy_blocked", "Experiment proposal must be ready or approved before launch."))
        backend_id = str(constraints.get("backend_id", self.backend_id))
        if backend_id != self.backend_id:
            issues.append(_issue("unsupported_backend", "unsupported_capability", f"Local lifecycle does not support backend '{backend_id}'."))
        compatibility = {name: str(constraints.get(name, "")).strip() for name in _COMPATIBILITY_FIELDS}
        compatibility["backend_id"] = backend_id
        for name in _COMPATIBILITY_FIELDS:
            if not compatibility.get(name):
                issues.append(_issue(f"missing_{name}", "missing_evidence", f"Prepared job requires {name}."))

        data_ref = Path(compatibility.get("data_contract_ref", ""))
        if compatibility.get("data_contract_ref") and not data_ref.is_file():
            issues.append(_issue("data_contract_missing", "artifact_integrity", "Prepared data artifact does not exist.", evidence_refs=[str(data_ref)]))
        elif data_ref.is_file() and compatibility.get("data_contract_hash") != _hash_file(data_ref):
            issues.append(_issue("data_contract_hash_mismatch", "artifact_integrity", "Prepared data content hash does not match the proposal.", evidence_refs=[str(data_ref)]))

        try:
            max_steps = int(constraints.get("max_steps", 0))
        except (TypeError, ValueError):
            max_steps = 0
        if max_steps <= 0 or max_steps > self.maximum_allowed_steps:
            issues.append(_issue("invalid_step_budget", "budget_exceeded", f"max_steps must be between 1 and {self.maximum_allowed_steps}."))
        try:
            learning_rate = float(constraints.get("learning_rate", 0.05))
        except (TypeError, ValueError):
            learning_rate = 0.0
        if not 0.0 < learning_rate <= 1.0:
            issues.append(_issue("invalid_learning_rate", "invalid_request", "learning_rate must be in (0, 1]."))

        try:
            step_delay_seconds = float(constraints.get("step_delay_seconds", 0.0))
        except (TypeError, ValueError):
            step_delay_seconds = -1.0
        if step_delay_seconds < 0.0 or step_delay_seconds > 1.0:
            issues.append(_issue("invalid_step_delay", "invalid_request", "step_delay_seconds must be between 0 and 1."))
        try:
            force_exit_code = int(constraints.get("force_exit_code", 0))
        except (TypeError, ValueError):
            force_exit_code = 1
            issues.append(_issue("invalid_force_exit_code", "invalid_request", "force_exit_code must be an integer."))
        raw_omitted = constraints.get("omit_artifacts", [])
        if not isinstance(raw_omitted, (list, tuple, set)):
            raw_omitted = []
            issues.append(_issue("invalid_omit_artifacts", "invalid_request", "omit_artifacts must be a list."))

        job: dict[str, Any] = {
            "run_id": proposal.run_id,
            "experiment_id": proposal.experiment_id,
            "lineage_id": proposal.lineage_id,
            "stage_name": proposal.stage_name,
            "artifact_root": str(run_root),
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "step_delay_seconds": step_delay_seconds,
            "force_exit_code": force_exit_code,
            "omit_artifacts": sorted({str(item) for item in raw_omitted}),
            "data_contract_ref": compatibility.get("data_contract_ref", ""),
            "compatibility": compatibility,
        }
        job["config_fingerprint"] = _canonical_hash({
            "compatibility": compatibility,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
        })
        return job, issues

    def _start(self, managed: _ManagedRun, resume_token: Path | None = None) -> None:
        log_ref = Path(str(managed.handle.metadata["log_ref"]))
        log_ref.parent.mkdir(parents=True, exist_ok=True)
        command = [self.python_executable, str(Path(__file__).with_name("fixture_worker.py")), "--job", str(managed.job_ref)]
        if resume_token is not None:
            command.extend(["--resume-token", str(resume_token)])
        managed.log_stream = log_ref.open("ab")
        managed.process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            stdout=managed.log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _stop(self, managed: _ManagedRun, action: str) -> TrainingRunHandle:
        if managed.process is None or managed.process.poll() is not None or managed.handle.status not in {"running", "resuming"}:
            return self._with_issue(managed.handle, _issue(
                f"cannot_{action}", "policy_blocked", f"Run must be active to {action}."
            ))
        managed.requested_action = action
        managed.handle.status = "interrupting"
        managed.handle.metadata["control_action"] = action
        try:
            os.killpg(managed.process.pid, signal.SIGINT if action == "interrupt" else signal.SIGTERM)
        except ProcessLookupError:
            self._finalize(managed)
            return self._with_issue(managed.handle, _issue(
                f"{action}_race", "runtime_failure", "The process exited before the control signal was delivered.", retryable=True
            ))
        self._persist_handle(managed.handle)
        return TrainingRunHandle.from_dict(managed.handle.to_dict())

    def _resume(self, managed: _ManagedRun, request: TrainingControlRequest) -> TrainingRunHandle:
        if managed.handle.status != "interrupted":
            return self._with_issue(managed.handle, _issue(
                "resume_state_invalid", "policy_blocked", "Only an interrupted run may be resumed."
            ))
        token_ref = Path(str(managed.handle.resume_token_ref or ""))
        issue = self._validate_resume(managed, token_ref, request.metadata)
        if issue:
            return self._with_issue(managed.handle, issue)
        managed.requested_action = None
        managed.handle.status = "resuming"
        try:
            self._start(managed, token_ref)
        except OSError as exc:
            if managed.log_stream is not None:
                managed.log_stream.close()
                managed.log_stream = None
            managed.handle.status = "failed"
            self._record_incident(
                managed,
                "resume_launch_failure",
                f"Resume subprocess could not start: {type(exc).__name__}: {exc}",
                [str(token_ref), str(managed.job_ref)],
            )
            self._persist_handle(managed.handle)
            return TrainingRunHandle.from_dict(managed.handle.to_dict())
        managed.handle.metadata["pid"] = managed.process.pid if managed.process else None
        self._persist_handle(managed.handle)
        return TrainingRunHandle.from_dict(managed.handle.to_dict())

    def _validate_resume(self, managed: _ManagedRun, token_ref: Path, requested: dict[str, object]) -> ContractIssue | None:
        evidence = [str(token_ref)]
        if not token_ref.is_file():
            return _issue("resume_token_missing", "missing_evidence", "Resume token is missing.", evidence_refs=evidence)
        try:
            token = json.loads(token_ref.read_text(encoding="utf-8"))
            job = json.loads(managed.job_ref.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _issue("resume_evidence_invalid", "artifact_integrity", f"Resume evidence is unreadable: {exc}", evidence_refs=evidence)
        checkpoint = Path(str(token.get("checkpoint_ref", "")))
        evidence.append(str(checkpoint))
        if not checkpoint.is_file() or not token.get("checkpoint_hash"):
            return _issue("checkpoint_integrity_missing", "artifact_integrity", "Resume requires a checkpoint and computed content hash.", evidence_refs=evidence)
        if _hash_file(checkpoint) != token["checkpoint_hash"]:
            return _issue("checkpoint_hash_mismatch", "artifact_integrity", "Checkpoint content hash validation failed.", evidence_refs=evidence)
        if token.get("config_fingerprint") != job.get("config_fingerprint"):
            return _issue("resume_config_mismatch", "incompatible_candidate", "Training configuration changed since interruption.", evidence_refs=evidence)
        token_compatibility = token.get("compatibility")
        if not isinstance(token_compatibility, dict):
            return _issue("resume_compatibility_missing", "missing_evidence", "Resume compatibility evidence is missing.", evidence_refs=evidence)
        for name in _COMPATIBILITY_FIELDS:
            expected = managed.compatibility.get(name)
            if token_compatibility.get(name) != expected:
                return _issue("resume_compatibility_mismatch", "incompatible_candidate", f"Resume {name} is incompatible.", evidence_refs=evidence)
            if name in requested and str(requested[name]) != expected:
                return _issue("resume_request_mismatch", "incompatible_candidate", f"Requested {name} does not match the interrupted run.", evidence_refs=evidence)
        return None

    def _finalize(self, managed: _ManagedRun) -> None:
        process = managed.process
        returncode = process.poll() if process is not None else None
        if managed.log_stream is not None:
            managed.log_stream.close()
            managed.log_stream = None
        root = managed.job_ref.parent
        result_ref = root / "runtime_result.json"
        result: dict[str, Any] = {}
        if result_ref.is_file():
            try:
                result = json.loads(result_ref.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                managed.handle.issues.append(_issue("runtime_result_invalid", "artifact_integrity", "Runtime result is malformed.", evidence_refs=[str(result_ref)]))
        worker_status = str(result.get("status", "failed"))
        if managed.requested_action == "interrupt" and worker_status == "interrupted":
            status = "interrupted"
        elif managed.requested_action == "cancel":
            status = "cancelled"
        elif returncode == 0 and worker_status == "completed":
            status = "completed"
        else:
            status = "failed"
            self._record_incident(managed, "non_zero_exit", f"Training process exited with code {returncode} and status {worker_status}.", [str(result_ref)])

        managed.handle.status = status
        managed.handle.metadata["returncode"] = returncode
        managed.handle.metadata["runtime_result_ref"] = str(result_ref)
        metrics_ref = root / "metrics_summary.json"
        record_ref = root / "checkpoint_record.json"
        events_ref = root / "events.jsonl"
        log_ref = root / "runtime.log"
        required = [events_ref, log_ref]
        if status == "completed":
            required.extend([metrics_ref, record_ref])
        missing = [path for path in required if not path.is_file()]
        if missing and status == "completed":
            status = managed.handle.status = "failed"
            self._record_incident(managed, "missing_required_artifact", "Training completed without all required artifacts.", [str(path) for path in missing])

        if record_ref.is_file():
            try:
                record = json.loads(record_ref.read_text(encoding="utf-8"))
                checkpoint = Path(str(record.get("checkpoint_ref", "")))
                content_hash = str(record.get("content_hash", ""))
                if not checkpoint.is_file() or not content_hash or _hash_file(checkpoint) != content_hash:
                    managed.handle.status = "failed"
                    self._record_incident(managed, "checkpoint_integrity_failure", "Checkpoint record failed content-hash verification.", [str(record_ref), str(checkpoint)])
                else:
                    managed.handle.checkpoint_refs = [str(checkpoint)]
                    managed.handle.resume_token_ref = str(root / "resume_token.json")
            except (OSError, json.JSONDecodeError) as exc:
                managed.handle.status = "failed"
                self._record_incident(managed, "checkpoint_record_invalid", f"Checkpoint record is unreadable: {exc}", [str(record_ref)])
        managed.handle.metrics_ref = str(metrics_ref) if metrics_ref.is_file() else None
        self._persist_handle(managed.handle)

    def _record_incident(self, managed: _ManagedRun, code: str, message: str, evidence_refs: list[str]) -> None:
        incident_ref = Path(str(managed.handle.metadata["incidents_ref"]))
        _append_jsonl(incident_ref, {
            "incident_id": f"inc-{managed.handle.run_id}-{code}",
            "run_id": managed.handle.run_id,
            "category": "runtime",
            "code": code,
            "message": message,
            "evidence_refs": evidence_refs,
        })
        managed.handle.issues.append(_issue(code, "runtime_failure" if code == "non_zero_exit" else "artifact_integrity", message, evidence_refs=evidence_refs, retryable=True))

    def _persist_handle(self, handle: TrainingRunHandle) -> None:
        if not _RUN_ID_PATTERN.fullmatch(handle.run_id):
            raise ValueError("refusing to persist a handle with an unsafe run_id")
        root = self.artifact_root / handle.run_id
        _write_json(root / "handle.json", handle.to_dict())

    def _restore(self, run_id: str) -> TrainingRunHandle | None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return None
        path = self.artifact_root / run_id / "handle.json"
        if not path.is_file():
            return None
        try:
            return TrainingRunHandle.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _failure_handle(self, proposal: ExperimentProposal, *issues: ContractIssue) -> TrainingRunHandle:
        return TrainingRunHandle(
            run_id=proposal.run_id,
            experiment_id=proposal.experiment_id,
            backend_id=self.backend_id,
            status="failed",
            issues=list(issues),
            metadata={"artifact_root": str(self.artifact_root)},
        )

    @staticmethod
    def _with_issue(handle: TrainingRunHandle, issue: ContractIssue) -> TrainingRunHandle:
        payload = handle.to_dict()
        payload["issues"] = [*payload.get("issues", []), issue.to_dict()]
        return TrainingRunHandle.from_dict(payload)


@dataclass(slots=True)
class FakeTrainingLifecycleService:
    """Deterministic in-memory lifecycle for consumer tests."""

    backend_id: str = "fake"
    _handles: dict[str, TrainingRunHandle] = field(default_factory=dict, init=False)

    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        handle = TrainingRunHandle(
            run_id=proposal.run_id,
            experiment_id=proposal.experiment_id,
            backend_id=self.backend_id,
            status="completed",
            checkpoint_refs=[f"fixture://checkpoints/{proposal.run_id}/final"],
            metrics_ref=f"fixture://metrics/{proposal.run_id}",
            event_stream_ref=f"fixture://events/{proposal.run_id}",
            metadata={"deterministic_fake": True},
        )
        self._handles[proposal.run_id] = handle
        return TrainingRunHandle.from_dict(handle.to_dict())

    def status(self, run_id: str) -> TrainingRunHandle:
        return TrainingRunHandle.from_dict(self._handles[run_id].to_dict())

    def control(self, request: TrainingControlRequest) -> TrainingRunHandle:
        return self.status(request.run_id)
