"""Governed subprocess lifecycle for optional Transformers causal-LM training."""

from __future__ import annotations

import hashlib
import json
import math
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

from hephaestus.backends.hf_causal_lm import (
    directory_content_identity,
    observable_device_memory,
    transformers_training_capability,
)
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import (
    ExperimentProposal,
    TrainingControlRequest,
    TrainingRunHandle,
)
from hephaestus.schemas.trainable_data_contract import TrainableDataContract

_ACTIVE_STATUSES = {"preparing", "queued", "running", "interrupting", "resuming"}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_SUPPORTED_ACTIONS = {"interrupt", "resume", "cancel", "status"}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMMUTABLE_REMOTE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_RESUME_FIELDS = (
    "backend_id",
    "model_id",
    "model_revision",
    "architecture_family",
    "tokenizer_id",
    "tokenizer_revision",
    "training_mode",
    "trainable_data_contract_ref",
    "trainable_data_contract_hash",
    "processed_dataset_ref",
    "processed_dataset_hash",
    "optimizer",
    "scheduler",
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _issue(
    code: str,
    category: str,
    message: str,
    *,
    evidence_refs: list[str] | None = None,
    retryable: bool = False,
    metadata: dict[str, object] | None = None,
) -> ContractIssue:
    return ContractIssue(
        code=code,
        category=category,
        message=message,
        retryable=retryable,
        blocking=True,
        evidence_refs=evidence_refs or [],
        metadata=metadata or {},
    )


def validate_checkpoint_manifest(checkpoint: str | Path) -> tuple[bool, str]:
    """Verify every checkpoint component and the canonical manifest hash."""

    root = Path(checkpoint)
    if not root.is_dir() or root.name.endswith(".partial"):
        return False, "checkpoint directory is missing or partial"
    manifest_ref = root / "checkpoint_manifest.json"
    if not manifest_ref.is_file():
        return False, "checkpoint manifest is missing"
    try:
        manifest = json.loads(manifest_ref.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "checkpoint manifest is unreadable"
    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        return False, "checkpoint component hashes are missing"
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_ref
    }
    if actual_paths != set(components):
        return False, "checkpoint component set differs from the manifest"
    for relative, expected in components.items():
        candidate = root / str(relative)
        if not candidate.is_file() or _hash_file(candidate) != expected:
            return False, f"checkpoint component hash mismatch: {relative}"
    if manifest.get("manifest_hash") != _canonical_hash(components):
        return False, "checkpoint manifest hash mismatch"
    if manifest.get("partial_write") is not False:
        return False, "checkpoint is not finalized"
    return True, str(manifest["manifest_hash"])


@dataclass(slots=True)
class _ManagedRun:
    handle: TrainingRunHandle
    job_ref: Path
    process: subprocess.Popen[bytes] | None = None
    log_stream: IO[bytes] | None = None
    requested_action: str | None = None
    compatibility: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class TransformersTrainingLifecycleService:
    """Run bounded full fine-tuning in an isolated optional-ML subprocess."""

    artifact_root: Path
    python_executable: str = sys.executable
    maximum_allowed_steps: int = 100_000
    maximum_dataset_bytes: int = 512 * 1024 * 1024
    maximum_rows: int = 1_000_000
    backend_id: str = "transformers_causal_lm"
    _runs: dict[str, _ManagedRun] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def launch(self, proposal: ExperimentProposal) -> TrainingRunHandle:
        with self._lock:
            if not _RUN_ID_PATTERN.fullmatch(proposal.run_id):
                return self._failure_handle(
                    proposal,
                    _issue("invalid_run_id", "invalid_request", "run_id has an unsafe or unsupported format."),
                )
            existing = self._runs.get(proposal.run_id)
            if existing and existing.handle.status in _ACTIVE_STATUSES:
                return self._with_issue(
                    existing.handle,
                    _issue("duplicate_active_run", "policy_blocked", "An active run already exists."),
                )
            run_root = self.artifact_root / proposal.run_id
            if run_root.exists():
                return self._failure_handle(
                    proposal,
                    _issue(
                        "existing_run_evidence",
                        "policy_blocked",
                        "Existing run evidence prevents implicit overwrite.",
                        evidence_refs=[str(run_root)],
                    ),
                )

            job, issues = self._prepare_job(proposal, run_root)
            if issues:
                handle = self._failure_handle(proposal, *issues)
                handle.metadata.update({"capability": job.get("capability", {})})
                self._persist_handle(handle)
                return handle

            run_root.mkdir(parents=True, exist_ok=False)
            job_ref = run_root / "prepared_job.json"
            _write_json(job_ref, job)
            events_ref = run_root / "events.jsonl"
            _append_jsonl(
                events_ref,
                {
                    "run_id": proposal.run_id,
                    "status": "preparing",
                    "category": "status",
                    "message": "Transformers training job validated",
                    "created_at_unix": time.time(),
                },
            )
            handle = TrainingRunHandle(
                run_id=proposal.run_id,
                experiment_id=proposal.experiment_id,
                backend_id=self.backend_id,
                status="preparing",
                event_stream_ref=str(events_ref),
                metadata={
                    "artifact_root": str(run_root),
                    "prepared_job_ref": str(job_ref),
                    "normalized_config_ref": str(run_root / "normalized_training_config.json"),
                    "resource_estimate_ref": str(run_root / "resource_estimate.json"),
                    "log_ref": str(run_root / "runtime.log"),
                    "checkpoint_record_ref": str(run_root / "checkpoint_record.json"),
                    "incidents_ref": str(run_root / "incidents.jsonl"),
                    "final_result_ref": str(run_root / "runtime_result.json"),
                    "config_fingerprint": job["config_fingerprint"],
                },
            )
            _write_json(Path(str(handle.metadata["normalized_config_ref"])), dict(job["training_config"]))
            _write_json(Path(str(handle.metadata["resource_estimate_ref"])), dict(job["resource_estimate"]))
            managed = _ManagedRun(handle, job_ref, compatibility=dict(job["compatibility"]))
            self._runs[proposal.run_id] = managed
            managed.handle.status = "queued"
            _append_jsonl(
                events_ref,
                {
                    "run_id": proposal.run_id,
                    "status": "queued",
                    "category": "status",
                    "message": "validated job queued for local subprocess launch",
                    "created_at_unix": time.time(),
                },
            )
            self._persist_handle(managed.handle)
            try:
                self._start(managed)
            except OSError as exc:
                self._close_log(managed)
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
                    issues=[_issue("invalid_run_id", "invalid_request", "run_id is unsafe.")],
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
                        issues=[_issue("run_not_found", "invalid_request", f"No run evidence for '{run_id}'.")],
                    )
                if restored.status in _ACTIVE_STATUSES:
                    pid = restored.metadata.get("pid")
                    if isinstance(pid, int) and self._pid_alive(pid):
                        restored.metadata["process_observation"] = "pid_alive_unattached"
                        return restored
                    restored.status = "failed"
                    restored.issues.append(
                        _issue(
                            "process_exit_evidence_missing",
                            "missing_evidence",
                            "The service restarted and cannot verify the process exit status.",
                            retryable=True,
                        )
                    )
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
            return self._with_issue(
                self.status(request.run_id),
                _issue("unsupported_control_action", "invalid_request", f"Unsupported action: {request.action}"),
            )
        if request.action == "status":
            return self.status(request.run_id)
        with self._lock:
            managed = self._runs.get(request.run_id)
            if managed is None:
                restored = self._restore(request.run_id)
                if restored is None or request.action != "resume":
                    return self.status(request.run_id)
                job_ref = Path(str(restored.metadata.get("prepared_job_ref", "")))
                try:
                    job = json.loads(job_ref.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return self._with_issue(
                        restored,
                        _issue("prepared_job_invalid", "artifact_integrity", "Prepared job is unreadable."),
                    )
                managed = _ManagedRun(restored, job_ref, compatibility=dict(job["compatibility"]))
                self._runs[request.run_id] = managed
            self.status(request.run_id)
            if request.action in {"interrupt", "cancel"}:
                return self._stop(managed, request.action)
            return self._resume(managed, request)

    def _prepare_job(
        self, proposal: ExperimentProposal, run_root: Path
    ) -> tuple[dict[str, Any], list[ContractIssue]]:
        constraints = dict(proposal.training_constraints)
        issues: list[ContractIssue] = []
        capability = transformers_training_capability()
        if not capability.supported:
            issues.append(
                _issue(
                    "transformers_training_unavailable",
                    "unsupported_capability",
                    "Optional Transformers training dependencies are unavailable.",
                    metadata={"missing_packages": list(capability.missing_packages)},
                )
            )
        if proposal.status not in {"ready", "approved"}:
            issues.append(
                _issue("experiment_not_ready", "policy_blocked", "Proposal must be ready or approved.")
            )
        backend_id = str(constraints.get("backend_id", self.backend_id)).strip()
        if backend_id != self.backend_id:
            issues.append(
                _issue("unsupported_backend", "unsupported_capability", f"Unsupported backend '{backend_id}'.")
            )

        compatibility = {
            field_name: str(constraints.get(field_name, "")).strip() for field_name in _RESUME_FIELDS
        }
        compatibility["backend_id"] = backend_id
        for field_name in _RESUME_FIELDS:
            if not compatibility[field_name]:
                issues.append(
                    _issue(
                        f"missing_{field_name}",
                        "missing_evidence",
                        f"Prepared job requires {field_name}.",
                    )
                )

        model_id = compatibility["model_id"]
        tokenizer_id = compatibility["tokenizer_id"]
        external = bool(model_id and not Path(model_id).expanduser().is_dir())
        self._validate_identity(
            kind="model",
            identifier=model_id,
            revision=compatibility["model_revision"],
            external=external,
            issues=issues,
        )
        tokenizer_external = bool(tokenizer_id and not Path(tokenizer_id).expanduser().is_dir())
        self._validate_identity(
            kind="tokenizer",
            identifier=tokenizer_id,
            revision=compatibility["tokenizer_revision"],
            external=tokenizer_external,
            issues=issues,
        )
        if bool(constraints.get("trust_remote_code", False)):
            issues.append(
                _issue(
                    "remote_code_forbidden",
                    "policy_blocked",
                    "Remote model code is disabled for this backend.",
                )
            )
        if external or tokenizer_external:
            self._validate_remote_policy(constraints, issues)

        data_summary = self._validate_data(compatibility, constraints, issues)
        training_config = self._normalize_training_config(
            compatibility, constraints, capability.framework_versions, data_summary, issues
        )
        resource_estimate = self._resource_estimate(training_config, data_summary, issues)
        fingerprint_payload = {
            "compatibility": compatibility,
            "training_config": training_config,
            "resource_estimate_inputs": resource_estimate.get("fingerprint_inputs", {}),
        }
        config_fingerprint = _canonical_hash(fingerprint_payload)
        training_config["config_fingerprint"] = config_fingerprint
        job: dict[str, Any] = {
            "run_id": proposal.run_id,
            "experiment_id": proposal.experiment_id,
            "lineage_id": proposal.lineage_id,
            "stage_name": proposal.stage_name,
            "artifact_root": str(run_root),
            "capability": capability.to_dict(),
            "compatibility": compatibility,
            "training_config": training_config,
            "resource_estimate": resource_estimate,
            "config_fingerprint": config_fingerprint,
            "resume_token_ref": None,
        }
        return job, issues

    def _validate_identity(
        self,
        *,
        kind: str,
        identifier: str,
        revision: str,
        external: bool,
        issues: list[ContractIssue],
    ) -> None:
        if not identifier or not revision:
            return
        if external:
            if not _IMMUTABLE_REMOTE_REVISION.fullmatch(revision):
                issues.append(
                    _issue(
                        f"{kind}_revision_not_immutable",
                        "artifact_integrity",
                        f"Remote {kind} revision must be a pinned 40-64 character commit hash.",
                    )
                )
            return
        try:
            actual = directory_content_identity(identifier)
        except ValueError as exc:
            issues.append(_issue(f"local_{kind}_missing", "artifact_integrity", str(exc)))
            return
        if actual != revision:
            issues.append(
                _issue(
                    f"local_{kind}_identity_mismatch",
                    "artifact_integrity",
                    f"Local {kind} content identity does not match its pinned revision.",
                    evidence_refs=[identifier],
                    metadata={"expected": revision, "observed": actual},
                )
            )

    @staticmethod
    def _validate_remote_policy(
        constraints: dict[str, object], issues: list[ContractIssue]
    ) -> None:
        required = {
            "external_provider_enabled": True,
            "external_download_enabled": True,
        }
        for name, expected in required.items():
            if constraints.get(name) is not expected:
                issues.append(
                    _issue(
                        "external_download_disabled",
                        "policy_blocked",
                        f"Remote acquisition requires explicit {name}=true.",
                    )
                )
        if "local_files_only" not in constraints:
            issues.append(
                _issue(
                    "offline_mode_unspecified",
                    "missing_evidence",
                    "Remote loading requires explicit local_files_only true or false.",
                )
            )
        if not str(constraints.get("cache_dir", "")).strip():
            issues.append(
                _issue(
                    "model_cache_unspecified",
                    "missing_evidence",
                    "Remote loading requires an explicit cache_dir.",
                )
            )
        if not str(constraints.get("license", "")).strip():
            issues.append(_issue("model_license_missing", "license_unknown", "Remote model license is missing."))
        if not str(constraints.get("provenance_ref", "")).strip():
            issues.append(
                _issue("model_provenance_missing", "provenance_unknown", "Remote model provenance is missing.")
            )
        approvals = constraints.get("approval_refs")
        if not isinstance(approvals, list) or not any(str(item).strip() for item in approvals):
            issues.append(
                _issue("model_download_approval_missing", "approval_required", "Remote download approval is missing.")
            )

    def _validate_data(
        self,
        compatibility: dict[str, str],
        constraints: dict[str, object],
        issues: list[ContractIssue],
    ) -> dict[str, object]:
        summary: dict[str, object] = {"rows": 0, "declared_tokens": 0, "bytes": 0}
        contract_ref = Path(compatibility["trainable_data_contract_ref"])
        evidence_ref = Path(str(constraints.get("processing_evidence_ref", "")))
        if not contract_ref.is_file():
            issues.append(
                _issue("trainable_data_contract_missing", "artifact_integrity", "Data contract is missing.")
            )
            return summary
        if _hash_file(contract_ref) != compatibility["trainable_data_contract_hash"]:
            issues.append(
                _issue(
                    "trainable_data_contract_hash_mismatch",
                    "artifact_integrity",
                    "Data contract hash mismatch.",
                    evidence_refs=[str(contract_ref)],
                )
            )
            return summary
        try:
            contract_payload = json.loads(contract_ref.read_text(encoding="utf-8"))
            contract = TrainableDataContract.from_dict(contract_payload)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            issues.append(
                _issue("trainable_data_contract_invalid", "artifact_integrity", f"Invalid data contract: {exc}")
            )
            return summary
        if contract.schema_version != "trainable-data.v1":
            issues.append(
                _issue("unsupported_data_schema", "unsupported_capability", "Only trainable-data.v1 is supported.")
            )
        processed_ref = Path(contract.processed_dataset_ref)
        if str(processed_ref) != compatibility["processed_dataset_ref"]:
            issues.append(
                _issue(
                    "processed_dataset_ref_mismatch",
                    "artifact_integrity",
                    "Contract and proposal reference different processed datasets.",
                )
            )
        if not processed_ref.is_file():
            issues.append(_issue("processed_dataset_missing", "artifact_integrity", "Processed JSONL is missing."))
            return summary
        if processed_ref.suffix.lower() != ".jsonl":
            issues.append(
                _issue("unsupported_processed_format", "unsupported_capability", "Processed data must be JSONL.")
            )
        observed_hash = _hash_file(processed_ref)
        if observed_hash != compatibility["processed_dataset_hash"]:
            issues.append(
                _issue(
                    "processed_dataset_hash_mismatch",
                    "artifact_integrity",
                    "Processed dataset content hash mismatch.",
                    evidence_refs=[str(processed_ref)],
                )
            )
        size = processed_ref.stat().st_size
        summary["bytes"] = size
        if size <= 0 or size > self.maximum_dataset_bytes:
            issues.append(
                _issue(
                    "processed_dataset_size_invalid",
                    "budget_exceeded",
                    f"Processed data must be 1-{self.maximum_dataset_bytes} bytes.",
                )
            )
        if not evidence_ref.is_file():
            issues.append(
                _issue(
                    "processing_evidence_missing",
                    "missing_evidence",
                    "Wrapper, boundary, and tokenizer evidence is required.",
                )
            )
        else:
            self._validate_processing_evidence(evidence_ref, compatibility, constraints, issues)
        rows = 0
        declared_tokens = 0
        try:
            with processed_ref.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    rows += 1
                    if rows > self.maximum_rows:
                        issues.append(
                            _issue("processed_row_limit_exceeded", "budget_exceeded", "Row bound exceeded.")
                        )
                        break
                    record = json.loads(line)
                    if not isinstance(record, dict) or not str(record.get("text", "")):
                        raise ValueError(f"row {rows} lacks non-empty text")
                    declared_tokens += max(0, int(record.get("token_count", 0)))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            issues.append(
                _issue("processed_dataset_invalid", "artifact_integrity", f"Processed JSONL is invalid: {exc}")
            )
        if rows == 0:
            issues.append(_issue("processed_dataset_empty", "artifact_integrity", "Processed data is empty."))
        summary.update({"rows": rows, "declared_tokens": declared_tokens, "contract_id": contract.contract_id})
        return summary

    @staticmethod
    def _validate_processing_evidence(
        evidence_ref: Path,
        compatibility: dict[str, str],
        constraints: dict[str, object],
        issues: list[ContractIssue],
    ) -> None:
        expected_hash = str(constraints.get("processing_evidence_hash", "")).strip()
        if not expected_hash or _hash_file(evidence_ref) != expected_hash:
            issues.append(
                _issue(
                    "processing_evidence_hash_mismatch",
                    "artifact_integrity",
                    "Processing evidence must have a matching content hash.",
                    evidence_refs=[str(evidence_ref)],
                )
            )
            return
        try:
            evidence = json.loads(evidence_ref.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            issues.append(_issue("processing_evidence_invalid", "artifact_integrity", "Evidence is unreadable."))
            return
        if evidence.get("processed_dataset_ref") != compatibility["processed_dataset_ref"]:
            issues.append(
                _issue("processing_dataset_ref_mismatch", "artifact_integrity", "Evidence references other data.")
            )
        if evidence.get("processed_content_hash") != compatibility["processed_dataset_hash"]:
            issues.append(
                _issue("processing_dataset_hash_mismatch", "artifact_integrity", "Evidence hash differs.")
            )
        wrapper = evidence.get("wrapper")
        boundary = evidence.get("prompt_target_boundary")
        if not isinstance(wrapper, dict) or not wrapper.get("template"):
            issues.append(_issue("wrapper_evidence_missing", "missing_evidence", "Wrapper evidence is missing."))
        if not isinstance(boundary, dict) or boundary.get("status") != "explicit":
            issues.append(
                _issue("prompt_target_boundary_missing", "missing_evidence", "Explicit boundaries are required.")
            )
        tokenizer = evidence.get("tokenizer_compatibility")
        if not isinstance(tokenizer, dict) or tokenizer.get("compatible") is not True:
            issues.append(
                _issue(
                    "tokenizer_compatibility_unverified",
                    "missing_evidence",
                    "Checked compatible tokenizer evidence is required.",
                )
            )
        elif str(tokenizer.get("tokenizer_ref", "")) != compatibility["tokenizer_id"]:
            issues.append(
                _issue("tokenizer_compatibility_mismatch", "incompatible_candidate", "Tokenizer evidence differs.")
            )

    def _normalize_training_config(
        self,
        compatibility: dict[str, str],
        constraints: dict[str, object],
        versions: dict[str, str | None],
        data_summary: dict[str, object],
        issues: list[ContractIssue],
    ) -> dict[str, object]:
        def integer(name: str, default: int, minimum: int = 0) -> int:
            try:
                value = int(constraints.get(name, default))
            except (TypeError, ValueError):
                value = minimum - 1
            if value < minimum:
                issues.append(_issue(f"invalid_{name}", "invalid_request", f"{name} must be >= {minimum}."))
            return value

        def number(name: str, default: float, minimum: float = 0.0) -> float:
            try:
                value = float(constraints.get(name, default))
            except (TypeError, ValueError):
                value = minimum - 1.0
            if value < minimum:
                issues.append(_issue(f"invalid_{name}", "invalid_request", f"{name} must be >= {minimum}."))
            return value

        max_steps = integer("max_steps", 0, 1)
        if max_steps > self.maximum_allowed_steps:
            issues.append(_issue("step_budget_exceeded", "budget_exceeded", "max_steps exceeds service bound."))
        device = str(constraints.get("device", "cpu")).lower()
        dtype = str(constraints.get("dtype", "float32")).lower()
        if device not in {"cpu", "cuda"}:
            issues.append(_issue("unsupported_device", "unsupported_capability", f"Unsupported device '{device}'."))
        if dtype not in {"float32", "float16", "bfloat16"}:
            issues.append(_issue("unsupported_dtype", "unsupported_capability", f"Unsupported dtype '{dtype}'."))
        if device == "cpu" and dtype != "float32":
            issues.append(
                _issue("cpu_mixed_precision_unsupported", "unsupported_capability", "CPU requires float32.")
            )
        mode = compatibility["training_mode"]
        if mode != "full_finetune":
            issues.append(
                _issue("unsupported_training_mode", "unsupported_capability", "Only full_finetune is supported.")
            )
        optimizer = compatibility["optimizer"].lower()
        scheduler = compatibility["scheduler"].lower()
        if optimizer != "adamw":
            issues.append(_issue("unsupported_optimizer", "unsupported_capability", "Only AdamW is supported."))
        if scheduler not in {"constant", "linear"}:
            issues.append(
                _issue("unsupported_scheduler", "unsupported_capability", "Scheduler must be constant or linear.")
            )
        if bool(constraints.get("shuffle", False)):
            issues.append(
                _issue(
                    "shuffle_not_supported",
                    "unsupported_capability",
                    "This bounded backend currently requires deterministic sequential loading.",
                )
            )
        loader_settings = constraints.get("loader_settings", {})
        if not isinstance(loader_settings, dict):
            loader_settings = {}
            issues.append(_issue("invalid_loader_settings", "invalid_request", "loader_settings must be an object."))
        loader_settings = dict(loader_settings)
        for estimate_field in ("parameter_count", "hidden_size"):
            if estimate_field in constraints:
                loader_settings[estimate_field] = constraints[estimate_field]
        max_epochs: float | None = None
        if constraints.get("max_epochs") is not None:
            max_epochs = number("max_epochs", 0.0, 1e-12)
        config: dict[str, object] = {
            "seed": integer("seed", 17, 0),
            "model_id": compatibility["model_id"],
            "model_revision": compatibility["model_revision"],
            "architecture_family": compatibility["architecture_family"],
            "tokenizer_id": compatibility["tokenizer_id"],
            "tokenizer_revision": compatibility["tokenizer_revision"],
            "vocabulary_size": integer("vocabulary_size", 0, 1),
            "special_token_ids": constraints.get("special_token_ids", {}),
            "context_length": integer("context_length", 0, 2),
            "parameter_count": integer("parameter_count", 0, 1),
            "hidden_size": integer("hidden_size", 0, 1),
            "trust_remote_code": False,
            "loader_settings": loader_settings,
            "local_files_only": bool(constraints.get("local_files_only", True)),
            "cache_dir": str(constraints.get("cache_dir", "")) or None,
            "dtype": dtype,
            "device": device,
            "training_mode": mode,
            "dataset_identity": compatibility["processed_dataset_hash"],
            "dataset_summary": data_summary,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "learning_rate": number("learning_rate", 5e-5, 1e-12),
            "warmup_steps": integer("warmup_steps", 0, 0),
            "batch_size": integer("batch_size", 1, 1),
            "gradient_accumulation_steps": integer("gradient_accumulation_steps", 1, 1),
            "max_steps": max_steps,
            "max_epochs": max_epochs,
            "gradient_clipping": number("gradient_clipping", 1.0, 0.0),
            "weight_decay": number("weight_decay", 0.0, 0.0),
            "mixed_precision": dtype != "float32",
            "dataloader_settings": {
                "shuffle": bool(constraints.get("shuffle", False)),
                "num_workers": 0,
                "drop_last": False,
            },
            "checkpoint_every_steps": integer("checkpoint_every_steps", max_steps, 1),
            "logging_every_steps": integer("logging_every_steps", 1, 1),
            "max_total_tokens": integer("max_total_tokens", 1_000_000, 1),
            "maximum_rows": self.maximum_rows,
            "processing_evidence_ref": str(constraints.get("processing_evidence_ref", "")),
            "processing_evidence_hash": str(constraints.get("processing_evidence_hash", "")),
            "prompt_masking": str(constraints.get("prompt_masking", "mask_prompt_for_prompt_target")),
            "ignored_label_token": -100,
            "eos_handling": "append_if_missing",
            "padding": "batch_right_padding",
            "truncation": "right_to_context_length",
            "framework_versions": versions,
            "environment_summary": {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "perfect_determinism_claimed": False,
            },
            "step_delay_seconds": number("step_delay_seconds", 0.0, 0.0),
            "simulate_oom": str(constraints.get("simulate_oom", "")),
        }
        if int(config["warmup_steps"]) > max_steps:
            issues.append(_issue("warmup_exceeds_steps", "invalid_request", "warmup_steps exceeds max_steps."))
        if float(config["step_delay_seconds"]) > 1.0:
            issues.append(_issue("invalid_step_delay", "invalid_request", "step_delay_seconds must be <= 1."))
        special = config["special_token_ids"]
        if not isinstance(special, dict) or special.get("eos_token_id") is None or special.get("pad_token_id") is None:
            issues.append(
                _issue(
                    "invalid_special_token_configuration",
                    "incompatible_candidate",
                    "Explicit EOS and padding token IDs are required.",
                )
            )
        return config

    @staticmethod
    def _resource_estimate(
        config: dict[str, object], data_summary: dict[str, object], issues: list[ContractIssue]
    ) -> dict[str, object]:
        try:
            parameters = int(config.get("parameter_count", 0))
        except (TypeError, ValueError):
            parameters = 0
        # parameter_count is declared by governed model discovery to avoid
        # opening model weights in the control process.
        loader = config.get("loader_settings")
        if isinstance(loader, dict):
            try:
                parameters = int(loader.get("parameter_count", parameters))
            except (TypeError, ValueError):
                parameters = 0
        if parameters <= 0:
            issues.append(
                _issue("parameter_count_missing", "missing_evidence", "Resource estimation requires parameter_count.")
            )
        bytes_per_parameter = 4
        parameter_memory = parameters * bytes_per_parameter
        gradient_memory = parameter_memory
        optimizer_memory = parameters * 8
        batch = int(config.get("batch_size", 1))
        context = int(config.get("context_length", 1))
        try:
            hidden_size = int(config.get("hidden_size", 0))
        except (TypeError, ValueError):
            hidden_size = 0
        activation_memory = batch * context * max(hidden_size, 1) * bytes_per_parameter * 12
        estimated_peak = parameter_memory + gradient_memory + optimizer_memory + activation_memory
        available, observation = observable_device_memory(str(config.get("device", "cpu")))
        if available is not None and estimated_peak > int(available * 0.8):
            issues.append(
                _issue(
                    "resource_estimate_exceeds_available_memory",
                    "budget_exceeded",
                    "Conservative estimated peak memory exceeds 80% of observable available memory.",
                )
            )
        expected_steps = int(config.get("max_steps", 0))
        if config.get("max_epochs") is not None:
            rows = int(data_summary.get("rows", 0))
            accumulation = int(config.get("gradient_accumulation_steps", 1))
            batches_per_epoch = math.ceil(rows / max(1, batch))
            steps_per_epoch = max(1, math.ceil(batches_per_epoch / max(1, accumulation)))
            epoch_steps = math.ceil(float(config["max_epochs"]) * steps_per_epoch)
            expected_steps = min(expected_steps, epoch_steps)
        return {
            "parameter_memory_bytes": parameter_memory,
            "optimizer_state_memory_bytes": optimizer_memory,
            "gradient_memory_bytes": gradient_memory,
            "activation_estimate_bytes": activation_memory,
            "estimated_peak_bytes": estimated_peak,
            "dataset_bytes": int(data_summary.get("bytes", 0)),
            "dataset_rows": int(data_summary.get("rows", 0)),
            "expected_steps": expected_steps,
            "expected_checkpoint_bytes": parameter_memory + optimizer_memory + parameter_memory,
            "available_device_memory_bytes": available,
            "available_memory_observation": observation,
            "uncertainty": "conservative_formula_not_runtime_measurement",
            "fingerprint_inputs": {
                "parameters": parameters,
                "hidden_size": hidden_size,
                "batch_size": batch,
                "context_length": context,
            },
        }

    def _start(self, managed: _ManagedRun, resume_token: Path | None = None) -> None:
        log_ref = Path(str(managed.handle.metadata["log_ref"]))
        log_ref.parent.mkdir(parents=True, exist_ok=True)
        command = [self.python_executable, str(Path(__file__).with_name("hf_worker.py")), "--job", str(managed.job_ref)]
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
        if managed.process is None or managed.process.poll() is not None or managed.handle.status not in {
            "running",
            "resuming",
        }:
            return self._with_issue(
                managed.handle,
                _issue(f"cannot_{action}", "policy_blocked", f"Run must be active to {action}."),
            )
        managed.requested_action = action
        managed.handle.status = "interrupting"
        managed.handle.metadata["control_action"] = action
        try:
            os.killpg(managed.process.pid, signal.SIGINT if action == "interrupt" else signal.SIGTERM)
        except ProcessLookupError:
            self._finalize(managed)
            return self._with_issue(
                managed.handle,
                _issue(f"{action}_race", "runtime_failure", "Process exited before signal delivery."),
            )
        self._persist_handle(managed.handle)
        return TrainingRunHandle.from_dict(managed.handle.to_dict())

    def _resume(self, managed: _ManagedRun, request: TrainingControlRequest) -> TrainingRunHandle:
        if managed.handle.status != "interrupted":
            return self._with_issue(
                managed.handle,
                _issue("resume_state_invalid", "policy_blocked", "Only interrupted runs may resume."),
            )
        token_ref = Path(str(managed.handle.resume_token_ref or ""))
        issue = self._validate_resume(managed, token_ref, request.metadata)
        if issue:
            return self._with_issue(managed.handle, issue)
        managed.requested_action = None
        managed.handle.status = "resuming"
        try:
            self._start(managed, token_ref)
        except OSError as exc:
            self._close_log(managed)
            managed.handle.status = "failed"
            self._record_incident(
                managed,
                "resume_launch_failure",
                f"Resume subprocess could not start: {type(exc).__name__}: {exc}",
                [str(token_ref)],
            )
        managed.handle.metadata["pid"] = managed.process.pid if managed.process else None
        self._persist_handle(managed.handle)
        return TrainingRunHandle.from_dict(managed.handle.to_dict())

    @staticmethod
    def _validate_resume(
        managed: _ManagedRun, token_ref: Path, requested: dict[str, object]
    ) -> ContractIssue | None:
        if not token_ref.is_file():
            return _issue("resume_token_missing", "missing_evidence", "Resume token is missing.")
        try:
            token = json.loads(token_ref.read_text(encoding="utf-8"))
            job = json.loads(managed.job_ref.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _issue("resume_evidence_invalid", "artifact_integrity", "Resume evidence is unreadable.")
        checkpoint_ref = Path(str(token.get("checkpoint_ref", "")))
        valid, observed_manifest_hash = validate_checkpoint_manifest(checkpoint_ref)
        if not valid:
            return _issue(
                "checkpoint_integrity_failure",
                "artifact_integrity",
                observed_manifest_hash,
                evidence_refs=[str(checkpoint_ref)],
            )
        if token.get("checkpoint_manifest_hash") != observed_manifest_hash:
            return _issue("checkpoint_manifest_mismatch", "artifact_integrity", "Resume manifest differs.")
        fingerprint = str(job.get("config_fingerprint", ""))
        if token.get("config_fingerprint") != fingerprint:
            return _issue("resume_config_mismatch", "incompatible_candidate", "Config fingerprint differs.")
        if "config_fingerprint" in requested and str(requested["config_fingerprint"]) != fingerprint:
            return _issue("resume_request_mismatch", "incompatible_candidate", "Requested fingerprint differs.")
        token_compatibility = token.get("compatibility")
        if not isinstance(token_compatibility, dict):
            return _issue("resume_compatibility_missing", "missing_evidence", "Compatibility evidence is missing.")
        for field_name in _RESUME_FIELDS:
            expected = managed.compatibility.get(field_name)
            if token_compatibility.get(field_name) != expected:
                return _issue(
                    "resume_compatibility_mismatch",
                    "incompatible_candidate",
                    f"Resume {field_name} is incompatible.",
                )
            if field_name in requested and str(requested[field_name]) != expected:
                return _issue(
                    "resume_request_mismatch",
                    "incompatible_candidate",
                    f"Requested {field_name} differs.",
                )
        return None

    def _finalize(self, managed: _ManagedRun) -> None:
        returncode = managed.process.poll() if managed.process is not None else None
        self._close_log(managed)
        root = managed.job_ref.parent
        result_ref = root / "runtime_result.json"
        try:
            result = json.loads(result_ref.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {"status": "failed", "error_type": "missing_or_invalid_result"}
        worker_status = str(result.get("status", "failed"))
        if managed.requested_action == "interrupt" and worker_status == "interrupted":
            status = "interrupted"
        elif managed.requested_action == "cancel" and worker_status == "cancelled":
            status = "cancelled"
        elif returncode == 0 and worker_status == "completed":
            status = "completed"
        else:
            status = "failed"
            error_type = str(result.get("error_type", "runtime_failure"))
            incident_code = "out_of_memory" if error_type in {"cpu_oom", "gpu_oom"} else "non_zero_exit"
            self._record_incident(
                managed,
                incident_code,
                f"Training exited with code {returncode}, status {worker_status}, error {error_type}.",
                [str(result_ref)],
                metadata={
                    "device": result.get("device"),
                    "attempted_batch_size": result.get("batch_size"),
                    "attempted_context_length": result.get("context_length"),
                    "resource_estimate_ref": managed.handle.metadata.get("resource_estimate_ref"),
                    "suggested_reductions": result.get("suggested_reductions", []),
                },
            )
        managed.handle.status = status
        managed.handle.metadata.update({"returncode": returncode, "runtime_result_ref": str(result_ref)})
        metrics_ref = root / "metrics_summary.json"
        record_ref = root / "checkpoint_record.json"
        required = [root / "events.jsonl", root / "runtime.log", result_ref]
        if status == "completed":
            required.extend([metrics_ref, record_ref, root / "final_result.json"])
        if status == "interrupted":
            required.extend([record_ref, root / "resume_token.json"])
        missing = [path for path in required if not path.is_file()]
        if missing and status in {"completed", "interrupted"}:
            managed.handle.status = "failed"
            self._record_incident(
                managed,
                "missing_required_artifact",
                "Run reached a terminal state without required evidence.",
                [str(path) for path in missing],
            )
        if record_ref.is_file():
            try:
                record = json.loads(record_ref.read_text(encoding="utf-8"))
                checkpoint_ref = Path(str(record.get("checkpoint_ref", "")))
                valid, observed_hash = validate_checkpoint_manifest(checkpoint_ref)
                if not valid or record.get("checkpoint_manifest_hash") != observed_hash:
                    raise ValueError(observed_hash)
                managed.handle.checkpoint_refs = [str(checkpoint_ref)]
                token_ref = root / "resume_token.json"
                managed.handle.resume_token_ref = str(token_ref) if token_ref.is_file() else None
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                managed.handle.status = "failed"
                self._record_incident(
                    managed,
                    "checkpoint_integrity_failure",
                    f"Checkpoint evidence failed verification: {exc}",
                    [str(record_ref)],
                )
        managed.handle.metrics_ref = str(metrics_ref) if metrics_ref.is_file() else None
        self._persist_handle(managed.handle)

    def _record_incident(
        self,
        managed: _ManagedRun,
        code: str,
        message: str,
        evidence_refs: list[str],
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        incident_ref = Path(str(managed.handle.metadata["incidents_ref"]))
        _append_jsonl(
            incident_ref,
            {
                "incident_id": f"inc-{managed.handle.run_id}-{code}",
                "run_id": managed.handle.run_id,
                "category": "runtime",
                "code": code,
                "message": message,
                "evidence_refs": evidence_refs,
                "metadata": metadata or {},
            },
        )
        issue_category = (
            "artifact_integrity"
            if code in {"missing_required_artifact", "checkpoint_integrity_failure"}
            else "runtime_failure"
        )
        managed.handle.issues.append(
            _issue(
                code,
                issue_category,
                message,
                evidence_refs=evidence_refs,
                retryable=True,
                metadata=metadata,
            )
        )

    def _persist_handle(self, handle: TrainingRunHandle) -> None:
        if not _RUN_ID_PATTERN.fullmatch(handle.run_id):
            raise ValueError("refusing to persist unsafe run_id")
        _write_json(self.artifact_root / handle.run_id / "handle.json", handle.to_dict())

    def _restore(self, run_id: str) -> TrainingRunHandle | None:
        path = self.artifact_root / run_id / "handle.json"
        if not path.is_file():
            return None
        try:
            return TrainingRunHandle.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return None

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

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _close_log(managed: _ManagedRun) -> None:
        if managed.log_stream is not None:
            managed.log_stream.close()
            managed.log_stream = None
