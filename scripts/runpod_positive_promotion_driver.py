#!/usr/bin/env python3
"""RunPod cycle driver for the generic ``hephaestus run`` production loop.

The GitHub runner owns the durable production-loop state and recovery policy.
Each infrastructure attempt launches one real RunPod Pod. A controlled first
bootstrap failure may be enabled to prove that the generic loop tears down the
failed Pod and retries the exact same scientific program without changing model,
data, decoding, or evaluation variables.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_positive_promotion_proof as launcher  # noqa: E402

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider  # noqa: E402
from hephaestus.production.loop import ProductionCycleResult  # noqa: E402
from hephaestus.production.recovery import (  # noqa: E402
    InfrastructureRecoveryController,
    RecoverableInfrastructureError,
)
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _controlled_bootstrap_failure_shell() -> str:
    return r'''set -Eeuo pipefail
ATTEMPT_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_PROOF_RUN_ID}/attempt-${HEPHAESTUS_ATTEMPT}"
mkdir -p "$ATTEMPT_DIR"
cat > "$ATTEMPT_DIR/driver_result.json" <<EOF
{
  "result_version": "positive-real-model-promotion-proof.v1",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "proof_run_id": "${HEPHAESTUS_PROOF_RUN_ID}",
  "attempt": "${HEPHAESTUS_ATTEMPT}",
  "status": "pod_bootstrap_failed",
  "error": "controlled RunPod bootstrap failure before scientific progress",
  "controlled_failure_injection": true,
  "optimizer_steps": 0,
  "checkpoint_created": false,
  "training_performed": false,
  "original_research_lineage_mutated": false
}
EOF
sync
exit 71
'''


def _create_pod(
    execution: RunPodExecutionAdapter,
    *,
    proof_run_id: str,
    repo_sha: str,
    attempt: int,
    controlled_bootstrap_failure: bool,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    shell = _controlled_bootstrap_failure_shell() if controlled_bootstrap_failure else launcher.pod_shell()

    def create_once(gpu_ids: list[str]) -> dict[str, Any]:
        body: dict[str, object] = {
            "name": f"hephaestus-positive-promotion-{proof_run_id}-a{attempt}"[:180],
            "computeType": "GPU",
            "gpuCount": 1,
            "gpuTypeIds": list(gpu_ids),
            "gpuTypePriority": "availability",
            "cloudType": "SECURE",
            "dataCenterIds": [launcher.DATACENTER_ID],
            "dataCenterPriority": "custom",
            "imageName": launcher.IMAGE,
            "containerDiskInGb": 24,
            "networkVolumeId": launcher.VOLUME_ID,
            "volumeMountPath": "/workspace",
            "dockerStartCmd": ["bash", "-lc", shell],
            "interruptible": False,
            "env": {
                "HEPHAESTUS_PROOF_RUN_ID": proof_run_id,
                "HEPHAESTUS_REPO_SHA": repo_sha,
                "HEPHAESTUS_ATTEMPT": str(attempt),
                "HEPHAESTUS_OPERATOR_APPROVAL_REF": launcher.APPROVAL_REF,
            },
        }
        return execution._create_pod(body)  # integration-owned scheduler projection

    return launcher.create_with_capacity_retries(create_once)


def _recoverable(exc: BaseException, *, details: dict[str, object] | None = None) -> RecoverableInfrastructureError | None:
    message = f"{type(exc).__name__}: {exc}"
    classified = InfrastructureRecoveryController.classify_message(message)
    if classified is not None:
        classified.details.update(details or {})
        return classified
    lowered = message.casefold()
    if isinstance(exc, TimeoutError):
        return RecoverableInfrastructureError(
            "stale_execution_sentinel",
            message,
            details=details,
        )
    if any(token in lowered for token in ("s3", "object store", "slowdown", "service unavailable", "http 503")):
        return RecoverableInfrastructureError(
            "temporary_object_store_unavailable",
            message,
            details=details,
        )
    return None


def _latest_remote_cycle(result: dict[str, Any]) -> dict[str, Any]:
    program = result.get("program_state")
    if not isinstance(program, dict):
        raise RuntimeError("remote proof result has no production program state")
    metadata = program.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("remote proof program state has no metadata")
    latest = metadata.get("latest_cycle")
    if not isinstance(latest, dict):
        raise RuntimeError("remote proof program state has no latest cycle")
    return latest


def _phase_evidence(result: dict[str, Any]) -> dict[str, list[str]]:
    latest = _latest_remote_cycle(result)
    raw = latest.get("phase_evidence")
    if not isinstance(raw, dict):
        raise RuntimeError("remote proof latest cycle has no phase evidence")
    normalized: dict[str, list[str]] = {}
    for phase, refs in raw.items():
        if not isinstance(refs, list) or not refs:
            raise RuntimeError(f"remote proof phase evidence is incomplete: {phase}")
        normalized[str(phase)] = [str(ref) for ref in refs]
    return normalized


@dataclass(slots=True)
class RunPodPositivePromotionDriver:
    prove_bootstrap_recovery_once: bool = True
    proof_run_id: str = field(default_factory=lambda: f"positive-real-model-promotion-001-{os.environ.get('GITHUB_RUN_ID', str(int(time.time())))}")
    repo_sha: str = field(default_factory=lambda: _required("GITHUB_SHA"))
    attempt_rows: list[dict[str, object]] = field(default_factory=list)
    verification: dict[str, object] | None = None
    last_error: str | None = None

    def _write_launcher_record(self) -> None:
        launcher.atomic_json(
            Path("positive_promotion_launcher.json"),
            {
                "launcher_version": "positive-real-model-promotion-generic-loop.v1",
                "created_at": _now(),
                "repo_sha": self.repo_sha,
                "proof_run_id": self.proof_run_id,
                "volume_id": launcher.VOLUME_ID,
                "datacenter_id": launcher.DATACENTER_ID,
                "generic_cli": "hephaestus run",
                "generic_recovery_owned": True,
                "scientific_variables_changed_on_retry": False,
                "attempts": self.attempt_rows,
                "error": self.last_error,
                "status": "verified" if self.verification is not None else "running",
            },
        )

    def execute_cycle(self, *, runtime: Any, state: Any, cycle_index: int) -> ProductionCycleResult:
        if cycle_index != 1:
            raise RuntimeError("RunPod positive-promotion launch driver is a single scientific program cycle")
        _required("RUNPOD_API_KEY")
        attempt = int(state.metadata.get("infrastructure_attempt", 1) or 1)
        controlled_failure = bool(self.prove_bootstrap_recovery_once and attempt == 1)
        row: dict[str, object] = {
            "attempt": attempt,
            "started_at": _now(),
            "pod_id": None,
            "capacity_selection": [],
            "observations": [],
            "controlled_bootstrap_failure": controlled_failure,
            "scientific_variables_changed": False,
            "optimizer_steps": 0,
            "checkpoint_created": False,
        }
        pod_id: str | None = None
        execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
        client = launcher.base.s3_client()

        try:
            try:
                client.head_bucket(Bucket=launcher.VOLUME_ID)
            except BaseException as exc:
                recoverable = _recoverable(exc, details={"phase": "object_store_preflight"})
                if recoverable is not None:
                    raise recoverable from exc
                raise

            pod, capacity = _create_pod(
                execution,
                proof_run_id=self.proof_run_id,
                repo_sha=self.repo_sha,
                attempt=attempt,
                controlled_bootstrap_failure=controlled_failure,
            )
            row["capacity_selection"] = capacity
            pod_id = str(pod["id"])
            row["pod_id"] = pod_id

            try:
                result, observations = launcher.wait_for_result(
                    client,
                    execution,
                    proof_run_id=self.proof_run_id,
                    attempt=attempt,
                    pod_id=pod_id,
                )
            except BaseException as exc:
                recoverable = _recoverable(exc, details={"phase": "wait_for_result", "pod_id": pod_id})
                if recoverable is not None:
                    raise recoverable from exc
                raise

            row["observations"] = observations
            row["terminal_status"] = result.get("status")
            row["terminal_error"] = result.get("error")

            if result.get("status") == "pod_bootstrap_failed":
                raise RecoverableInfrastructureError(
                    "pod_bootstrap_failed",
                    str(result.get("error") or "RunPod bootstrap failed before scientific progress"),
                    optimizer_steps=int(result.get("optimizer_steps", 0) or 0),
                    checkpoint_created=bool(result.get("checkpoint_created", False)),
                    details={
                        "pod_id": pod_id,
                        "attempt": attempt,
                        "controlled_failure_injection": bool(result.get("controlled_failure_injection", False)),
                    },
                )
            if result.get("status") != "completed":
                remote_error = RuntimeError(
                    f"remote scientific proof failed: {result.get('status')}: {result.get('error', '')}"
                )
                recoverable = _recoverable(remote_error, details={"phase": "remote_terminal", "pod_id": pod_id})
                if recoverable is not None:
                    raise recoverable from remote_error
                raise remote_error

            verification = launcher.verify_proof(client, self.proof_run_id, result)
            launcher.atomic_json(Path("positive_promotion_verification.json"), verification)
            self.verification = verification
            row["verified"] = True
            latest = _latest_remote_cycle(result)
            certified_ref = str(verification.get("certified_checkpoint_ref") or "")
            if not certified_ref:
                raise RuntimeError("verified remote proof has no certified checkpoint reference")
            phase_evidence = _phase_evidence(result)
            confidence = float(latest.get("confidence", 0.95) or 0.95)

            return ProductionCycleResult(
                cycle_id=f"{self.proof_run_id}-runpod-launch",
                run_id=f"{self.proof_run_id}-attempt-{attempt}",
                experiment_id=f"experiment-{self.proof_run_id}",
                status="completed",
                judge_action="promote_checkpoint",
                checkpoint_ref=certified_ref,
                confidence=confidence,
                promotion_allowed=True,
                certification_state="certification_passed",
                approval_status="approved",
                approval_ref=launcher.APPROVAL_REF,
                comparison_ref=str(latest.get("comparison_ref") or "") or None,
                phase_evidence=phase_evidence,
                evidence={
                    "proof_run_id": self.proof_run_id,
                    "runpod_attempt": attempt,
                    "remote_terminal_status": result.get("status"),
                    "remote_program_state": result.get("program_state"),
                    "independent_verification_ref": result.get("independent_verification_ref"),
                    "independent_verification_sha256": result.get("independent_verification_sha256"),
                    "certified_model_manifest": verification.get("certified_model_manifest"),
                    "generic_recovery_owned": True,
                    "scientific_variables_changed_on_retry": False,
                },
            )
        except RecoverableInfrastructureError as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            row["recoverable"] = True
            row["failure_code"] = exc.code
            row["launcher_error"] = self.last_error
            row["optimizer_steps"] = exc.optimizer_steps
            row["checkpoint_created"] = exc.checkpoint_created
            raise
        except BaseException as exc:
            recoverable = _recoverable(exc, details={"phase": "runpod_launch", "pod_id": pod_id})
            if recoverable is not None:
                self.last_error = f"{type(recoverable).__name__}: {recoverable}"
                row["recoverable"] = True
                row["failure_code"] = recoverable.code
                row["launcher_error"] = self.last_error
                raise recoverable from exc
            self.last_error = f"{type(exc).__name__}: {exc}"
            row["recoverable"] = False
            row["launcher_error"] = self.last_error
            raise
        finally:
            if pod_id:
                row["teardown"] = launcher.base.delete_pod(execution, pod_id)
            else:
                row["teardown"] = {"deleted": False, "not_created": True}
            row["completed_at"] = _now()
            self.attempt_rows.append(row)
            self._write_launcher_record()


def build_driver(config: dict[str, object]) -> RunPodPositivePromotionDriver:
    return RunPodPositivePromotionDriver(
        prove_bootstrap_recovery_once=bool(config.get("prove_bootstrap_recovery_once", True)),
    )
