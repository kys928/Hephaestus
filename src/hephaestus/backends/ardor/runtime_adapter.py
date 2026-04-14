from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hephaestus.backends.base import BackendRunResult, PreparedBackendJob
from hephaestus.backends.ardor.launcher import ArdorLaunchOutcome
from hephaestus.runtime.event_stream import events_from_process_output
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


@dataclass(slots=True)
class ArdorRuntimeAdapter:
    def normalize_run(self, prepared_job: PreparedBackendJob, outcome: ArdorLaunchOutcome) -> BackendRunResult:
        if outcome.status != "launched":
            return BackendRunResult(
                run_id=prepared_job.run_id,
                status="failed",
                events=[
                    RuntimeEvent(
                        event_id=f"{prepared_job.run_id}-launch-failure",
                        run_id=prepared_job.run_id,
                        step=0,
                        category=RuntimeEventCategory.INCIDENT,
                        message=f"ardor_launch_failure status={outcome.status} detail={outcome.stderr}",
                        payload_ref=outcome.contract_ref,
                    )
                ],
                artifact_refs=[ref for ref in [outcome.contract_ref] if ref],
                checkpoint_candidates=[],
                intermediate_eval={},
            )

        events = events_from_process_output(prepared_job.run_id, outcome.stdout, outcome.stderr)
        contract_ref = str(outcome.contract_ref or "")
        contract_path = Path(contract_ref) if contract_ref else None
        if contract_path is None or not contract_path.exists():
            events.append(
                RuntimeEvent(
                    event_id=f"{prepared_job.run_id}-missing-contract",
                    run_id=prepared_job.run_id,
                    step=0,
                    category=RuntimeEventCategory.INCIDENT,
                    message="ardor_missing_runtime_contract",
                    payload_ref=contract_ref or None,
                )
            )
            return BackendRunResult(prepared_job.run_id, "failed", events, [], [], {})

        try:
            payload = json.loads(contract_path.read_text())
        except json.JSONDecodeError:
            events.append(
                RuntimeEvent(
                    event_id=f"{prepared_job.run_id}-malformed-contract",
                    run_id=prepared_job.run_id,
                    step=0,
                    category=RuntimeEventCategory.INCIDENT,
                    message="ardor_malformed_output_contract",
                    payload_ref=contract_ref,
                )
            )
            return BackendRunResult(prepared_job.run_id, "failed", events, [contract_ref], [], {})

        ardor_status = str(payload.get("status", "")).strip()
        status = self._map_status(prepared_job.run_id, ardor_status, events, contract_ref)

        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            events.append(
                RuntimeEvent(
                    event_id=f"{prepared_job.run_id}-bad-artifacts",
                    run_id=prepared_job.run_id,
                    step=0,
                    category=RuntimeEventCategory.INCIDENT,
                    message="ardor_malformed_artifacts_section",
                    payload_ref=contract_ref,
                )
            )
            return BackendRunResult(prepared_job.run_id, "failed", events, [contract_ref], [], {})

        intermediate_eval = {
            "metrics_ref": str(artifacts.get("metrics_ref", "")),
            "probe_ref": str(artifacts.get("probe_ref", "")),
            "deterministic_ref": str(artifacts.get("deterministic_ref", "")),
            "runtime_log_ref": str(artifacts.get("runtime_log_ref", "")),
        }

        checkpoint_refs = artifacts.get("checkpoint_refs", [])
        if not isinstance(checkpoint_refs, list):
            checkpoint_refs = []
            events.append(
                RuntimeEvent(
                    event_id=f"{prepared_job.run_id}-bad-checkpoints",
                    run_id=prepared_job.run_id,
                    step=0,
                    category=RuntimeEventCategory.INCIDENT,
                    message="ardor_malformed_checkpoint_refs",
                    payload_ref=contract_ref,
                )
            )

        checkpoint_scores = payload.get("checkpoint_scores", {})
        if not isinstance(checkpoint_scores, dict):
            checkpoint_scores = {}

        checkpoint_candidates: list[dict[str, object]] = []
        for ref in checkpoint_refs:
            ref_str = str(ref)
            if not ref_str:
                continue
            candidate: dict[str, object] = {"checkpoint_ref": ref_str}
            if ref_str in checkpoint_scores:
                candidate["probe_score"] = float(checkpoint_scores[ref_str])
            checkpoint_candidates.append(candidate)

        artifact_refs = [contract_ref]
        for key in ("metrics_ref", "probe_ref", "deterministic_ref", "runtime_log_ref"):
            ref = str(artifacts.get(key, ""))
            if ref:
                artifact_refs.append(ref)
        artifact_refs.extend(str(ref) for ref in checkpoint_refs if str(ref))

        status = self._validate_artifacts(
            run_id=prepared_job.run_id,
            status=status,
            artifacts=artifacts,
            artifact_refs=artifact_refs,
            events=events,
        )
        if outcome.returncode not in (0, None):
            status = "failed"

        return BackendRunResult(
            run_id=prepared_job.run_id,
            status=status,
            events=events,
            artifact_refs=artifact_refs,
            checkpoint_candidates=checkpoint_candidates,
            intermediate_eval=intermediate_eval,
        )

    def _map_status(self, run_id: str, ardor_status: str, events: list[RuntimeEvent], payload_ref: str) -> str:
        mapped = {
            "succeeded": "completed",
            "failed": "failed",
            "partial": "failed",
            "unsupported": "failed",
        }
        if ardor_status in mapped:
            if ardor_status in {"partial", "unsupported"}:
                events.append(
                    RuntimeEvent(
                        event_id=f"{run_id}-{ardor_status}",
                        run_id=run_id,
                        step=0,
                        category=RuntimeEventCategory.INCIDENT,
                        message=f"ardor_runtime_state_{ardor_status}",
                        payload_ref=payload_ref,
                    )
                )
            return mapped[ardor_status]

        events.append(
            RuntimeEvent(
                event_id=f"{run_id}-unsupported-state",
                run_id=run_id,
                step=0,
                category=RuntimeEventCategory.INCIDENT,
                message=f"ardor_unsupported_runtime_state:{ardor_status or 'missing'}",
                payload_ref=payload_ref,
            )
        )
        return "failed"

    def _validate_artifacts(
        self,
        *,
        run_id: str,
        status: str,
        artifacts: dict[str, object],
        artifact_refs: list[str],
        events: list[RuntimeEvent],
    ) -> str:
        required_on_success = ["metrics_ref", "deterministic_ref"]
        for key in required_on_success:
            ref = str(artifacts.get(key, ""))
            if status == "completed" and not ref:
                events.append(
                    RuntimeEvent(
                        event_id=f"{run_id}-missing-{key}",
                        run_id=run_id,
                        step=0,
                        category=RuntimeEventCategory.INCIDENT,
                        message=f"ardor_missing_{key}",
                        payload_ref=None,
                    )
                )
                status = "failed"

        checkpoint_refs = artifacts.get("checkpoint_refs", [])
        if status == "completed" and (not isinstance(checkpoint_refs, list) or not checkpoint_refs):
            events.append(
                RuntimeEvent(
                    event_id=f"{run_id}-missing-checkpoints",
                    run_id=run_id,
                    step=0,
                    category=RuntimeEventCategory.INCIDENT,
                    message="ardor_missing_checkpoint_refs",
                    payload_ref=None,
                )
            )
            status = "failed"

        for ref in artifact_refs:
            if ref and not Path(ref).exists():
                events.append(
                    RuntimeEvent(
                        event_id=f"{run_id}-missing-artifact-{Path(ref).name}",
                        run_id=run_id,
                        step=0,
                        category=RuntimeEventCategory.INCIDENT,
                        message=f"ardor_missing_artifact_ref:{ref}",
                        payload_ref=ref,
                    )
                )
                status = "failed"

        return status
