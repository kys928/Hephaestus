from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.base import BackendRunResult, PreparedBackendJob
from hephaestus.config_loader import ConfigError
from hephaestus.schemas.runtime_event import RuntimeEvent, RuntimeEventCategory


@dataclass(slots=True)
class ArdorRuntimeAdapter:
    def launch(self, prepared_job: PreparedBackendJob) -> BackendRunResult:
        runner = str(prepared_job.execution_spec.get("runner", ""))
        if runner != "ardor_api":
            raise ConfigError("Ardor runtime adapter expects runner=ardor_api")

        simulate_only = bool(prepared_job.execution_spec.get("simulate_only", True))
        if not simulate_only:
            raise NotImplementedError("Real Ardor job submission is not yet supported in this environment")

        run_id = prepared_job.run_id
        endpoint = str(prepared_job.execution_spec.get("endpoint", ""))
        queue = str(prepared_job.execution_spec.get("queue", ""))
        payload_ref = f"{prepared_job.artifact_root}/ardor_launch_request.json"
        events = [
            RuntimeEvent(
                event_id=f"{run_id}-ardor-queued",
                run_id=run_id,
                step=0,
                category=RuntimeEventCategory.INCIDENT,
                message=f"prepared_only_unsupported_execution endpoint={endpoint} queue={queue}",
                payload_ref=payload_ref,
            )
        ]
        return BackendRunResult(
            run_id=run_id,
            status="failed",
            events=events,
            artifact_refs=[payload_ref],
            checkpoint_candidates=[],
            intermediate_eval={},
        )
