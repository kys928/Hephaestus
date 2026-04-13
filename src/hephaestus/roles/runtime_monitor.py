from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.backends.base import BackendRunResult, ExecutionBackend
from hephaestus.policy.runtime_policy import RuntimePolicy
from hephaestus.runtime.health_checks import count_deterministic_failures, count_incidents
from hephaestus.runtime.incident_manager import incident_from_event, launch_failure_incident
from hephaestus.runtime.stop_logic import stop_recommendation
from hephaestus.schemas.incident_record import IncidentRecord
from hephaestus.schemas.runtime_event import RuntimeEvent
from hephaestus.schemas.stage_profile import StageProfile


def _stop_sensitivity_for_stage(stage_profile: StageProfile | None) -> str:
    if stage_profile is None:
        return "normal"
    strictness = str(stage_profile.strictness).lower()
    if strictness == "strict":
        return "high"
    if strictness == "lenient":
        return "normal"
    return "normal"


@dataclass(slots=True)
class RuntimeMonitorResult:
    outcome: str
    recommendation: str
    events: list[RuntimeEvent]
    incidents: list[IncidentRecord]
    training_outputs: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeMonitorRole:
    backend: ExecutionBackend
    runtime_policy: RuntimePolicy
    name: str = "runtime_monitor"

    def run(
        self,
        *,
        run_id: str,
        experiment_plan: dict[str, object],
        training_plan: dict[str, object],
        launch_config: dict[str, object],
        data_contract: dict[str, object],
        stage_profile: StageProfile | None = None,
    ) -> RuntimeMonitorResult:
        prepared_job = self.backend.prepare_training_job(
            experiment_plan=experiment_plan,
            data_contract=data_contract,
            training_plan=training_plan,
            launch_config=launch_config,
        )
        launch_result = self.backend.launch_training(prepared_job)
        stop_sensitivity = _stop_sensitivity_for_stage(stage_profile)
        return self._from_launch_result(run_id, launch_result, stop_sensitivity)

    def _from_launch_result(self, run_id: str, launch_result: BackendRunResult, stop_sensitivity: str) -> RuntimeMonitorResult:
        events = launch_result.events
        incidents = [incident for event in events if (incident := incident_from_event(event))]

        if launch_result.status != "completed":
            incidents.append(launch_failure_incident(run_id, f"backend status={launch_result.status}"))

        if launch_result.status != "completed":
            outcome = "hard_abort"
        else:
            outcome = self.runtime_policy.classify(
                incident_count=count_incidents(events),
                deterministic_failures=count_deterministic_failures(events),
                stop_sensitivity=stop_sensitivity,
            )
        return RuntimeMonitorResult(
            outcome=outcome,
            recommendation=stop_recommendation(outcome),
            events=events,
            incidents=incidents,
            training_outputs={
                "status": launch_result.status,
                "checkpoint_candidates": launch_result.checkpoint_candidates,
                "intermediate_eval": launch_result.intermediate_eval,
                "artifact_refs": launch_result.artifact_refs,
            },
        )
