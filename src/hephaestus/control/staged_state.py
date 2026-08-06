"""Transport-safe state and dependency contracts for staged orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from hephaestus.control.autonomous_experiment import IntegrationRecordSink
from hephaestus.control.spine import SPINE_ORDER, SpinePhase
from hephaestus.storage.base import ArtifactStore, StateRepository


GOVERNED_AUTONOMOUS_MODE = "governed_autonomous"

STEP_STATUSES = frozenset(
    {
        "completed",
        "blocked",
        "inconclusive",
        "retryable_failure",
        "terminal_failure",
        "cancelled",
        "interrupted",
        "approval_pending",
    }
)
RESUMABLE_STATUSES = frozenset(
    {"blocked", "inconclusive", "retryable_failure", "interrupted", "approval_pending"}
)
TERMINAL_TRAINING_STATUSES = frozenset({"completed", "failed", "cancelled"})

WORKFLOW_STATE_COLLECTION = "staged_workflow_states"
WORKFLOW_RECORD_COLLECTION = "staged_workflow_records"


PHASE_SUBSTEPS: dict[SpinePhase, tuple[str, ...]] = {
    SpinePhase.JUDGE_ENTRY: (
        "judge_entry_decision",
        "evidence_collection",
        "evidence_based_diagnosis",
        "readiness_to_plan",
    ),
    SpinePhase.PLANNER: (
        "diagnosis_handoff",
        "intervention_ranking",
        "experiment_proposal",
        "discovery_request_preparation",
        "approval_requirement_discovery",
    ),
    SpinePhase.DATA_ACQUISITION_AUDIT: (
        "dataset_discovery",
        "dataset_selection",
        "acquisition_approval_gate",
        "dataset_acquisition",
        "license_provenance_evidence",
        "dataset_audit",
        "manifest_production",
    ),
    SpinePhase.DATA_PREPROCESSOR: (
        "approved_source_preprocessing",
        "deduplication_contamination",
        "tokenizer_compatibility",
        "trainable_data_contract",
    ),
    SpinePhase.TRAINING_ENGINEER: (
        "model_discovery",
        "model_selection",
        "model_approval_gate",
        "training_input_binding",
        "launch_configuration",
        "lifecycle_launch",
    ),
    SpinePhase.RUNTIME_MONITOR: (
        "training_status_poll",
        "runtime_observation",
        "runtime_evidence",
        "bounded_recovery_advice",
        "runtime_control_governance",
    ),
    SpinePhase.EVALUATOR: (
        "checkpoint_resolution",
        "generation_prompt_materialization",
        "baseline_generation",
        "candidate_generation",
        "semantic_comparison",
        "deterministic_regression_evidence",
        "repeatability_variance_evidence",
        "human_review_references",
    ),
    SpinePhase.JUDGE_EXIT: (
        "governed_verdict",
        "action_boundary",
        "action_approval_gate",
        "promotion_gate",
        "action_application",
        "replay_evidence",
    ),
}


REQUIRED_RECORD_KINDS: dict[str, str] = {
    "judge_entry_decision": "judge_entry",
    "evidence_based_diagnosis": "diagnosis_report",
    "intervention_ranking": "intervention_proposal",
    "experiment_proposal": "experiment_proposal",
    "discovery_request_preparation": "search_request",
    "dataset_discovery": "dataset_discovery_result",
    "dataset_selection": "dataset_selection_decision",
    "dataset_acquisition": "acquisition_receipt",
    "manifest_production": "dataset_manifest",
    "approved_source_preprocessing": "preprocessing_report",
    "trainable_data_contract": "trainable_data_contract",
    "model_discovery": "model_discovery_result",
    "model_selection": "model_selection_decision",
    "lifecycle_launch": "training_run_handle",
    "training_status_poll": "training_run_handle",
    "runtime_evidence": "runtime_evidence",
    "bounded_recovery_advice": "recovery_decision",
    "baseline_generation": "generation_report",
    "candidate_generation": "generation_report",
    "semantic_comparison": "experiment_comparison",
    "deterministic_regression_evidence": "deterministic_regression_evidence",
    "repeatability_variance_evidence": "repeatability_variance_evidence",
    "human_review_references": "human_review_references",
    "governed_verdict": "judge_exit",
    "promotion_gate": "promotion_gate_report",
    "action_application": "action_decision",
}


@dataclass(frozen=True, slots=True)
class StagedOutputRecord:
    kind: str
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class StagedOperationResult:
    status: str = "completed"
    output_refs: tuple[str, ...] = ()
    records: tuple[StagedOutputRecord, ...] = ()
    blocking_issues: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    resumable: bool | None = None


@dataclass(frozen=True, slots=True)
class StagedOperationRequest:
    workflow_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    phase: str
    substep: str
    operation_id: str
    attempt: int
    input_refs: tuple[str, ...]
    prior_outputs: dict[str, dict[str, object]]


class StagedOperationService(Protocol):
    def execute(self, request: StagedOperationRequest) -> StagedOperationResult: ...


@dataclass(frozen=True, slots=True)
class StagedApprovalRequest:
    request_id: str
    workflow_id: str
    run_id: str
    operation_id: str
    phase: str
    substep: str
    subject_ref: str
    requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requirements"] = list(self.requirements)
        payload["status"] = "pending"
        return payload


@dataclass(frozen=True, slots=True)
class StagedApprovalDecision:
    request_id: str
    operation_id: str
    subject_ref: str
    status: str
    approval_ref: str
    requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requirements"] = list(self.requirements)
        return payload


class StagedApprovalService(Protocol):
    def decision_for(self, request: StagedApprovalRequest) -> StagedApprovalDecision | None: ...


@dataclass(slots=True)
class StagedAutonomousServices:
    """Injected subsystem adapters; missing capabilities block explicitly."""

    judge_entry: StagedOperationService | None = None
    evidence_collector: StagedOperationService | None = None
    diagnosis: StagedOperationService | None = None
    plan_readiness: StagedOperationService | None = None
    planner: StagedOperationService | None = None
    dataset_discovery: StagedOperationService | None = None
    dataset_selection: StagedOperationService | None = None
    dataset_acquisition: StagedOperationService | None = None
    data_preprocessor: StagedOperationService | None = None
    model_discovery: StagedOperationService | None = None
    model_selection: StagedOperationService | None = None
    training_lifecycle: StagedOperationService | None = None
    runtime_monitor: StagedOperationService | None = None
    generation: StagedOperationService | None = None
    evaluator: StagedOperationService | None = None
    recovery: StagedOperationService | None = None
    judge_exit: StagedOperationService | None = None
    action_executor: StagedOperationService | None = None


@dataclass(slots=True)
class StagedAutonomousDependencies:
    services: StagedAutonomousServices
    state_repository: StateRepository
    approval_service: StagedApprovalService | None = None
    record_sink: IntegrationRecordSink | None = None
    artifact_store: ArtifactStore | None = None
    job_queue: object | None = None


@dataclass(slots=True)
class StagedStepState:
    phase: str
    substep: str
    operation_id: str
    status: str = "pending"
    attempt: int = 0
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    approval_request: dict[str, object] | None = None
    resumable: bool = True
    completion_marker: bool = False
    output: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StagedStepState":
        return cls(
            phase=str(payload.get("phase", "")),
            substep=str(payload.get("substep", "")),
            operation_id=str(payload.get("operation_id", "")),
            status=str(payload.get("status", "pending")),
            attempt=int(payload.get("attempt", 0)),
            input_refs=[str(item) for item in payload.get("input_refs", [])],
            output_refs=[str(item) for item in payload.get("output_refs", [])],
            blocking_issues=[str(item) for item in payload.get("blocking_issues", [])],
            approval_request=(
                dict(payload["approval_request"])
                if isinstance(payload.get("approval_request"), dict)
                else None
            ),
            resumable=bool(payload.get("resumable", True)),
            completion_marker=bool(payload.get("completion_marker", False)),
            output=dict(payload.get("output", {})),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StagedWorkflowState:
    workflow_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    mode: str = GOVERNED_AUTONOMOUS_MODE
    current_phase: str = SpinePhase.JUDGE_ENTRY.value
    current_substep: str = PHASE_SUBSTEPS[SpinePhase.JUDGE_ENTRY][0]
    status: str = "pending"
    steps: dict[str, StagedStepState] = field(default_factory=dict)
    phase_order: list[str] = field(default_factory=lambda: [phase.value for phase in SPINE_ORDER])
    blocking_issues: list[str] = field(default_factory=list)
    resumable: bool = True
    completion_marker: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StagedWorkflowState":
        raw_steps = payload.get("steps", {})
        if not isinstance(raw_steps, dict):
            raw_steps = {}
        steps = {
            str(key): StagedStepState.from_dict(dict(value))
            for key, value in raw_steps.items()
            if isinstance(value, dict)
        }
        return cls(
            workflow_id=str(payload.get("workflow_id", "")),
            run_id=str(payload.get("run_id", "")),
            lineage_id=str(payload.get("lineage_id", "")),
            stage_name=str(payload.get("stage_name", "")),
            mode=str(payload.get("mode", GOVERNED_AUTONOMOUS_MODE)),
            current_phase=str(payload.get("current_phase", SpinePhase.JUDGE_ENTRY.value)),
            current_substep=str(payload.get("current_substep", PHASE_SUBSTEPS[SpinePhase.JUDGE_ENTRY][0])),
            status=str(payload.get("status", "pending")),
            steps=steps,
            phase_order=[str(item) for item in payload.get("phase_order", [])],
            blocking_issues=[str(item) for item in payload.get("blocking_issues", [])],
            resumable=bool(payload.get("resumable", True)),
            completion_marker=bool(payload.get("completion_marker", False)),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["steps"] = {key: step.to_dict() for key, step in self.steps.items()}
        return payload
