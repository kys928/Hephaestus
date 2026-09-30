"""Hephaestus control and integration entry points."""

from .autonomous_experiment import (
    ApprovalAwareDatasetSelectionService,
    AutonomousExperimentCoordinator,
    DiscoveryBundle,
    GuardedTrainingLifecycleService,
    InMemoryIntegrationRecordSink,
    IntegratedDiagnosisService,
    PlanningBundle,
    TruthNormalizingEvidenceAdapter,
    normalize_diagnostic_truth_values,
)
from .evaluator_boundary import (
    DECISION_TO_ACTION,
    EvaluatorPolicyProjection,
    action_for_evaluator_decision,
    project_evaluator_action,
)
from .production_autonomy import ProductionAutonomyCoordinator
from .staged_autonomous import (
    GovernedStagedOrchestrator,
    build_staged_autonomous_orchestrator,
)
from .staged_state import (
    GOVERNED_AUTONOMOUS_MODE,
    PHASE_SUBSTEPS,
    StagedApprovalDecision,
    StagedApprovalRequest,
    StagedApprovalService,
    StagedAutonomousDependencies,
    StagedAutonomousServices,
    StagedOperationRequest,
    StagedOperationResult,
    StagedOperationService,
    StagedOutputRecord,
    StagedStepState,
    StagedWorkflowState,
)

__all__ = [
    "ApprovalAwareDatasetSelectionService",
    "AutonomousExperimentCoordinator",
    "DiscoveryBundle",
    "GuardedTrainingLifecycleService",
    "InMemoryIntegrationRecordSink",
    "IntegratedDiagnosisService",
    "PlanningBundle",
    "ProductionAutonomyCoordinator",
    "TruthNormalizingEvidenceAdapter",
    "normalize_diagnostic_truth_values",
    "DECISION_TO_ACTION",
    "EvaluatorPolicyProjection",
    "action_for_evaluator_decision",
    "project_evaluator_action",
    "GOVERNED_AUTONOMOUS_MODE",
    "PHASE_SUBSTEPS",
    "GovernedStagedOrchestrator",
    "StagedApprovalDecision",
    "StagedApprovalRequest",
    "StagedApprovalService",
    "StagedAutonomousDependencies",
    "StagedAutonomousServices",
    "StagedOperationRequest",
    "StagedOperationResult",
    "StagedOperationService",
    "StagedOutputRecord",
    "StagedStepState",
    "StagedWorkflowState",
    "build_staged_autonomous_orchestrator",
]
