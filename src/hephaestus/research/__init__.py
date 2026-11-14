"""Bounded research jobs, distinct from model-training campaigns and promotions."""

from .autonomous_study import (
    CertifiedDiagnosisObserver,
    EvaluationOnlyResearchJob,
    ReadOnlyS3SourceReader,
    ResearchAdmissionError,
    ResearchObjective,
    SourceCatalog,
    SourceDescriptor,
    VerifiedSourceAcquirer,
)

__all__ = (
    "CertifiedDiagnosisObserver",
    "EvaluationOnlyResearchJob",
    "ReadOnlyS3SourceReader",
    "ResearchAdmissionError",
    "ResearchObjective",
    "SourceCatalog",
    "SourceDescriptor",
    "VerifiedSourceAcquirer",
)

from .constrained_repair import (
    BoundedPatchProposer,
    ConstrainedSelfRepair,
    GitHubReviewBranchPublisher,
    OfflineDockerPytestRunner,
    RepairBoundaryError,
    RepairScope,
)
from .durable_jobs import ResearchObjectiveJobHandler, submit_research_objective
from .self_repair_controller import AutonomousDiagnosisRepairEpisode, LoopbackOllamaPatchModel
