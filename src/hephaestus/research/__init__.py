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
