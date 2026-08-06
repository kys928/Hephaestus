"""Frozen evaluation generation bridge."""

from .backends import (
    DeterministicFakeGenerationBackend,
    GenerationBackend,
    GenerationBackendError,
    TransformersCausalLMGenerationBackend,
    load_generation_instructions,
)
from .models import (
    GeneratedText,
    GenerationPlan,
    GenerationReport,
    GenerationResult,
    GenerationSample,
    GenerationTask,
)
from .service import EvaluationGenerationService, generation_evidence_refs
from .staged import StagedExperimentEvaluationAdapter, StagedGenerationAdapter

__all__ = [
    "DeterministicFakeGenerationBackend",
    "EvaluationGenerationService",
    "GeneratedText",
    "GenerationBackend",
    "GenerationBackendError",
    "GenerationPlan",
    "GenerationReport",
    "GenerationResult",
    "GenerationSample",
    "GenerationTask",
    "StagedExperimentEvaluationAdapter",
    "StagedGenerationAdapter",
    "TransformersCausalLMGenerationBackend",
    "generation_evidence_refs",
    "load_generation_instructions",
]
