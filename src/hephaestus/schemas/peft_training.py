"""Explicit LoRA recipe and immutable parent-adapter identity."""
from dataclasses import dataclass, field
from ._base import JsonSchema


@dataclass(slots=True)
class PeftTrainingSpec(JsonSchema):
    rank: int
    alpha: float
    dropout: float
    target_modules: list[str]
    parent_adapter_ref: str | None = None
    parent_adapter_hash: str | None = None
    # Cross-run adapter warm starts reset optimization state. Faithful state
    # continuation is the lifecycle's interrupted-run resume, not this option.
    continuation_kind: str = "fresh"
    bias: str = "none"
