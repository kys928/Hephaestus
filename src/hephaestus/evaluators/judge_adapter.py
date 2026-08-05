"""Optional judge-model boundary for semantic evaluation."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class JudgeModelAdapter(Protocol):
    """Return bounded dimension scores for one already-generated response.

    Implementations may call a model, but the evaluation service does not know
    how that happens. Tests and offline runs can inject a deterministic fake.
    Missing dimensions are treated as missing evidence, never as zero or pass.
    """

    def score(self, task: Mapping[str, object], response: str) -> Mapping[str, float]: ...
