from __future__ import annotations

from dataclasses import dataclass, field

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest


@dataclass(slots=True)
class FakeDatasetProvider:
    """Deterministic, network-free provider for tests and simulations."""

    candidates: tuple[DatasetCandidate, ...] = field(default_factory=tuple)
    provider_id: str = "fake"
    failure_message: str | None = None

    def search(self, request: DatasetSearchRequest) -> tuple[DatasetCandidate, ...]:
        del request
        if self.failure_message:
            raise RuntimeError(self.failure_message)
        return tuple(self.candidates)
