"""Provider and selector protocols for governed discovery."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSearchRequest,
    DatasetSelectionDecision,
    ModelCandidate,
    ModelSearchRequest,
    ModelSelectionDecision,
)


@runtime_checkable
class DatasetDiscoveryProvider(Protocol):
    provider_id: str
    def search(self, request: DatasetSearchRequest) -> Sequence[DatasetCandidate]: ...


@runtime_checkable
class ModelDiscoveryProvider(Protocol):
    provider_id: str
    def search(self, request: ModelSearchRequest) -> Sequence[ModelCandidate]: ...


@runtime_checkable
class DatasetSelectionService(Protocol):
    def select(
        self, request: DatasetSearchRequest, candidates: Sequence[DatasetCandidate]
    ) -> DatasetSelectionDecision: ...


@runtime_checkable
class ModelSelectionService(Protocol):
    def select(
        self, request: ModelSearchRequest, candidates: Sequence[ModelCandidate]
    ) -> ModelSelectionDecision: ...
