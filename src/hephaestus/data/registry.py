from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from hephaestus.interfaces.discovery import DatasetDiscoveryProvider
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest

from .normalization import normalize_dataset_candidate


@dataclass(frozen=True, slots=True)
class DatasetDiscoveryResult:
    request_id: str
    candidates: tuple[DatasetCandidate, ...] = ()
    provider_ids: tuple[str, ...] = ()
    issues: tuple[ContractIssue, ...] = ()


@dataclass(slots=True)
class DatasetProviderRegistry:
    """Allowlisted provider registry and auditable discovery entry point."""

    providers: dict[str, DatasetDiscoveryProvider] = field(default_factory=dict)
    provider_allowlist: set[str] = field(default_factory=set)

    def register(self, provider: DatasetDiscoveryProvider) -> None:
        provider_id = str(provider.provider_id).strip().casefold()
        if not provider_id:
            raise ValueError("dataset provider_id must not be empty")
        if provider_id in self.providers:
            raise ValueError(f"dataset provider already registered: {provider_id}")
        self.providers[provider_id] = provider

    def discover(self, request: DatasetSearchRequest) -> DatasetDiscoveryResult:
        requested = {value.strip().casefold() for value in request.provider_allowlist if value.strip()}
        configured = {value.strip().casefold() for value in self.provider_allowlist if value.strip()}
        constraints_active = bool(requested or configured)
        allowed = requested & configured if requested and configured else requested or configured
        provider_ids = (
            sorted(provider_id for provider_id in self.providers if provider_id in allowed)
            if constraints_active
            else []
        )
        issues: list[ContractIssue] = []
        candidates: list[DatasetCandidate] = []
        seen: set[str] = set()

        if not constraints_active:
            issues.append(
                ContractIssue(
                    code="dataset_provider_allowlist_required",
                    category="policy_blocked",
                    message="dataset discovery requires a configured or request-level provider allowlist",
                    retryable=False,
                    blocking=True,
                )
            )

        if requested:
            missing = sorted(requested - set(self.providers))
            for provider_id in missing:
                issues.append(
                    ContractIssue(
                        code="dataset_provider_not_registered",
                        category="provider_unavailable",
                        message=f"requested dataset provider is not registered: {provider_id}",
                        retryable=False,
                        blocking=False,
                        metadata={"provider_id": provider_id},
                    )
                )
            if configured:
                disallowed = sorted((requested & set(self.providers)) - configured)
                for provider_id in disallowed:
                    issues.append(
                        ContractIssue(
                            code="dataset_provider_not_allowlisted",
                            category="policy_blocked",
                            message=f"requested dataset provider is not in the configured allowlist: {provider_id}",
                            retryable=False,
                            blocking=False,
                            metadata={"provider_id": provider_id},
                        )
                    )

        for provider_id in provider_ids:
            provider = self.providers[provider_id]
            try:
                discovered: Iterable[DatasetCandidate] = provider.search(request)
                for raw_candidate in discovered:
                    candidate = normalize_dataset_candidate(raw_candidate, provider_id=provider_id)
                    if candidate.candidate_id in seen:
                        issues.append(
                            ContractIssue(
                                code="duplicate_candidate_id",
                                category="internal_contract_violation",
                                message=f"duplicate candidate_id ignored: {candidate.candidate_id}",
                                metadata={"provider_id": provider_id},
                            )
                        )
                        continue
                    seen.add(candidate.candidate_id)
                    candidates.append(candidate)
            except Exception as exc:  # providers are an explicit failure boundary
                issues.append(
                    ContractIssue(
                        code="dataset_provider_failure",
                        category="provider_unavailable",
                        message=f"provider {provider_id} failed: {type(exc).__name__}: {exc}",
                        retryable=True,
                        blocking=False,
                        metadata={"provider_id": provider_id, "exception_type": type(exc).__name__},
                    )
                )

        candidates.sort(key=lambda item: item.candidate_id)
        if not candidates:
            issues.append(
                ContractIssue(
                    code="no_dataset_candidates",
                    category="candidate_not_found",
                    message="no dataset candidates were discovered",
                    retryable=bool(provider_ids),
                    blocking=True,
                )
            )
        return DatasetDiscoveryResult(
            request_id=request.request_id,
            candidates=tuple(candidates),
            provider_ids=tuple(provider_ids),
            issues=tuple(issues),
        )
