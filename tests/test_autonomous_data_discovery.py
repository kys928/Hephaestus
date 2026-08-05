from __future__ import annotations

from hephaestus.data.normalization import normalize_dataset_candidate
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.data.selection import DeterministicDatasetSelectionService
from hephaestus.providers.datasets import FakeDatasetProvider
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest


def _request(**overrides: object) -> DatasetSearchRequest:
    values: dict[str, object] = {
        "request_id": "search-1",
        "diagnosis_report_id": "diag-1",
        "problem_statement": "English instruction following coverage",
        "capability_targets": ["instruction_following"],
        "required_languages": ["en"],
        "required_domains": ["general"],
        "required_formats": ["jsonl"],
        "license_allowlist": ["mit", "apache-2.0"],
        "provider_allowlist": ["fake"],
    }
    values.update(overrides)
    return DatasetSearchRequest(**values)


def _candidate(candidate_id: str, **overrides: object) -> DatasetCandidate:
    values: dict[str, object] = {
        "candidate_id": candidate_id,
        "provider_id": "fake",
        "dataset_id": f"instructions-{candidate_id}",
        "revision": "v1",
        "task_types": ["instruction_following"],
        "languages": ["en"],
        "domains": ["general"],
        "format_profile": {"record_format": "jsonl"},
        "estimated_rows": 1000,
        "estimated_bytes": 100_000,
        "license": "mit",
        "provenance": {"source": "fixture"},
        "trust_level": "verified",
        "compatibility": {"model_compatible": True, "tokenizer_compatible": True},
        "evidence_refs": [f"fixture://{candidate_id}"],
    }
    values.update(overrides)
    return DatasetCandidate(**values)


def test_fake_provider_discovery_normalizes_and_preserves_multiple_candidates() -> None:
    provider = FakeDatasetProvider(candidates=(_candidate("c2"), _candidate("c1"), _candidate("c3")))
    registry = DatasetProviderRegistry(provider_allowlist={"fake"})
    registry.register(provider)

    result = registry.discover(_request())

    assert [candidate.candidate_id for candidate in result.candidates] == ["c1", "c2", "c3"]
    assert result.provider_ids == ("fake",)
    assert not [issue for issue in result.issues if issue.blocking]


def test_candidate_normalization_and_selection_are_deterministic() -> None:
    candidate = _candidate(
        "c1",
        languages=["EN", "en"],
        domains=["General", "general"],
        metadata={"z": 1, "a": ["second", "first"]},
    )
    first = normalize_dataset_candidate(candidate)
    second = normalize_dataset_candidate(candidate)
    selector = DeterministicDatasetSelectionService()

    decision_a = selector.select(_request(), [first, _candidate("c2")])
    decision_b = selector.select(_request(), [_candidate("c2"), second])

    assert first.to_dict() == second.to_dict()
    assert first.languages == ["en"]
    assert decision_a.to_dict() == decision_b.to_dict()
    assert decision_a.status == "selected"
    assert decision_a.selected_candidate_ids == ["c1"]
    assert set(decision_a.metadata["candidate_audits"]) == {"c1", "c2"}


def test_unknown_license_blocks_selection_and_requests_approval() -> None:
    candidate = _candidate("unknown-license", license=None)

    decision = DeterministicDatasetSelectionService().select(_request(), [candidate])

    assert decision.status == "blocked"
    assert decision.selected_candidate_ids == []
    assert "unknown_license:unknown-license" in decision.required_approvals
    assert decision.issues[0].category == "approval_required"


def test_incompatible_or_policy_denied_candidates_produce_inconclusive() -> None:
    incompatible = _candidate("bad-model", compatibility={"model_compatible": False})
    denied = _candidate("denied-license", license="gpl-3.0")

    decision = DeterministicDatasetSelectionService().select(_request(), [incompatible, denied])

    assert decision.status == "inconclusive"
    assert "incompatible:model_compatible" in decision.rejected_candidates["bad-model"]
    assert "license_not_allowlisted:gpl-3.0" in decision.rejected_candidates["denied-license"]
    assert decision.selected_candidate_ids == []


def test_provider_failure_is_a_contract_issue_not_an_exception() -> None:
    registry = DatasetProviderRegistry(provider_allowlist={"fake"})
    registry.register(FakeDatasetProvider(failure_message="offline"))

    result = registry.discover(_request())

    assert result.candidates == ()
    assert any(issue.category == "provider_unavailable" for issue in result.issues)
    assert any(issue.category == "candidate_not_found" and issue.blocking for issue in result.issues)


def test_provider_allowlist_intersection_cannot_fall_back_to_all_providers() -> None:
    registry = DatasetProviderRegistry(provider_allowlist={"local_fixture"})
    registry.register(FakeDatasetProvider(candidates=(_candidate("c1"),)))

    result = registry.discover(_request(provider_allowlist=["fake"]))

    assert result.candidates == ()
    assert result.provider_ids == ()
    assert any(issue.category == "policy_blocked" for issue in result.issues)


def test_discovery_requires_an_explicit_provider_allowlist() -> None:
    registry = DatasetProviderRegistry()
    registry.register(FakeDatasetProvider(candidates=(_candidate("c1"),)))

    result = registry.discover(_request(provider_allowlist=[]))

    assert result.candidates == ()
    assert any(issue.code == "dataset_provider_allowlist_required" and issue.blocking for issue in result.issues)
