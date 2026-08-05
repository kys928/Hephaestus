from __future__ import annotations

from hephaestus.interfaces import ModelDiscoveryProvider, ModelSelectionService
from hephaestus.providers.models import (
    DeterministicModelSelectionService,
    ExternalModelRegistryProvider,
    FakeModelProvider,
)
from hephaestus.schemas.discovery_contract import ModelSearchRequest


def _request(**overrides: object) -> ModelSearchRequest:
    values: dict[str, object] = {
        "request_id": "models-1",
        "diagnosis_report_id": "diagnosis-1",
        "problem_statement": "Need a bounded causal-LM smoke test.",
        "task_requirements": ["causal_lm", "smoke_test"],
        "architecture_constraints": {"allowed_families": ["tiny_linear_lm"], "min_context_length": 32},
        "tokenizer_constraints": {"tokenizer_ref": "fixture://byte-tokenizer/v1"},
        "runtime_constraints": {
            "backend": "local_fixture",
            "max_memory_gb": 0.1,
            "smoke_test_required": True,
            "checkpoint_integrity": "content_hash",
        },
        "budget_constraints": {"max_parameters": 10, "max_runtime_seconds": 10},
        "license_allowlist": ["internal-test-only"],
    }
    values.update(overrides)
    return ModelSearchRequest(**values)


def test_fake_provider_conforms_and_returns_explicit_candidates() -> None:
    provider = FakeModelProvider()
    assert isinstance(provider, ModelDiscoveryProvider)
    candidates = provider.search(_request())
    assert [candidate.model_id for candidate in candidates] == ["tiny-char-lm", "tiny-unknown-license"]
    selected = candidates[0]
    assert selected.revision == "fixture-v1"
    assert selected.score_components["metadata_completeness"] == 1.0
    assert selected.runtime_requirements["supported_backends"] == ["local_fixture", "fake"]


def test_model_ranking_is_deterministic_and_preserves_scores() -> None:
    request = _request()
    candidates = FakeModelProvider().search(request)
    selector = DeterministicModelSelectionService()
    assert isinstance(selector, ModelSelectionService)
    first = selector.select(request, candidates)
    second = selector.select(request, list(reversed(candidates)))
    assert first.to_dict() == second.to_dict()
    assert first.status == "selected"
    assert first.selected_candidate_id == "fake_models:tiny-char-lm@fixture-v1"
    assert first.metadata["score_components"][first.selected_candidate_id]["total"] > 0


def test_unknown_license_and_incompatible_runtime_never_silently_pass() -> None:
    request = _request(license_allowlist=[])
    candidates = FakeModelProvider().search(request)
    unknown_only = [candidate for candidate in candidates if candidate.license is None]
    decision = DeterministicModelSelectionService().select(request, unknown_only)
    assert decision.status == "blocked"
    assert "license_unknown" in decision.rejected_candidates[unknown_only[0].candidate_id]
    assert decision.issues[0].category == "license_unknown"

    incompatible_request = _request(runtime_constraints={"backend": "gpu_cluster"})
    incompatible = DeterministicModelSelectionService().select(incompatible_request, candidates[:1])
    assert incompatible.status == "blocked"
    assert "backend_incompatible" in incompatible.rejected_candidates[candidates[0].candidate_id]


def test_unavailable_requested_revision_is_rejected() -> None:
    request = _request(metadata={"model_id": "tiny-char-lm", "revision": "not-present"})
    candidates = FakeModelProvider().search(request)
    assert len(candidates) == 1
    decision = DeterministicModelSelectionService().select(request, candidates)
    assert decision.status == "blocked"
    assert "revision_unavailable" in decision.rejected_candidates[candidates[0].candidate_id]


def test_external_registry_is_explicitly_opt_in() -> None:
    provider = ExternalModelRegistryProvider(fetch_entries=lambda _request: [])
    try:
        provider.search(_request())
    except RuntimeError as exc:
        assert "opt-in" in str(exc)
    else:
        raise AssertionError("disabled external registry must not execute")
