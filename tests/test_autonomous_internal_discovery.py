import json
from dataclasses import replace
from hashlib import sha256
import pytest
pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers
from hephaestus.data.internal_screening import InternalSourceScanner
from hephaestus.data.intelligence import AutonomousDatasetIntelligence, derive_data_requirements
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.providers.datasets.approved_registry import EvidenceBoundSourceInspector
from hephaestus.schemas.data_intelligence import DiscoveryBudget
from hephaestus.schemas.discovery_contract import DatasetCandidate
from hephaestus.schemas.internal_research import InternalDatasetReview
from hephaestus.policy.internal_research import put_record
from hephaestus.policy.dataset_review import candidate_sha256
from test_internal_research_policy import setup_policy


class FixtureSemanticChecker:
    """Test-only task semantic hook; live implementation arrives with generated task validator."""
    def check(self, rows, refs):
        return {"overlap": 0, "method": "test_only", "limitations": ["test-only semantic checker"]}


class FixtureProvider:
    provider_id = "test_registry"
    def __init__(self, candidates):
        self.candidates = candidates
    def search(self, request):
        return self.candidates


def prepared(tmp_path):
    artifacts, store, policy, admission = setup_policy(tmp_path)
    tok = Tokenizer(models.WordLevel({"<unk>": 0, "add": 1, "seven": 2, "eight": 3,
        "fifteen": 4}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    digest = "sha256:" + sha256(path.read_bytes()).hexdigest()
    policy = replace(policy, tokenizer_sha256=digest, allowed_licenses=("cc0-1.0",))
    policy_ref = put_record(artifacts, policy.to_dict())
    scanner = InternalSourceScanner(artifacts, path, digest, FixtureSemanticChecker())
    admission = replace(admission, policy_ref=policy_ref, trusted_policy_refs=frozenset({policy_ref}), scanner=scanner)
    raw = b'{"prompt":"add seven eight","target":"fifteen"}\n'
    source = artifacts.put_bytes(raw, media_type="application/jsonl").artifact_ref
    files = {"source.jsonl": source}
    candidate = DatasetCandidate("source", "test_registry", "authored-source", revision="a" * 40,
        license="cc0-1.0", task_types=["arithmetic"], languages=["en"], domains=["math"], metadata={"acquisition_files": ["source.jsonl"]})
    rights = put_record(artifacts, {"schema": "data-rights-evidence.v1", "license": candidate.license,
        "source_file_hashes": files, "training_permitted": True, "test_only": True})
    policy = replace(policy, trusted_rights_refs=(rights,))
    ref = put_record(artifacts, policy.to_dict())
    admission = replace(admission, policy_ref=ref, trusted_policy_refs=frozenset({ref}))
    scan = scanner.scan(files, protocol_ref=policy.protocol_ref, processing_profile_sha256=policy.processing_profile_sha256)
    review = InternalDatasetReview(candidate_sha256(candidate), files, policy.processing_profile_sha256,
        digest, policy.protocol_ref, "arithmetic", rights, put_record(artifacts, scan))
    review_ref = put_record(artifacts, review.to_dict())
    scope = {"experiment_id": policy.experiment_id, "workflow_id": policy.workflow_id, "run_id": policy.run_id,
        "processing_profile_sha256": policy.processing_profile_sha256, "tokenizer_sha256": digest}
    return artifacts, admission, candidate, review, review_ref, scope


def test_actual_source_screen_and_rejection_continue_to_valid_alternative(tmp_path):
    artifacts, admission, candidate, review, ref, scope = prepared(tmp_path)
    bad = replace(candidate, candidate_id="ambiguous-first", license=None)
    registry = DatasetProviderRegistry(provider_allowlist={"test_registry"})
    registry.register(FixtureProvider([bad, candidate]))
    request = derive_data_requirements({"report_id": "diag", "problem_statement": "arithmetic deficit",
        "capability_gaps": ["arithmetic"]}, stage="targeted_repair", objective="arithmetic",
        tokenizer_ref="frozen", architecture={"context": 128}, providers=["test_registry"], license_allowlist=["CC0-1.0"])
    intelligence = AutonomousDatasetIntelligence(registry, EvidenceBoundSourceInspector(artifacts, {candidate.candidate_id: ref}),
        admission, artifacts, DiscoveryBudget())
    decision, decision_ref = intelligence.discover_select(request, **scope)
    assert decision.selected_candidate_ids == ["source"]
    assert "ambiguous-first" in decision.rejected_candidates
    assert decision.metadata["examined_candidates"] == 2
    assert len(decision.evidence_refs) == 3
    assert admission.authorize(candidate, ref, **scope).startswith("sha256:")
    for k in ("experiment_id", "workflow_id", "run_id", "processing_profile_sha256", "tokenizer_sha256"):
        with pytest.raises(PermissionError):
            admission.authorize(candidate, ref, **{**scope, k: "substitution"})


@pytest.mark.parametrize("change", ["source", "tokenizer", "pass_flag", "rights", "policy", "semantic"])
def test_forged_scans_substitution_and_policy_downgrade_fail(tmp_path, change):
    artifacts, admission, candidate, review, ref, scope = prepared(tmp_path)
    if change == "source":
        candidate = replace(candidate, revision="b" * 40)
    elif change == "tokenizer":
        admission.scanner.tokenizer_path.write_text("{}")
    elif change == "policy":
        admission = replace(admission, trusted_policy_refs=frozenset())
    elif change == "rights":
        review.rights_ref = put_record(artifacts, {"training_permitted": True})
        ref = put_record(artifacts, review.to_dict())
    else:
        from hephaestus.storage.verified import read_verified_json
        scan = read_verified_json(artifacts, review.scan_ref)
        scan["total_tokens" if change == "pass_flag" else "semantic_overlap"] = 900 if change == "pass_flag" else 1
        review.scan_ref = put_record(artifacts, scan)
        ref = put_record(artifacts, review.to_dict())
    with pytest.raises((PermissionError, ValueError)):
        admission.authorize(candidate, ref, **scope)


def test_budget_and_natural_language_requirement_do_not_weaken_admission(tmp_path):
    artifacts, admission, candidate, _, ref, scope = prepared(tmp_path)
    candidate.metadata["synthetic"] = True
    registry = DatasetProviderRegistry(provider_allowlist={"test_registry"})
    registry.register(FixtureProvider([candidate]))
    request = derive_data_requirements({"report_id": "d", "problem_statement": "pretrain", "capability_gaps": ["arithmetic"]},
        stage="base", objective="pretraining", tokenizer_ref="t", architecture={"context": 128},
        providers=["test_registry"], license_allowlist=["CC0-1.0"])
    service = AutonomousDatasetIntelligence(registry, EvidenceBoundSourceInspector(artifacts, {"source": ref}), admission, artifacts, DiscoveryBudget())
    decision, _ = service.discover_select(request, **scope)
    assert decision.status == "inconclusive"
    assert "cannot substitute" in decision.rejected_candidates["source"]
    for value in (0, True, float("inf")):
        with pytest.raises(ValueError):
            DiscoveryBudget(maximum_candidates=value).validate()
