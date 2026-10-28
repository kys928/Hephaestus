#!/usr/bin/env python3
"""Real metadata triage + original validated source selection. No model training."""
import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
from huggingface_hub import snapshot_download
from hephaestus.data.generated_factory import (prepare_original_source, generator_code_sha256,
    OriginalTaskRegistryProvider, materialize_authorized_tokens, load_authorized_tokens)
from hephaestus.data.generated_validation import OriginalTaskSemanticChecker
from hephaestus.data.internal_screening import InternalSourceScanner
from hephaestus.data.intelligence import AutonomousDatasetIntelligence, derive_data_requirements
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.llm.cpu_causal_research import MODEL_ID, MODEL_REVISION
from hephaestus.policy.internal_research import InternalDatasetAdmissionPolicy, put_record
from hephaestus.providers.datasets.approved_registry import ApprovedHubRegistryProvider, EvidenceBoundSourceInspector
from hephaestus.schemas.data_intelligence import DiscoveryBudget
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.schemas.internal_research import InternalResearchPolicy
from hephaestus.state.prospective_evaluation import ProspectiveEvaluationStore
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import read_verified_json


def prepare_inputs(root, *, tokenizer_path, include_public=True):
    artifacts = FileSystemArtifactStore(root / "artifacts")
    repository = SQLiteStateRepository(root / "state.sqlite")
    evaluation = ProspectiveEvaluationStore(repository, artifacts)
    tokenizer_digest = "sha256:" + sha256(tokenizer_path.read_bytes()).hexdigest()
    scanner = InternalSourceScanner(artifacts, tokenizer_path, tokenizer_digest, OriginalTaskSemanticChecker())
    profile = "sha256:" + sha256(b'original-prompt-Answer-boundary-no-eos-no-truncation-v1').hexdigest()
    spec = GeneratedDataSpec(20261028)
    candidate, review_ref, protocol_ref, generator_ref = prepare_original_source(
        artifacts, evaluation, scanner, spec, experiment_id="internal-original-repair",
        processing_profile_sha256=profile)
    policy = InternalResearchPolicy("bounded-original-research-v1", "user-implementation-mandate",
        "internal-original-repair", "cpu-research-workflow", "cpu-research-run",
        protocol_ref, profile, tokenizer_digest, 8*1024*1024, 2000, 2, 600,
        ("cc0-1.0", "mit"), allowed_generator_code_sha256=(generator_code_sha256(),),
        allowed_capabilities=spec.families)
    policy_ref = put_record(artifacts, policy.to_dict())
    admission = InternalDatasetAdmissionPolicy(artifacts, evaluation, policy_ref,
        frozenset({policy_ref}), scanner)
    providers = ["original_task_registry"] + (["approved_hub_registry"] if include_public else [])
    registry = DatasetProviderRegistry(provider_allowlist=set(providers))
    registry.register(OriginalTaskRegistryProvider((candidate,)))
    if include_public:
        registry.register(ApprovedHubRegistryProvider(("Salesforce/wikitext", "openai/gsm8k"), artifacts))
    request = derive_data_requirements({"report_id": "user-requested-targets-20261028",
        "problem_statement": "Implementation validation for requested arithmetic, JSON, symbolic and reasoning repair capabilities.",
        "capability_gaps": list(spec.families), "epistemic_status": "requested_targets_not_new_model_measurements"},
        stage="targeted_repair", objective="verifiable_task_repair", tokenizer_ref=str(tokenizer_path),
        architecture={"context_length": 2048, "model_id": MODEL_ID, "revision": MODEL_REVISION},
        providers=providers, license_allowlist=policy.allowed_licenses)
    scope = {"experiment_id": policy.experiment_id, "workflow_id": policy.workflow_id,
        "run_id": policy.run_id, "processing_profile_sha256": profile, "tokenizer_sha256": tokenizer_digest}
    intelligence = AutonomousDatasetIntelligence(registry, EvidenceBoundSourceInspector(artifacts,
        {candidate.candidate_id: review_ref}), admission, artifacts, DiscoveryBudget())
    decision, decision_ref = intelligence.discover_select(request, **scope)
    if decision.selected_candidate_ids != [candidate.candidate_id]:
        raise RuntimeError("no eligible original source selected: " + str(decision.rejected_candidates))
    if include_public and len(decision.rejected_candidates) < 2:
        raise RuntimeError("real public candidates were not individually examined/rejected")
    materialized = materialize_authorized_tokens(admission, candidate, review_ref, **scope)
    tokens = load_authorized_tokens(admission, candidate, review_ref, materialized, **scope)
    review = read_verified_json(artifacts, review_ref)
    scan = read_verified_json(artifacts, review["scan_ref"])
    report = {"schema": "internal-data-intelligence-executed.v1", "decision_ref": decision_ref,
        "policy_ref": policy_ref, "protocol_ref": protocol_ref, "review_ref": review_ref,
        "generator_ref": generator_ref, "materialization_ref": materialized,
        "selected_source": candidate.to_dict(), "decision": decision.to_dict(),
        "scan": scan, "trainable_records": len(tokens),
        "input_tokens": sum(len(r["input_ids"]) for r in tokens),
        "supervised_tokens": sum(sum(t != -100 for t in r["labels"]) for r in tokens),
        "external_certification": False, "historical_reference_recreated": False,
        "training_executed": False, "campaign_progress_claimed": False,
        "scope": scope}
    (root / "data_intelligence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return admission, candidate, review_ref, materialized, scope, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/hephaestus-model-cache"))
    parser.add_argument("--no-public", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new input evidence directory required")
    args.output.mkdir(parents=True)
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(args.cache_dir),
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))
    *_, report = prepare_inputs(args.output, tokenizer_path=snapshot / "tokenizer.json", include_public=not args.no_public)
    print(json.dumps({"selected": report["decision"]["selected_candidate_ids"],
        "rejected": report["decision"]["rejected_candidates"], "rows": report["trainable_records"],
        "unknown_tokens": report["scan"]["unknown_tokens"], "decision_ref": report["decision_ref"]}, sort_keys=True))


if __name__ == "__main__":
    main()
