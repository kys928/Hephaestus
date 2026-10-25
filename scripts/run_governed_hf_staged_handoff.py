#!/usr/bin/env python3
"""Live read-only shortlisted HF -> persisted staged operations 10/11.

Imports the ORIGINAL frozen Planner context, explicitly NOT nine newly
executed stages. No fabricated approvals, corpus downloads, training or GPU.
The resulting partial handoff is never a resumed or complete 45-stage run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tokenizers import Tokenizer
from hephaestus.control.spine import SpinePhase
from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import StagedAutonomousServices, StagedAutonomousDependencies, WORKFLOW_RECORD_COLLECTION
from hephaestus.policy.dataset_review import IndependentDatasetReviewPolicy, candidate_sha256
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.providers.datasets import HuggingFaceDatasetProvider
from hephaestus.schemas.dataset_review import DatasetSourceReview, DatasetScreeningAccessScope
from hephaestus.schemas.discovery_contract import DatasetCandidate
from hephaestus.storage.verified import canonical_bytes, read_verified_json

import run_schema_aware_public_preview_review as study
from run_first_autonomous_dataset_discovery import (
    load_frozen_chain, TOKENIZER_ID, TOKENIZER_KEY, get_s3_bytes, s3_client, required,
)

OUTPUT = Path("governed_hf_handoff")
SOURCES = {
    "vara201/ecommerce-orders-structured-output": "train.parquet",
    "Short-Answer-Feedback/saf_communication_networks_english": "data/train-",
    "Salesforce/wikitext": "wikitext-2-raw-v1/train-",
}


@dataclass(slots=True)
class ReviewedPublicShortlistProvider:
    """Live-resolved source metadata, scoped to the reviewed shortlist."""
    candidates: tuple[DatasetCandidate, ...]
    provider_id: str = "huggingface"

    def search(self, request):
        return list(self.candidates)


def preview_with_identity(dataset, config, split, identities):
    url = "https://datasets-server.huggingface.co/first-rows?" + urlencode({
        "dataset": dataset, "config": config, "split": split})
    with urlopen(Request(url, headers={"User-Agent": "hephaestus-readonly-staged-handoff/1"}), timeout=20) as response:
        data = response.read(2 * 1024 * 1024 + 1)
        revision = response.headers.get("x-revision")
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("public preview exceeds byte ceiling")
    identities[dataset] = {"response_sha256": hashlib.sha256(data).hexdigest(),
        "response_bytes": len(data), "server_revision": revision,
        "scope": "public_first_100_train_rows"}
    payload = json.loads(data)
    if payload.get("dataset") != dataset or payload.get("config") != config or payload.get("split") != split:
        raise ValueError("preview response does not match the scoped dataset/config/train split")
    return [dict(row["row"]) for row in payload.get("rows", [])[:100]
        if isinstance(row, dict) and isinstance(row.get("row"), dict) and not row.get("truncated_cells")]


def main():
    original, diagnosis, intervention = load_frozen_chain()
    raw = get_s3_bytes(s3_client(), required("RUNPOD_NETWORK_VOLUME_ID"), TOKENIZER_KEY)
    sha = hashlib.sha256(raw).hexdigest()
    if sha != study.EXPECTED_TOKENIZER_BYTES_SHA256 or original.tokenizer_ref != TOKENIZER_ID:
        raise ValueError("frozen tokenizer identity mismatch")
    tok = Tokenizer.from_str(raw.decode("utf-8"))
    OUTPUT.mkdir(exist_ok=True)
    rt = ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=OUTPUT / "state", artifact_root=OUTPUT / "artifacts")).build()
    def put(value):
        return rt.artifact_store.put_bytes(canonical_bytes(value), media_type="application/json").artifact_ref
    provider = HuggingFaceDatasetProvider(enable_network=True, timeout_seconds=20)
    identities, candidates, review_refs, scope_requests, reviews = {}, [], {}, [], []
    study.hf_first_rows = lambda dataset, config, split: preview_with_identity(dataset, config, split, identities)
    for dataset, path_prefix in SOURCES.items():
        snapshot = provider.resolve_revision(dataset, "main")
        files = provider.enumerate_files(snapshot)
        selected_files = [f for f in files if f.relative_path.startswith(path_prefix)
                          and f.relative_path.endswith(".parquet")]
        if len(selected_files) != 1:
            raise ValueError("reviewed source does not resolve to exactly one train file")
        file = selected_files[0]
        candidate = DatasetCandidate(
            candidate_id="huggingface:" + dataset + "@" + snapshot.resolved_revision,
            provider_id="huggingface", dataset_id=dataset, revision=snapshot.resolved_revision,
            splits=["train"], languages=["en"], task_types=["text-generation"],
            domains=[study.REVIEW[dataset]["task"]], format_profile={"record_format": "parquet"},
            estimated_bytes=file.size_bytes, license=snapshot.license, trust_level="external_metadata",
            provenance={"sha": snapshot.resolved_revision, "dataset_card_ref": snapshot.dataset_card_ref},
            compatibility={"remote_code_required": snapshot.remote_code_required},
            metadata={"acquisition_files": [file.relative_path],
                "review_scope": "explicit_shortlist_revalidation_not_new_broad_search"})
        # review_one resolves main independently; reject a race rather than
        # asserting that the card/file/preview necessarily share one revision.
        measured = study.review_one(dataset, study.REVIEW[dataset], provider, tok)
        preview_identity = identities.get(dataset, {})
        measured["preview_response_identity"] = preview_identity
        measured_ref = put(measured)
        tokenizer = measured.get("tokenizer", {})
        counts = measured.get("counters", {})
        schema_rows, english = counts.get("schema_rows", 0), counts.get("english_heuristic_pass", 0)
        task = study.REVIEW[dataset]["task"]
        target_evidence = (
            counts.get("json_object_or_array_target", 0) == english and english >= 10
            if task == "structured_json" else
            counts.get("explicit_length_constraint", 0) >= 10
            if task == "short_answer" else
            measured.get("prefix_next_window_study", {}).get("candidate_windows", 0) > 0)
        review = DatasetSourceReview(
            candidate_sha256(candidate), "automated-public-preview-review-not-independent-clearance",
            "public_preview",
            measured.get("immutable_revision") == snapshot.resolved_revision == preview_identity.get("server_revision"),
            "pass" if schema_rows else "unverified",
            "pass" if english >= 10 else "unverified",
            "pass" if target_evidence else "unverified", "declared_only", [measured_ref],
            TOKENIZER_ID, sha, tokenizer.get("tokens_after", 0), tokenizer.get("unknowns_after", 0),
            "conservative-English-heuristic+NFKC-whitespace;required-source-fields-v2",
            limitations=["public preview only", "source-level rights not independently cleared",
                "English heuristic is not language certification", "no full-corpus PII or protected-eval access"])
        review_refs[candidate.candidate_id] = put(review.to_dict())
        candidates.append(candidate)
        reviews.append({"dataset_id": dataset, "revision": candidate.revision,
            "review_ref": review_refs[candidate.candidate_id], "measured_ref": measured_ref,
            "revision_verified": review.revision_verified, "total_tokens": review.total_tokens,
            "unknown_tokens": review.unknown_tokens, "rights_status": review.rights_status,
            "file": file.to_dict(), "capability_status": review.capability_status})
        # A concrete proposal, not access authorization. Protected identities
        # must be supplied by their custodian before this can be accepted.
        scope = DatasetScreeningAccessScope(candidate_sha256(candidate), [file.relative_path],
            min(32 * 1024 * 1024, (file.size_bytes or 32 * 1024 * 1024)),
            ["pii", "exact_near_duplicate", "semantic_contamination", "frozen_evaluation_leakage", "tokenizer_recheck"],
            reference_manifest_refs={})
        scope_requests.append({"dataset_id": dataset, "revision": candidate.revision,
            "scope": scope.to_dict(), "eligible_for_authorization_now": False,
            "remaining_conditions": ["source_rights", "frozen_reference_manifest_identities",
                "independent_custodian_read_scope", "original_tokenizer_compatibility"]})
    policy = IndependentDatasetReviewPolicy(rt.artifact_store, TOKENIZER_ID, sha)
    rt.dataset_registry.providers = {"huggingface": ReviewedPublicShortlistProvider(tuple(candidates))}
    rt.dataset_registry.provider_allowlist = {"huggingface"}
    services = rt.with_discovery_services(StagedAutonomousServices(),
        require_planner_approval_receipt=True, dataset_review_policy=policy,
        candidate_review_refs=review_refs)
    services = rt.with_huggingface_data_factory(services, review_policy=policy)
    orchestrator = build_staged_autonomous_orchestrator(workflow_id="live-hf-shortlist-handoff",
        run_id="live-hf-shortlist-review", lineage_id=diagnosis.lineage_id, stage_name=diagnosis.stage_name,
        dependencies=StagedAutonomousDependencies(services=services,
            state_repository=rt.state_repository, artifact_store=rt.artifact_store))
    # Persist exact prior context, but do not fabricate steps 1-9 completion.
    orchestrator._persist_record(phase=SpinePhase.PLANNER, substep="discovery_request_preparation",
        operation_id="imported-frozen-planner-context-not-a-new-operation", kind="search_request", payload={
            "dataset_request": original.to_dict(), "model_request": None, "discovery_required": True,
            "origin": "imported_original_frozen_planner_not_executed_in_this_handoff"})
    orchestrator._persist_record(phase=SpinePhase.PLANNER, substep="approval_requirement_discovery",
        operation_id="imported-frozen-planner-approvals-not-granted", kind="approval_requirements", payload={
            "intervention_id": intervention.intervention_id,
            "required_approvals": list(intervention.metadata.get("required_approvals", []))})
    state = orchestrator._load_or_create()
    for step in ("dataset_discovery", "dataset_selection"):
        result = orchestrator._execute_step(state, SpinePhase.DATA_ACQUISITION_AUDIT, step, None)
        state.steps["data_acquisition_audit." + step] = result
        state.current_phase, state.current_substep, state.status = "data_acquisition_audit", step, result.status
        state.blocking_issues, state.resumable = list(result.blocking_issues), result.resumable
        orchestrator._persist_state(state)
        if not result.completion_marker:
            break
    records = rt.state_repository.all(WORKFLOW_RECORD_COLLECTION)
    for record in records:
        assert read_verified_json(rt.artifact_store, record["artifact_ref"]) == {
            k: v for k, v in record.items() if k != "artifact_ref"}
    result = {"schema": "live-hf-staged-handoff.v1", "source_sha": os.getenv("GITHUB_SHA"),
        "original_request_id": original.request_id, "original_diagnosis_id": diagnosis.report_id,
        "original_frozen_context_imported_not_rerun": True, "candidate_reviews": reviews,
        "scope_requests": scope_requests, "workflow_state": state.to_dict(),
        "verified_workflow_record_refs": [r["artifact_ref"] for r in records],
        "new_full_campaign_execution": False, "existing_cpu_campaign_progress": "11/45 unchanged",
        "corpus_access_authorized": False, "corpus_acquired": False,
        "training_executed": False, "scientific_certificate_issued": False}
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_reviews": reviews, "handoff_status": state.status,
        "cpu_campaign": "11/45 unchanged", "corpus_acquired": False}, sort_keys=True))


if __name__ == "__main__":
    main()
