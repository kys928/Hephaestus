#!/usr/bin/env python3
"""Independent real-Hub candidate challenge against the frozen scientific request.

Uses the original first-failure diagnosis and exactly the same frozen S3
tokenizer, read-only. Does NOT download a train set, issue approval or train.
Named contrast searches are independently declared, not represented as a
new autonomous Planner recommendation. No live text rows are stored.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from hephaestus.data.audit import audit_candidate
from hephaestus.providers.datasets import HuggingFaceDatasetProvider

from run_first_autonomous_dataset_discovery import (
    TOKENIZER_ID, TOKENIZER_KEY, load_frozen_chain, s3_client,
    get_s3_bytes, required, derived_request, enrich_candidate,
)

CONTRASTS = (
    "OpenAssistant/oasst1",
    "HuggingFaceH4/ultrachat_200k",
    "allenai/tulu-3-sft-mixture",
)
OUT = Path("independent_dataset_contrast_review.json")


def main() -> None:
    frozen, diagnosis, intervention = load_frozen_chain()
    raw = get_s3_bytes(
        s3_client(), required("RUNPOD_NETWORK_VOLUME_ID"), TOKENIZER_KEY)
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    assert frozen.tokenizer_ref == TOKENIZER_ID
    provider = HuggingFaceDatasetProvider(
        enable_network=True, max_results=25, timeout_seconds=25)
    attempts = []
    for number, desired in enumerate(CONTRASTS, 1):
        # These are independently specified comparison probes, not a
        # retroactive change to the frozen Planner's original discovery queries.
        child = derived_request(frozen, desired.split("/")[-1], 100 + number)
        attempt = {"dataset_id": desired, "search_query": child.problem_statement}
        try:
            listed = provider.search(child)
            exact = [candidate for candidate in listed
                     if candidate.dataset_id.casefold() == desired.casefold()]
            attempt["search_result_count"] = len(listed)
            if len(exact) != 1:
                attempt.update(status="not_discovered_in_live_search",
                               review="insufficient_provider_evidence")
            else:
                candidate, proof = enrich_candidate(provider, exact[0], tokenizer)
                attempt["provider_review"] = proof
                # The review is meaningful even when a conservative preview
                # rejects the provider record.
                attempt["resolved_revision"] = proof.get("resolved_revision")
                attempt["declared_license"] = proof.get("license")
                if candidate is None:
                    attempt.update(status="not_admitted_by_preview",
                                   review="requires_manual_investigation")
                else:
                    audit = audit_candidate(frozen, candidate)
                    preview = candidate.metadata.get("preview_audit", {})
                    attempt.update(
                        status="bounded_preview_audited",
                        revision=candidate.revision,
                        dataset_metadata=candidate.to_dict(),
                        independent_metadata_audit=audit.to_dict(),
                        preview_coverage=preview.get("capability_coverage", {}),
                        tokenizer_audit=preview.get("tokenizer", {}),
                        review="manual_corpus_license_contamination_and_heldout_audit_required",
                    )
            if desired == "allenai/tulu-3-sft-mixture":
                attempt["license_caveat"] = (
                    "Research collection has additional third-party and "
                    "noncommercial source subset restrictions; no blanket "
                    "commercial or training license clearance."
                )
            if desired == "HuggingFaceH4/ultrachat_200k":
                attempt["provenance_caveat"] = (
                    "Synthetic ChatGPT conversation source; third-party "
                    "generation terms and train/eval leakage need independent review."
                )
            if desired == "OpenAssistant/oasst1":
                attempt["structure_caveat"] = (
                    "Message-tree / per-turn rows must reconstruct assistant "
                    "targets, speaker roles, and filters; standalone text rows "
                    "are not verified prompt-response training records."
                )
        except Exception as exc:
            attempt.update(status="provider_review_error",
                           error_type=type(exc).__name__,
                           error_message=str(exc)[:400])
        attempts.append(attempt)
    result = {
        "schema": "hephaestus-independent-hub-candidate-contrast.v1",
        "original_diagnosis_report_id": diagnosis.report_id,
        "original_dataset_search_request_id": frozen.request_id,
        "original_intervention_id": intervention.intervention_id,
        "independent_contrast_search_not_original_planner_selection": True,
        "raw_tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_tokenizer_identity": TOKENIZER_ID,
        "training_launched": False,
        "dataset_acquired": False,
        "approval_issued": False,
        "benchmark_holdout_compared": False,
        "full_corpus_contamination_audited": False,
        "contrasts": attempts,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",
                   encoding="utf-8")
    print(json.dumps({
        "contrasts": [{
            "dataset_id": x["dataset_id"],
            "status": x["status"],
            "resolved_revision": x.get("resolved_revision"),
            "declared_license": x.get("declared_license"),
            "preview": x.get("preview_coverage"),
        } for x in attempts],
        "training_launched": False,
        "approval_issued": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
