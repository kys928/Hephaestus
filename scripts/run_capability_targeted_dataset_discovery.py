#!/usr/bin/env python3
"""Capability-stratified read-only real Hugging Face discovery.

Grounded in the *unchanged SHA-pinned* diagnosis, Planner request, and training
tokenizer of the original failed baseline. This is an independently motivated
follow-up search, not a rewrite of the original Planner choice.

Do not download corpora, train, alter model/eval parameters, approve sources,
execute dataset code, or persist sampled raw text. Preview measurements remain
bounded proxies requiring independent full-corpus evaluation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tokenizers import Tokenizer
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.data.audit import audit_candidate
from hephaestus.providers.datasets import HuggingFaceDatasetProvider
from run_first_autonomous_dataset_discovery import (
    TOKENIZER_ID, TOKENIZER_KEY, load_frozen_chain, s3_client,
    get_s3_bytes, required, derived_request, enrich_candidate,
    hf_splits, hf_first_rows, row_prompt_response,
)

# All tracks MUST be searched independently. No shared discovery cap may
# exhaust the budget before structured and continuation searches begin.
TRACK_QUERIES = {
    "structured_json": ("json response", "json instruction", "structured output", "json"),
    "constrained_short_answer": ("short answer", "concise answer", "instruction constraint", "qa"),
    "termination": ("assistant response", "chat conversation", "instruction following", "dialogue"),
    "causal_lm_continuation": ("text completion", "story completion", "language modeling", "wikitext"),
}
MAX_METADATA_PER_QUERY = 8
MAX_TRACK_CANDIDATES = 14
MAX_PREVIEWS_PER_TRACK = 8
ROW_LIMIT = 100
OUTPUT = Path("capability_targeted_discovery_20261025.json")

CONSTRAINT_PHRASES = (
    "one sentence", "two sentences", "exactly", "at most", "no more than",
    "no longer than", "under ", "in one word", "short answer",
    "briefly", "reply with", "respond with", "maximum ",
)
SUFFIX_KEYS = ("completion", "continuation", "suffix", "next_text")
PREFIX_KEYS = ("prompt", "prefix", "context")


def strict_preview_metrics(rows: list[dict], tokenizer: Tokenizer) -> dict:
    """Count *structural proxies* rather than claiming model capabilities.

    A single "text" field is deliberately NOT evidence of a paired causal
    continuation. Terminal punctuation is NOT proof of an EOS/EOT label.
    Uncertainty is explicit and each denominator is the full preview rows.
    """
    n = len(rows)
    counts = {
        "instruction_response_pairs": 0, "parseable_json_targets": 0,
        "explicit_length_or_format_constraints": 0,
        "constrained_short_targets_64_tokens_or_less": 0,
        "terminal_punctuation_proxy": 0,
        "explicit_prompt_suffix_pairs_proxy": 0,
        "single_text_rows_not_continuation_proof": 0,
    }
    for row in rows:
        prompts, responses, _ = row_prompt_response(row)
        prompts = [value for value in prompts if value.strip()]
        responses = [value for value in responses if value.strip()]
        has_pair = bool(prompts and responses)
        if has_pair:
            counts["instruction_response_pairs"] += 1
        structured = False
        for response in responses:
            try:
                structured = structured or isinstance(json.loads(response.strip()), (list, dict))
            except (ValueError, TypeError):
                pass
        if structured and has_pair:
            counts["parseable_json_targets"] += 1
        constraint = any(
            phrase in prompt.casefold()
            for prompt in prompts for phrase in CONSTRAINT_PHRASES
        )
        if constraint:
            counts["explicit_length_or_format_constraints"] += 1
            if has_pair and any(0 < len(tokenizer.encode(r).ids) <= 64 for r in responses):
                counts["constrained_short_targets_64_tokens_or_less"] += 1
        if responses and any(
            len(r.strip()) >= 4 and r.rstrip().endswith((".", "!", "?", "}", "]"))
            for r in responses
        ):
            counts["terminal_punctuation_proxy"] += 1
        prefix = any(isinstance(row.get(key), str) and row[key].strip() for key in PREFIX_KEYS)
        suffix = any(isinstance(row.get(key), str) and row[key].strip() for key in SUFFIX_KEYS)
        if prefix and suffix:
            counts["explicit_prompt_suffix_pairs_proxy"] += 1
        if isinstance(row.get("text"), str) and row["text"].strip() and not (prefix and suffix):
            counts["single_text_rows_not_continuation_proof"] += 1
    rates = {key: round(value / max(n, 1), 6) for key, value in counts.items()}
    return {
        "rows_checked": n,
        "counts": counts,
        "rates": rates,
        "evidence_scope": "up_to_100_public_preview_rows_not_full_corpus",
        "caveats": {
            "structured_json": "valid JSON object or list in an observed assistant response, not a reliable schema-conformance test",
            "constrained_short_answer": "explicit prompt constraint plus <=64 frozen-token response; does not verify exact instruction compliance",
            "termination": "final punctuation proxy only, NOT EOS/EOT or causal-model stop-behavior evidence",
            "causal_lm_continuation": "paired prompt/prefix plus suffix/completion proxy only; NOT tested next-token modeling or holdout performance",
        },
    }


TRACK_SIGNAL = {
    "structured_json": "parseable_json_targets",
    "constrained_short_answer": "constrained_short_targets_64_tokens_or_less",
    "termination": "terminal_punctuation_proxy",
    "causal_lm_continuation": "explicit_prompt_suffix_pairs_proxy",
}


def rank_status(track: str, audit: dict, *, declared_en: bool,
                known_license: bool, immutably_pinned: bool,
                tokenizer_ok: bool) -> str:
    if not immutably_pinned or not known_license:
        return "metadata_provenance_review_required"
    if not tokenizer_ok:
        return "frozen_tokenizer_incompatible_in_preview"
    if not declared_en:
        return "english_language_not_verified"
    if audit["rows_checked"] < 20:
        return "insufficient_preview_rows"
    if audit["counts"][TRACK_SIGNAL[track]] == 0:
        return "target_capability_not_observed_in_preview"
    # A single hit in a 100-row sample is not credible target coverage.
    # Count and sample-frequency gates must BOTH be satisfied.
    minimum_examples = max(3, (audit["rows_checked"] + 19) // 20)
    if audit["counts"][TRACK_SIGNAL[track]] < minimum_examples:
        return "target_coverage_too_sparse_in_preview"
    if track == "termination":
        return "terminal_punctuation_proxy_not_verified_eos_eot"
    return "provisional_target_review_only_not_approved"


def collect_tracks(base, tokenizer: Tokenizer, provider: HuggingFaceDatasetProvider) -> dict:
    registry = DatasetProviderRegistry(provider_allowlist={"huggingface"})
    registry.register(provider)
    frozen_targets = set(base.capability_targets)
    required_targets = {
        "structured JSON response", "bounded response length", "termination",
        "causal-LM continuation",
    }
    if not required_targets.issubset(frozen_targets):
        raise ValueError("target study is not linked to the unchanged original diagnostic")
    cache = {}
    tracks = {}
    for index, (track, queries) in enumerate(TRACK_QUERIES.items(), 1):
        attempts, candidates = [], {}
        query_groups = []
        for position, query in enumerate(queries, 1):
            child = derived_request(base, query, index * 100 + position)
            discovery = registry.discover(child)
            attempts.append({
                "query": query, "request_id": child.request_id,
                "candidate_count": len(discovery.candidates),
                "blocking_issues": [x.to_dict() for x in discovery.issues],
            })
            per_query = sorted(discovery.candidates, key=lambda c: (
                0 if "en" in c.languages else 1,
                0 if c.license and c.license.casefold() not in {"unknown", "other", "cc"} else 1,
                0 if c.revision else 1,
                -int(c.metadata.get("downloads") or 0), c.dataset_id.casefold(),
            ))
            query_groups.append(per_query)
            for candidate in per_query:
                candidates.setdefault(candidate.dataset_id, candidate)
        # Round-robin across the four QUERIES within this capability.
        # A single popular generic search must not starve the structured
        # output / targeted answer / continuation search results.
        ranked = []
        seen = set()
        depth = max((len(group) for group in query_groups), default=0)
        for position in range(depth):
            for group in query_groups:
                if position < len(group):
                    candidate = group[position]
                    if candidate.dataset_id not in seen:
                        ranked.append(candidate)
                        seen.add(candidate.dataset_id)
                if len(ranked) >= MAX_TRACK_CANDIDATES:
                    break
            if len(ranked) >= MAX_TRACK_CANDIDATES:
                break
        reviews = []
        for candidate in ranked[:MAX_PREVIEWS_PER_TRACK]:
            if candidate.dataset_id not in cache:
                enriched, provider_audit = enrich_candidate(provider, candidate, tokenizer)
                record = {
                    "provider_audit": provider_audit,
                    "enriched": enriched.to_dict() if enriched is not None else None,
                }
                if enriched is not None:
                    try:
                        splits = hf_splits(candidate.dataset_id)
                        train = next((s for s in splits if str(s.get("split")) == "train"), None)
                        chosen = train or (splits[0] if splits else None)
                        if chosen is None:
                            raise ValueError("no split for structural preview")
                        rows = hf_first_rows(
                            candidate.dataset_id,
                            str(chosen.get("config") or "default"),
                            str(chosen.get("split") or "train"),
                        )[:ROW_LIMIT]
                        record["structure"] = strict_preview_metrics(rows, tokenizer)
                        record["metadata_policy_audit"] = audit_candidate(base, enriched).to_dict()
                    except Exception as exc:
                        record["structure_error"] = type(exc).__name__
                cache[candidate.dataset_id] = record
            observed = cache[candidate.dataset_id]
            audit = observed.get("structure")
            enriched = observed.get("enriched")
            item = {
                "dataset_id": candidate.dataset_id,
                "candidate_id": candidate.candidate_id,
                "hub_revision": observed["provider_audit"].get("resolved_revision") or candidate.revision,
                "hub_declared_license": observed["provider_audit"].get("license") or candidate.license,
                "hub_card_ref": observed["provider_audit"].get("dataset_card_ref"),
                "languages_from_provider_tags": candidate.languages,
                "preview_provider_status": observed["provider_audit"].get("status"),
                "preview_rejection_reason": observed["provider_audit"].get("reason"),
                "preview_summary": observed["provider_audit"].get("preview"),
                "strict_structural_preview": audit,
                "structural_preview_error": observed.get("structure_error"),
                "candidate_metadata": enriched if enriched is not None else candidate.to_dict(),
                "original_domain_audit": observed.get("metadata_policy_audit"),
            }
            if audit:
                unknown_fraction = (observed["provider_audit"].get("preview") or {}).get("tokenizer", {}).get("unknown_token_fraction")
                ok = isinstance(unknown_fraction, (int, float)) and unknown_fraction <= .01
                license_value = item["hub_declared_license"]
                sha = item["hub_revision"] or ""
                status = rank_status(
                    track, audit,
                    declared_en="en" in candidate.languages,
                    known_license=bool(
                        license_value and str(license_value).casefold().strip()
                        not in {"unknown", "other", "unspecified", "none", "cc", "n/a"}
                    ),
                    immutably_pinned=bool(re.fullmatch(r"[a-fA-F0-9]{40}", sha)),
                    tokenizer_ok=ok,
                )
                item["target_preview_count"] = audit["counts"][TRACK_SIGNAL[track]]
                item["target_preview_rate"] = audit["rates"][TRACK_SIGNAL[track]]
                item["tokenizer_within_frozen_1pct"] = ok
                item["review_status"] = status
            else:
                item["target_preview_count"] = 0
                item["target_preview_rate"] = 0.0
                item["tokenizer_within_frozen_1pct"] = False
                item["review_status"] = "no_strict_structural_preview"
            reviews.append(item)
        reviews.sort(key=lambda v: (
            0 if v["review_status"] == "provisional_target_review_only_not_approved" else 1,
            -v["target_preview_rate"], -v["target_preview_count"],
            v["dataset_id"].casefold(),
        ))
        tracks[track] = {
            "queries": attempts,
            "unique_metadata_candidates": len(candidates),
            "track_metadata_shortlist_count": len(ranked),
            "reviewed_preview_count": len(reviews),
            "signal": TRACK_SIGNAL[track],
            "observed_target_in_previews": any(v["target_preview_count"] for v in reviews),
            "provisionally_reviewable_count": sum(
                v["review_status"] == "provisional_target_review_only_not_approved"
                for v in reviews
            ),
            "candidate_reviews": reviews,
        }
    return tracks


def main() -> None:
    original, diagnosis, intervention = load_frozen_chain()
    raw = get_s3_bytes(
        s3_client(), required("RUNPOD_NETWORK_VOLUME_ID"), TOKENIZER_KEY)
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    if original.tokenizer_ref != TOKENIZER_ID:
        raise ValueError("frozen request tokenizer changed")
    tracks = collect_tracks(
        original, tokenizer,
        HuggingFaceDatasetProvider(enable_network=True, max_results=MAX_METADATA_PER_QUERY, timeout_seconds=18),
    )
    payload = {
        "schema": "hephaestus-targeted-read-only-discovery.v1",
        "source_kind": "independent_followup_research_not_original_planner_selection",
        "frozen_diagnosis_id": diagnosis.report_id,
        "frozen_dataset_request_id": original.request_id,
        "frozen_intervention_id": intervention.intervention_id,
        "unchanged_primary_variable": intervention.primary_variable,
        "frozen_tokenizer_identity": TOKENIZER_ID,
        "frozen_tokenizer_actual_file_sha256": hashlib.sha256(raw).hexdigest(),
        "search_policy": {
            "metadata_results_per_query": MAX_METADATA_PER_QUERY,
            "distinct_metadata_limit_per_track": MAX_TRACK_CANDIDATES,
            "preview_limit_per_track": MAX_PREVIEWS_PER_TRACK,
            "preview_rows_per_candidate_limit": ROW_LIMIT,
            "per_track_budget_not_global_cap": True,
        },
        "tracks": tracks,
        "acquisition_executed": False,
        "approval_issued": False,
        "training_launched": False,
        "gpu_allocated": False,
        "full_corpus_audit_performed": False,
        "heldout_contamination_checked": False,
        "independent_license_clearance": False,
        "automatic_approval_allowed": False,
        "method_caveat": "Capability-specific rates are bounded public preview *proxies*, not measured language-model improvements.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen_request": original.request_id,
        "tracks": {
            key: {
                "metadata_candidates": value["unique_metadata_candidates"],
                "reviewed": value["reviewed_preview_count"],
                "observed_target": value["observed_target_in_previews"],
                "provisionally_reviewable": value["provisionally_reviewable_count"],
                "top_candidates": [{
                    "dataset_id": row["dataset_id"],
                    "target_rate": row["target_preview_rate"],
                    "review_status": row["review_status"],
                } for row in value["candidate_reviews"][:3]],
            } for key, value in tracks.items()
        },
        "approved": False, "training_launched": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
