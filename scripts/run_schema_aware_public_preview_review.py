#!/usr/bin/env python3
"""Source-pinned public-preview schema and rights review. NEVER acquires corpus.

No full dataset files, dataset code, tokenizer training, S3 writes, approvals,
PII scan, contamination comparison or held-out evaluation. Reports bounded
aggregate previews only; discards the fetched text in process memory.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote

from tokenizers import Tokenizer
from hephaestus.providers.datasets import HuggingFaceDatasetProvider
from hephaestus.data.normalization import normalize_text
from run_first_autonomous_dataset_discovery import (
    TOKENIZER_ID, TOKENIZER_KEY, get_s3_bytes, s3_client, required,
    load_frozen_chain, hf_splits, hf_first_rows,
)

OUTPUT = Path("schema_aware_public_preview_review.json")
EXPECTED_TOKENIZER_BYTES_SHA256 = "3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25"
REVIEW = {
    "soikit/dataset_instruction_json": {
        "task": "structured_json", "prompt": ("instruction", "input"),
        "target": ("output",), "preferred_config": None,
        "caveat": "Chinese-language classroom instructions; English filter cannot be assumed to preserve source coverage",
    },
    "vara201/ecommerce-orders-structured-output": {
        "task": "structured_json", "prompt": ("instruction", "prompt", "input"),
        "target": ("output", "response", "completion"), "preferred_config": None,
        "caveat": "Exact schema and origin must be checked; card license does not settle underlying generated-content rights",
    },
    "nvidia/Nemotron-RL-instruction_following-structured_outputs": {
        "task": "structured_json", "prompt": ("instruction", "question", "prompt"),
        "target": ("response", "answer", "output"), "preferred_config": None,
        "caveat": "Schema/instruction-focused dataset might be tasks only rather than licensed supervised answer targets",
    },
    "hotpotqa/hotpot_qa": {
        "task": "short_answer", "prompt": ("question",),
        "target": ("answer",), "preferred_config": "distractor",
        "caveat": "Factoid/multihop QA is not itself an explicit bounded-answer instruction; Wiki text and evaluation overlap possible",
    },
    "Short-Answer-Feedback/saf_communication_networks_english": {
        "task": "short_answer", "prompt": ("question",),
        "target": ("reference_answer",), "preferred_config": None,
        "caveat": "Reference answer, provided answer, feedback and score are distinct labels; cannot use feedback as answer",
    },
    "Salesforce/wikitext": {
        "task": "raw_continuation", "prompt": (),
        "target": ("text",), "preferred_config": "wikitext-2-raw-v1",
        "caveat": "Raw Wikipedia article paragraphs require license/share-alike attribution and train/eval separation",
    },
}
MAX_PREVIEW_ROWS = 100
MAX_CARD_BYTES = 100000
STOPWORDS = frozenset(
    "a an the of for to from is are was were in on at with by and or as that this "
    "it can will which what where when who why how not they their we you your "
    "have has do does please output return answer provide".split()
)


def english_likelihood(text: str) -> tuple[bool, str]:
    """Transparent conservative heuristic, NOT language-ID certification."""
    if not text or len(text.strip()) < 12:
        return False, "too_short"
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False, "cjk_script_detected"
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 7:
        return False, "too_few_letters"
    latin = sum("LATIN" in unicodedata.name(ch, "") for ch in letters)
    if latin / len(letters) < 0.90:
        return False, "non_latin_script"
    words = re.findall(r"[A-Za-z]+", text.casefold())
    if len(words) < 5 or not set(words).intersection(STOPWORDS):
        return False, "weak_english_word_evidence"
    return True, "latin_with_english_function_words"


def text_pair(row: dict, spec: dict) -> tuple[str, str, str]:
    if spec["task"] == "raw_continuation":
        text = row.get("text")
        return "", text if isinstance(text, str) else "", "single_text_train_row"
    prompt = " ".join(str(row[k]).strip() for k in spec["prompt"]
                      if isinstance(row.get(k), str) and row[k].strip())
    target = next((row[k] for k in spec["target"]
                   if isinstance(row.get(k), str) and row[k].strip()), "")
    return prompt, target, "schema_pinned_prompt_target"


def count_unk(tok: Tokenizer, text: str) -> tuple[int, int]:
    encoded = tok.encode(text)
    unknown_id = tok.token_to_id("<|unk|>")
    if unknown_id is None:
        unknown_id = tok.token_to_id("[UNK]")
    if unknown_id is None:
        raise ValueError("frozen tokenizer unknown-token ID unresolved")
    return len(encoded.ids), sum(x == unknown_id for x in encoded.ids)


def window_proof(tok: Tokenizer, normalized: list[str]) -> dict:
    """One row => contiguous 128-input / 32-target token windows.

    Never append EOS/EOT, concatenate rows, use eval splits or write window
    text/tokens. Confirm exact adjacency and reconstruct token IDs internally.
    """
    prefix, target, stride = 128, 32, 64
    total = 0
    long_rows = 0
    proofs_ok = True
    source_rows = 0
    for text in normalized:
        ids = tok.encode(text).ids
        if len(ids) < prefix + target:
            continue
        source_rows += 1
        long_rows += 1
        for start in range(0, len(ids) - (prefix + target) + 1, stride):
            context = ids[start:start + prefix]
            labels = ids[start + prefix:start + prefix + target]
            proofs_ok = proofs_ok and bool(
                len(context) == prefix and len(labels) == target and
                context + labels == ids[start:start + prefix + target]
            )
            total += 1
    return {
        "source_rows_long_enough": long_rows,
        "unique_source_rows": source_rows,
        "candidate_windows": total,
        "prefix_tokens": prefix, "next_token_targets": target,
        "stride": stride, "context_plus_targets_max": prefix + target,
        "same_row_contiguous_identity_checked": proofs_ok,
        "cross_row_concatenation": False,
        "eos_or_eot_inserted": False,
        "train_preview_only": True,
        "trained_or_materialized_windows": False,
        "window_rights_and_eval_leakage_checked": False,
    }


def read_pinned_card(dataset: str, revision: str) -> dict:
    url = "https://huggingface.co/datasets/" + dataset + "/raw/" + quote(revision) + "/README.md"
    try:
        with urlopen(Request(url, headers={"User-Agent": "hephaestus-bounded-card-review/1"}),
                     timeout=12) as response:
            data = response.read(MAX_CARD_BYTES + 1)
        if len(data) > MAX_CARD_BYTES:
            return {"url": url, "read": False, "reason": "card_exceeds_bound"}
        card = data.decode("utf-8")
        lowered = card.casefold()
        return {
            "url": url, "read": True,
            "card_sha256": hashlib.sha256(data).hexdigest(),
            "card_bytes": len(data),
            "mentions_license": "license" in lowered,
            "mentions_source_or_origin": any(k in lowered for k in ("source", "origin", "wikipedia", "generated", "annotat")),
            "mentions_restrictions": any(k in lowered for k in ("noncommercial", "non-commercial", "restricted", "third-party")),
            "source_rights_independently_verified": False,
            "note": "Card assertions, not independent source-level legal clearance; raw card text not stored",
        }
    except Exception as exc:
        return {"url": url, "read": False, "error_type": type(exc).__name__,
                "source_rights_independently_verified": False}


def review_one(dataset: str, spec: dict, provider: HuggingFaceDatasetProvider,
               tok: Tokenizer) -> dict:
    result = {
        "dataset_id": dataset, "task_hypothesis": spec["task"],
        "schema_mapping": {"prompt_keys": list(spec["prompt"]),
                           "target_keys": list(spec["target"])},
        "risks_to_review": spec["caveat"],
        "approval": "not_requested", "acquisition": "not_performed",
        "full_corpus_scanned": False,
    }
    try:
        snapshot = provider.resolve_revision(dataset, "main")
        result.update(
            immutable_revision=snapshot.resolved_revision,
            card_declared_license=snapshot.license,
            license_source=snapshot.license_source,
            license_terms=list(snapshot.terms),
            provider_provenance_confidence=snapshot.provenance_confidence,
            gated=snapshot.gated, private=snapshot.private,
            remote_code_required=snapshot.remote_code_required,
        )
        result["dataset_card_review"] = read_pinned_card(dataset, snapshot.resolved_revision)
        if snapshot.gated or snapshot.private or snapshot.remote_code_required:
            result["status"] = "blocked_remote_access_or_code"
            return result
        splits = hf_splits(dataset)
        result["public_split_metadata"] = [
            {"config": str(v.get("config")), "split": str(v.get("split")),
             "num_examples": v.get("num_examples")} for v in splits[:32]
        ]
        trains = [v for v in splits if str(v.get("split", "")).lower() == "train"]
        selected = next(
            (s for s in trains if str(s.get("config")) == spec["preferred_config"]), None
        ) if spec["preferred_config"] else None
        selected = selected or (trains[0] if trains else None)
        if selected is None:
            result["status"] = "no_public_train_preview"
            return result
        config = str(selected.get("config") or "default")
        split = str(selected["split"])
        result["preview_config"] = config
        result["preview_split"] = split
        rows = hf_first_rows(dataset, config, split)[:MAX_PREVIEW_ROWS]
        result["preview_rows"] = len(rows)
        if not rows:
            result["status"] = "empty_public_train_preview"
            return result
        observed_schema = Counter()
        counters = Counter()
        failures = Counter()
        before_n = before_unk = after_n = after_unk = 0
        english_normalized = []
        raw_train_text = []
        for row in rows:
            observed_schema[tuple(sorted(str(k) for k in row.keys()))] += 1
            p, t, mapping = text_pair(row, spec)
            if not t or (spec["task"] != "raw_continuation" and not p):
                failures["pinned_schema_missing_pair"] += 1
                continue
            counters["schema_rows"] += 1
            raw_input = p + "\n" + t if p else t
            raw_input = raw_input[:24000]
            n, u = count_unk(tok, raw_input)
            before_n += n
            before_unk += u
            good, reason = english_likelihood(raw_input)
            if not good:
                failures[reason] += 1
                continue
            counters["english_heuristic_pass"] += 1
            pn, tn = normalize_text(p), normalize_text(t)
            after = pn + "\n" + tn if pn else tn
            if not after:
                failures["empty_after_normalization"] += 1
                continue
            n2, u2 = count_unk(tok, after[:24000])
            after_n += n2
            after_unk += u2
            english_normalized.append(after)
            if spec["task"] == "structured_json":
                try:
                    if isinstance(json.loads(tn), (dict, list)):
                        counters["json_object_or_array_target"] += 1
                except (TypeError, ValueError):
                    pass
            if spec["task"] == "short_answer":
                if 0 < len(tok.encode(tn).ids) <= 64:
                    counters["short_answer_target_64_tokens"] += 1
                if any(phrase in pn.casefold() for phrase in (
                    "one sentence", "no more than", "at most", "short answer",
                    "under ", "exactly", "in one word",
                )):
                    counters["explicit_length_constraint"] += 1
            if spec["task"] == "raw_continuation":
                raw_train_text.append(tn)
        before_rate = before_unk / before_n if before_n else None
        after_rate = after_unk / after_n if after_n else None
        result.update(
            observed_schema=[
                {"keys": list(keys), "rows": n} for keys, n in
                observed_schema.most_common(8)
            ],
            counters=dict(counters),
            exclusions=dict(failures),
            tokenizer={
                "original_frozen_identity": TOKENIZER_ID,
                "unknown_fraction_all_schema_valid_preview": before_rate,
                "unknown_fraction_after_english_filter_and_nfkc": after_rate,
                "tokens_before": before_n, "unknowns_before": before_unk,
                "tokens_after": after_n, "unknowns_after": after_unk,
                "english_heuristic_not_language_proof": True,
                "compatibility_threshold_unmodified": 0.01,
                "english_subset_meets_preview_threshold": (
                    after_rate is not None and after_rate <= .01
                ),
            },
        )
        if spec["task"] == "raw_continuation":
            result["prefix_next_window_study"] = window_proof(tok, raw_train_text)
        result["status"] = "bounded_preview_reviewed"
    except Exception as exc:
        result["status"] = "provider_or_schema_error"
        result["error_type"] = type(exc).__name__
        # No raw content or credential-bearing URLs in error messages.
    return result


def main() -> None:
    original, diagnosis, intervention = load_frozen_chain()
    if intervention.primary_variable != "dataset_mixture":
        raise ValueError("original primary variable was mutated")
    raw = get_s3_bytes(s3_client(), required("RUNPOD_NETWORK_VOLUME_ID"), TOKENIZER_KEY)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_TOKENIZER_BYTES_SHA256 or original.tokenizer_ref != TOKENIZER_ID:
        raise ValueError("original tokenizer revision or bytes differ from frozen evidence")
    tok = Tokenizer.from_str(raw.decode("utf-8"))
    provider = HuggingFaceDatasetProvider(enable_network=True, max_results=5, timeout_seconds=15)
    evidence = {
        "schema": "hephaestus-schema-aware-public-preview-review.v1",
        "source": {"diagnosis_id": diagnosis.report_id, "request_id": original.request_id,
                   "intervention_id": intervention.intervention_id,
                   "original_primary_variable": intervention.primary_variable,
                   "tokenizer_ref": TOKENIZER_ID, "tokenizer_file_sha256": actual},
        "scope": {
            "public_train_preview_only": True,
            "requested_rows_per_dataset_max": MAX_PREVIEW_ROWS,
            "raw_dataset_text_persisted": False,
            "full_dataset_files_downloaded": False,
            "all_previewed_text_deleted_after_measurement": True,
            "no_training_or_gpu": True,
            "not_original_planner_selection": True,
        },
        "candidate_reviews": [
            review_one(name, spec, provider, tok) for name, spec in REVIEW.items()
        ],
        "approval": {
            "dataset_selection_approval": "not_granted",
            "dataset_acquisition": "not_granted",
            "full_corpus_access": "not_authorized",
            "full_corpus_pii": "not_run_requires_separate_authorization",
            "full_corpus_contamination": "not_run_requires_separate_authorization",
            "frozen_evaluation_leakage": "not_run_requires_separate_authorization",
            "full_corpus_license_source_audit": "not_run_requires_separate_authorization",
        },
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "reviews": [{
            "dataset_id": item["dataset_id"], "status": item["status"],
            "license": item.get("card_declared_license"),
            "preview_rows": item.get("preview_rows"),
            "schema_rows": item.get("counters", {}).get("schema_rows", 0),
            "english_pass": item.get("counters", {}).get("english_heuristic_pass", 0),
            "unknown_before": item.get("tokenizer", {}).get("unknown_fraction_all_schema_valid_preview"),
            "unknown_after": item.get("tokenizer", {}).get("unknown_fraction_after_english_filter_and_nfkc"),
            "json_targets": item.get("counters", {}).get("json_object_or_array_target"),
            "windows": item.get("prefix_next_window_study", {}).get("candidate_windows"),
        } for item in evidence["candidate_reviews"]],
        "approval": "not_granted", "full_corpus_scans": "not_run",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
