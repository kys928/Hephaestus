"""Conservative WikiText train-row selection evidence; never source approval.

Receives actual pinned source rows and original tokenizer. Emits aggregate
counts and a position/content commitment, never selected text or weights.
No protected-reference access and no trainable output materialization.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Callable

from hephaestus.data.quarantine_screening import PII, normalized
from hephaestus.data.normalization import normalize_text


def _public_overlap(text: str, prompts: tuple[str, ...]) -> bool:
    content = normalized(text).casefold()
    words = content.split()
    grams = {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}
    for prompt in prompts:
        reference = normalized(prompt).casefold()
        if reference and reference in content:
            return True
        ref_words = reference.split()
        if grams and any(
            " ".join(ref_words[i:i + 5]) in grams
            for i in range(max(0, len(ref_words) - 4))
        ):
            return True
    return False


def screen_continuation_rows(
    rows: list[dict], tokenizer, unknown_id: int,
    language_check: Callable[[str], bool], *,
    public_prompts: tuple[str, ...] = (),
    prefix_tokens: int = 128,
    target_tokens: int = 32,
    stride_tokens: int = 64,
) -> dict[str, object]:
    if (prefix_tokens <= 0 or target_tokens <= 0 or stride_tokens <= 0
            or type(unknown_id) is not int or unknown_id < 0):
        raise ValueError("invalid frozen-tokenizer continuation window policy")
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    total_tokens = unknown_tokens = windows = 0
    digest = sha256()
    selected = 0
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            rejected["schema_invalid"] += 1
            continue
        # Preserve the existing newline-sensitive field-normalization profile.
        content = normalize_text(row["text"])
        if not content.strip():
            rejected["empty"] += 1
            continue
        if any(pattern.search(content) for pattern in PII.values()):
            rejected["pii_regex_candidate_excluded"] += 1
            continue
        key = normalized(content).casefold()
        if key in seen:
            rejected["normalized_duplicate_excluded"] += 1
            continue
        seen.add(key)  # duplicates cannot gain eligibility by occurring later
        if not language_check(content):
            rejected["language_heuristic_rejected"] += 1
            continue
        if _public_overlap(content, public_prompts):
            rejected["public_probe_overlap_excluded"] += 1
            continue
        ids = tokenizer.encode(content).ids
        if len(ids) < prefix_tokens + target_tokens:
            rejected["insufficient_contiguous_tokens"] += 1
            continue
        unknown = ids.count(unknown_id)
        if unknown * 100 > len(ids):
            rejected["frozen_unknown_fraction_exceeds_1pct"] += 1
            continue
        selected += 1
        total_tokens += len(ids)
        unknown_tokens += unknown
        windows += 1 + (len(ids) - prefix_tokens - target_tokens) // stride_tokens
        # Commitment fixes source row index and normalized *actual* bytes,
        # but does not emit source text, token sequences or a training file.
        digest.update(f"{position}\t{sha256(content.encode('utf-8')).hexdigest()}\n".encode())
    if sum(rejected.values()) + selected != len(rows):
        raise AssertionError("source row inventory not exhaustive")
    if unknown_tokens * 100 > total_tokens:
        raise AssertionError("frozen tokenizer ceiling exceeded in screened subset")
    return {
        "schema": "wikitext-continuation-subset-screen.v1",
        "source_rows_examined": len(rows),
        "accepted_unique_rows": selected,
        "rejected_counts": dict(sorted(rejected.items())),
        "eligible_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "frozen_unknown_fraction": (unknown_tokens / total_tokens if total_tokens else None),
        "maximum_frozen_unknown_fraction": 0.01,
        "within_row_prefix_tokens": prefix_tokens,
        "within_row_target_tokens": target_tokens,
        "within_row_stride_tokens": stride_tokens,
        "eligible_within_row_windows": windows,
        "selection_commitment_sha256": "sha256:" + digest.hexdigest(),
        "no_cross_row_stitching": True,
        "newlines_preserved": True,
        "no_eos_inserted": True,
        "public_prompt_overlap_checked_not_sealed_holdout": True,
        "pii_regex_filtered_not_pii_certified": True,
        "dataset_approved": False,
        "training_allowed": False,
        "acquisition_allowed": False,
        "protected_reference_audits_complete": False,
    }
