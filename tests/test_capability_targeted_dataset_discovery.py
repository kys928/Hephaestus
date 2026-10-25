"""Offline contract checks for read-only capability-stratified Hub review.

The real Hub access and original S3 tokenizer are separately exercised in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
pytest.importorskip("boto3")
pytest.importorskip("tokenizers")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_capability_targeted_dataset_discovery import (
    TRACK_QUERIES, TRACK_SIGNAL, MAX_PREVIEWS_PER_TRACK,
    rank_status, strict_preview_metrics,
)


def _tokenizer():
    tok = Tokenizer(WordLevel(vocab={"[UNK]": 0, "Answer": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return tok


def test_independent_queries_cover_all_four_frozen_deficits():
    assert set(TRACK_QUERIES) == {
        "structured_json", "constrained_short_answer",
        "termination", "causal_lm_continuation",
    }
    assert all(len(queries) >= 3 for queries in TRACK_QUERIES.values())
    assert len(set(sum((list(v) for v in TRACK_QUERIES.values()), []))) >= 10
    assert MAX_PREVIEWS_PER_TRACK >= 4
    assert len(set(TRACK_SIGNAL.values())) == 4


def test_plain_text_does_not_falsely_pass_causal_continuation_or_json():
    audit = strict_preview_metrics([
        {"text": "An ordinary snippet ends in a period."},
        {"text": "An isolated assistant conversation message."},
    ], _tokenizer())
    assert audit["counts"]["single_text_rows_not_continuation_proof"] == 2
    assert audit["counts"]["explicit_prompt_suffix_pairs_proxy"] == 0
    assert audit["counts"]["parseable_json_targets"] == 0
    assert audit["counts"]["terminal_punctuation_proxy"] == 0


def test_explicit_response_structure_not_prompt_json_is_counted():
    audit = strict_preview_metrics([
        {"instruction": 'The user mentions {"json":true}, but asks for prose.',
         "response": "No JSON answer."},
        {"instruction": "Return JSON with exactly one item",
         "response": '{"answer": "done"}'},
        {"instruction": "Reply in one sentence", "response": "Done."},
        {"prompt": "Once upon a time", "completion": "the next event happened."},
    ], _tokenizer())
    counts = audit["counts"]
    assert counts["parseable_json_targets"] == 1
    assert counts["constrained_short_targets_64_tokens_or_less"] == 2
    assert counts["explicit_prompt_suffix_pairs_proxy"] == 1
    assert counts["terminal_punctuation_proxy"] >= 2


def test_policy_gate_cannot_promote_zero_target_or_missing_provenance():
    audit = strict_preview_metrics(
        [{"instruction": "Say hello", "response": "Hello."}] * 100, _tokenizer())
    assert rank_status(
        "structured_json", audit, declared_en=True, known_license=True,
        immutably_pinned=True, tokenizer_ok=True,
    ) == "target_capability_not_observed_in_preview"
    assert rank_status(
        "termination", audit, declared_en=True, known_license=True,
        immutably_pinned=True, tokenizer_ok=False,
    ) == "frozen_tokenizer_incompatible_in_preview"
    assert rank_status(
        "termination", audit, declared_en=True, known_license=False,
        immutably_pinned=True, tokenizer_ok=True,
    ) == "metadata_provenance_review_required"
    assert rank_status(
        "termination", audit, declared_en=True, known_license=True,
        immutably_pinned=True, tokenizer_ok=True,
    ) == "terminal_punctuation_proxy_not_verified_eos_eot"
    sparse = strict_preview_metrics(
        [{"instruction": "Return JSON", "response": '{"value": 1}'}]
        + [{"instruction": "Describe the topic", "response": "Some prose."}] * 99,
        _tokenizer(),
    )
    assert rank_status(
        "structured_json", sparse, declared_en=True, known_license=True,
        immutably_pinned=True, tokenizer_ok=True,
    ) == "target_coverage_too_sparse_in_preview"
    assert rank_status(
        "structured_json", sparse, declared_en=True, known_license=False,
        immutably_pinned=True, tokenizer_ok=True,
    ) == "metadata_provenance_review_required"
