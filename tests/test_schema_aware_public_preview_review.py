"""Fail-closed checks for public-preview schema, English and tokenizer studies."""
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

from run_schema_aware_public_preview_review import (
    REVIEW, english_likelihood, text_pair, count_unk, window_proof, MAX_PREVIEW_ROWS,
)


def _tok():
    t = Tokenizer(WordLevel(
        vocab={"[UNK]":0, "The":1, "the":2, "quick":3, "brown":4,
               "fox":5, "jumps":6, "over":7, "dog":8, ".":9},
        unk_token="[UNK]",
    ))
    t.pre_tokenizer = Whitespace()
    return t


def test_schema_mapping_keeps_feedback_out_of_reference_answer():
    v = REVIEW["Short-Answer-Feedback/saf_communication_networks_english"]
    prompt, target, name = text_pair({
        "question": "What is the main benefit of routing?",
        "reference_answer": "It can reduce delays.",
        "provided_answer": "Something unrelated.",
        "answer_feedback": "Incorrect.",
    }, v)
    assert prompt == "What is the main benefit of routing?"
    assert target == "It can reduce delays."
    assert name == "schema_pinned_prompt_target"


def test_hotpotqa_and_json_fields_mapped_without_inference():
    q = REVIEW["hotpotqa/hotpot_qa"]
    assert text_pair({"question": "Which hotel was founded earlier?",
                      "answer": "The first hotel.",
                      "context": {"title": ["Hotel A"]}}, q)[1] == "The first hotel."
    j = REVIEW["soikit/dataset_instruction_json"]
    assert text_pair({"instruction": "Return a JSON answer",
                      "input": "Given this task", "output": '{"a":1}'}, j)[1] == '{"a":1}'
    assert text_pair({"text": "Unrelated raw record"}, j)[1] == ""


def test_conservative_english_filter_excludes_chinese_and_non_latin():
    assert english_likelihood("Return the answer in one sentence with a useful example.")[0]
    assert not english_likelihood("分析给定老师话语提取JSON对象。")[0]
    assert not english_likelihood("こんにちは、今日は明日です。")[0]
    assert not english_likelihood("ABC XYZ")[0]


def test_contiguous_windows_are_training_study_only():
    t = _tok()
    long = " ".join(["The quick brown fox jumps over the dog ."] * 45)
    proof = window_proof(t, [long, "short"])
    assert proof["candidate_windows"] > 0
    assert proof["same_row_contiguous_identity_checked"]
    assert proof["unique_source_rows"] == 1
    assert not proof["cross_row_concatenation"]
    assert not proof["eos_or_eot_inserted"]
    assert not proof["trained_or_materialized_windows"]


def test_unk_is_measured_with_original_tokenizer_not_silently_replaced():
    n, unknown = count_unk(_tok(), "The supercalifragilistic quick")
    assert n > 0 and unknown > 0


def test_scope_is_bounded_and_source_coverage_explicit():
    assert MAX_PREVIEW_ROWS <= 100
    assert {"structured_json", "short_answer", "raw_continuation"} == {
        spec["task"] for spec in REVIEW.values()}
    assert "Salesforce/wikitext" in REVIEW
