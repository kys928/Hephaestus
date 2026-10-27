from dataclasses import dataclass

import pytest
from hephaestus.data.continuation_eligibility import screen_continuation_rows


@dataclass
class Tokens:
    ids: list[int]


class FrozenTokenizer:
    def encode(self, text):
        if "unknown" in text:
            return Tokens([99] * 3 + [1] * 158)
        return Tokens([1] * 168)


def run(rows, **kwargs):
    return screen_continuation_rows(
        rows, FrozenTokenizer(), 99, lambda text: "nonenglish" not in text,
        prefix_tokens=128, target_tokens=32, **kwargs)


def test_preserves_original_rows_and_rejects_privacy_dup_and_reference_overlap():
    rows = [
        {"text":"some ample continuation contents"},
        {"text":" some   ample continuation contents "},  # exact normalized duplicate
        {"text":"confidential alice@example.org mentioned"},
        {"text":"contains nonenglish tokens"},
        {"text":"a published public prompt word sequence exactly"},
        {"text":"unknown token fraction exceeds ceiling"},
        {"text":""},
        {"text":None},
    ]
    result = run(rows, public_prompts=("public prompt word sequence exactly",))
    assert result["source_rows_examined"] == len(rows)
    assert result["accepted_unique_rows"] == 1
    assert result["eligible_within_row_windows"] == 1
    assert result["unknown_tokens"] == 0
    assert result["rejected_counts"] == {
        "normalized_duplicate_excluded": 1,
        "pii_regex_candidate_excluded": 1,
        "language_heuristic_rejected": 1,
        "public_probe_overlap_excluded": 1,
        "frozen_unknown_fraction_exceeds_1pct": 1,
        "empty": 1,
        "schema_invalid": 1,
    }
    assert not result["training_allowed"] and not result["acquisition_allowed"]


def test_commitment_is_sensitive_to_true_row_positions_and_original_text():
    one = run([{"text":"alpha message"}])
    two = run([{"text":"different alpha message"}])
    three = run([{"text":""}, {"text":"alpha message"}])
    assert one["selection_commitment_sha256"] != two["selection_commitment_sha256"]
    assert one["selection_commitment_sha256"] != three["selection_commitment_sha256"]


def test_insufficient_tokens_and_unknown_limit_never_get_windows():
    class TooShort:
        def encode(self, _):
            return Tokens([1] * 159)
    result = screen_continuation_rows(
        [{"text":"non-sensitive paragraph"}], TooShort(), 99,
        lambda t: True,
    )
    assert result["eligible_within_row_windows"] == 0
    assert result["rejected_counts"]["insufficient_contiguous_tokens"] == 1


@pytest.mark.parametrize("kwargs", [
    {"prefix_tokens":0}, {"target_tokens":0}, {"stride_tokens":0},
    {"unknown_id":-1}, {"unknown_id":True},
])
def test_invalid_scientific_window_contract_fails(kwargs):
    cfg=dict(prefix_tokens=128,target_tokens=32,stride_tokens=64,unknown_id=99)
    cfg.update(kwargs)
    with pytest.raises(ValueError):
        screen_continuation_rows([{"text":"safe"}],FrozenTokenizer(),
            cfg.pop("unknown_id"),lambda t:True,**cfg)
