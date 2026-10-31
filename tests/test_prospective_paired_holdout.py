"""#152: strictly paired held-out evaluation; no model or credentials required."""
from hashlib import sha256

import pytest

from hephaestus.evaluation.paired_holdout import (
    exact_mcnemar_p, paired_binary_statistics, score_prediction,
    score_precommitted_holdout,
)
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.storage.verified import canonical_bytes


def test_strict_json_rejects_duplicate_keys_nonfinite_wrong_types_and_fences():
    good = '{"result":4}'
    assert score_prediction(good, good) == {"valid_json": True, "exact_correct": True}
    assert score_prediction('{"result":"4"}', good) == {
        "valid_json": True, "exact_correct": False,
    }
    assert not score_prediction('{"result":4,"result":4}', good)["valid_json"]
    assert not score_prediction('{"result":NaN}', good)["valid_json"]
    assert not score_prediction('garbage'+good, good)["valid_json"]
    assert not score_prediction('```json'+good+'```', good)["valid_json"]
    assert score_prediction('{"result":true}', '{"result":1}')["exact_correct"] is False


def test_exact_mcnemar_counts_discordance_not_unpaired_averages():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(10, 0) == pytest.approx(2 / 2**10)
    assert exact_mcnemar_p(3, 3) == 1.0
    with pytest.raises(ValueError):
        exact_mcnemar_p(-1, 4)
    stat = paired_binary_statistics([False]*10, [True]*10, bootstrap_draws=100)
    assert stat["wins"] == 10 and stat["losses"] == 0
    assert stat["paired_absolute_gain"] == 1
    assert stat["paired_bootstrap_95_ci"] == [1, 1]


@pytest.fixture(scope="module")
def original_private_like_holdout():
    # Synthetic fixture only; not the real unreleased confirmatory evaluation.
    splits, _ = generate_original_splits(GeneratedDataSpec(
        seed=812351, train_per_family=24, development_per_family=6,
        evaluation_per_family=512, difficulty=1000,
    ))
    exposure = {row["sample_id"] for row in splits["train"]+splits["development"]}
    items = sorted(({
        "sample_id": row["sample_id"], "family": row["family"],
        "prompt": row["prompt"], "target": row["target"]
    } for row in splits["evaluation"]), key=lambda r: r["sample_id"])
    commitment = "sha256:" + sha256(canonical_bytes(items)).hexdigest()
    rows = [dict(row, baseline_raw='not-json', candidate_raw=row["target"],
                 baseline_eos=True, candidate_eos=True,
                 baseline_truncated=False, candidate_truncated=False)
            for row in splits["evaluation"]]
    return rows, commitment, exposure


def test_complete_balanced_mock_holdout_passes_only_as_internal_effect(original_private_like_holdout):
    rows, commitment, exposed = original_private_like_holdout
    result = score_precommitted_holdout(
        rows, expected_commitment=commitment, exposure_sample_ids=exposed,
    )
    assert result["paired_exact"]["count"] == 2048
    assert result["internal_effect_supported"] is True
    assert result["external_certification"] is False
    assert result["checkpoint_bytes_verified"] is False
    assert result["model_promoted"] is False


def test_exposure_and_holdout_integrity_fail_closed(original_private_like_holdout):
    rows, commitment, exposed = original_private_like_holdout
    with pytest.raises(PermissionError, match="commitment"):
        score_precommitted_holdout(rows, expected_commitment="sha256:"+"a"*64,
                                  exposure_sample_ids=exposed)
    with pytest.raises(ValueError, match="exposed"):
        score_precommitted_holdout(rows, expected_commitment=commitment,
                                  exposure_sample_ids=exposed | {rows[0]["sample_id"]})
    with pytest.raises(ValueError, match="complete"):
        score_precommitted_holdout(rows[:4], expected_commitment=commitment,
                                  exposure_sample_ids=exposed)
