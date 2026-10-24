"""Scientific-state assessment exercises real content-addressed artifacts, no ML doubles."""
import json
from pathlib import Path

import pytest

from hephaestus.evaluation.scientific_state_certification import assess_scientific_states
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json


STATES = (
    "equivalent", "recheck_required", "regressed", "scientific_rejection",
    "improved", "incomplete_evidence", "inconclusive", "not_passed",
)


def _put(store, payload):
    return store.put_bytes(canonical_bytes(payload), media_type="application/json").artifact_ref


def _fixture(tmp_path, *, altered=None, contaminated=False, omit_diagnostic=False):
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    cases = [{"case_id": f"sealed-{i}", "gold_state": state,
              "contrast_id": f"pair-{i // 2}", "protected": i in (0, 1, 2, 3)}
             for i, state in enumerate(STATES)]
    pack_ref = _put(store, {
        "schema": "scientific-state-holdout.v1",
        "split": "sealed_holdout", "cases": cases,
    })
    diagnostic_ref = _put(store, {
        "schema": "scientific-diagnostic-cases.v1",
        "case_ids": ["observed-D-1"] + (["sealed-0"] if contaminated else []),
    })
    settings_ref = _put(store, {"decoding": "greedy", "temperature": 0.0, "version": 1})
    baselines = []
    candidates = []
    for seed in (13, 47):
        baseline = [STATES[i] if i in (0, 1, 2) else "wrong" for i in range(8)]
        candidate = list(STATES)
        if altered == "protected_regression":
            candidate[0] = "wrong"
        if altered == "low_contrast":
            candidate[2] = "wrong"
            candidate[4] = "wrong"
        def record(run, checkpoint, states):
            return {
                "schema": "scientific-state-predictions.v1",
                "pack_ref": pack_ref,
                "run_id": run, "checkpoint_ref": checkpoint,
                "seed": seed, "settings_ref": settings_ref,
                "predictions": [
                    {"case_id": f"sealed-{i}", "state": state}
                    for i, state in enumerate(states)
                ],
            }
        b = record("parent", "checkpoint://V3.1", baseline)
        c = record("candidate", "checkpoint://D2", candidate)
        if altered == "seed_mismatch" and seed == 47:
            c["seed"] = 48
        if altered == "settings_mismatch":
            c["settings_ref"] = _put(store, {"decoding": "sampling"})
        if altered == "missing_case":
            c["predictions"].pop()
        if altered == "duplicate_case":
            c["predictions"].append(c["predictions"][0])
        baselines.append(_put(store, b))
        candidates.append(_put(store, c))
    kwargs = {
        "approved_pack_ref": pack_ref,
        "baseline_refs": baselines, "candidate_refs": candidates,
        "diagnostic_manifest_ref": None if omit_diagnostic else diagnostic_ref,
        "minimum_accuracy": 0.75, "minimum_contrast_accuracy": 0.75,
        "minimum_per_state_accuracy": 0.5, "minimum_matched_seeds": 2,
    }
    return store, kwargs


def test_clean_scientific_contrast_is_eligible_for_review_but_not_certified(tmp_path):
    store, arguments = _fixture(tmp_path)
    result = assess_scientific_states(store, **arguments)
    assert result.eligible_for_review
    assert result.candidate_accuracy == result.candidate_contrast_accuracy == 1.0
    assert result.baseline_accuracy == 3 / 8
    assert result.protected_regressions == 0
    receipt = read_verified_json(store, result.artifact_ref)
    assert receipt["certification_state"] == "not_certified"
    assert receipt["holdout_unseen_provenance_independently_verified"] is False
    assert receipt["matched_seeds"] == [13, 47]
    assert receipt["criteria"]["minimum_matched_seeds"] == 2


def test_protected_regression_rejects_even_high_overall_accuracy(tmp_path):
    store, arguments = _fixture(tmp_path, altered="protected_regression")
    result = assess_scientific_states(store, **arguments)
    assert result.candidate_accuracy == 7 / 8
    assert result.candidate_accuracy > result.baseline_accuracy
    assert not result.eligible_for_review
    assert result.protected_regressions == 2
    assert "protected_regression_detected" in result.blocking_issues


def test_overlap_with_diagnostic_cases_fails_closed(tmp_path):
    store, arguments = _fixture(tmp_path, contaminated=True)
    result = assess_scientific_states(store, **arguments)
    assert not result.eligible_for_review
    assert "known_holdout_contamination" in result.blocking_issues


def test_missing_diagnostic_manifest_blocks_eligibility(tmp_path):
    store, arguments = _fixture(tmp_path, omit_diagnostic=True)
    result = assess_scientific_states(store, **arguments)
    assert not result.eligible_for_review
    assert "diagnostic_overlap_not_verified" in result.blocking_issues


@pytest.mark.parametrize("altered", [
    "seed_mismatch", "settings_mismatch", "missing_case", "duplicate_case",
])
def test_changed_settings_seeds_or_case_coverage_is_rejected(tmp_path, altered):
    store, arguments = _fixture(tmp_path, altered=altered)
    with pytest.raises(ValueError):
        assess_scientific_states(store, **arguments)


def test_immutable_prediction_modification_fails_before_scoring(tmp_path):
    store, arguments = _fixture(tmp_path)
    prediction_ref = arguments["candidate_refs"][0]
    path = store._path_for_digest(prediction_ref.split(":")[1])
    path.write_bytes(b'{"schema":"scientific-state-predictions.v1","fake":true}')
    with pytest.raises((ValueError, Exception), match="hash|integrity|mismatch"):
        assess_scientific_states(store, **arguments)


def test_unknown_threshold_and_one_seed_rejected(tmp_path):
    store, arguments = _fixture(tmp_path)
    with pytest.raises(ValueError, match="threshold"):
        assess_scientific_states(store, **{**arguments, "minimum_accuracy": float("nan")})
    with pytest.raises(ValueError, match="insufficient matched"):
        assess_scientific_states(store, **{
            **arguments, "baseline_refs": arguments["baseline_refs"][:1],
            "candidate_refs": arguments["candidate_refs"][:1],
        })


def test_bad_or_unpaired_holdout_manifest_rejected(tmp_path):
    store, arguments = _fixture(tmp_path)
    bad = _put(store, {
        "schema":"scientific-state-holdout.v1","split":"sealed_holdout",
        "cases":[{"case_id":"a","gold_state":"equivalent","contrast_id":"x","protected":True}],
    })
    with pytest.raises(ValueError, match="contrast"):
        assess_scientific_states(store, **{**arguments, "approved_pack_ref": bad})
