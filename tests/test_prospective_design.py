import pytest

from hephaestus.evaluation.prospective_design import arithmetic_capacity, build_custodian_holdout, holdout_commitment
from hephaestus.schemas.prospective_study import ProspectiveDesign


def test_default_old_domain_is_insufficient_and_new_declared_range_has_capacity():
    assert arithmetic_capacity(20)["evaluation"] == 79
    assert sum(arithmetic_capacity(20).values()) == 820
    assert arithmetic_capacity(100)["evaluation"] >= 512


def test_public_fixture_rehearsal_is_distinct_from_confirmatory_design():
    rehearsal = ProspectiveDesign(purpose="rehearsal", cases_per_family=2, max_new_tokens=16)
    rows = build_custodian_holdout(rehearsal, custodian_seed=152001,
                                   exposure_sample_ids={"sha256:" + "a" * 64})
    assert len(rows) == 8
    assert holdout_commitment(rows) == holdout_commitment(list(reversed(rows)))
    assert rehearsal.commitment() != ProspectiveDesign().commitment()
    with pytest.raises(ValueError):
        ProspectiveDesign(cases_per_family=2)
    with pytest.raises(ValueError):
        ProspectiveDesign(difficulty=1000)
    with pytest.raises(ValueError):
        ProspectiveDesign(training_seeds=(1117, 1117, 3337))


def test_exposure_exclusion_and_commitment_bind_actual_answers():
    design = ProspectiveDesign(purpose="rehearsal", cases_per_family=2, max_new_tokens=16)
    exposure = {"sha256:" + "b" * 64}
    first = build_custodian_holdout(design, custodian_seed=152002, exposure_sample_ids=exposure)
    second = build_custodian_holdout(design, custodian_seed=152002,
        exposure_sample_ids=exposure | {first[0]["sample_id"]})
    assert first[0]["sample_id"] not in {r["sample_id"] for r in second}
    broken = [{**first[0], "target": "{}"}, *first[1:]]
    with pytest.raises(ValueError):
        holdout_commitment(broken)


def test_power_simulator_has_correct_degenerate_boundaries():
    pytest.importorskip("numpy")
    from hephaestus.evaluation.prospective_design import planning_power
    assert planning_power(win_probability=0, loss_probability=0, trials=100)["statistical_gate_power"] == 0
    assert planning_power(win_probability=1, loss_probability=0, trials=100)["statistical_gate_power"] == 1
    with pytest.raises(ValueError):
        planning_power(win_probability=.2, loss_probability=.1, assumed_family_icc=1, trials=100)
