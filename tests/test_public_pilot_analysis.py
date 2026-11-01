from copy import deepcopy

import pytest

from hephaestus.evaluation.public_pilot_analysis import paired_template_diagnostics


def sample(sid, correct, template="original-canonical.v1"):
    return {"sample_id": sid, "family": "arithmetic", "template_id": template,
            "terminated_exact": correct, "typed_exact": correct,
            "valid_json": True, "eos": True, "truncated": False}


def test_template_comparison_uses_only_matching_canonical_subset():
    # The unpaired canonical success must not inflate the paired original rate.
    rows = [sample("paired", False), sample("unpaired", True),
            sample("paired", True, "task-wrapper-sensitivity.v1")]
    result = paired_template_diagnostics(rows)
    assert result["all_canonical_units"] == 2
    assert result["paired_canonical_units"] == 1
    assert result["original_on_same_cases"]["terminated_exact"] == 0
    assert result["wrapper_on_same_cases"]["terminated_exact"] == 1
    assert result["terminated_exact_wrapper_wins"] == 1
    assert result["additional_independent_units"] == 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "foreign_family", "non_boolean"])
def test_incomplete_or_forged_pairing_cannot_produce_a_sensitivity_result(defect):
    rows = [sample("paired", True), sample("paired", False, "task-wrapper-sensitivity.v1")]
    if defect == "missing":
        rows = rows[1:]
    elif defect == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif defect == "foreign_family":
        rows[1]["family"] = "symbolic"
    else:
        rows[1]["eos"] = 1
    with pytest.raises(ValueError):
        paired_template_diagnostics(rows)
