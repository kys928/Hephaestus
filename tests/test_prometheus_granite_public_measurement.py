"""No-GPU negative gates for conditional public Granite Evaluator measurement.

CPU doubles are deliberately *never* accepted as authentic research outputs.
"""
from __future__ import annotations

import json

import pytest

from hephaestus.development.prometheus_granite_public_measurement import (
    MEASUREMENT_ID, SCHEMA, TOOL_OWNER, admit_decision, run_public_measurement,
)
from hephaestus.development.prometheus_scientific_decision import REF_PARENT, REF_D
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError


def _decision():
    return {
        "schema":"prometheus-authentic-scientific-decision.v1",
        "research_owner":"HEPHAESTUS_ACTUAL_FROZEN_ROLE_INFERENCE",
        "source_parent_report_sha256":REF_PARENT,
        "source_historical_d_report_sha256":REF_D,
        "training_executed":False,
        "independent_certification":False,
        "model_diagnosis_receipt_ref":"sha256:"+"1"*64,
        "model_planner_receipt_ref":"sha256:"+"2"*64,
        "independently_admitted_intervention":{
            "intervention_kind":"collect_more_evidence",
            "primary_variable":"diagnostic_measurement",
            "metadata":{"model_inference_receipt_ref":"sha256:"+"2"*64},
        },
    }


def _write(tmp_path, document):
    root=tmp_path/"decision"
    root.mkdir(exist_ok=True)
    (root/"scientific-decision.json").write_text(json.dumps(document),encoding="utf-8")
    return root


def test_hand_authored_decision_cannot_trigger_real_measurement(tmp_path):
    root=_write(tmp_path,_decision())
    with pytest.raises((AttributeError,TypeError,PermissionError)):
        admit_decision(object(),root)


@pytest.mark.parametrize("modifier", [
    lambda x: x.update(schema="simulated-completion"),
    lambda x: x.update(research_owner="SOL"),
    lambda x: x.update(source_parent_report_sha256="sha256:"+"0"*64),
    lambda x: x.update(source_historical_d_report_sha256="sha256:"+"0"*64),
    lambda x: x.update(training_executed=True),
    lambda x: x.update(independent_certification=True),
])
def test_reject_untrusted_or_wrong_scope_before_inference_receipts(tmp_path,modifier):
    source=_decision()
    modifier(source)
    root=_write(tmp_path,source)
    with pytest.raises(PinnedEvidenceError):
        admit_decision(object(),root)


def test_stop_and_unimplemented_training_actions_do_not_measure(tmp_path):
    for action in ("stop","replace_or_mix_dataset","repair_evaluation"):
        root=tmp_path/action
        x=_decision()
        x["independently_admitted_intervention"]["intervention_kind"]=action
        _write(root,x)
        with pytest.raises(PermissionError,match="did not select"):
            admit_decision(object(),root/"decision")


def test_wrong_variable_not_a_supported_public_diagnostic(tmp_path):
    x=_decision()
    x["independently_admitted_intervention"]["primary_variable"]="optimizer"
    root=_write(tmp_path,x)
    with pytest.raises(PermissionError,match="diagnostic measurement"):
        admit_decision(object(),root)


def test_no_decision_means_no_gpu_inference(tmp_path):
    with pytest.raises(PinnedEvidenceError,match="no bounded"):
        run_public_measurement(object(),
            decision_root=tmp_path/"missing",output=tmp_path/"no-output")
    assert not (tmp_path/"no-output").exists()


def test_scope_constant_declares_public_only():
    assert SCHEMA.endswith("public-development-measurement.v1")
    assert TOOL_OWNER == "HEPHAESTUS_DETERMINISTIC_EXECUTION_POLICY"
    assert MEASUREMENT_ID == "public-sixstate-paired-counterfactual-v1"
