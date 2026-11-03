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
from hephaestus.llm.role_contract import RoleAnswer
from hephaestus.policy.model_proposals import planner_selection
import hephaestus.development.prometheus_granite_public_measurement as measurement_module


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
    root.mkdir(parents=True,exist_ok=True)
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


def test_public_measurement_binds_action_to_actual_planner_output(tmp_path, monkeypatch):
    # Offline stub isolates the second admission check. It is NOT evidence
    # of model inference or permission to run an actual CUDA measurement.
    actual = RoleAnswer(
        "collect_more_evidence", "request_recheck", "diagnostic_measurement",
        0.6, (REF_PARENT,), (), "Public development evidence is insufficient.",
    )
    source = _decision()
    selection = planner_selection(actual, source["model_planner_receipt_ref"]).to_dict()
    source["model_planner_selection"] = selection
    source["independently_admitted_intervention"]["metadata"]["model_selection"] = selection

    class Contract:
        def parse(self, role, raw, allowed_refs):
            assert role == "planner" and allowed_refs == {REF_PARENT}
            return actual

    class Reasoner:
        contract = Contract()

    monkeypatch.setattr(measurement_module, "_require_actual_gpu_receipt",
        lambda reasoner, ref, role: {
            "raw_output": "offline-stub-only", "evidence_refs": [REF_PARENT],
        })
    root = _write(tmp_path / "valid", source)
    assert admit_decision(Reasoner(), root) == source

    tampered = _decision()
    tampered["model_planner_selection"] = dict(selection, intervention_kind="stop")
    tampered["independently_admitted_intervention"]["metadata"]["model_selection"] = selection
    root = _write(tmp_path / "tampered", tampered)
    with pytest.raises(PinnedEvidenceError, match="original Planner output"):
        admit_decision(Reasoner(), root)


def test_scope_constant_declares_public_only():
    assert SCHEMA.endswith("public-development-measurement.v1")
    assert TOOL_OWNER == "HEPHAESTUS_DETERMINISTIC_EXECUTION_POLICY"
    assert MEASUREMENT_ID == "public-sixstate-paired-counterfactual-v1"
