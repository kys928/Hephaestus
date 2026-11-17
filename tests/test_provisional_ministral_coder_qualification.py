"""Development qualification: typed fault, finite strategy, model proposal and no unsafe handoff.

A declared fake model and runner exercise the controller only. These tests
DO NOT certify live Ministral inference or the rootless Docker executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import subprocess

import pytest

from hephaestus.research.constrained_repair import SandboxReceipt,RepairBoundaryError
from hephaestus.research.exact_span_repair import ExactSpanPatchProposer
from hephaestus.research.provisional_coder import (
    ProvisionalMinistralCoder,PARENT_ID,PARENT_REVISION,
)
from hephaestus.research.provisional_qualification import (
    ProvisionalCoderQualification,ReviewedRepairCase,
)
from hephaestus.production.original_cuda_identity import registry_digest
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import read_verified_json


_SOURCE="src/hephaestus/diagnosis/probes.py"
_TESTS="tests/test_targeted_post_failure_probes.py"
_NODE=_TESTS+"::test_null_text_not_counted"
_BAD='text = str(row.get("text", ""))'
_GOOD='text = row.get("text") or ""'


def _checkout(root):
    directory=root/"working"
    source=directory/_SOURCE
    fixed=directory/_TESTS
    source.parent.mkdir(parents=True)
    fixed.parent.mkdir(parents=True)
    source.write_text("def analyze_dataset_task_coverage(row):\n"
                      "    "+_BAD+"\n"
                      "    return bool(text)\n")
    fixed.write_text("def test_null_text_not_counted():\n"
                     "    from hephaestus.diagnosis.probes import analyze_dataset_task_coverage\n"
                     "    assert analyze_dataset_task_coverage({'text':None}) is False\n")
    for args in (
        ["git","init","-q",str(directory)],
        ["git","-C",str(directory),"config","user.name","unit-example"],
        ["git","-C",str(directory),"config","user.email","unit@example.invalid"],
        ["git","-C",str(directory),"add","."],
        ["git","-C",str(directory),"commit","-qm","pinned genuine failing example"],
    ):
        subprocess.run(args,check=True,timeout=10)
    commit=subprocess.run(["git","-C",str(directory),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True).stdout.strip()
    case=ReviewedRepairCase(
        commit,"sha256:"+sha256(source.read_bytes()).hexdigest(),
        "sha256:"+sha256(fixed.read_bytes()).hexdigest(),
        _NODE,"analyze_dataset_task_coverage")
    return directory,source,fixed,case


@dataclass
class DeclaredFakeIndependentRunner:
    calls: list[tuple[str,str]]
    collection_error: bool=False

    def run(self,checkout,*,fixed_test):
        source=(checkout/_SOURCE).read_text()
        outcome="passed" if _GOOD in source else "failed"
        if self.collection_error:
            output="ERROR collecting tests/test_targeted_post_failure_probes.py\nModuleNotFoundError"
        elif outcome=="failed":
            output=f"FAILED {_NODE} - AssertionError: assert True is False"
        else:
            output=f"1 passed in 0.02s"
        self.calls.append((outcome,output))
        if self.collection_error:
            return SandboxReceipt("failed",2,"sha256:"+sha256(output.encode()).hexdigest(),
                                  output,"offline-docker-pytest.v1")
        return SandboxReceipt(outcome,0 if outcome=="passed" else 1,
            "sha256:"+sha256(output.encode()).hexdigest(),output,
            "offline-docker-pytest.v1")


class DeclaredFakeCoder:
    def __init__(self,*,first_bad=False):
        self.calls=0
        self.first_bad=first_bad
    def generate_patch_json(self,prompt):
        self.calls+=1
        if self.first_bad and self.calls==1:
            new_text='text = str(row.get("text", "")) + " "'
        else:
            if self.calls>1:
                assert "previous_failed_patch_tests" in prompt
            new_text=_GOOD
        return json.dumps({
            "old_text":_BAD,"new_text":new_text,
            "hypothesis":"Missing text should remain empty rather than becoming the string None.",
        })


def _qualifier(root,*,runner=None,coder=None,attempts=1):
    artifacts=FileSystemArtifactStore(root/"artifacts")
    repo=SQLiteStateRepository(root/"records.db")
    runner=runner or DeclaredFakeIndependentRunner([])
    coder=coder or DeclaredFakeCoder()
    return ProvisionalCoderQualification(artifacts,repo,runner,
        ExactSpanPatchProposer(coder),maximum_attempts=attempts),runner,coder


def test_exact_parent_identity_without_loading_or_tuning_planner_lora(tmp_path):
    obj=ProvisionalMinistralCoder(tmp_path/"hf-model-cache",registry_digest())
    assert PARENT_ID=="mistralai/Ministral-3-8B-Reasoning-2512"
    assert PARENT_REVISION=="81eaece1948f3875421d9a45bc55487d10e2d894"
    assert obj._model is None and obj.last_receipt is None
    with pytest.raises(PermissionError):
        ProvisionalMinistralCoder(tmp_path/"hf-model-cache","sha256:"+"0"*64)


def test_controller_identifies_fault_chooses_fixed_strategy_and_quarantines_fake_result(tmp_path):
    checkout,source,fixed,case=_checkout(tmp_path)
    qualification,runner,coder=_qualifier(tmp_path)
    result=qualification.run(
        objective="Investigate and repair Diagnostician missing text measurement",
        checkout=checkout,case=case,run_id="dev-study",lineage_id="dev-lineage")
    assert result["status"]=="blocked_unverified_candidate"
    assert result["model_invocations"]==1
    assert result["trial_attempts"][0]["red_to_green_verified"] is True
    assert result["actual_offline_docker_verified"] is False
    assert result["real_parent_inference_proven"] is False
    assert result["verified_review_handoff_ref"] is None
    assert result["first_attempt_correct"] is False
    assert coder.calls==1 and len(runner.calls)==3
    assert _BAD in source.read_text()
    strategy=read_verified_json(qualification.artifacts,result["authorized_strategy_ref"])
    assert strategy["decision"]=="minimal_exact_span"
    assert strategy["model_may_edit_live_code"] is False
    assert strategy["model_may_choose_tests"] is False
    with pytest.raises(PermissionError):
        qualification.verify_review_handoff(result)
    repeated=qualification.run(
        objective="Investigate and repair Diagnostician missing text measurement",
        checkout=checkout,case=case,run_id="dev-study",lineage_id="dev-lineage")
    assert repeated==result and coder.calls==1


def test_two_attempts_are_local_not_two_full_research_pipeline_passes(tmp_path):
    checkout,source,fixed,case=_checkout(tmp_path)
    qualification,runner,coder=_qualifier(tmp_path,
        coder=DeclaredFakeCoder(first_bad=True),attempts=2)
    result=qualification.run(
        objective="Investigate and repair Diagnostician missing text measurement",
        checkout=checkout,case=case,run_id="dev-retry",lineage_id="dev-lineage")
    assert result["model_invocations"]==2
    assert [t["red_to_green_verified"] for t in result["trial_attempts"]]==[False,True]
    assert result["verified_review_handoff_ref"] is None
    assert result["status"]=="blocked_unverified_candidate"
    assert _BAD in source.read_text()
    assert len(runner.calls)==5


def test_import_or_collection_failure_is_not_mistaken_for_real_defect(tmp_path):
    checkout,source,fixed,case=_checkout(tmp_path)
    qualification,runner,coder=_qualifier(tmp_path,
        runner=DeclaredFakeIndependentRunner([],collection_error=True))
    result=qualification.run(
        objective="Investigate and repair Diagnostician missing text measurement",
        checkout=checkout,case=case,run_id="bad-infra",lineage_id="dev-lineage")
    assert result["status"]=="blocked_no_authenticated_semantic_defect"
    assert result["model_invocations"]==0
    assert coder.calls==0 and len(runner.calls)==1


def test_tampered_original_source_or_frozen_test_is_refused_before_model(tmp_path):
    checkout,source,fixed,case=_checkout(tmp_path)
    qualification,runner,coder=_qualifier(tmp_path)
    fixed.write_text(fixed.read_text()+"\n# unreviewed test change\n")
    with pytest.raises(RepairBoundaryError,match="fixed regression test changed"):
        qualification.run(objective="Investigate Diagnostician missing text measurement",
            checkout=checkout,case=case,run_id="tampered",lineage_id="dev-lineage")
    assert coder.calls==0 and runner.calls==[]


def test_model_cannot_expand_repair_scope_to_unreviewed_target(tmp_path):
    checkout,source,fixed,case=_checkout(tmp_path)
    qualification,runner,coder=_qualifier(tmp_path)
    bad=ReviewedRepairCase(case.source_commit,case.source_sha256,
        case.fixed_test_sha256,_NODE,"unreviewed_other_function")
    with pytest.raises(RepairBoundaryError,match="unrecognized"):
        qualification.run(objective="Investigate Diagnostician missing text measurement",
            checkout=checkout,case=bad,run_id="scope",lineage_id="dev-lineage")
    assert coder.calls==0


def test_phase_handoff_requires_real_parent_receipt_even_if_development_sandbox_claims_green(tmp_path):
    checkout,source,fixed,case=_checkout(tmp_path)
    qualification,_,_=_qualifier(tmp_path)
    result=qualification.run(
        objective="Investigate Diagnostician missing text measurement",
        checkout=checkout,case=case,run_id="hand-off",lineage_id="dev-lineage")
    with pytest.raises(PermissionError,match="no verified result"):
        qualification.verify_review_handoff(result,require_real_parent=True)


def test_temporary_coder_declaration_is_base_only_and_not_certified():
    declaration=json.loads(
        Path("configs/models/provisional_ministral_coder_v1.json").read_text())
    assert declaration["status"]=="temporary_uncertified_not_promoted"
    assert declaration["role"]=="coder"
    assert declaration["model_id"]==PARENT_ID
    assert declaration["parent_revision"]==PARENT_REVISION
    assert declaration["source_parent_role"]=="planner"
    assert declaration["adapter"] is None
    assert declaration["reuse_planner_lora"] is False
    assert declaration["automatic_role_dispatch_enabled"] is False
    assert declaration["gates"]["model_inference_verified"] is False
    assert declaration["gates"]["full_system_certified"] is False
    assert declaration["repair_scope"]["maximum_attempts_default"]==1
