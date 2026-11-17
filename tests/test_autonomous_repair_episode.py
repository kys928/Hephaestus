"""One objective -> discover actual red -> model patch -> sandbox green -> draft PR."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json

import pytest

from hephaestus.research.constrained_repair import (
    BoundedPatchProposer, RepairBoundaryError,
)
from hephaestus.research.self_repair_controller import AutonomousDiagnosisRepairEpisode
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository

from test_constrained_research_self_repair import (
    _repo, _candidate, DeclaredFakeRunnerForContractTests, FakeReviewPublisher,
)


class FakeCoder:
    def __init__(self):
        self.called=0
    def generate_patch_json(self,prompt):
        self.called+=1
        import re
        path=re.search(r"APPROVED_PATH=([^\n]+)",prompt).group(1)
        base=re.search(r"EXPECTED_SHA256=(sha256:[a-f0-9]{64})",prompt).group(1)
        evidence=re.search(r"EVIDENCE_REF=(sha256:[a-f0-9]{64})",prompt).group(1)
        return json.dumps({
            "path":path,"expected_source_sha256":base,
            "replacement":"def diagnostic_value():\n    return 'fixed'\n",
            "evidence_ref":evidence,
            "hypothesis":"Verified diagnosed test failure from unchanged baseline needs a small correction.",
        })


def _controller(tmp_path,coder=None,runner=None):
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    store=SQLiteStateRepository(tmp_path/"state.db")
    coder=coder or FakeCoder()
    runner=runner or DeclaredFakeRunnerForContractTests([])
    publisher=FakeReviewPublisher()
    controller=AutonomousDiagnosisRepairEpisode(
        artifacts,store,runner,BoundedPatchProposer(coder),publisher)
    return controller,coder,runner,publisher


def test_autonomous_self_repair_from_goal_only_does_not_mutate_main(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    controller,coder,runner,publisher=_controller(tmp_path)
    objective="Investigate and repair the Diagnostician coverage-probe regression"
    report=controller.run(objective,checkout=checkout)
    assert report["status"]=="draft_review_pr_created"
    assert report["model_invocations"]==1
    assert report["production_mutated"] is False
    assert report["pull_request"]=="https://github.com/example/Hephaestus/pull/100"
    assert report["verified_fault_ref"].startswith("sha256:")
    assert len(runner.runs)==3
    assert coder.called==1 and len(publisher.calls)==1
    assert source.read_text().endswith("return 'wrong'\n")
    again=controller.run(objective,checkout=checkout)
    assert again==report and coder.called==1 and len(runner.runs)==3


def test_no_actual_red_regression_means_no_model_and_no_code_edits(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    controller,coder,runner,publisher=_controller(tmp_path)
    # This is a deliberately bounded fake unit observer, not a real coder.
    class AlwaysPass:
        def run(self,checkout,*,fixed_test):
            from hephaestus.research.constrained_repair import SandboxReceipt
            return SandboxReceipt("passed",0,"sha256:"+"a"*64,"unit test only","offline-docker-pytest.v1")
    controller.runner=AlwaysPass()
    report=controller.run("Investigate and repair the Diagnostician regression",
                          checkout=checkout)
    assert report["status"]=="no_verified_regression"
    assert report["model_invocations"]==0
    assert not publisher.calls and coder.called==0


def test_unknown_goal_does_not_access_source_or_invoke_model(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    controller,coder,runner,publisher=_controller(tmp_path)
    with pytest.raises(RepairBoundaryError,match="unsupported"):
        controller.run("Rewrite every Hephaestus system policy immediately",
                       checkout=checkout)
    assert coder.called==0 and runner.runs==[] and not publisher.calls


def test_unusable_sandbox_result_never_proposes_code_or_publishes(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    controller,coder,runner,publisher=_controller(tmp_path)
    class UncertainRunner:
        def run(self,checkout,*,fixed_test):
            from hephaestus.research.constrained_repair import SandboxReceipt
            return SandboxReceipt("timeout",124,"sha256:"+"b"*64,"timed out","offline-docker-pytest.v1")
    controller.runner=UncertainRunner()
    with pytest.raises(RepairBoundaryError,match="uncertain"):
        controller.run("Investigate and repair a Diagnostician coverage regression",
                       checkout=checkout)
    assert coder.called==0 and not publisher.calls


def test_failed_candidate_does_not_publish_review_pr(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    controller,coder,runner,publisher=_controller(tmp_path)
    class NeverGreen:
        def run(self,checkout,*,fixed_test):
            from hephaestus.research.constrained_repair import SandboxReceipt
            return SandboxReceipt("failed",1,"sha256:"+"c"*64,"still red","offline-docker-pytest.v1")
    controller.runner=NeverGreen()
    result=controller.run("Investigate and repair Diagnostician coverage reliability",
                          checkout=checkout)
    assert result["status"]=="repair_candidate_rejected"
    assert result["pull_request"] is None
    assert coder.called==1 and not publisher.calls



def test_autonomous_repair_uses_failed_sandbox_evidence_for_one_bounded_retry(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    class FeedbackCoder:
        def __init__(self):
            self.called=0
            self.saw_feedback=False
        def generate_patch_json(self,prompt):
            import re
            self.called+=1
            path=re.search(r"APPROVED_PATH=([^\n]+)",prompt).group(1)
            sha=re.search(r"EXPECTED_SHA256=(sha256:[a-f0-9]{64})",prompt).group(1)
            ref=re.search(r"EVIDENCE_REF=(sha256:[a-f0-9]{64})",prompt).group(1)
            if self.called==1:
                replacement="def diagnostic_value():\n    return 'still-wrong'\n"
            else:
                assert "candidate_test_status" in prompt
                assert "still red" in prompt
                self.saw_feedback=True
                replacement="def diagnostic_value():\n    return 'fixed'\n"
            return json.dumps({"path":path,"expected_source_sha256":sha,
                "replacement":replacement,"evidence_ref":ref,
                "hypothesis":"An independently failing original test requires the exact observed value."})
    coder=FeedbackCoder()
    controller,_,runner,publisher=_controller(tmp_path,coder=coder)
    controller.maximum_model_attempts=2
    result=controller.run("Investigate and repair the Diagnostician coverage-probe regression",
                          checkout=checkout)
    assert result["model_invocations"]==2
    assert result["status"]=="draft_review_pr_created"
    assert len(result["trial_refs"])==2
    assert coder.saw_feedback
    assert len(runner.runs)==5
    assert len(publisher.calls)==1
    assert source.read_text().endswith("return 'wrong'\n")


def test_autonomous_repair_does_not_loop_on_same_failed_patch(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    class RepeatingCoder:
        def __init__(self):
            self.called=0
        def generate_patch_json(self,prompt):
            import re
            self.called+=1
            path=re.search(r"APPROVED_PATH=([^\n]+)",prompt).group(1)
            sha=re.search(r"EXPECTED_SHA256=(sha256:[a-f0-9]{64})",prompt).group(1)
            ref=re.search(r"EVIDENCE_REF=(sha256:[a-f0-9]{64})",prompt).group(1)
            return json.dumps({"path":path,"expected_source_sha256":sha,
                "replacement":"def diagnostic_value():\n    return 'still-wrong'\n",
                "evidence_ref":ref,
                "hypothesis":"The recorded incorrect value should be replaced with a corrected value."})
    coder=RepeatingCoder()
    controller,_,runner,publisher=_controller(tmp_path,coder=coder)
    controller.maximum_model_attempts=2
    result=controller.run("Investigate and repair the Diagnostician coverage-probe regression",
                          checkout=checkout)
    assert result["status"]=="repair_candidate_rejected"
    assert result["model_invocations"]==2
    assert len(result["trial_refs"])==1
    assert len(runner.runs)==3
    assert len(publisher.calls)==0


def test_autonomous_repair_has_finite_iteration_cap(tmp_path):
    checkout,source,scope=_repo(tmp_path)
    controller,_,runner,publisher=_controller(tmp_path)
    controller.maximum_model_attempts=99
    with pytest.raises(RepairBoundaryError,match="one or two"):
        controller.run("Investigate and repair Diagnostician coverage reliability",
                       checkout=checkout)
    assert runner.runs==[]
