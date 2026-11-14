"""Bounded self-repair contracts; explicitly simulated inference/test backend.

These tests exercise real Git checkout hash checks, temp-workspace patch
isolation, source receipt validation, and PR-only publication admission.
They do not claim that a fake sandbox is independent container execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import subprocess

import pytest

from hephaestus.research.constrained_repair import (
    BoundedPatchProposer, ConstrainedSelfRepair, GitHubReviewBranchPublisher,
    PatchCandidate, RepairBoundaryError, RepairScope, SandboxReceipt,
)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


def _repo(tmp_path):
    workspace=tmp_path/"workspace"
    source=workspace/"src"/"hephaestus"/"diagnosis"/"probes.py"
    tests=workspace/"tests"/"test_targeted_post_failure_probes.py"
    source.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    source.write_text("def diagnostic_value():\n    return 'wrong'\n")
    tests.write_text("def test_stable():\n    assert True\n")
    (workspace/"pyproject.toml").write_text("[build-system]\nrequires=[]\n")
    for args in (
        ["git","init","-q",str(workspace)],
        ["git","-C",str(workspace),"config","user.email","bot@invalid.example"],
        ["git","-C",str(workspace),"config","user.name","local-review"],
        ["git","-C",str(workspace),"add","."],
        ["git","-C",str(workspace),"commit","-qm","frozen original"],
    ):
        subprocess.run(args,check=True,timeout=10)
    sha=subprocess.run(["git","-C",str(workspace),"rev-parse","HEAD"],
                       capture_output=True,text=True,check=True).stdout.strip()
    scope=RepairScope(sha,"src/hephaestus/diagnosis/probes.py",
                     "tests/test_targeted_post_failure_probes.py")
    return workspace,source,scope


@dataclass
class DeclaredFakeRunnerForContractTests:
    runs: list[tuple[str,str]]

    def run(self,checkout,*,fixed_test):
        content=(checkout/"src/hephaestus/diagnosis/probes.py").read_text()
        status="failed" if "'wrong'" in content else "passed"
        self.runs.append((str(checkout),status))
        return SandboxReceipt(status,1 if status=="failed" else 0,
            "sha256:"+hashlib.sha256(status.encode()).hexdigest(),"contract simulation only",
            "offline-docker-pytest.v1")


@dataclass
class DeclaredModelResponseForContractTests:
    response: str

    def generate_patch_json(self,prompt):
        assert "APPROVED_PATH=" in prompt and "VERIFIED_FAULT=" in prompt
        return self.response


class FakeReviewPublisher:
    def __init__(self):
        self.calls=[]
    def publish(self,*,scope,candidate,verified_trial):
        self.calls.append((scope,candidate,verified_trial))
        return "https://github.com/example/Hephaestus/pull/100"


def _candidate(original,scope,ref):
    return PatchCandidate(
        path=scope.allowed_file,
        expected_source_sha256="sha256:"+hashlib.sha256(original).hexdigest(),
        replacement="def diagnostic_value():\n    return 'fixed'\n",
        evidence_ref=ref,
        hypothesis="A demonstrably incorrect diagnostic return value should be repaired.",
    )


def _fault(artifacts,scope):
    return artifacts.put_bytes(canonical_bytes({
        "schema":"reviewed-code-failure.v1", "status":"verified_failed",
        "target_file":scope.allowed_file, "reproduction_test":scope.fixed_test,
        "scientific_certification":False,
    })).artifact_ref


def test_model_can_propose_patch_and_controller_tests_without_mutating_original(tmp_path):
    repo,source,scope=_repo(tmp_path)
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    ref=_fault(artifacts,scope)
    patch=_candidate(source.read_bytes(),scope,ref)
    model=BoundedPatchProposer(DeclaredModelResponseForContractTests(
        json.dumps(asdict(patch))))
    proposed=model.propose(scope=scope,original=source.read_bytes(),
                           verified_fault={"status":"verified_failed","target_file":scope.allowed_file},
                           evidence_ref=ref)
    assert proposed==patch
    runner=DeclaredFakeRunnerForContractTests([])
    repair=ConstrainedSelfRepair(artifacts,runner)
    trial=repair.trial(checkout=repo,scope=scope,candidate=patch,verified_fault_ref=ref)
    assert trial["status"]=="reviewable_patch"
    assert trial["red_to_green_verified"] is True
    assert trial["production_source_mutated"] is False
    assert source.read_text().endswith("return 'wrong'\n")
    assert len(runner.runs)==2
    publisher=FakeReviewPublisher()
    assert repair.submit_for_review(
        scope=scope,candidate=patch,trial=trial,publisher=publisher).endswith("/pull/100")
    assert len(publisher.calls)==1


@pytest.mark.parametrize("bad_path",[
    "src/hephaestus/control/orchestrator.py", "configs/eval_packs/semantic_behavior_v1.yaml",
    "src/hephaestus/diagnosis/../../policy/action_registry.py",
    "tests/test_targeted_post_failure_probes.py", ".env", "/tmp/another.py",
])
def test_operator_scope_cannot_enable_forbidden_self_modification(tmp_path,bad_path):
    _,_,scope=_repo(tmp_path)
    with pytest.raises(RepairBoundaryError):
        RepairScope(scope.base_commit,bad_path,scope.fixed_test).validate()


def test_source_drift_refused_even_if_candidate_is_python(tmp_path):
    _,source,scope=_repo(tmp_path)
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    patch=_candidate(source.read_bytes(),scope,_fault(artifacts,scope))
    altered=PatchCandidate(patch.path,"sha256:"+"0"*64,patch.replacement,patch.evidence_ref,patch.hypothesis)
    with pytest.raises(RepairBoundaryError,match="stale"):
        altered.validate(scope,source.read_bytes())


def test_model_cannot_choose_its_own_file_or_suite(tmp_path):
    _,source,scope=_repo(tmp_path)
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    ref=_fault(artifacts,scope)
    patch=_candidate(source.read_bytes(),scope,ref)
    candidate={**asdict(patch),"path":"src/hephaestus/policy/action_registry.py"}
    with pytest.raises(RepairBoundaryError,match="outside operator"):
        BoundedPatchProposer(DeclaredModelResponseForContractTests(json.dumps(candidate))).propose(
            scope=scope,original=source.read_bytes(),
            verified_fault={"status":"verified_failed","target_file":scope.allowed_file},evidence_ref=ref)


def test_forged_failure_receipt_is_not_accepted(tmp_path):
    repo,source,scope=_repo(tmp_path)
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    fake_ref=artifacts.put_bytes(canonical_bytes({"schema":"reviewed-code-failure.v1",
        "status":"claimed_failed","target_file":scope.allowed_file})).artifact_ref
    patch=_candidate(source.read_bytes(),scope,fake_ref)
    repair=ConstrainedSelfRepair(artifacts,DeclaredFakeRunnerForContractTests([]))
    with pytest.raises(RepairBoundaryError,match="not independently"):
        repair.trial(checkout=repo,scope=scope,candidate=patch,verified_fault_ref=fake_ref)


def test_dirty_or_wrong_git_commit_refused_before_running_untrusted_code(tmp_path):
    repo,source,scope=_repo(tmp_path)
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    ref=_fault(artifacts,scope)
    candidate=_candidate(source.read_bytes(),scope,ref)
    runner=DeclaredFakeRunnerForContractTests([])
    repair=ConstrainedSelfRepair(artifacts,runner)
    source.write_text(source.read_text()+"\n# unauthorized source change\n")
    with pytest.raises(RepairBoundaryError,match="dirty"):
        repair.trial(checkout=repo,scope=scope,candidate=candidate,verified_fault_ref=ref)
    assert runner.runs==[]


def test_rejected_trial_cannot_be_published(tmp_path):
    repo,source,scope=_repo(tmp_path)
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    ref=_fault(artifacts,scope)
    patch=_candidate(source.read_bytes(),scope,ref)
    class AlwaysGreen:
        def run(self,checkout,*,fixed_test):
            return SandboxReceipt("passed",0,"sha256:"+"a"*64,"","offline-docker-pytest.v1")
    repair=ConstrainedSelfRepair(artifacts,AlwaysGreen())
    trial=repair.trial(checkout=repo,scope=scope,candidate=patch,verified_fault_ref=ref)
    assert trial["status"]=="rejected_patch"
    publisher=FakeReviewPublisher()
    with pytest.raises(RepairBoundaryError,match="no accepted"):
        repair.submit_for_review(scope=scope,candidate=patch,trial=trial,publisher=publisher)
    assert publisher.calls==[]


def test_review_publisher_cannot_write_if_remote_main_changed(tmp_path,monkeypatch):
    repo,source,scope=_repo(tmp_path)
    patch=_candidate(source.read_bytes(),scope,"sha256:"+"a"*64)
    publisher=GitHubReviewBranchPublisher("example","Hephaestus","dedicated-test-token")
    calls=[]
    def fake_request(self,method,route,body=None):
        calls.append((method,route,body))
        return {"object":{"sha":"1"*40}}
    monkeypatch.setattr(GitHubReviewBranchPublisher,"_request",fake_request)
    with pytest.raises(RepairBoundaryError,match="main advanced"):
        publisher.publish(scope=scope,candidate=patch,
            verified_trial={"status":"reviewable_patch","red_to_green_verified":True,
                            "replacement_sha256":"sha256:"+"b"*64})
    assert len(calls)==1 and calls[0][0]=="GET"
