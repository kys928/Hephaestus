"""Single-scope provisional Coder qualification with non-propagating local errors.

Native services make the decisions, not the model's declaration:
independently reproduce defect -> admit authorized repair strategy ->
obtain an untrusted parent-Model patch -> run unchanged baseline and
candidate independently -> only then write a verified review handoff.

This proves the control boundary with test doubles, NOT actual original-model
inference or entire-stack science. Genuine parent inference is separately
required to claim the live capability level.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol
import json
import re
import subprocess
import tempfile

from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes,read_verified_json
from .constrained_repair import (
    ConstrainedSelfRepair, OfflineDockerPytestRunner, RepairBoundaryError,
    RepairScope, SandboxRunner, _prepare_copy,
)
from .exact_span_repair import ExactSpanPatchProposer
from .provisional_coder import ProvisionalMinistralCoder, PARENT_ID, PARENT_REVISION

_COMMIT=re.compile(r"[0-9a-f]{40}\Z")
_SHA=re.compile(r"sha256:[0-9a-f]{64}\Z")
_TEST_NODE=re.compile(
    r"tests/test_targeted_post_failure_probes\.py::test_[A-Za-z0-9_]+\Z")
_REGRESSION="tests/test_targeted_post_failure_probes.py"
_SOURCE="src/hephaestus/diagnosis/probes.py"
_COLLECTION="provisional_coder_qualifications"


@dataclass(frozen=True,slots=True)
class ReviewedRepairCase:
    """Signed/reviewed fixture identity is supplied by the trusted caller."""

    source_commit: str
    source_sha256: str
    failing_test_nodeid: str
    regression_entrypoint: str

    def validate(self) -> None:
        if (not _COMMIT.fullmatch(self.source_commit)
                or not _SHA.fullmatch(self.source_sha256)
                or not _TEST_NODE.fullmatch(self.failing_test_nodeid)
                or self.regression_entrypoint!="analyze_dataset_task_coverage"):
            raise RepairBoundaryError("unrecognized reviewed Diagnostician repair case")


@dataclass(slots=True)
class ProvisionalCoderQualification:
    """One fail-closed cycle; no test edit, merge or automatic training action."""

    artifacts: ArtifactStore
    repository: StateRepository
    runner: SandboxRunner
    proposer: ExactSpanPatchProposer
    maximum_attempts: int=1

    def run(self, *, objective: str, checkout: Path, case: ReviewedRepairCase,
            run_id: str, lineage_id: str) -> dict[str,object]:
        case.validate()
        if (not isinstance(objective,str) or not 16<=len(objective.strip())<=800
                or "diagnos" not in objective.lower()
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",run_id)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}",lineage_id)
                or type(self.maximum_attempts) is not int or not 1<=self.maximum_attempts<=2):
            raise RepairBoundaryError("provisional Coder objective/lineage/attempt scope invalid")
        original_root=Path(checkout)
        source=original_root/_SOURCE
        if source.is_symlink() or not source.is_file():
            raise RepairBoundaryError("no actual reviewed diagnostic source")
        actual=source.read_bytes()
        if "sha256:"+sha256(actual).hexdigest()!=case.source_sha256:
            raise RepairBoundaryError("source has changed since independent repair review")
        head=subprocess.run(["git","-C",str(original_root),"rev-parse","HEAD"],
            capture_output=True,text=True,timeout=10,check=False)
        dirty=subprocess.run(["git","-C",str(original_root),"status","--porcelain",
                              "--untracked-files=all"],
            capture_output=True,text=True,timeout=10,check=False)
        if (head.returncode!=0 or head.stdout.strip()!=case.source_commit
                or dirty.returncode!=0 or dirty.stdout.strip()):
            raise RepairBoundaryError("repair subject is not exact clean reviewed Git checkout")
        scope=RepairScope(case.source_commit,_SOURCE,_REGRESSION)
        scope.validate()
        key="qual-"+sha256(canonical_bytes({
            "case":case.__dict__ if hasattr(case,"__dict__") else {
                "source_commit":case.source_commit,"source_sha256":case.source_sha256,
                "failing_test_nodeid":case.failing_test_nodeid,
                "regression_entrypoint":case.regression_entrypoint},
            "objective":objective.strip(),"run_id":run_id,"lineage_id":lineage_id,
            "attempts":self.maximum_attempts,
        })).hexdigest()[:24]
        previous=self.repository.get_latest(_COLLECTION,"qualification_id",key)
        if previous is not None:
            report=read_verified_json(self.artifacts,str(previous["report_ref"]))
            if report.get("qualification_id")!=key:
                raise RepairBoundaryError("previous qualification identity differs")
            return report

        if type(self.runner) is not OfflineDockerPytestRunner:
            execution_scope="contract_test_double_not_real_sandbox"
        else:
            execution_scope="offline_rootless_docker"

        with tempfile.TemporaryDirectory(prefix="hephaestus-reviewed-fault-") as directory:
            copied=Path(directory)/"unchanged"
            copied.mkdir()
            _prepare_copy(original_root,copied)
            baseline=self.runner.run(copied,fixed_test=_REGRESSION)
        output=baseline.output_excerpt
        bad=(baseline.status!="failed" or baseline.exit_code!=1
             or baseline.worker!="offline-docker-pytest.v1"
             or case.failing_test_nodeid not in output
             or "AssertionError" not in output
             or any(text in output for text in (
                 "ERROR collecting","ImportError","ModuleNotFoundError",
                 "INTERNALERROR","no tests ran","OSError:")))
        if bad:
            # An import error, sandbox crash, missing test, or unrelated green
            # cannot authorize either the Coder or the next pipeline phase.
            report={
                "schema":"provisional-coder-qualification.v1",
                "qualification_id":key,"run_id":run_id,"lineage_id":lineage_id,
                "status":"blocked_no_authenticated_semantic_defect",
                "source_commit":case.source_commit,
                "original_source_sha256":case.source_sha256,
                "model_invocations":0,"verified_review_handoff_ref":None,
                "first_attempt_correct":False,"repaired_after_feedback":False,
                "real_parent_inference_proven":False,
                "stack_certified":False,
            }
        else:
            fault={
                "schema":"reviewed-code-failure.v1",
                "status":"verified_failed",
                "source_commit":case.source_commit,
                "source_sha256":case.source_sha256,
                "target_file":_SOURCE,"fixed_test":_REGRESSION,
                "failing_test_nodeid":case.failing_test_nodeid,
                "regression_entrypoint":case.regression_entrypoint,
                "baseline_test_output_sha256":baseline.output_sha256,
                "reviewed_strategy":"minimal_exact_span",
                "test_overwrite_authorized":False,
                "independent_scientific_certification":False,
            }
            evidence_ref=self.artifacts.put_bytes(
                canonical_bytes(fault),media_type="application/json").artifact_ref
            strategy={
                "schema":"admitted-repair-strategy.v1",
                "source_commit":case.source_commit,
                "source_sha256":case.source_sha256,
                "verified_fault_ref":evidence_ref,
                "allowed_file":_SOURCE,
                "fixed_test":_REGRESSION,
                "decision":"minimal_exact_span",
                "model_may_choose_tests":False,
                "model_may_edit_live_code":False,
                "automated_merge_allowed":False,
            }
            strategy_ref=self.artifacts.put_bytes(
                canonical_bytes(strategy),media_type="application/json").artifact_ref
            engine=ConstrainedSelfRepair(self.artifacts,self.runner)
            trials=[]
            feedback=[]
            seen=set()
            winner=None
            real_parent=False
            attempt_count=0
            for number in range(1,self.maximum_attempts+1):
                attempt_count=number
                proposal_fault={**fault,
                    "original_semantic_failure_excerpt":output[-1000:],
                    "previous_failed_patch_tests":feedback}
                try:
                    patch=self.proposer.propose(
                        scope=scope,original=actual,verified_fault=proposal_fault,
                        evidence_ref=evidence_ref)
                except (RepairBoundaryError,ValueError) as exc:
                    feedback.append({"attempt":number,"status":"invalid_model_proposal",
                                     "reason_type":type(exc).__name__})
                    break
                if patch.replacement in seen:
                    feedback.append({"attempt":number,"status":"duplicate_rejected"})
                    break
                seen.add(patch.replacement)
                coder=self.proposer.model
                receipt=getattr(coder,"last_receipt",None)
                validated_real=(type(coder) is ProvisionalMinistralCoder
                    and isinstance(receipt,dict)
                    and receipt.get("schema")=="provisional-parent-coder-inference.v1"
                    and receipt.get("model_id")==PARENT_ID
                    and receipt.get("revision")==PARENT_REVISION
                    and receipt.get("adapter_used") is False
                    and receipt.get("device")=="cuda:0"
                    and receipt.get("certified_coder") is False
                    and isinstance(receipt.get("output_sha256"),str)
                    and _SHA.fullmatch(receipt["output_sha256"]))
                if number==1:
                    real_parent=validated_real
                else:
                    real_parent=real_parent and validated_real
                receipt_ref=(self.artifacts.put_bytes(canonical_bytes(receipt),
                    media_type="application/json").artifact_ref if validated_real else None)
                trial=engine.trial(
                    checkout=original_root,scope=scope,candidate=patch,
                    verified_fault_ref=evidence_ref)
                trials.append({
                    "attempt":number,"trial_ref":trial["trial_ref"],
                    "coder_receipt_ref":receipt_ref,
                    "candidate_sha256":trial["replacement_sha256"],
                    "red_to_green_verified":trial["red_to_green_verified"],
                    "sandbox_scope":execution_scope,
                })
                if trial["red_to_green_verified"]:
                    winner=(number,trial,receipt_ref)
                    break
                feedback.append({
                    "attempt":number,"status":"failed_independent_sandbox",
                    "tested_patch_sha256":trial["replacement_sha256"],
                    "sandbox_failure_excerpt":trial["candidate"]["output_excerpt"][-1200:],
                    "instruction":"The original regression is still failing. Propose a distinct fix.",
                })
            accepted=winner is not None and execution_scope=="offline_rootless_docker"
            reviewed_handoff_ref=None
            if accepted:
                number,trial,receipt_ref=winner
                handoff={
                    "schema":"coder-review-handoff.v1",
                    "source_commit":case.source_commit,
                    "source_sha256":case.source_sha256,
                    "run_id":run_id,"lineage_id":lineage_id,
                    "strategy_ref":strategy_ref,"fault_ref":evidence_ref,
                    "trial_ref":trial["trial_ref"],
                    "test_nodeid":case.failing_test_nodeid,
                    "accepted_candidate_sha256":trial["replacement_sha256"],
                    "attempt":number,
                    "authentic_parent_coder_inference":real_parent,
                    "coder_inference_receipt_ref":receipt_ref,
                    "first_attempt_correct":number==1,
                    "automated_merge_authorized":False,
                    "production_source_mutated":False,
                    "full_stack_certified":False,
                }
                reviewed_handoff_ref=self.artifacts.put_bytes(
                    canonical_bytes(handoff),media_type="application/json").artifact_ref
            report={
                "schema":"provisional-coder-qualification.v1",
                "qualification_id":key,
                "run_id":run_id,"lineage_id":lineage_id,
                "source_commit":case.source_commit,
                "original_source_sha256":case.source_sha256,
                "status":("verified_review_handoff" if accepted
                          else "blocked_unverified_candidate"),
                "verified_fault_ref":evidence_ref,
                "authorized_strategy_ref":strategy_ref,
                "model_invocations":attempt_count,
                "first_attempt_correct":bool(accepted and winner[0]==1),
                "repaired_after_feedback":bool(accepted and winner[0]>1),
                "trial_attempts":trials,
                "verified_review_handoff_ref":reviewed_handoff_ref,
                "real_parent_inference_proven":real_parent,
                "actual_offline_docker_verified":execution_scope=="offline_rootless_docker",
                "stack_certified":False,
                "automatic_merge_authorized":False,
            }
        published=self.artifacts.put_bytes(
            canonical_bytes(report),media_type="application/json")
        self.repository.append(_COLLECTION,{
            "qualification_id":key,"report_ref":published.artifact_ref,
            "status":report["status"]})
        return report

    def verify_review_handoff(self, report: dict[str,object]) -> dict[str,object]:
        ref=report.get("verified_review_handoff_ref")
        if not isinstance(ref,str) or not _SHA.fullmatch(ref):
            raise PermissionError("no verified result may cross phase boundary")
        handoff=read_verified_json(self.artifacts,ref)
        if (handoff["schema"]!="coder-review-handoff.v1"
                or handoff["source_commit"]!=report["source_commit"]
                or handoff["run_id"]!=report["run_id"]
                or handoff["lineage_id"]!=report["lineage_id"]
                or handoff["full_stack_certified"] is not False
                or handoff["automated_merge_authorized"] is not False):
            raise PermissionError("review handoff has changed identity or authority")
        trial=read_verified_json(self.artifacts,str(handoff["trial_ref"]))
        strategy=read_verified_json(self.artifacts,str(handoff["strategy_ref"]))
        fault=read_verified_json(self.artifacts,str(handoff["fault_ref"]))
        if (trial.get("status")!="reviewable_patch"
                or trial.get("red_to_green_verified") is not True
                or trial.get("replacement_sha256")!=handoff["accepted_candidate_sha256"]
                or trial.get("verified_fault_ref")!=handoff["fault_ref"]
                or strategy.get("verified_fault_ref")!=handoff["fault_ref"]
                or strategy.get("decision")!="minimal_exact_span"
                or fault.get("failing_test_nodeid")!=handoff["test_nodeid"]):
            raise PermissionError("review handoff lacks original authenticated red-to-green evidence")
        return handoff
