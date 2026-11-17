"""Opt-in bounded autonomous code-repair episode with no manually authored case list.

The controller discovers an actual red regression by executing one registered
suite in the external sandbox. It records a source-hashed fault, requests one
model-generated candidate, independently reexecutes red->green, and submits
a *draft review PR only*. No unrestricted loops or mutation of running code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json
import re
import subprocess
import urllib.request
import urllib.error

from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from .constrained_repair import (
    BoundedPatchProposer, ConstrainedSelfRepair, GitHubPatchPublisher,
    RepairBoundaryError, RepairScope, SandboxRunner, _prepare_copy,
)

_REPAIR_EPISODES="autonomous_diagnosis_repair_episodes"
# This registry is controlled by source review. The model cannot broaden it.
_APPROVED_REPAIR_TARGETS = {
    "diagnosis_coverage_probe": (
        "src/hephaestus/diagnosis/probes.py",
        "tests/test_targeted_post_failure_probes.py",
    ),
}


@dataclass(slots=True)
class LoopbackOllamaPatchModel:
    """Optional local coder; text proposal only, no tool or filesystem access."""

    model: str
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    timeout_seconds: int = 120
    last_receipt: dict[str,object] | None = field(default=None,init=False,repr=False)

    def generate_patch_json(self, prompt: str) -> str:
        if (not re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}",self.model)
                or self.endpoint != "http://127.0.0.1:11434/api/generate"):
            raise RepairBoundaryError("coder endpoint/model outside fixed loopback policy")
        request=urllib.request.Request(self.endpoint,
            data=canonical_bytes({
                "model":self.model,"prompt":prompt,"stream":False,
                "format":"json","options":{"temperature":0,"num_predict":1024},
            }),
            headers={"Content-Type":"application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request,timeout=self.timeout_seconds) as result:
                raw=result.read(150_001)
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError):
            raise RepairBoundaryError("reviewed local coder inference unavailable") from None
        if len(raw)>150_000:
            raise RepairBoundaryError("local coder response too large")
        doc=json.loads(raw)
        if (not isinstance(doc,dict) or doc.get("model")!=self.model
                or doc.get("done") is not True
                or not isinstance(doc.get("response"),str)
                or type(doc.get("eval_count")) is not int
                or doc["eval_count"]<1
                or len(doc["response"])>16000):
            raise RepairBoundaryError("local coder inference receipt lacks matching actual model completion")
        self.last_receipt={
            "schema":"real-local-coder-inference.v1",
            "model":doc["model"],
            "prompt_sha256":"sha256:"+sha256(prompt.encode()).hexdigest(),
            "output_sha256":"sha256:"+sha256(doc["response"].encode()).hexdigest(),
            "generated_token_count":doc["eval_count"],
            "prompt_eval_count":doc.get("prompt_eval_count"),
            "total_duration_ns":doc.get("total_duration"),
            "source":"loopback-ollama-json-generator",
            "independent_scientific_certification":False,
        }
        return doc["response"]


@dataclass(slots=True)
class AutonomousDiagnosisRepairEpisode:
    """One code repair attempt. Allowed scope is registered, not language-inferred."""

    artifacts: ArtifactStore
    repository: StateRepository
    runner: SandboxRunner
    proposer: BoundedPatchProposer
    publisher: GitHubPatchPublisher
    maximum_model_attempts: int = 1

    def run(self, objective: str, *, checkout: Path) -> dict[str, object]:
        if (not isinstance(objective,str) or not 16<=len(objective.strip())<=800
                or "diagnos" not in objective.casefold()
                or not any(word in objective.casefold() for word in ("repair","fix","investigat"))):
            raise RepairBoundaryError("unsupported autonomous self-repair research objective")
        if type(self.maximum_model_attempts) is not int or not 1<=self.maximum_model_attempts<=2:
            raise RepairBoundaryError("bounded self-repair permits one or two model attempts only")
        root=Path(checkout)
        head=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
            capture_output=True,text=True,timeout=10,check=False)
        if head.returncode!=0 or not re.fullmatch(r"[a-f0-9]{40}",head.stdout.strip()):
            raise RepairBoundaryError("repair requires original Git checkout")
        dirty=subprocess.run(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],
            capture_output=True,text=True,timeout=10,check=False)
        if dirty.returncode!=0 or dirty.stdout.strip():
            raise RepairBoundaryError("repair requires clean reviewed Git checkout")
        target, fixed_test = _APPROVED_REPAIR_TARGETS["diagnosis_coverage_probe"]
        scope=RepairScope(head.stdout.strip(),target,fixed_test)
        scope.validate()
        source=root/target
        if source.is_symlink() or not source.is_file():
            raise RepairBoundaryError("registered repair source unavailable")
        original=source.read_bytes()
        episode_id="repair-"+sha256(canonical_bytes({
            "goal":objective.strip(),"commit":scope.base_commit,
            "source_sha256":sha256(original).hexdigest(),"target":target,
            "maximum_model_attempts":self.maximum_model_attempts,
        })).hexdigest()[:24]
        prior=self.repository.get_latest(_REPAIR_EPISODES,"episode_id",episode_id)
        if prior is not None:
            report=read_verified_json(self.artifacts,str(prior.get("report_ref")))
            if report.get("episode_id")!=episode_id:
                raise RepairBoundaryError("previous repair episode identity drift")
            return report

        # The checker does not receive model output. We run on a copy, never
        # execute potentially patched source in the main production process.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="hephaestus-fault-discovery-") as tmp:
            copy=Path(tmp)/"source"
            copy.mkdir()
            _prepare_copy(root,copy)
            baseline=self.runner.run(copy,fixed_test=fixed_test)
        if baseline.worker!="offline-docker-pytest.v1":
            raise RepairBoundaryError("source failure not verified by the reviewed sandbox class")
        if baseline.status=="passed":
            report={
                "schema":"autonomous-diagnosis-repair-episode.v1",
                "episode_id":episode_id,"status":"no_verified_regression",
                "objective":objective.strip(),"target":target,
                "source_commit":scope.base_commit,
                "baseline_test_receipt_sha256":baseline.output_sha256,
                "model_invocations":0,"production_mutated":False,
                "pull_request":None,"independent_scientific_certification":False,
            }
        elif baseline.status!="failed":
            raise RepairBoundaryError("sandbox baseline uncertain; code repair not permitted")
        else:
            fault={
                "schema":"reviewed-code-failure.v1",
                "status":"verified_failed","target_file":target,
                "fixed_test":fixed_test,"source_commit":scope.base_commit,
                "source_sha256":"sha256:"+sha256(original).hexdigest(),
                "baseline_test_receipt_sha256":baseline.output_sha256,
                "independent_scientific_certification":False,
            }
            # Model text may improve after receiving its OWN failed sandbox
            # result, but cannot modify fault authority or the unchanged test.
            failure_ref=self.artifacts.put_bytes(canonical_bytes(fault),media_type="application/json").artifact_ref
            repair_engine=ConstrainedSelfRepair(self.artifacts,self.runner)
            trials=[]
            seen_replacements=set()
            feedback=[]
            accepted=None
            attempt_count=0
            for attempt in range(1,self.maximum_model_attempts+1):
                attempt_count=attempt
                contextual_fault={**fault,
                    "original_regression_output_excerpt":baseline.output_excerpt[-1400:],
                    "prior_failed_model_attempts":feedback}
                candidate=self.proposer.propose(
                    scope=scope,original=original,
                    verified_fault=contextual_fault,evidence_ref=failure_ref)
                replacement_digest="sha256:"+sha256(candidate.replacement.encode()).hexdigest()
                if replacement_digest in seen_replacements:
                    # Another copy of a failed patch is not a new experiment.
                    break
                seen_replacements.add(replacement_digest)
                trial=repair_engine.trial(
                    checkout=root,scope=scope,candidate=candidate,
                    verified_fault_ref=failure_ref)
                trials.append(trial["trial_ref"])
                if trial["red_to_green_verified"]:
                    accepted=(candidate,trial)
                    break
                feedback.append({
                    "attempt":attempt,
                    "model_hypothesis":candidate.hypothesis,
                    "candidate_replacement_sha256":replacement_digest,
                    "candidate_test_status":trial["candidate"]["status"],
                    "candidate_failure_output_excerpt":
                        trial["candidate"]["output_excerpt"][-1400:],
                    "instruction":"Do not repeat the failed patch. The original regression remains red.",
                })
            url=None
            if accepted is not None:
                candidate,trial=accepted
                url=repair_engine.submit_for_review(
                    scope=scope,candidate=candidate,trial=trial,publisher=self.publisher)
            report={
                "schema":"autonomous-diagnosis-repair-episode.v1",
                "episode_id":episode_id,
                "status":"draft_review_pr_created" if url else "repair_candidate_rejected",
                "objective":objective.strip(),"target":target,"source_commit":scope.base_commit,
                "verified_fault_ref":failure_ref,
                "trial_ref":trials[-1] if trials else None,
                "trial_refs":trials,
                "baseline_test_receipt_sha256":baseline.output_sha256,
                "model_invocations":attempt_count,
                "maximum_model_attempts":self.maximum_model_attempts,
                "production_mutated":False,
                "pull_request":url,"independent_scientific_certification":False,
            }
        evidence=self.artifacts.put_bytes(canonical_bytes(report),media_type="application/json")
        self.repository.append(_REPAIR_EPISODES,{
            "episode_id":episode_id, "report_ref":evidence.artifact_ref,
            "status":report["status"],
        })
        return report
