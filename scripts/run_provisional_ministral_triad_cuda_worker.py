#!/usr/bin/env python3
"""Single approved, authentic three-model CUDA research experiment.

This Pod holds no API credentials, performs no model-proposed shell execution,
and writes only research artifacts. Off-Pod independent Docker decides whether
any patch is correct. This is an integration qualification, NOT general science
or automatic model certification/promotion.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import argparse
import json
import os
import re
import time
import traceback

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from hephaestus.llm.role_contract import RoleEvidence, RoleTask
from hephaestus.production.original_cuda_identity import registry_digest, verified_original_roles
from hephaestus.research.constrained_repair import RepairScope
from hephaestus.research.exact_span_repair import ExactSpanPatchProposer
from hephaestus.research.provisional_coder import ProvisionalMinistralCoder
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


SOURCE="src/hephaestus/diagnosis/probes.py"
MODEL_CACHE=Path("/tmp/hephaestus-triad-hf-model-cache")
ADAPTER_CACHE=Path("/tmp/hephaestus-triad-role-adapters")
_ALLOWED_ROOT=Path("/workspace/hephaestus/operations/ministral-triad")


class MountedOriginalVolumeGet:
    """Read-only S3-get-shaped accessor, constrained to one mounted original volume."""

    def get_object(self, *, Bucket: str, Key: str):
        if (Bucket!="cviwpryzao" or not isinstance(Key,str)
                or not Key.startswith("hephaestus/scientific/v1/role_mastery/")
                or ".." in Key.split("/") or Key.startswith("/")):
            raise PermissionError("worker cannot read arbitrary mounted source")
        path=Path("/workspace")/Key
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("trusted role adapter volume object missing")
        return {"Body":BytesIO(path.read_bytes())}


def put_json(root: Path, name: str, obj: dict):
    target=root/name
    if target.exists() or not re.fullmatch(r"[a-z0-9._-]+\.json",name):
        raise PermissionError("cannot replace immutable triad research output")
    encoded=canonical_bytes(obj)
    temporary=root/(name+".partial")
    with temporary.open("xb") as out:
        out.write(encoded)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary,target)
    return "sha256:"+sha256(encoded).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-tag",required=True)
    p.add_argument("--source-commit",required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[0-9]{6,20}",a.run_tag) or not re.fullmatch(r"[0-9a-f]{40}",a.source_commit):
        raise PermissionError("worker identity must match exact operator run and source")
    work=_ALLOWED_ROOT/a.run_tag
    if not work.is_dir() or work.is_symlink():
        raise PermissionError("protected operator-mounted run directory unavailable")
    case=json.loads((work/"frozen-case.json").read_bytes())
    if (case.get("schema")!="heph-real-triad-approved-case-v1"
            or case.get("run_tag")!=a.run_tag
            or case.get("source_commit")!=a.source_commit
            or case.get("source_file")!=SOURCE
            or case.get("issue_nodeid")!="tests/test_targeted_post_failure_probes.py::test_null_text_is_not_literal_supervised_text"
            or not re.fullmatch(r"sha256:[a-f0-9]{64}",str(case.get("source_sha256","")))
            or not re.fullmatch(r"sha256:[a-f0-9]{64}",str(case.get("fixed_test_sha256","")))):
        raise PermissionError("operator-frozen triad case is incomplete")
    root=Path(__file__).resolve().parents[1]
    source=(root/SOURCE).read_bytes()
    if sha256(source).hexdigest()!=case["source_sha256"][7:]:
        raise PermissionError("worker source digest differs from admitted original")
    if b'text = str(row.get("text", ""))' not in source:
        raise PermissionError("original genuine production defect absent from reviewed source")
    result={
        "schema":"real-original-ministral-triad-worker-v1",
        "run_tag":a.run_tag,
        "source_commit":a.source_commit,
        "source_sha256":case["source_sha256"],
        "original_five_role_registry_sha256":registry_digest(),
        "coder_uses_planner_lora":False,
        "code_modified":False,
        "automatic_promotion":False,
        "steps":[],
        "status":"in_progress",
    }
    started=time.monotonic()
    artifacts=FileSystemArtifactStore(work/"model-artifacts")
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise PermissionError("one genuine CUDA device required; no CPU substitute")
        result["gpu_model"]=torch.cuda.get_device_name(0)
        result["gpu_compute_capability"]=list(torch.cuda.get_device_capability(0))
        stack,contract=verified_original_roles(registry_digest())
        ref="sha256:"+sha256(canonical_bytes(case)).hexdigest()
        evidence=EvidenceRecord(
            ref=ref,kind="artifact",
            run_id="triad-"+a.run_tag,lineage_id="research-"+a.run_tag,
            sha256=ref,observed_sha256=ref,
            facts={"provenance_valid":True,"artifact_hash_match":True,
                   "evidence_complete":True})
        reg=EvidenceRegistry({ref:evidence})
        scene=(
            "Actual reviewed production defect in analyze_dataset_task_coverage: "
            "for the original source line text = str(row.get('text', '')), a "
            "record with text=None is incorrectly counted as containing supervised "
            "text. An independent frozen pytest regression reproduces the incorrect "
            "text_rows=1 rather than expected text_rows=0. This is a bounded "
            "evidence-based code repair investigation; do not invent a successful "
            "fix or select new data. Diagnose source semantics. "
            f"Source SHA256: {case['source_sha256']}; reviewed failure node: {case['issue_nodeid']}."
        )
        generator=PinnedTransformersRoleGenerator(
            MODEL_CACHE,device="cuda:0",allow_remote_pinned_download=True)
        reasoner=CertifiedRoleReasoner(
            stack=stack,contract=contract,
            adapters=VerifiedRoleAdapters(MountedOriginalVolumeGet(),"cviwpryzao",ADAPTER_CACHE),
            generator=generator,artifact_store=artifacts)
        previous=None
        for role in ("diagnosis","planner"):
            situation=scene
            if role=="planner":
                situation+=(
                    "\nThe original Diagnostician's observed advisory diagnosis is "
                    +json.dumps(previous,sort_keys=True)
                    +"\nThe operator permits ONLY a single minimal exact-span edit to "
                    "the in-scope diagnosis Python function, with unchanged tests and "
                    "external independent red-to-green Docker verification. "
                    "Choose an appropriate bounded intervention consistent with the "
                    "frozen Planner action vocabulary. Your answer is advisory; "
                    "the original finite control policy decides admission."
                )
            task=RoleTask(
                role=role,situation=situation,
                evidence=(RoleEvidence(ref,scene[:2200]),),
                run_id="triad-"+a.run_tag,lineage_id="research-"+a.run_tag,
                stage_name="bounded_diagnosis_code_repair")
            context=DeterministicRoleContext(
                run_id=task.run_id,lineage_id=task.lineage_id,
                stage_name=task.stage_name,
                approval_status="approved_for_review_only",
                stage_allowed_actions=("request_recheck",),
                verified_facts={"evidence_complete":True,
                                "provenance_valid":True,
                                "artifact_hash_match":True})
            observation=reasoner.propose(task,reg,context=context,
                max_new_tokens=384)
            previous=observation.answer.to_dict()
            result["steps"].append({
                "role":role,"model_id":observation.model_id,
                "revision":observation.revision,
                "adapter_sha256":observation.adapter_sha256,
                "inference_receipt_ref":observation.inference_receipt_ref,
                "model_decision":previous,
                "real_inference":True,
            })
            put_json(work,role+"-observation.json",result["steps"][-1])
        generator.close()
        del generator,reasoner
        import gc
        gc.collect()
        coder=ProvisionalMinistralCoder(
            MODEL_CACHE,registry_digest(),maximum_new_tokens=768)
        proposer=ExactSpanPatchProposer(coder)
        verified_fault={
            "schema":"reviewed-code-failure.v1",
            "status":"verified_failed",
            "target_file":SOURCE,
            "regression_entrypoint":"analyze_dataset_task_coverage",
            "fixed_test":"tests/test_targeted_post_failure_probes.py",
            "failing_test_nodeid":case["issue_nodeid"],
            "source_sha256":case["source_sha256"],
            "original_semantic_failure_excerpt":case["baseline_failure_excerpt"],
            "diagnosis_observation":result["steps"][0]["model_decision"],
            "planner_observation":result["steps"][1]["model_decision"],
            "reviewed_strategy":"minimal_exact_span",
            "independent_test_mutation_authorized":False,
        }
        # This SHA is operator-admitted evidence, not a model-authored action.
        fault_ref="sha256:"+sha256(canonical_bytes(verified_fault)).hexdigest()
        put_json(work,"verified-fault.json",verified_fault)
        patch=proposer.propose(
            scope=RepairScope(a.source_commit,SOURCE,
                "tests/test_targeted_post_failure_probes.py"),
            original=source,verified_fault=verified_fault,evidence_ref=fault_ref)
        inference=coder.last_receipt
        if (not isinstance(inference,dict)
                or inference.get("model_id")!="mistralai/Ministral-3-8B-Reasoning-2512"
                or inference.get("adapter_used") is not False):
            raise PermissionError("real coder inference receipt invalid")
        put_json(work,"coder-observation.json",{
            "model_id":inference["model_id"],
            "revision":inference["revision"],
            "receipt":inference,
            "patch_hypothesis":patch.hypothesis,
            "patch_source_sha256":patch.expected_source_sha256,
            "patched_sha256":"sha256:"+sha256(patch.replacement.encode()).hexdigest(),
        })
        put_json(work,"candidate.json",{
            "path":patch.path,
            "expected_source_sha256":patch.expected_source_sha256,
            "replacement":patch.replacement,
            "hypothesis":patch.hypothesis,
            "evidence_ref":patch.evidence_ref,
        })
        result["steps"].append({
            "role":"coder","model_id":inference["model_id"],
            "revision":inference["revision"],
            "adapter_used":False,
            "prompt_sha256":inference["prompt_sha256"],
            "output_sha256":inference["output_sha256"],
            "real_inference":True,
            "independently_verified":False,
        })
        result["status"]="candidate_proposed_pending_offpod_verification"
        coder.close()
    except Exception as exc:
        result["status"]="inconclusive_failure"
        result["error_class"]=type(exc).__name__
        # Do not write secrets/raw exception text into the public CI output.
        raise
    finally:
        result["elapsed_seconds"]=round(time.monotonic()-started,3)
        result["model_success_count"]=len(result["steps"])
        put_json(work,"worker-result.json",result)
        print("AUTHENTIC_TRIAD_WORKER_RESULT="+result["status"],flush=True)
        print("AUTHENTIC_TRIAD_MODELS_SUCCEEDED="+str(result["model_success_count"]),flush=True)


if __name__=="__main__":
    main()
