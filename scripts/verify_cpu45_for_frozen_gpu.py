#!/usr/bin/env python3
"""Offline original five-role CUDA replay admission: inspect ONE authentic CPU 45/45.

Never invokes GPU/RunPod, grants no approvals and never manufactures a certificate.
An eligible CPU archive and signed original role registry still do not constitute
a GPU execution or independent historical-source review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from zipfile import ZipFile

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import PHASE_SUBSTEPS, WORKFLOW_RECORD_COLLECTION, WORKFLOW_STATE_COLLECTION

ROLE_NAMES = {"controller", "diagnosis", "planner", "evaluator", "judge"}
EXPECTED_RECIPE = "chat_json_r8_full_epoch_96_v1"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _files(archive):
    names = [x.filename for x in archive.infolist()]
    if len(set(names)) != len(names) or "archive_sha256.json" not in names:
        raise PermissionError("missing or duplicate CPU archive inventory")
    content = {}
    manifest = json.loads(archive.read("archive_sha256.json"))
    names = [x for x in names if not x.endswith("/")]
    if not isinstance(manifest, dict) or set(manifest) != set(names) - {"archive_sha256.json"}:
        raise PermissionError("original CPU archive inventory mismatches SHA index")
    for info in archive.infolist():
        name = info.filename
        if name.endswith("/"):
            continue
        if (name.startswith("/") or ".." in Path(name).parts or
                not isinstance(manifest.get(name), str) and name!="archive_sha256.json"):
            raise PermissionError("unsafe archive member")
        if name == "archive_sha256.json":
            continue
        with archive.open(info) as stream:
            digest = hashlib.sha256()
            while True:
                piece = stream.read(1024*1024)
                if not piece:
                    break
                digest.update(piece)
        if digest.hexdigest() != manifest[name]:
            raise PermissionError("original CPU archive byte SHA mismatch for "+name)
    return manifest


def verify_campaign_archive(path):
    with ZipFile(path) as z:
        manifest = _files(z)
        if not {"state.sqlite", "cpu_campaign_report.json"}.issubset(manifest):
            raise PermissionError("original SQLite and scientific report required")
        report = json.loads(z.read("cpu_campaign_report.json"))
        e = report.get("evidence") or {}
        t = e.get("actual_training") or {}
        p = e.get("prospective") or {}
        if (report.get("status")!="completed" or report.get("completed_operations")!=45 or
                e.get("verified_operations")!=45 or e.get("verified_domain_operations")!=40 or
                e.get("training_recipe")!=EXPECTED_RECIPE or
                t.get("training_recipe")!=EXPECTED_RECIPE or
                t.get("completed_optimizer_steps")!=96 or
                t.get("weights_changed") is not True or
                t.get("full_dataset_epoch_completed") is not True or
                len(set(t.get("consumed_sample_indices",[]))) != 96 or
                e.get("claim_scope")!="internal_prospective_generated_task_research" or
                e.get("five_frozen_cuda_roles_executed") is not False or
                e.get("independent_scientific_certification") is not False or
                e.get("new_independent_holdout_claimed") is not False or
                p.get("count")!=24):
            raise PermissionError("CPU archive is not one authentic completed native 96-update campaign")
        if set(e.get("research_role_inferences",[]))!=ROLE_NAMES:
            raise PermissionError("CPU archive missing original five advisory-role receipts")
        with tempfile.TemporaryDirectory() as tmp:
            db_path=Path(tmp)/"state.sqlite"
            db_path.write_bytes(z.read("state.sqlite"))
            db=sqlite3.connect(db_path.as_uri()+"?mode=ro&immutable=1", uri=True)
            try:
                if db.execute("PRAGMA quick_check").fetchone()[0]!="ok":
                    raise PermissionError("original CPU SQLite integrity failed")
                rows=db.execute("SELECT collection,payload_json FROM state_records").fetchall()
            finally:
                db.close()
    states=[json.loads(s) for kind,s in rows if kind==WORKFLOW_STATE_COLLECTION]
    workflow = report["workflow_id"]
    completed=[s for s in states if s.get("workflow_id")==workflow and
               s.get("completion_marker") is True and s.get("status")=="completed"]
    if not completed:
        raise ValueError("original SQLite lacks final completed native workflow")
    state=completed[-1]
    expected=[phase.value+"."+substep for phase in SPINE_ORDER for substep in PHASE_SUBSTEPS[phase]]
    steps=state.get("steps",{})
    if len(expected)!=45 or set(steps)!=set(expected):
        raise ValueError("original CPU workflow lacks exact native 45-stage binding")
    ids=set()
    for key in expected:
        step=steps[key]
        if step.get("status")!="completed" or step.get("completion_marker") is not True or not step.get("operation_id"):
            raise ValueError("original native operation not completed: "+key)
        ids.add(step["operation_id"])
    original_receipts = [json.loads(s) for kind,s in rows if kind == WORKFLOW_RECORD_COLLECTION]
    native = [r for r in original_receipts if r.get("workflow_id")==workflow and
              r.get("kind")=="substep_result" and r.get("payload",{}).get("completion_marker") is True]
    if len(ids)!=45 or len(native)!=45 or {r.get("operation_id") for r in native}!=ids:
        raise ValueError("45 unique original SQLite completion receipts required")
    original_steps=[json.loads(s) for kind,s in rows if kind=="cpu_research_optimizer_steps"]
    if len(original_steps)!=96 or set(x.get("completed_steps") for x in original_steps)!=set(range(1,97)):
        raise ValueError("all 96 real original optimizer receipts must exist")
    return {
        "source_cpu_archive_sha256":"sha256:"+_sha(Path(path).read_bytes()),
        "original_native_completed_operations":45,
        "original_domain_operations":40,
        "actual_optimizer_receipts":96,
        "native_trace_ref":e["native_trace_ref"],
        "policy_ref":e["policy_ref"],
        "run_id":state["run_id"],
        "baseline_correct":p["correct_before"],
        "candidate_correct":p["correct_after"],
        "baseline_json":p["json_before"],
        "candidate_json":p["json_after"],
        "candidate_checkpoint_sha256":t["adapter_sha256"],
        "all_original_source_members_verified":True,
    }


def verify_original_roles(path):
    raw=Path(path).read_bytes()
    record=json.loads(raw)
    roles=record.get("roles") or {}
    if (record.get("record_id")!="hephaestus-role-model-stack-v1-certification" or
            record.get("status")!="certified_and_artifact_verified" or set(roles)!=ROLE_NAMES or
            record.get("decision",{}).get("automatic_role_dispatch_enabled") is not False):
        raise PermissionError("frozen original five-role registry drift")
    for x in roles.values():
        if not isinstance(x.get("revision"),str) or len(x["revision"])!=40 or (
                not isinstance(x.get("adapter_sha256"),str) or len(x["adapter_sha256"])!=64):
            raise PermissionError("missing original frozen adapter or model revision")
    return {"original_cuda_role_registry_sha256":"sha256:"+_sha(raw),
            "original_cuda_roles":{k:{"model_id":v["model_id"],"revision":v["revision"],
                                     "adapter_sha256":v["adapter_sha256"]}
                                   for k,v in sorted(roles.items())}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cpu-archive", type=Path, required=True)
    parser.add_argument("--roles",type=Path,default=Path("docs/evidence/role-model-stack-v1/certification.json"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite original GPU acceptance proposal")
    measured=verify_campaign_archive(args.cpu_archive)
    registry=verify_original_roles(args.roles)
    intent={"schema":"original-five-cuda-replay-readiness.v1", **measured, **registry,
        "gpu_roles_must_execute_same_original_45_operation_graph":True,
        "corrected_target_model_separate_from_cuda_role_stack":True,
        "gpu_run_performed":False,"gpu_allocation_authorized":False,
        "external_certification_claimed":False,"external_original_private_reference_review":False,
        "historic_granite_comparisons_verified":False,"independent_signed_attestation_received":False,
        "issue_102_complete":False,
        "required_next_steps":["independently scoped original CUDA 45-stage composer with all five original adapters",
          "exact source policy, runtime quota and GPU allocation approvals",
          "complete real GPU execution with source and per-stage hashes",
          "human-authenticated historical train/development/holdout review and Granite comparisons"]}
    args.output.write_text(json.dumps(intent,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"cpu_native_operations":45,"gpu_executed":False,"certified":False}))


if __name__=="__main__":
    main()
