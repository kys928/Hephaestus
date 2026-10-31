#!/usr/bin/env python3
"""Independent cold ZIP readback of an original full five-role CUDA campaign.

Does not run training or use the original live SQLite/optimizer services.
Only existing immutable source members, SHA commitments, original completion
rows and real CUDA training/inference receipts can produce a positive result.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import sqlite3
import tempfile
from zipfile import ZipFile
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_state import PHASE_SUBSTEPS,WORKFLOW_RECORD_COLLECTION,WORKFLOW_STATE_COLLECTION
from hephaestus.production.signed_cuda_grants import expected_gpu_operation_map
from hephaestus.production.original_cuda_identity import verified_original_roles,registry_digest
from hephaestus.storage.verified import canonical_bytes


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_original_role_receipt(row, *, artifact, frozen, contract, state):
    """Reparse original bytes and bind accepted/rejected attempts to one source.

    A status flag is not sufficient scientific evidence. This CPU readback
    checks the frozen parser and CUDA provenance without executing a model.
    """
    role=row["role"]
    source_ref=row["domain_observation_ref"]
    source=artifact(source_ref)
    if (source.get("schema")!="tier-a-gpu-role-domain-observation.v1"
            or source.get("workflow_id")!=state["workflow_id"]
            or source.get("operation_id")!=row["operation_id"]
            or any(source.get(k)!=state[k] for k in ("run_id","lineage_id","stage_name"))):
        raise PermissionError("cold CUDA role source belongs to a different operation")
    receipt=artifact(row["inference_receipt_ref"])
    def provenance(doc):
        execution=doc.get("execution",{})
        if (doc.get("schema")!="role-inference.v1" or doc.get("role")!=role
                or any(doc.get(k)!=state[k] for k in ("run_id","lineage_id","stage_name"))
                or doc.get("model_id")!=frozen.model_id
                or doc.get("base_revision")!=frozen.revision
                or doc.get("adapter_sha256")!=frozen.adapter.sha256
                or doc.get("adapter_source")!=frozen.adapter.s3_key
                or doc.get("tokenizer_id")!=frozen.model_id
                or doc.get("tokenizer_revision")!=frozen.revision
                or doc.get("profile")!=contract.profile(role)
                or doc.get("evidence_refs")!=[source_ref]
                or doc.get("generator")!="PinnedTransformersRoleGenerator"
                or execution.get("device")!="cuda:0"
                or not execution.get("gpu_name") or not execution.get("cuda_version")):
            raise PermissionError("cold original role source/model/CUDA provenance mismatch")
        for key in ("input_token_ids","generated_token_ids"):
            ids=execution.get(key)
            if not isinstance(ids,list) or not ids or any(type(v) is not int or v<0 for v in ids):
                raise PermissionError("cold original CUDA inference lacks numeric token evidence")
        for text_key,digest_key in (("prompt","prompt_sha256"),("raw_output","output_sha256")):
            if not isinstance(doc.get(text_key),str) or _sha(doc[text_key].encode())!=doc.get(digest_key):
                raise PermissionError("cold original CUDA inference text digest mismatch")
    provenance(receipt)
    answer=contract.parse(role,receipt["raw_output"],{source_ref})
    if receipt.get("status")!="accepted" or receipt.get("answer")!=answer.to_dict():
        raise PermissionError("cold original accepted answer differs from strict raw parse")
    rejected_refs=row.get("rejected_inference_receipt_refs",[])
    if (not isinstance(rejected_refs,list) or len(rejected_refs)>1
            or row.get("bounded_format_repair_attempts",0)!=len(rejected_refs)):
        raise PermissionError("cold original role exceeded the single format retry")
    for ref in rejected_refs:
        rejected=artifact(ref)
        provenance(rejected)
        if (ref==row["inference_receipt_ref"] or rejected.get("status")!="rejected"
                or rejected.get("seed")!=receipt.get("seed")
                or rejected.get("max_new_tokens")!=receipt.get("max_new_tokens")
                or rejected["execution"].get("decoding")!=receipt["execution"].get("decoding")):
            raise PermissionError("cold rejected CUDA inference retry provenance mismatch")
        try:
            contract.parse(role,rejected["raw_output"],{source_ref})
        except ValueError as exc:
            expected=("role response has invalid primary_variable" if role=="controller" else
                      "role response cites missing, duplicate or empty evidence")
            if str(exc)!=expected or rejected.get("error")!=expected:
                raise PermissionError("cold CUDA retry has an unapproved original rejection") from exc
        else:
            raise PermissionError("cold rejected CUDA inference was not actually rejected")

def verify_original_cuda_archive(path, *, expected_sha256):
    if "sha256:"+_sha(Path(path).read_bytes())!=expected_sha256:
        raise PermissionError("independently pinned remote GPU archive hash mismatch")
    with ZipFile(path) as z:
        members=[i.filename for i in z.infolist() if not i.is_dir()]
        if (len(set(members))!=len(members) or
                "archive_sha256.json" not in members):
            raise PermissionError("duplicate/missing original archive inventory")
        index=json.loads(z.read("archive_sha256.json"))
        if not isinstance(index,dict) or set(index)!=set(members)-{"archive_sha256.json"}:
            raise PermissionError("original GPU archive member set differs")
        for item in z.infolist():
            if item.is_dir():
                continue
            name=item.filename
            if name.startswith("/") or ".." in PurePosixPath(name).parts:
                raise PermissionError("unsafe GPU evidence archive member")
            if name=="archive_sha256.json":
                continue
            digest=hashlib.sha256()
            with z.open(item) as stream:
                for piece in iter(lambda:stream.read(1024*1024),b""):
                    digest.update(piece)
            if digest.hexdigest()!=index[name]:
                raise PermissionError("original CUDA evidence member SHA mismatch "+name)
        reports=[]
        for name in index:
            if name.startswith("cuda-native-report-") and name.endswith(".json"):
                if _sha(z.read(name))!=name[len("cuda-native-report-"):-len(".json")]:
                    raise PermissionError("GPU report filename hash mismatch")
                d=json.loads(z.read(name))
                if d.get("status")=="completed" and d.get("completed_operations")==45:
                    reports.append(d)
        if len(reports)!=1 or not isinstance(reports[0].get("verified"),dict):
            raise PermissionError("exactly one actual verified GPU 45/45 report required")
        report=reports[0];e=report["verified"]
        if (e.get("schema")!="tier-a-native-original-five-role-cuda-verified-45.v1"
                or e.get("actual_cuda_optimizer_steps")!=96
                or e.get("native_operations")!=45
                or e.get("independent_certification") is not False
                or e.get("protected_historical_references_reviewed") is not False):
            raise PermissionError("claimed original GPU science not scoped to generated research")
        with tempfile.TemporaryDirectory() as tmp:
            dbpath=Path(tmp)/"state.sqlite"
            dbpath.write_bytes(z.read("state.sqlite"))
            con=sqlite3.connect("file:"+str(dbpath)+"?mode=ro&immutable=1",uri=True)
            try:
                if con.execute("PRAGMA quick_check").fetchone()[0]!="ok":
                    raise PermissionError("original archived GPU SQLite integrity failed")
                rows=[(name,json.loads(payload)) for name,payload in con.execute(
                    "SELECT collection,payload_json FROM state_records ORDER BY sequence").fetchall()]
            finally:
                con.close()
        states=[r for k,r in rows if k==WORKFLOW_STATE_COLLECTION and
                r.get("workflow_id")==report["workflow_id"] and
                r.get("completion_marker") is True and r.get("status")=="completed"]
        if not states:
            raise PermissionError("original real GPU run lacks native completed SQLite state")
        state=states[-1]
        if state.get("run_id")!=report["run_id"]:
            raise PermissionError("archived GPU scientific run ID differs")
        expected=[(p.value,s) for p in SPINE_ORDER for s in PHASE_SUBSTEPS[p]]
        if len(expected)!=45 or set(state.get("steps",{}))!={
                p+"."+s for p,s in expected}:
            raise PermissionError("original native GPU graph changed")
        operation_ids={}
        for p,s in expected:
            step=state["steps"][p+"."+s]
            if (step.get("phase")!=p or step.get("substep")!=s
                    or step.get("status")!="completed"
                    or step.get("completion_marker") is not True):
                raise PermissionError("uncompleted original GPU native stage "+p+"."+s)
            operation_ids[step["operation_id"]]=p+"."+s
        if len(operation_ids)!=45:
            raise PermissionError("duplicate original CUDA native operation")
        evidence=[r for k,r in rows if k==WORKFLOW_RECORD_COLLECTION and
                  r.get("workflow_id")==report["workflow_id"]]
        completions=[r for r in evidence if r.get("kind")=="substep_result" and
                     r.get("payload",{}).get("completion_marker") is True]
        if len(completions)!=45 or set(r.get("operation_id") for r in completions)!=set(operation_ids):
            raise PermissionError("45 original native SQLite completion receipts missing")
        if [(r.get("phase"),r.get("substep")) for r in completions]!=expected:
            raise PermissionError("original CUDA operations did not complete in dependency order")
        # Cross-check the exact original recorded operation with immutable SHA object bytes.
        def artifact(ref):
            if not isinstance(ref,str) or not ref.startswith("sha256:"):
                raise PermissionError("invalid content-addressed original GPU artifact")
            digest=ref[7:]
            name=f"artifacts/objects/sha256/{digest[:2]}/{digest}"
            if name not in index:
                raise PermissionError("original GPU artifact not present in byte-verified archive")
            value=z.read(name)
            if _sha(value)!=digest:
                raise PermissionError("original GPU artifact digest mismatch")
            return json.loads(value)
        for row in completions:
            expected_row={key:value for key,value in row.items() if key!="artifact_ref"}
            if artifact(row["artifact_ref"])!=expected_row:
                raise PermissionError("original native CUDA completion diverges from immutable artifact")
        role_rows=[r["payload"] for r in evidence if r.get("kind")=="original_cuda_advisory_role"]
        frozen_stack,contract=verified_original_roles(registry_digest())
        approved=expected_gpu_operation_map(report["workflow_id"])
        actual_roles={role for role in approved.values() if role not in {
            "target_evaluation","target_training"}}
        if len(role_rows)!=5 or {r.get("role") for r in role_rows}!=actual_roles:
            raise PermissionError("actual original five CUDA role receipt set differs")
        for row in role_rows:
            frozen=frozen_stack.resolve(row["role"])
            evidence_role=artifact(row["inference_receipt_ref"])
            if (row.get("model_id")!=frozen.model_id
                    or row.get("base_revision")!=frozen.revision
                    or row.get("adapter_sha256")!=frozen.adapter.sha256
                    or row.get("device")!="cuda:0"
                    or row.get("advisory_only") is not True
                    or evidence_role.get("status")!="accepted"
                    or evidence_role.get("execution",{}).get("device")!="cuda:0"
                    or evidence_role.get("adapter_sha256")!=row.get("adapter_sha256")
                    or evidence_role.get("model_id")!=row.get("model_id")):
                raise PermissionError("original model inference not independently proven on CUDA")
            verify_original_role_receipt(row,artifact=artifact,frozen=frozen,contract=contract,state=state)
        step_rows=[r for k,r in rows if k=="cuda_target_optimizer_steps" and
                   r.get("run_id")==report["run_id"]]
        if len(step_rows)!=96 or {
                r.get("completed_steps") for r in step_rows}!=set(range(1,97)):
            raise PermissionError("all original CUDA AdamW optimizer receipts required")
        for n,row in enumerate(sorted(step_rows,key=lambda r:r["completed_steps"]),start=1):
            receipt=artifact(row["receipt_ref"])
            if (receipt.get("completed_steps")!=n or
                    receipt.get("execution_device")!="cuda:0"
                    or receipt.get("run_id")!=report["run_id"]
                    or receipt.get("chat_trainable_ref")!=e["exact_supervised_training_ref"]
                    or receipt.get("training_recipe")!="chat_json_r8_full_epoch_96_v1"):
                raise PermissionError("original CUDA optimizer receipt or data lineage differs")
            # The Pod stored an absolute path, but relocation must not change it.
            prefix=f"cuda-target-checkpoints/step-{n}/"
            mf=prefix+"checkpoint_manifest.json"
            if mf not in index or "sha256:"+index[mf]!=receipt.get("checkpoint_manifest_sha256"):
                raise PermissionError("actual original GPU optimizer manifest not retained")
            manifest=json.loads(z.read(mf))
            components=manifest.get("components")
            if (not isinstance(components,dict) or manifest.get("partial_write") is not False
                or manifest.get("manifest_hash")!="sha256:"+_sha(canonical_bytes(components))):
                raise PermissionError("final original CUDA optimizer manifest invalid")
            for filename,expectedhash in components.items():
                if "sha256:"+index.get(prefix+filename,"")!=expectedhash:
                    raise PermissionError("CUDA optimizer component hash differs on remote cold readback")
            if "sha256:"+index.get(prefix+"adapter_model.safetensors","")!=receipt.get("adapter_sha256"):
                raise PermissionError("CUDA adapter not the original saved actual optimizer step")
    return {
        "schema":"independent-cold-original-cuda-replay-review.v1",
        "source_gpu_archive_sha256":expected_sha256,
        "verified_native_operations":45,
        "verified_original_cuda_role_inferences":5,
        "verified_real_cuda_optimizer_steps":96,
        "external_historical_certification":False,
        "issue_102_complete":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--expected-archive-sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("cold GPU readback evidence is immutable")
    res=verify_original_cuda_archive(a.archive,
                                   expected_sha256=a.expected_archive_sha256)
    a.output.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
    print(json.dumps(res,sort_keys=True))


if __name__=="__main__":
    main()
