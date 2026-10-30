"""No fake CUDA campaign can pass the independent original cold verifier."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import json

import pytest
from hephaestus.storage.verified import canonical_bytes

import importlib.util
ROOT=Path(__file__).resolve().parents[1]

def _load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

freeze=_load_script("freeze_original_tier_a_cuda_archive").freeze
verify_original_cuda_archive=_load_script("cold_verify_original_tier_a_cuda_archive").verify_original_cuda_archive
verify_original_role_receipt=_load_script("cold_verify_original_tier_a_cuda_archive").verify_original_role_receipt


def test_cold_checker_refuses_empty_or_fabricated_45(tmp_path):
    archive=tmp_path/"fake.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("archive_sha256.json","{}")
        z.writestr("state.sqlite",b"not a real 45-operation database")
    sha="sha256:"+sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        verify_original_cuda_archive(archive,expected_sha256=sha)


def test_cold_checker_requires_separately_pinned_archive_sha(tmp_path):
    archive=tmp_path/"a.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("archive_sha256.json","{}")
    with pytest.raises(PermissionError,match="independently pinned"):
        verify_original_cuda_archive(archive,expected_sha256="sha256:"+"0"*64)


def test_freeze_refuses_claimed_success_without_actual_complete_science(tmp_path):
    root=tmp_path/"unverified-original"
    root.mkdir()
    fake={"status":"completed","completed_operations":45,
          "verified":{"native_operations":43,"actual_cuda_optimizer_steps":96}}
    from hashlib import sha256
    raw=canonical_bytes(fake)
    (root/("cuda-native-report-"+sha256(raw).hexdigest()+".json")).write_bytes(raw)
    with pytest.raises(PermissionError,match="exactly one"):
        freeze(root,tmp_path/"release.zip")


def test_freeze_never_overwrites_existing_original_archive(tmp_path):
    root=tmp_path/"workflow"
    root.mkdir()
    archive=tmp_path/"archive.zip"
    archive.write_bytes(b"existing evidence")
    with pytest.raises(PermissionError,match="fresh archive"):
        freeze(root,archive)
    assert archive.read_bytes()==b"existing evidence"


def _cold_role_fixture(role):
    """Explicit structural fixture; never authentic GPU execution evidence."""
    from hephaestus.production.original_cuda_identity import verified_original_roles,registry_digest
    stack,contract=verified_original_roles(registry_digest())
    frozen=stack.resolve(role)
    state={"workflow_id":"fixture","run_id":"fixture-run","lineage_id":"fixture-lineage",
           "stage_name":"targeted_repair"}
    docs={}
    def put(doc):
        ref="sha256:"+sha256(canonical_bytes(doc)).hexdigest()
        docs[ref]=doc
        return ref
    source=put({"schema":"tier-a-gpu-role-domain-observation.v1",**state,"operation_id":"fixture-op"})
    vocab=contract.profile(role)["vocabulary"]
    answer={"decision":vocab["decision"][0],"action":vocab["action"][0],
            "primary_variable":vocab["primary_variable"][0],"confidence":0.5,
            "evidence_refs":[source],"uncertainties":[],"rationale":"Fixture assessment only."}
    raw=json.dumps(answer)
    receipt={"schema":"role-inference.v1",**state,"role":role,"status":"accepted",
        "model_id":frozen.model_id,"base_revision":frozen.revision,
        "adapter_sha256":frozen.adapter.sha256,"adapter_source":frozen.adapter.s3_key,
        "tokenizer_id":frozen.model_id,"tokenizer_revision":frozen.revision,
        "profile":contract.profile(role),"evidence_refs":[source],
        "generator":"PinnedTransformersRoleGenerator","prompt":"fixture prompt",
        "prompt_sha256":sha256(b"fixture prompt").hexdigest(),"raw_output":raw,
        "output_sha256":sha256(raw.encode()).hexdigest(),"answer":answer,
        "execution":{"device":"cuda:0","gpu_name":"fixture-only","cuda_version":"fixture-only",
                     "input_token_ids":[1],"generated_token_ids":[2],"test_double":True}}
    row={"role":role,"domain_observation_ref":source,"operation_id":"fixture-op",
         "inference_receipt_ref":put(receipt),"bounded_format_repair_attempts":0,
         "rejected_inference_receipt_refs":[]}
    return row,docs,receipt,frozen,contract,state,put


@pytest.mark.parametrize("role",["controller","diagnosis","planner","evaluator","judge"])
def test_cold_role_checker_reparses_all_five_original_contracts(role):
    row,docs,_,frozen,contract,state,_=_cold_role_fixture(role)
    verify_original_role_receipt(row,artifact=docs.__getitem__,frozen=frozen,contract=contract,state=state)


@pytest.mark.parametrize("change",["foreign_run","false_answer","invented_ref","bad_hash","no_tokens","cpu"])
def test_cold_role_checker_rejects_resealed_but_false_acceptance(change):
    row,docs,receipt,frozen,contract,state,put=_cold_role_fixture("judge")
    if change=="foreign_run": receipt["run_id"]="other"
    if change=="false_answer": receipt["answer"]={**receipt["answer"],"confidence":0.9}
    if change=="invented_ref":
        response=json.loads(receipt["raw_output"])
        response["evidence_refs"]=["sha256:"+"0"*64]
        receipt["raw_output"]=json.dumps(response)
        receipt["output_sha256"]=sha256(receipt["raw_output"].encode()).hexdigest()
    if change=="bad_hash": receipt["output_sha256"]="0"*64
    if change=="no_tokens": receipt["execution"]["generated_token_ids"]=[]
    if change=="cpu": receipt["execution"]["device"]="cpu"
    row["inference_receipt_ref"]=put(receipt)
    with pytest.raises((PermissionError,ValueError)):
        verify_original_role_receipt(row,artifact=docs.__getitem__,frozen=frozen,contract=contract,state=state)
