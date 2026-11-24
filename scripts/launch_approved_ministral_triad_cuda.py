#!/usr/bin/env python3
"""Off-Pod controller for ONE user-authorized prepaid original Ministral triad.

No old grants. One GPU, pinned datacenter and volume, finite prepaid ceiling,
persistent source/test hashes, no model secret, independent termination watchdog.
Do not reuse this controller for a new job without a distinct GitHub run ID.
"""
from __future__ import annotations

from decimal import Decimal,ROUND_FLOOR
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import argparse,json,os,re,shutil,subprocess,tempfile,time

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from hephaestus.production.original_cuda_identity import verified_original_roles,registry_digest
from hephaestus.research.constrained_repair import (
    RepairScope,PatchCandidate,ConstrainedSelfRepair,OfflineDockerPytestRunner)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

BASE="hephaestus/operations/ministral-triad"
SKU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
REGRESSION="tests/test_targeted_post_failure_probes.py"
SOURCE="src/hephaestus/diagnosis/probes.py"
TEST_NAME="test_null_text_is_not_literal_supervised_text"
NODEID=REGRESSION+"::"+TEST_NAME
# THIS TEST IS REVIEWED BEFORE PAID MODEL INFERENCE. Never edit it using model output.
FROZEN_TEST=(
    "\n\ndef test_null_text_is_not_literal_supervised_text():\n"
    "    from hephaestus.diagnosis.probes import analyze_dataset_task_coverage\n"
    "    _, measurement = analyze_dataset_task_coverage([{'text': None}], {}, source_ref='sha256:real-input')\n"
    "    assert measurement['text_rows'] == 0\n"
)
MAX_SECONDS=75*60
MAX_TOTAL_USD=Decimal("4.00")
RESERVE_USD=Decimal("1.00")
MAX_QUOTED_GPU_USD_PER_HOUR=Decimal("3.25")
OTHER_STORAGE_RATE=Decimal("0.08")
MIN_SECONDS=30*60


def digest(data: bytes) -> str:
    return "sha256:"+sha256(data).hexdigest()


def read_limited(s3,bucket,key,limit=1_000_000):
    stream=s3.get_object(Bucket=bucket,Key=key)["Body"]
    try:
        raw=stream.read(limit+1)
    finally:
        stream.close()
    if len(raw)>limit:
        raise PermissionError("external experiment artifact exceeds review limit")
    return raw


def upload_once(s3,bucket,key,raw):
    try:
        s3.head_object(Bucket=bucket,Key=key)
    except Exception as exc:
        code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
        if code not in {"404","NoSuchKey","NotFound"}:
            raise
    else:
        raise PermissionError("repeat GPU campaign run directory exists; never overwrite")
    s3.put_object(Bucket=bucket,Key=key,Body=raw)
    if digest(read_limited(s3,bucket,key,max(len(raw)+5,1024)))!=digest(raw):
        raise PermissionError("S3 operator stage bytes differ from original")
    return digest(raw)


def independent_failure_preflight(tmp,*,docker,code_root):
    """Admit the genuine existing defect BEFORE touching paid RunPod."""
    assert Path(code_root/SOURCE).is_file()
    original=(code_root/SOURCE).read_bytes()
    if b'text = str(row.get("text", ""))' not in original:
        raise PermissionError("the claimed real historical defect is already fixed")
    trial=tmp/"reviewed-repair-checkout"
    trial.mkdir()
    for directory in ("src","tests","configs"):
        shutil.copytree(code_root/directory,trial/directory,
                        ignore=shutil.ignore_patterns("__pycache__","*.pyc",".git"))
    shutil.copy2(code_root/"pyproject.toml",trial/"pyproject.toml")
    fixed=trial/REGRESSION
    if TEST_NAME in fixed.read_text():
        raise PermissionError("test name collision invalidates frozen independent regression")
    fixed.write_text(fixed.read_text()+FROZEN_TEST)
    for args in (
        ["git","init","-q",str(trial)],
        ["git","-C",str(trial),"config","user.name","independent-repair-review"],
        ["git","-C",str(trial),"config","user.email","research@example.invalid"],
        ["git","-C",str(trial),"add","src","tests","configs","pyproject.toml"],
        ["git","-C",str(trial),"commit","-qm","freeze original source and independent semantic regression"],
    ):
        subprocess.run(args,check=True,timeout=20)
    head=subprocess.check_output(["git","-C",str(trial),"rev-parse","HEAD"],
        text=True).strip()
    image=subprocess.check_output([docker,"image","inspect","--format={{.Id}}",
        "hephaestus-triad-review-sandbox:local"],text=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}",image):
        raise PermissionError("independent Docker image lacks immutable digest")
    runner=OfflineDockerPytestRunner(image,
        docker_executable=docker,timeout_seconds=180)
    baseline=runner.run(trial,fixed_test=REGRESSION)
    if (baseline.status!="failed" or baseline.exit_code!=1
            or baseline.worker!="offline-docker-pytest.v1"
            or NODEID not in baseline.output_excerpt
            or "ERROR collecting" in baseline.output_excerpt
            or not ("AssertionError" in baseline.output_excerpt
                    or re.search(r"(?m)^E\s+assert\b",baseline.output_excerpt))):
        raise PermissionError("independently frozen semantic baseline did not fail as expected")
    print("TRIAD_INDEPENDENT_ORIGINAL_BASELINE_RED=PASS",flush=True)
    return trial,head,image,{
        "schema":"heph-real-triad-approved-case-v1",
        "source_file":SOURCE,
        "source_sha256":digest(original),
        "fixed_test_sha256":digest(fixed.read_bytes()),
        "issue_nodeid":NODEID,
        "baseline_output_sha256":baseline.output_sha256,
        "baseline_failure_excerpt":baseline.output_excerpt[-1100:],
        "reviewed_strategy":"minimal_exact_span",
        "model_may_modify_tests":False,
        "model_may_execute_source":False,
        "auto_merge":False,
    }


def command_for(source_commit,code_digest,run_tag):
    if not re.fullmatch(r"[a-f0-9]{40}",source_commit):
        raise PermissionError("source commit identity invalid")
    prefix="/workspace/"+BASE+"/"+run_tag
    code_tar=prefix+"/code.tar.gz"
    logs=prefix+"/worker-stdout.log"
    exit_receipt=prefix+"/worker-exit.txt"
    root="/tmp/hephaestus-triad-pinned-code"
    # There are intentionally NO API keys or Docker mounts in this worker.
    shell=(
        "set -Eeuo pipefail; umask 077; "
        f"trap 'x=$?; printf \"%s %s\\n\" \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"$x\" > \"{exit_receipt}\"' EXIT; "
        f"exec >> '{logs}' 2>&1; "
        f"mkdir -p '{root}'; "
        f"test \"$(sha256sum '{code_tar}' | cut -d' ' -f1)\" = '{code_digest[7:]}'; "
        f"tar -xzf '{code_tar}' -C '{root}'; "
        f"cd '{root}'; "
        "python -m pip install -q --ignore-installed 'cryptography>=44,<47'; "
        "python -m pip install -q -e '.[role-inference,independent-certification]'; "
        "python -u scripts/run_provisional_ministral_triad_cuda_worker.py "
        f"--run-tag '{run_tag}' --source-commit '{source_commit}'; "
        "echo ORIGINAL_MINISTRAL_TRIAD_WORKER_COMPLETED"
    )
    env={
        "HF_HOME":"/tmp/hephaestus-hf",
        "HF_HUB_DISABLE_TELEMETRY":"1",
        "HF_HUB_ENABLE_HF_TRANSFER":"0",
        "PYTHONUNBUFFERED":"1",
    }
    return ["bash","-lc",shell],env


def cold_verify_candidate(s3,bucket,prefix,trial,head,image,case,docker):
    """Original CUDA output is only a proposal: execute nothing on the Pod."""
    result=json.loads(read_limited(s3,bucket,prefix+"/worker-result.json"))
    if (result.get("schema")!="real-original-ministral-triad-worker-v1"
            or result.get("status")!="candidate_proposed_pending_offpod_verification"
            or result.get("source_sha256")!=case["source_sha256"]
            or result.get("code_modified") is not False
            or result.get("original_five_role_registry_sha256")!=registry_digest()):
        raise PermissionError("GPU research result is absent or outside exact frozen scope")
    stack,_=verified_original_roles(registry_digest())
    steps=result.get("steps")
    if (not isinstance(steps,list) or [x.get("role") for x in steps]
            !=["diagnosis","planner","coder"]):
        raise PermissionError("three genuine frozen model inference steps not established")
    for role in ("diagnosis","planner"):
        record=json.loads(read_limited(s3,bucket,prefix+"/"+role+"-observation.json"))
        spec=stack.resolve(role)
        ref=record.get("inference_receipt_ref")
        if (not isinstance(ref,str) or not re.fullmatch(r"sha256:[0-9a-f]{64}",ref)
                or record.get("model_id")!=spec.model_id
                or record.get("revision")!=spec.revision
                or record.get("adapter_sha256")!=spec.adapter.sha256):
            raise PermissionError("role identity/adapter proof mismatched original certified registry")
        key=prefix+"/model-artifacts/objects/sha256/"+ref[7:9]+"/"+ref[7:]
        raw=read_limited(s3,bucket,key,800_000)
        receipt=json.loads(raw)
        if (digest(raw)!=ref or receipt.get("status")!="accepted"
                or receipt.get("role")!=role
                or receipt.get("model_id")!=spec.model_id
                or receipt.get("adapter_sha256")!=spec.adapter.sha256
                or receipt.get("generator")!="PinnedTransformersRoleGenerator"
                or receipt.get("execution",{}).get("device")!="cuda:0"
                or receipt.get("run_id")!="triad-"+case["run_tag"]):
            raise PermissionError("authentic frozen original role GPU inference receipt unavailable")
        print("TRIAD_ORIGINAL_"+role.upper()+"_CUDA_RECEIPT=VERIFIED",flush=True)
    coder=json.loads(read_limited(s3,bucket,prefix+"/coder-observation.json"))
    cr=coder["receipt"]
    if (cr.get("schema")!="provisional-parent-coder-inference.v1"
            or cr.get("model_id")!="mistralai/Ministral-3-8B-Reasoning-2512"
            or cr.get("revision")!=stack.resolve("planner").revision
            or cr.get("adapter_used") is not False
            or cr.get("device")!="cuda:0"
            or cr.get("certified_coder") is not False):
        raise PermissionError("parent-only genuine Coder receipt invalid")
    print("TRIAD_ORIGINAL_PARENT_CODER_INFERENCE=RECEIPT_VERIFIED",flush=True)
    patch_dict=json.loads(read_limited(s3,bucket,prefix+"/candidate.json",100_000))
    fault_data=read_limited(s3,bucket,prefix+"/verified-fault.json")
    fault=json.loads(fault_data)
    if (fault.get("schema")!="reviewed-code-failure.v1"
            or fault.get("status")!="verified_failed"
            or fault.get("source_sha256")!=case["source_sha256"]
            or fault.get("fixed_test")!=REGRESSION
            or fault.get("failing_test_nodeid")!=NODEID):
        raise PermissionError("model-consumed regression did not match original frozen test")
    evidence=FileSystemArtifactStore(trial.parent/"cold-verifier-artifacts")
    ref=evidence.put_bytes(fault_data,media_type="application/json").artifact_ref
    if ref!=patch_dict.get("evidence_ref"):
        raise PermissionError("candidate proposal not bound to actual model-consumed fault")
    patch=PatchCandidate(**patch_dict)
    scope=RepairScope(head,SOURCE,REGRESSION)
    # The independent Docker sandbox is the ONLY executor of Coder-authored code.
    checker=ConstrainedSelfRepair(
        evidence,OfflineDockerPytestRunner(
            image,docker_executable=docker,timeout_seconds=180))
    checked=checker.trial(
        checkout=trial,scope=scope,candidate=patch,verified_fault_ref=ref)
    if not checked["red_to_green_verified"]:
        print("TRIAD_REAL_MODELS_PROPOSED_BUT_PATCH_REJECTED=true",flush=True)
        print("TRIAD_SANDBOX_RESULT_REF="+str(checked["trial_ref"]),flush=True)
        return False
    print("TRIAD_REAL_MODELS_RED_TO_GREEN_VERIFIED=PASS",flush=True)
    print("TRIAD_REVIEW_ONLY_TRIAL_REF="+str(checked["trial_ref"]),flush=True)
    print("TRIAD_FIRST_ATTEMPT_CORRECT=true",flush=True)
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-commit",required=True)
    p.add_argument("--run-tag",required=True)
    args=p.parse_args()
    if (not re.fullmatch(r"[a-f0-9]{40}",args.source_commit)
            or not re.fullmatch(r"[0-9]{6,20}",args.run_tag)
            or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
               !=args.source_commit):
        raise PermissionError("GitHub original source must be exact pinned checkout")
    cfg=RunPodConfig.from_env()
    if cfg.network_volume_id!="cviwpryzao" or cfg.datacenter_id!="EU-CZ-1":
        raise PermissionError("model qualification can't use a different private volume")
    api_key=os.environ.get("RUNPOD_API_KEY")
    if not api_key or not os.environ.get("RUNPOD_S3_ACCESS_KEY_ID") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise PermissionError("operator GitHub secrets unavailable; no spending")
    provider=ReadOnlyRunPodPricing(bearer_token=api_key)
    account=provider.account()
    quoted=provider.quote_gpu(gpu_id=SKU,datacenter_id=cfg.datacenter_id)
    if account.autopay_enabled or quoted.price_usd_per_hour>MAX_QUOTED_GPU_USD_PER_HOUR or quoted.gpu_memory_gb<48:
        raise PermissionError("prepaid balance or GPU price/VRAM outside user-approved safe experiment")
    combined=quoted.price_usd_per_hour+account.already_spending_usd_per_hour+OTHER_STORAGE_RATE
    remaining=min(MAX_TOTAL_USD,account.credit_balance_usd-RESERVE_USD)
    allowed_seconds=min(MAX_SECONDS,int((remaining*Decimal(3600)/combined).to_integral_value(rounding=ROUND_FLOOR)))
    if allowed_seconds<MIN_SECONDS:
        raise PermissionError("prepaid budget cannot safely cover an original three-model qualification")
    adapter=RunPodExecutionAdapter(cfg,EnvironmentSecretsProvider(),
        state_root=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"triad-original-single-allocation")
    existing=adapter.list_pods()
    own="hephaestus-ministral-triad-"+args.run_tag+"-"
    if any(str(x.get("name","")).startswith(own) for x in existing):
        raise PermissionError("an existing triad Pod must be independently reconciled, not duplicated")
    if adapter.select_gpu([SKU])!=SKU:
        raise PermissionError("current secure EU-CZ-1 GPU unavailable")
    s3=RunPodNetworkVolumeStorage.from_boto3(cfg,EnvironmentSecretsProvider()).client
    roles,_=verified_original_roles(registry_digest())
    for role in ("diagnosis","planner"):
        spec=roles.resolve(role)
        h=s3.head_object(Bucket=cfg.network_volume_id,Key=spec.adapter.s3_key)
        if int(h["ContentLength"])!=spec.adapter.bytes:
            raise PermissionError("original frozen source adapters no longer match certified sizes")
    if not Path("/usr/bin/docker").is_file():
        raise PermissionError("off-Pod independent Docker verifier unavailable")
    docker="/usr/bin/docker"
    prefix=BASE+"/"+args.run_tag
    with tempfile.TemporaryDirectory(prefix="triad-offpod-independent-") as tmpdir:
        root=Path(tmpdir)
        checkout,head,image,case=independent_failure_preflight(
            root,docker=docker,code_root=Path.cwd())
        case.update(run_tag=args.run_tag,source_commit=args.source_commit)
        archive=root/"code.tar.gz"
        subprocess.run(["git","archive","--format=tar.gz","--output",str(archive),"HEAD"],
            check=True,timeout=60)
        code=archive.read_bytes()
        code_hash=upload_once(s3,cfg.network_volume_id,prefix+"/code.tar.gz",code)
        upload_once(s3,cfg.network_volume_id,prefix+"/frozen-case.json",canonical_bytes(case))
        print("TRIAD_GIT_SOURCE_SHA="+args.source_commit,flush=True)
        print("TRIAD_EXACT_CODE_ARCHIVE_SHA="+code_hash,flush=True)
        print("TRIAD_PINNED_TEST_SHA="+case["fixed_test_sha256"],flush=True)
        print("TRIAD_ORIGINAL_ADAPTER_S3_VERIFIED=true",flush=True)
        print("TRIAD_POD_GPU_SKU="+SKU,flush=True)
        print("TRIAD_QUOTED_GPU_USD_PER_HOUR="+str(quoted.price_usd_per_hour),flush=True)
        print("TRIAD_HARD_MAX_SECONDS="+str(allowed_seconds),flush=True)
        print("TRIAD_MAX_TOTAL_SPENDING_ENVELOPE_USD="+str(MAX_TOTAL_USD),flush=True)
        pod=None
        try:
            cmd,env=command_for(args.source_commit,code_hash,args.run_tag)
            # Exactly one allocation intent. Unknown response NEVER issues another POST.
            pod=adapter.create_bounded_gpu_pod(
                name="hephaestus-ministral-triad-"+args.run_tag,
                image_name=IMAGE,
                gpu_type_ids=[SKU],docker_start_cmd=cmd,env=env,
                container_disk_in_gb=110,
                operation_id="original-ministral-triad-"+args.run_tag)
            print("TRIAD_REAL_RUNPOD_POD_CREATED=true",flush=True)
            print("TRIAD_REAL_RUNPOD_POD_ID="+str(pod["id"]),flush=True)
            deadline=time.monotonic()+allowed_seconds
            while time.monotonic()<deadline:
                time.sleep(18)
                try:
                    status=read_limited(s3,cfg.network_volume_id,prefix+"/worker-exit.txt",500).decode()
                except Exception as exc:
                    code=getattr(exc,"response",{}).get("Error",{}).get("Code","")
                    if str(code) not in {"404","NoSuchKey","NotFound"}:
                        raise
                else:
                    fields=status.strip().split()
                    if len(fields)!=2 or not fields[-1].isdigit():
                        raise PermissionError("GPU worker exit status malformed")
                    exit_code=int(fields[-1])
                    print("TRIAD_GPU_WORKER_EXIT_CODE="+str(exit_code),flush=True)
                    if exit_code:
                        raise RuntimeError("actual model worker failed; exact stage receipts retained")
                    break
                if time.monotonic()+180>deadline:
                    print("TRIAD_GPU_WORKER_BUDGET_APPROACHING=true",flush=True)
                if int((deadline-time.monotonic())//60)%3==0:
                    latest=provider.account()
                    if latest.autopay_enabled or latest.credit_balance_usd<=RESERVE_USD:
                        raise PermissionError("prepaid balance reserve reached")
            else:
                raise TimeoutError("paid model qualification exceeded hard bounded wall clock")
        finally:
            if pod is not None:
                adapter.delete_pod(str(pod["id"]))
                print("TRIAD_REAL_RUNPOD_POD_DELETED=true",flush=True)
                owned=[x for x in adapter.list_pods()
                       if str(x.get("name","")).startswith(own)]
                if owned:
                    raise PermissionError("exact triad Pod still listed after teardown")
        print("TRIAD_OFFPOD_COLD_VERIFICATION_STARTED=true",flush=True)
        passed=cold_verify_candidate(s3,cfg.network_volume_id,prefix,
            checkout,head,image,case,docker)
        if not passed:
            raise RuntimeError("genuine Coder proposed a patch but independent red-to-green gate rejected it")


if __name__=="__main__":
    main()
