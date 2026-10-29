#!/usr/bin/env python3
"""User-delegated full original 45/45 GPU science, governed OFF the Pod.

The user explicitly authorized the presently available prepaid RunPod balance
for this exact original research run (2026-10-29). This GitHub Actions operator
is separate from the target/role models; its ephemeral Ed25519 compute-grant
signature is **not** an independent human scientific attestation. No auto-pay,
new debt, role substitution, or historical certification is authorized.

A separate GitHub Actions watchdog must run in parallel. Never execute this
file alone without the external watchdog and original source staging.
"""
from __future__ import annotations
from datetime import datetime,timedelta,timezone
from decimal import Decimal,ROUND_FLOOR
from hashlib import sha256
from pathlib import Path
import argparse,json,os,time,boto3

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.storage.verified import canonical_bytes

SOURCE_POLICY="sha256:79b949c946b9c979eb4ab9830aad53280c4ded25cee5d3bb978d3eddbcc22dbb"
TRAINABLE="sha256:ee98849e7141cfdaba13eafc4647cc82707da82c8cdc9dd0ec88c7caac413b73"
WORKFLOW="cuda-json-45-original-five-20261028"
RUN="cuda-json-45-new-1117"
PREFIX="hephaestus/scientific/original_cuda_45/"+WORKFLOW
GPU="NVIDIA GeForce RTX 4090"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
RESERVE_USD=Decimal("0.30")
MAX_SECONDS=4*3600
MIN_SECONDS=3600


def decide_budget(account,rate):
    if account.autopay_enabled:
        raise PermissionError("prepaid RunPod research refuses auto-pay and new liabilities")
    remaining=account.credit_balance_usd-RESERVE_USD
    combined=rate.price_usd_per_hour+account.already_spending_usd_per_hour
    if combined<=0 or remaining<=0:
        raise PermissionError("no safely spendable prepaid funding")
    max_seconds=min(MAX_SECONDS,int((remaining*Decimal(3600)/combined).to_integral_value(
        rounding=ROUND_FLOOR)))
    if max_seconds<MIN_SECONDS:
        raise PermissionError("less than one hour of bounded original campaign budget remains")
    return max_seconds


def client_s3(config):
    from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
    return RunPodNetworkVolumeStorage.from_boto3(
        config,EnvironmentSecretsProvider()).client


def require_source(s3, bucket):
    raw=s3.get_object(Bucket=bucket,Key=PREFIX+"/cuda-grant-unsigned-request.json")["Body"].read()
    doc=json.loads(raw)
    if (doc.get("schema")!="tier-a-original-cuda-compute-authorization.v1"
        or doc.get("status")!="requested"
        or doc.get("workflow_id")!=WORKFLOW or doc.get("run_id")!=RUN
        or doc.get("source_policy_ref")!=SOURCE_POLICY
        or doc.get("reviewed_trainable_ref")!=TRAINABLE
        or len(doc.get("operations",{}))!=9):
        raise PermissionError("private source scope or exact nine native GPU operations changed")
    s3.head_object(Bucket=bucket,Key=PREFIX+"/state.sqlite")
    s3.head_object(Bucket=bucket,Key=PREFIX+"/context_ref.txt")
    return doc


def grant(s3,bucket,source_doc,max_seconds,run_tag):
    private=Ed25519PrivateKey.generate()
    public=private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    auth={**source_doc,
          "status":"approved",
          "operator_id":"github-actions-offpod-user-delegated-20261029",
          "expires_at":(datetime.now(timezone.utc)+timedelta(
              seconds=max_seconds+1200)).isoformat(),
          "maximum_gpu_seconds":max_seconds}
    body=canonical_bytes({
        "authorization":auth,"signature_hex":private.sign(canonical_bytes(auth)).hex()})
    if not run_tag.isdecimal():
        raise PermissionError("external grant requires exact GitHub run ID")
    grant_name="external-original-cuda-grant-"+run_tag+".json"
    key=PREFIX+"/"+grant_name
    try:
        s3.head_object(Bucket=bucket,Key=key)
    except Exception as exc:
        code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
        if code not in {"404","NoSuchKey","NotFound"}:
            raise
        s3.put_object(Bucket=bucket,Key=key,Body=body)
    else:
        # Never change an existing authorized grant or start a duplicate Pod.
        raise PermissionError("a previous GPU spending grant exists: reconcile run first")
    if s3.get_object(Bucket=bucket,Key=key)["Body"].read()!=body:
        raise PermissionError("original offpod grant was not persisted byte-identically")
    return public.hex(),"sha256:"+sha256(public).hexdigest(),grant_name


def pod_command(commit_sha,grant_public,grant_sha,max_seconds,grant_filename="external-original-cuda-grant.json"):
    if (not isinstance(commit_sha,str) or len(commit_sha)!=40
        or any(ch not in "0123456789abcdef" for ch in commit_sha)):
        raise PermissionError("GitHub original scientific code must be pinned by immutable SHA")
    root="/workspace/"+PREFIX
    base="/tmp/hephaestus-original-cuda-code"
    # Never inject RunPod API or S3 credentials into the GPU Pod.
    command=(
        "set -Eeuo pipefail; umask 077; "
        "trap 'code=$?; printf \"%s\\\\n\" \"$code\" > "
        "/workspace/hephaestus/scientific/original_cuda_45/worker_exit_status.txt' EXIT; "
        "git clone -q --depth 1 --branch feature/original-cuda-45-full-budget-launch-20261029 "
        "https://github.com/kys928/Hephaestus "+base+"; "
        "cd "+base+"; "
        'test "$(git rev-parse HEAD)" = "'+commit_sha+'"; '
        "python -m pip install -q -e '.[peft-training,role-inference,independent-certification]'; "
        "python - <<'PY'\n"
        "from huggingface_hub import snapshot_download\n"
        "from hephaestus.llm.cpu_causal_research import MODEL_ID,MODEL_REVISION\n"
        "snapshot_download(MODEL_ID,revision=MODEL_REVISION,cache_dir='/tmp/cuda-target-cache')\n"
        "PY\n"
        "python -u scripts/run_original_tier_a_cuda_worker.py --resume "
        "--workspace "+root+" "
        "--config configs/research/tier_a_original_cuda_45_v1.json "
        "--target-model-cache /tmp/cuda-target-cache "
        "--original-role-model-cache /tmp/cuda-original-role-cache "
        "--original-adapter-cache /tmp/cuda-verified-adapters "
        "--signed-approval-file "+root+"/"+grant_filename+" "
        "--maximum-gpu-seconds "+str(max_seconds)+"; "
        "python scripts/freeze_original_tier_a_cuda_archive.py "
        "--workspace "+root+" "
        "--archive /workspace/hephaestus/scientific/original_cuda_45/"
        "original_frozen_45_archive.zip; "
        "sha256sum /workspace/hephaestus/scientific/original_cuda_45/"
        "original_frozen_45_archive.zip | awk '{print $1}' > "
        "/workspace/hephaestus/scientific/original_cuda_45/"
        "original_frozen_archive_sha256.txt; "
        "echo FULL_NATIVE_CUDA_45_FINISHED"
    )
    env={
        "HEPHAESTUS_TRUSTED_GPU_OPERATOR_PUBLIC_KEY_HEX":grant_public,
        "HEPHAESTUS_TRUSTED_GPU_OPERATOR_PUBLIC_KEY_SHA256":grant_sha,
        "HEPHAESTUS_ENABLE_PINNED_MODEL_DOWNLOAD":"1",
        "RUNPOD_NETWORK_VOLUME_ID":"cviwpryzao",
        "HF_HOME":"/tmp/hephaestus-hf",
        "HF_HUB_DISABLE_TELEMETRY":"1",
    }
    return ["bash","-lc",command],env


def monitor(s3, bucket, adapter, pod, *, deadline, provider):
    key="hephaestus/scientific/original_cuda_45/original_frozen_archive_sha256.txt"
    exit_key="hephaestus/scientific/original_cuda_45/worker_exit_status.txt"
    last_balance_check=0
    while time.monotonic()<deadline:
        time.sleep(40)
        try:
            output=s3.get_object(Bucket=bucket,Key=key)["Body"].read().decode().strip()
            if len(output)==64 and all(c in "0123456789abcdef" for c in output):
                print("ORIGINAL_CUDA45_ARCHIVE_CREATED=true",flush=True)
                return True
        except Exception as exc:
            code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
            if code not in {"404","NoSuchKey","NotFound"}:
                raise
        try:
            exit_status=s3.get_object(Bucket=bucket,Key=exit_key)["Body"].read().decode().strip()
            if exit_status and exit_status!="0":
                raise RuntimeError("real GPU science process failed; preserve all receipts and terminate Pod")
            if exit_status=="0":
                raise RuntimeError("GPU worker returned success without an authenticated frozen archive")
        except RuntimeError:
            raise
        except Exception as exc:
            code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
            if code not in {"404","NoSuchKey","NotFound"}:
                raise
        if time.monotonic()-last_balance_check>180:
            actual=provider.account()
            last_balance_check=time.monotonic()
            if actual.autopay_enabled or actual.credit_balance_usd<=RESERVE_USD:
                raise PermissionError("prepaid RunPod balance reserve reached; stop GPU immediately")
        current=adapter.get_pod(pod["id"])
        if str(current.get("desiredStatus","")).upper() in {"EXITED","TERMINATED","STOPPED"}:
            raise RuntimeError("original GPU Pod stopped before complete scientific report")
    raise TimeoutError("original GPU science exceeded independently bounded prepaid allowance")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--commit-sha",required=True)
    p.add_argument("--run-tag",required=True)
    args=p.parse_args()
    if not args.run_tag.isdecimal():
        raise PermissionError("native GPU launch must come from a real GitHub Actions run ID")
    config=RunPodConfig.from_env()
    if config.datacenter_id!="EU-CZ-1" or config.network_volume_id!="cviwpryzao":
        raise PermissionError("original certified EU-CZ-1 Hephaestus volume changed")
    provider=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    account=provider.account()
    quoted=provider.quote_gpu(gpu_id=GPU,datacenter_id=config.datacenter_id)
    if quoted.gpu_memory_gb<24 or quoted.price_usd_per_hour>Decimal("0.89"):
        raise PermissionError("original full-campaign GPU price increased above live RTX 4090 $0.89/h quote")
    seconds=decide_budget(account,quoted)
    s3=client_s3(config)
    original=require_source(s3,config.network_volume_id)
    print("USER_AUTHORIZED_PREPAID_FULL_GPU_CAMPAIGN=true")
    print("ORIGINAL_FULL_CUDA45_GPU_RENTAL_CEILING_SECONDS="+str(seconds))
    print("RUNPOD_ACCOUNT_FINANCIAL_AMOUNTS_REDACTED=true")
    print("ORIGINAL_GPU_ROLE_SOURCE_HASH_VERIFIED=true")
    grant_public,grant_sha,grant_name=grant(s3,config.network_volume_id,original,seconds,args.run_tag)
    command,env=pod_command(args.commit_sha,grant_public,grant_sha,seconds,grant_name)
    from hephaestus.providers.runpod.execution import UrllibRunPodTransport,RunPodApiError
    class DiagnosticTransport(UrllibRunPodTransport):
        def request(self,method,url,*,headers,json_body=None):
            try:
                return super().request(method,url,headers=headers,json_body=json_body)
            except RunPodApiError as error:
                if method=="POST" and error.status_code in (400,422):
                    # Provider response may include account identifiers; classify
                    # error category only. Never print raw response or headers.
                    import json
                    try:
                        data=json.loads(error.__cause__.read())
                        detail=data.get("detail",data.get("error",""))
                        if isinstance(detail,list):
                            fields=[str(part) for e in detail[:6]
                                    for part in e.get("loc",[]) if isinstance(part,(str,int))]
                            category="validation_fields_"+",".join(fields)[:100]
                        else:
                            msg=str(detail).lower()
                            category=next((name for name,tokens in {
                                "insufficient_credit":("balance","credit","fund"),
                                "unavailable_capacity":("unavailable","no longer any instances","capacity"),
                                "container_disk":("disk","storage"),
                                "network_mount":("volume","mount"),
                                "gpu_type":("gpu type","gpu id"),
                                "container_image":("image","container"),
                            }.items() if any(k in msg for k in tokens)), "provider_validation_unknown")
                        print("RUNPOD_POD_REJECTED_VALIDATION_CATEGORY="+category,flush=True)
                    except Exception:
                        print("RUNPOD_POD_REJECTED_VALIDATION_CATEGORY=unparseable",flush=True)
                raise
    adapter=RunPodExecutionAdapter(
        config=config,secrets=EnvironmentSecretsProvider(),
        transport=DiagnosticTransport(),
        state_root=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"hephaestus-original-gpu-controller")
    pod=None
    try:
        pod=adapter.create_bounded_gpu_pod(
            name="hephaestus-original-five-fullcuda45-"+args.run_tag,
            image_name=IMAGE,gpu_type_ids=[GPU],
            docker_start_cmd=command,env=env,container_disk_in_gb=90,
            operation_id="original-full-cuda45-"+args.run_tag)
        print("ORIGINAL_FULL_CUDA45_POD_CREATED=true",flush=True)
        print("ORIGINAL_GPU_POD_ID="+str(pod["id"]),flush=True)
        done=monitor(s3,config.network_volume_id,adapter,pod,
                     deadline=time.monotonic()+seconds,provider=provider)
        if not done:raise RuntimeError("no original 45 evidence verified")
    finally:
        if pod is not None:
            try:
                adapter.delete_pod(str(pod["id"]))
                print("ORIGINAL_GPU_POD_DELETED=true",flush=True)
            except Exception:
                print("CRITICAL_ORIGINAL_GPU_POD_DELETE_FAILED_REQUIRES_EXTERNAL_WATCHDOG",
                      flush=True)
                raise
    # Cold verification is conducted by the *off-Pod* controller after
    # stopping billing. Report success only after actual SHA member readback.
    archive_key="hephaestus/scientific/original_cuda_45/original_frozen_45_archive.zip"
    digest_key="hephaestus/scientific/original_cuda_45/original_frozen_archive_sha256.txt"
    expected=s3.get_object(Bucket=config.network_volume_id,Key=digest_key)["Body"].read().decode().strip()
    if len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
        raise PermissionError("GPU archive digest missing or malformed")
    archive=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"full_original_cuda_45_cold.zip"
    s3.download_file(config.network_volume_id,archive_key,str(archive))
    from cold_verify_original_tier_a_cuda_archive import verify_original_cuda_archive
    verified=verify_original_cuda_archive(archive,expected_sha256="sha256:"+expected)
    if (verified["verified_native_operations"]!=45
        or verified["verified_original_cuda_role_inferences"]!=5
        or verified["verified_real_cuda_optimizer_steps"]!=96):
        raise PermissionError("independent original five-CUDA-role cold replay incomplete")
    print("ORIGINAL_CUDA45_INDEPENDENT_COLD_REPLAY=SUCCESS",flush=True)
    print("ORIGINAL_CUDA45_REAL_NATIVE_OPERATIONS=45",flush=True)
    print("ORIGINAL_CUDA45_REAL_GPU_OPTIMIZER_STEPS=96",flush=True)
    print("ORIGINAL_CUDA45_ORIGINAL_CERTIFIED_ROLE_GPU_INFERENCES=5",flush=True)
    print("ISSUE102_INDEPENDENT_HISTORICAL_CERTIFICATION=false",flush=True)


if __name__=="__main__":
    main()
