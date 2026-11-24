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
import argparse,json,os,re,shutil,subprocess,tempfile,time,urllib.request,urllib.error,zipfile,tarfile,base64

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter,RunPodApiError
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from hephaestus.production.original_cuda_identity import verified_original_roles,registry_digest
from hephaestus.research.constrained_repair import (
    RepairScope,PatchCandidate,ConstrainedSelfRepair,OfflineDockerPytestRunner)
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes

BASE="hephaestus/operations/ministral-triad"
SKU="NVIDIA RTX PRO 4500 Blackwell"
IMAGE="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GLOBAL_ID="cmw51pjsk000007l9a0mv1jga"
GPU_SKUS=("NVIDIA GeForce RTX 5090","NVIDIA RTX PRO 4500 Blackwell","NVIDIA RTX A6000","NVIDIA A40","NVIDIA RTX 6000 Ada Generation","NVIDIA L40","NVIDIA L40S","NVIDIA RTX PRO 6000 Blackwell Server Edition")
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
MAX_QUOTED_GPU_USD_PER_HOUR=Decimal("2.69")
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


def require_live_watchdog(s3,bucket,prefix,*,wait_seconds=0):
    key=prefix+"/watchdog-heartbeat.json"
    deadline=time.monotonic()+wait_seconds
    while True:
        try:
            heartbeat=json.loads(read_limited(s3,bucket,key,2000))
            if (heartbeat.get("schema")=="external-triad-watchdog-heartbeat.v1"
                    and heartbeat.get("run_tag")==prefix.split("/")[-1]
                    and type(heartbeat.get("heartbeat_epoch")) is int
                    and 0<=time.time()-heartbeat["heartbeat_epoch"]<85
                    and heartbeat.get("remaining_until_hard_stop",0)>180):
                return
        except Exception as exc:
            code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
            if code not in {"404","NoSuchKey","NotFound"}:
                raise
        if time.monotonic()>=deadline:
            raise PermissionError("independent external Pod-termination watchdog not live; no paid Pod permitted")
        time.sleep(8)


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
    tmp.chmod(0o755)
    trial=tmp/"reviewed-repair-checkout"
    trial.mkdir()
    trial.chmod(0o755)
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


def command_for(source_commit,code_digest,case_digest,run_tag,storage):
    if not re.fullmatch(r"[a-f0-9]{40}",source_commit):
        raise PermissionError("source commit identity invalid")
    # Existing authenticated RunPod S3 transport, confined to trusted bootstrap.
    # The inference subprocess receives no storage/provider keys, no shell tool.
    bootstrap=r"""
import hashlib,os,pathlib,subprocess,sys,tarfile,time,zipfile
from io import BytesIO
tag=os.environ["TRIAD_RUN_TAG"]
root=pathlib.Path("/tmp/hephaestus-triad-pinned-code")
work=pathlib.Path("/tmp/hephaestus-triad-local")/tag
root.mkdir(exist_ok=False)
work.mkdir(parents=True,exist_ok=False)
access=os.environ.pop("TRIAD_BRIDGE_ACCESS")
secret=os.environ.pop("TRIAD_BRIDGE_SECRET")
endpoint=os.environ.pop("TRIAD_BRIDGE_ENDPOINT")
region=os.environ.pop("TRIAD_BRIDGE_REGION")
bucket=os.environ.pop("TRIAD_BRIDGE_BUCKET")
prefix="hephaestus/operations/ministral-triad/"+tag+"/"
# Never pass credentials to pip, worker or its descendants.
child_env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ("SECRET","TOKEN","API_KEY","ACCESS_KEY","PASSWORD","CREDENTIAL"))}
subprocess.run([sys.executable,"-m","pip","install","-q","boto3>=1.35,<2"],env=child_env,check=True)
import boto3
from botocore.config import Config
s3=boto3.client("s3",endpoint_url=endpoint,region_name=region,aws_access_key_id=access,
    aws_secret_access_key=secret,config=Config(signature_version="s3v4",s3={"addressing_style":"path"},
    connect_timeout=15,read_timeout=45,retries={"mode":"standard","max_attempts":3}))
del access,secret
def get(name,limit,optional=False):
    try: body=s3.get_object(Bucket=bucket,Key=prefix+name)["Body"]
    except Exception as exc:
        if optional and str(getattr(exc,"response",{}).get("Error",{}).get("Code","")) in ("404","NoSuchKey","NotFound"):return None
        raise
    try: raw=body.read(limit+1)
    finally: body.close()
    if len(raw)>limit:raise PermissionError("oversize exact-prefix artifact")
    return raw
def verified(name,expected,limit):
    raw=get(name,limit)
    if "sha256:"+hashlib.sha256(raw).hexdigest()!=expected:
        raise PermissionError("immutable off-Pod source digest mismatch")
    return raw
def put(name,raw):
    if len(raw)>10_000_000:raise PermissionError("oversize evidence")
    s3.put_object(Bucket=bucket,Key=prefix+name,Body=raw)
    if get(name,len(raw))!=raw:raise RuntimeError("evidence readback failed")
def atomic(path,raw):
    tmp=path.with_suffix(".partial")
    with tmp.open("xb") as f:f.write(raw)
    os.replace(tmp,path)
code=1
proc=None
try:
    raw=verified("code.tar.gz",os.environ["TRIAD_CODE_SHA256"],100_000_000)
    case=verified("frozen-case.json",os.environ["TRIAD_CASE_SHA256"],100_000)
    (work/"frozen-case.json").write_bytes(case)
    with tarfile.open(fileobj=BytesIO(raw),mode="r:gz") as archive:
        for item in archive.getmembers():
            if item.name.startswith("/") or ".." in pathlib.PurePosixPath(item.name).parts or not (item.isfile() or item.isdir()):
                raise PermissionError("unsafe source archive member")
        archive.extractall(root)
    os.chdir(root)
    put("worker-phase.json",b'{"phase":"source_verified"}')
    subprocess.run([sys.executable,"-m","pip","install","-q","--ignore-installed","cryptography>=44,<47"],env=child_env,check=True)
    subprocess.run([sys.executable,"-m","pip","install","-q","-e",".[role-inference,independent-certification]"],env=child_env,check=True)
    put("worker-phase.json",b'{"phase":"runtime_installed"}')
    proc=subprocess.Popen([sys.executable,"-u","scripts/run_provisional_ministral_triad_cuda_worker.py",
        "--run-tag",tag,"--source-commit",os.environ["TRIAD_SOURCE_COMMIT"]],env=child_env)
    sent=set()
    while True:
        for n in (1,2):
            name="attempt-"+str(n)+".zip"
            p=work/name
            if n not in sent and p.is_file() and not p.is_symlink():
                put(name,p.read_bytes());sent.add(n)
            feedback=work/("feedback-"+str(n)+".json")
            if n in sent and not feedback.exists():
                value=get(feedback.name,8000,optional=True)
                if value is not None:atomic(feedback,value)
        status=proc.poll()
        if status is not None:
            code=status
            break
        time.sleep(5)
except Exception as exc:
    print("TRIAD_BOOTSTRAP_FAILURE_CLASS="+type(exc).__name__,flush=True)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
finally:
    try:
        out=BytesIO()
        with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for p in work.rglob("*"):
                if p.is_file() and not p.is_symlink() and p.name!="frozen-case.json" and p.stat().st_size<=1_000_000:
                    z.write(p,str(p.relative_to(work)))
        put("final-evidence.zip",out.getvalue())
    finally:
        put("worker-exit.txt",str(code).encode())
sys.exit(code)
"""
    env={"HF_HOME":"/tmp/hephaestus-hf","HF_HUB_DISABLE_TELEMETRY":"1",
         "HF_HUB_ENABLE_HF_TRANSFER":"0","PYTHONUNBUFFERED":"1",
         "RUNPOD_GLOBAL_VOLUME_ID":GLOBAL_ID,"TRIAD_RUN_TAG":run_tag,
         "TRIAD_SOURCE_COMMIT":source_commit,"TRIAD_CODE_SHA256":code_digest,
         "TRIAD_CASE_SHA256":case_digest,**storage}
    encoded=base64.b64encode(bootstrap.encode()).decode("ascii")
    # Match the actual successful Global Volume boot: bash -lc, no entrypoint override.
    return ["bash","-lc","exec python -u -c \"import base64;exec(base64.b64decode('"+encoded+"'))\""],env


class BundleReader:
    """Read immutable bounded archive in memory; never extract model-created files."""
    def __init__(self,raw,prefix):
        self.z=zipfile.ZipFile(BytesIO(raw))
        self.prefix=prefix+"/"
        members=self.z.infolist()
        names=[m.filename for m in members]
        if (len(members)>100 or len(set(names))!=len(names)
                or sum(m.file_size for m in members)>10_000_000
                or any(m.filename.startswith("/") or ".." in Path(m.filename).parts for m in members)):
            raise PermissionError("invalid evidence bundle")
    def get_object(self,*,Bucket,Key):
        if not Key.startswith(self.prefix):
            raise PermissionError("bundle key outside scope")
        return {"Body":BytesIO(self.z.read(Key[len(self.prefix):]))}


def optional_read(s3,bucket,key,limit):
    try:return read_limited(s3,bucket,key,limit)
    except Exception as exc:
        if str(getattr(exc,"response",{}).get("Error",{}).get("Code","")) in {"404","NoSuchKey","NotFound"}:
            return None
        raise


def trusted_storage_bridge(s3,bucket,prefix,cfg):
    sample=b"hephaestus-global-triad-sdk-transport-v1"
    upload_once(s3,bucket,prefix+"/transport-probe.bin",sample)
    print("TRIAD_AUTHENTICATED_S3_TRANSPORT=PASS",flush=True)
    return {"TRIAD_BRIDGE_ACCESS":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
            "TRIAD_BRIDGE_SECRET":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
            "TRIAD_BRIDGE_ENDPOINT":cfg.s3_endpoint_url,
            "TRIAD_BRIDGE_REGION":cfg.datacenter_id,"TRIAD_BRIDGE_BUCKET":bucket}


def cold_verify_candidate(s3,bucket,prefix,trial,head,image,case,docker):
    """Original CUDA output is only a proposal: execute nothing on the Pod."""
    result=json.loads(read_limited(s3,bucket,prefix+"/worker-result.json"))
    if (result.get("schema")!="real-original-ministral-triad-worker-v1"
            or result.get("status") not in ("candidate_proposed_pending_offpod_verification","invalid_candidate")
            or result.get("global_volume_id")!=GLOBAL_ID
            or result.get("run_tag")!=case["run_tag"]
            or result.get("source_commit")!=case["source_commit"]
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
    if result["status"]=="invalid_candidate":
        return {"red_to_green_verified":False,"candidate":{"output_excerpt":"The generated proposal failed the exact-span JSON/schema boundary: "+str(coder.get("proposal_error"))},"trial_ref":digest(canonical_bytes({"invalid_candidate":coder,"attempt":result["attempt"]}))}
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
        return checked
    print("TRIAD_REAL_MODELS_RED_TO_GREEN_VERIFIED=PASS",flush=True)
    print("TRIAD_REVIEW_ONLY_TRIAL_REF="+str(checked["trial_ref"]),flush=True)
    return checked


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-commit",required=True)
    p.add_argument("--run-tag",required=True)
    args=p.parse_args()
    if (not re.fullmatch(r"[a-f0-9]{40}",args.source_commit)
            or not re.fullmatch(r"[0-9]{6,20}",args.run_tag)
            or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=args.source_commit):
        raise PermissionError("exact operator-pinned source required")
    cfg=RunPodConfig.from_env()
    if cfg.network_volume_id!="cviwpryzao":
        raise PermissionError("original independent receipt store required")
    # cfg's Czech ID applies ONLY to S3 evidence. No datacenter constraint in
    # the GPU allocation. The pre-existing Global Volume is the sole mount.
    provider=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    adapter=RunPodExecutionAdapter(cfg,EnvironmentSecretsProvider(),
        state_root=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"global-triad-operations")
    import boto3
    from botocore.config import Config
    # RunPod requires SigV4; boto3 presigning can otherwise default to SigV2.
    s3=boto3.client("s3",endpoint_url=cfg.s3_endpoint_url,region_name=cfg.datacenter_id,
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4",s3={"addressing_style":"path"},
                      retries={"mode":"standard","max_attempts":3}))
    bucket=cfg.network_volume_id
    prefix=BASE+"/"+args.run_tag
    own="hephaestus-ministral-triad-"+args.run_tag+"-"
    def owned():return [p for p in adapter.list_pods() if str(p.get("name","")).startswith(own)]
    if owned():raise PermissionError("prior allocation needs independent reconciliation")
    account=provider.account()
    if account.autopay_enabled or account.credit_balance_usd<=RESERVE_USD:
        raise PermissionError("prepaid spending not approved")
    quotes=[]
    for sku in GPU_SKUS:
        try:q=provider.quote_gpu(gpu_id=sku,datacenter_id="GLOBAL")
        except RuntimeError:continue
        if q.gpu_memory_gb>=32 and q.price_usd_per_hour<=MAX_QUOTED_GPU_USD_PER_HOUR:
            quotes.append(q)
    preferred=("NVIDIA RTX PRO 6000 Blackwell Server Edition","NVIDIA GeForce RTX 5090","NVIDIA RTX PRO 4500 Blackwell","NVIDIA RTX 6000 Ada Generation","NVIDIA L40S","NVIDIA RTX A6000")
    quotes=[q for q in quotes if q.gpu_id in preferred]
    quotes.sort(key=lambda q:preferred.index(q.gpu_id))
    if not quotes:raise RuntimeError("no globally quoted GPU fits the fixed envelope")
    print("TRIAD_GLOBAL_ELIGIBLE_QUOTES="+str(len(quotes)),flush=True)
    docker="/usr/bin/docker"
    with tempfile.TemporaryDirectory(prefix="triad-global-review-") as tmpdir:
        root=Path(tmpdir)
        checkout,head,image,case=independent_failure_preflight(root,docker=docker,code_root=Path.cwd())
        case.update(run_tag=args.run_tag,source_commit=args.source_commit)
        archive=root/"code.tar.gz"
        subprocess.run(["git","archive","--format=tar.gz","--output",str(archive),"HEAD"],check=True,timeout=60)
        code_hash=upload_once(s3,bucket,prefix+"/code.tar.gz",archive.read_bytes())
        case_hash=upload_once(s3,bucket,prefix+"/frozen-case.json",canonical_bytes(case))
        storage=trusted_storage_bridge(s3,bucket,prefix,cfg)
        cmd,env=command_for(args.source_commit,code_hash,case_hash,args.run_tag,storage)
        require_live_watchdog(s3,bucket,prefix,wait_seconds=300)
        pod=None
        accepted=False
        try:
            for index,quoted in enumerate(quotes[:6],1):
                require_live_watchdog(s3,bucket,prefix)
                account=provider.account()
                combined=quoted.price_usd_per_hour+account.already_spending_usd_per_hour+OTHER_STORAGE_RATE
                remaining=min(MAX_TOTAL_USD,account.credit_balance_usd-RESERVE_USD)
                allowed_seconds=min(MAX_SECONDS,int((remaining*Decimal(3600)/combined).to_integral_value(rounding=ROUND_FLOOR)))
                if account.autopay_enabled or allowed_seconds<MIN_SECONDS:
                    raise PermissionError("safe prepaid GPU interval unavailable")
                if owned():raise PermissionError("unexpected owned Pod; no duplicate")
                name=own+"global-"+str(index)
                lease={"schema":"global-triad-offpod-lease.v1","run_tag":args.run_tag,
                       "name":name,"source_commit":args.source_commit,
                       "global_volume":GLOBAL_ID,"sku":quoted.gpu_id,
                       "quoted_usd_per_hour":str(quoted.price_usd_per_hour),
                       "maximum_cost_usd":str(MAX_TOTAL_USD),
                       "deadline_epoch":int(time.time())+allowed_seconds}
                upload_once(s3,bucket,prefix+"/allocation-intent-"+str(index)+".json",canonical_bytes(lease))
                # The watchdog reads this bounded active lease, separate from GPU.
                s3.put_object(Bucket=bucket,Key=prefix+"/active-lease.json",Body=canonical_bytes(lease))
                body={"name":name,"gpu":{"id":quoted.gpu_id,"count":1},"cloud":"SECURE",
                      "image":IMAGE,"disk":110,
                      "mounts":{"global":[{"volumeId":GLOBAL_ID,"path":"/workspace-global"}]},
                      "cmd":cmd,"env":env}
                print("TRIAD_GLOBAL_ALLOCATION_TRY="+quoted.gpu_id+" quote="+str(quoted.price_usd_per_hour),flush=True)
                try:
                    status,candidate=adapter._request("POST","pods",body)
                except RunPodApiError as exc:
                    # Only explicit rejected capacity may advance to a distinct SKU.
                    detail=""
                    try:detail=exc.__cause__.read(4096).decode(errors="replace").lower()
                    except Exception:pass
                    print("TRIAD_GLOBAL_PROVIDER_HTTP_STATUS="+str(exc.status_code),flush=True)
                    categories={"capacity":("capacity","unavailable","no available","no machines","stock","no gpu","not enough resources"),"funds":("balance","insufficient funds","payment","credit"),"invalid":("validation","invalid","required field","unsupported","not supported"),"global_mount":("global volume","global storage","mount")}
                    matched=[k for k,terms in categories.items() if any(x in detail for x in terms)]
                    print("TRIAD_GLOBAL_PROVIDER_CATEGORY="+(",".join(matched) if matched else "unclassified"),flush=True)
                    # A definite HTTP 400 capacity rejection plus independent zero-Pod listing
                    # permits the next distinct allowed SKU; no repeat of an ambiguous POST.
                    if exc.status_code in (400,409,503) and "capacity" in matched and "funds" not in matched and "invalid" not in matched and not owned():
                        print("TRIAD_GLOBAL_SKU_CAPACITY_REFUSED",flush=True)
                        continue
                    raise RuntimeError("ambiguous or invalid Global allocation; no retry") from None
                if status not in (200,201,202) or not isinstance(candidate,dict) or not candidate.get("id"):
                    raise RuntimeError("unconfirmed allocation; independent watchdog reconciles")
                pod=candidate
                print("TRIAD_REAL_RUNPOD_POD_CREATED=true",flush=True)
                print("TRIAD_REAL_RUNPOD_POD_ID="+str(pod["id"]),flush=True)
                # Independently confirm exact provider-side mount and price.
                live=adapter.get_pod(str(pod["id"]))
                if live.get("name")!=name or live.get("mounts")!={"global":[{"volumeId":GLOBAL_ID,"path":"/workspace-global"}]}:
                    raise PermissionError("provider mount identity differs from frozen Global request")
                actual=Decimal(str(live.get("costPerHr",live.get("cost","-1"))))
                if not actual.is_finite() or actual<=0 or actual>quoted.price_usd_per_hour:
                    raise PermissionError("actual provider rate exceeded quoted bound")
                print("TRIAD_ACTUAL_GLOBAL_VOLUME="+GLOBAL_ID,flush=True)
                print("TRIAD_ACTUAL_DATACENTER="+str(live.get("dataCenterId","unknown")),flush=True)
                print("TRIAD_ACTUAL_GPU="+quoted.gpu_id,flush=True)
                print("TRIAD_ACTUAL_HOURLY_USD="+str(actual),flush=True)
                break
            if pod is None:raise RuntimeError("global secure capacity exhausted within bounded candidates")
            deadline=lease["deadline_epoch"]
            attempt=1
            while time.time()<deadline:
                require_live_watchdog(s3,bucket,prefix)
                latest=provider.account()
                if latest.autopay_enabled or latest.credit_balance_usd<=RESERVE_USD:
                    raise PermissionError("prepaid reserve reached")
                if attempt<=2:
                    raw=optional_read(s3,bucket,prefix+"/attempt-"+str(attempt)+".zip",5_000_000)
                    if raw is not None:
                        reader=BundleReader(raw,prefix)
                        checked=cold_verify_candidate(reader,bucket,prefix,checkout,head,image,case,docker)
                        if (checkout/SOURCE).read_bytes()!=(Path.cwd()/SOURCE).read_bytes() or digest((checkout/REGRESSION).read_bytes())!=case["fixed_test_sha256"]:
                            raise PermissionError("frozen independent source or regression changed")
                        accepted=checked.get("red_to_green_verified") is True
                        verdict="accepted" if accepted else "retry" if attempt==1 else "rejected"
                        feedback={"run_tag":args.run_tag,"attempt":attempt,"verdict":verdict,
                                  "trial_ref":checked["trial_ref"],
                                  "failure_excerpt":str(checked.get("candidate",{}).get("output_excerpt",""))[-1200:]}
                        upload_once(s3,bucket,prefix+"/independent-trial-"+str(attempt)+".json",canonical_bytes(checked))
                        upload_once(s3,bucket,prefix+"/feedback-"+str(attempt)+".json",canonical_bytes(feedback))
                        print("TRIAD_INDEPENDENT_ATTEMPT_"+str(attempt)+"="+verdict,flush=True)
                        attempt=3 if accepted else attempt+1
                exit_raw=optional_read(s3,bucket,prefix+"/worker-exit.txt",500)
                if exit_raw is not None:
                    if exit_raw.strip()!=b"0":raise RuntimeError("GPU worker failed; final scoped evidence retained")
                    if not accepted:raise RuntimeError("no independent accepted three-model repair")
                    print("TRIAD_GENUINE_THREE_MODEL_GLOBAL_REPAIR=PASS",flush=True)
                    return
                time.sleep(10)
            raise TimeoutError("independently bounded global experiment deadline")
        finally:
            # Reconcile even on ambiguous POST, never create another Pod.
            for item in owned():
                adapter.delete_pod(str(item["id"]))
                print("TRIAD_REAL_RUNPOD_POD_DELETED=true",flush=True)
            if owned():raise PermissionError("exact campaign Pod remains; watchdog continues")
            s3.put_object(Bucket=bucket,Key=prefix+"/controller-finished.json",Body=canonical_bytes({"run_tag":args.run_tag,"accepted":accepted}))


if __name__=="__main__":
    try:main()
    except Exception as exc:
        # Never print signed URLs or provider request payloads.
        print("TRIAD_CONTROLLER_FAILURE_CLASS="+type(exc).__name__,flush=True)
        raise SystemExit(1) from None
