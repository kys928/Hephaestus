#!/usr/bin/env python3
"""Boot smoke for Hephaestus on fresh RunPod CPU Pod with Global + Czech storage.

Loads actual five pinned PEFT adapter safetensors tensors from Global storage,
executes Hephaestus's real Python CLI/console, and puts mutable SQLite on local
ephemeral disk with an integrity-checked cold backup on the Czech network volume.
Does not start training, mutate old campaigns, or enable LLM role dispatch.
"""
from __future__ import annotations
import hashlib, io, json, os, pathlib, shutil, socket, sqlite3, subprocess, sys, tarfile, tempfile, time, urllib.request, zipfile
from datetime import datetime, timezone

REGIONAL_ID="cviwpryzao"
GLOBAL_ID="cmw51pjsk000007l9a0mv1jga"
REGIONAL=pathlib.Path("/workspace-state")
GLOBAL=pathlib.Path("/workspace-global")
MANIFEST_SHA="129d690ef9ae24ef141fbb0c1ce48280f033b33f03b691073c20231d16677638"
EXPORT=GLOBAL/"hephaestus/hf-restores"/("export-"+MANIFEST_SHA[:24])
ARCHIVE=GLOBAL/"hephaestus/migrations/hf-czech-20261113"/"czech-only-cabc61d948c78122a25470ce.tar"
ORIGINAL=GLOBAL/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
REPO_SHA=os.environ.get("HEPHAESTUS_BOOT_CODE_SHA","")
BUF=2*1024*1024
def hash_file(p):
    h=hashlib.sha256();n=0
    with p.open("rb") as r:
        while b:=r.read(BUF):h.update(b);n+=len(b)
    return h.hexdigest(),n

def require(ok,why):
    if not ok:raise RuntimeError(why)

def main():
    rid=os.environ.get("BOOT_RUN_ID","")
    require(rid.isdecimal(),"Invalid boot run ID")
    require(os.getenv("RUNPOD_GLOBAL_VOLUME_ID")==GLOBAL_ID,"Wrong advertised Global Volume")
    require(os.getenv("RUNPOD_NETWORK_VOLUME_ID")==REGIONAL_ID,"Wrong advertised regional volume")
    require(REGIONAL.is_mount() and GLOBAL.is_mount(),"Both storage mounts must be real")
    require(ORIGINAL.is_file(),"Historical R9 archive marker missing from Global")
    require(EXPORT.is_dir(),"Curated HF restore path missing")
    require(REPO_SHA and len(REPO_SHA)==40 and all(ch in "0123456789abcdef" for ch in REPO_SHA),"Git source revision missing")

    base=REGIONAL/("hephaestus/runtime/boot-"+rid)
    require(not base.is_symlink(),"Runtime root must not be symlink")
    base.mkdir(parents=True,exist_ok=True)
    status_path=base/"boot-status.json"
    def status(stage,**kwargs):
        record={"schema":"hephaestus.boot-smoke.v1","run_id":rid,"phase":stage,
            "global_volume":GLOBAL_ID,"regional_volume":REGIONAL_ID,
            "updated_at":datetime.now(timezone.utc).isoformat(),**kwargs}
        # status is an operational report, not a scientific certificate.
        status_path.write_text(json.dumps(record,sort_keys=True)+"\n")
        import boto3
        client=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
            aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
        client.put_object(Bucket=REGIONAL_ID,Key="hephaestus/boot-status/"+rid+".json",
            Body=(json.dumps(record,sort_keys=True)+"\n").encode(),ContentType="application/json")
        print("HEPHAESTUS_BOOT_STATUS="+json.dumps({k:v for k,v in record.items() if k not in {"updated_at"}},sort_keys=True),flush=True)

    status("MOUNT_CONFIRMED")
    # Repo files go onto container disk. Global is never used for package install.
    archive_url="https://codeload.github.com/kys928/Hephaestus/zip/"+REPO_SHA
    raw=urllib.request.urlopen(archive_url,timeout=120).read()
    code_digest=hashlib.sha256(raw).hexdigest()
    zip_path=pathlib.Path("/tmp/hephaestus-source.zip")
    zip_path.write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extractall("/tmp/hephaestus-repo")
    children=[p for p in pathlib.Path("/tmp/hephaestus-repo").iterdir() if (p/"pyproject.toml").is_file()]
    require(len(children)==1,"Unexpected GitHub checkout tree")
    code=children[0]
    status("CODE_DOWNLOADED",github_commit=REPO_SHA,zip_sha256=code_digest)
    subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","--no-cache-dir","-e",str(code),"safetensors>=0.5,<1","numpy>=1.26,<3"],check=True,stdout=subprocess.DEVNULL)
    env=os.environ.copy()
    env["PYTHONUNBUFFERED"]="1"
    state_root=pathlib.Path("/tmp/hephaestus-console-state")
    state_root.mkdir(exist_ok=True)
    from hephaestus.state.run_store import RunStore
    existing=RunStore(state_root).get("boot-"+rid)
    if existing is not None:
        require(existing.get("status")=="completed",
                "Preexisting demo run exists but is not completed")
    else:
        demo=subprocess.run([sys.executable,"-m","hephaestus.cli.create_demo_state",
            "--state-root",str(state_root),"--run-id","boot-"+rid,"--format","json"],
            env=env,text=True,capture_output=True,timeout=90)
        if demo.returncode:
            # CLI contains only a synthetic public demo; expose the final
            # traceback line to distinguish a coding error from an interrupted run.
            tail=(demo.stderr or "").splitlines()[-1:] or ["no stderr"]
            raise RuntimeError("Demo boot subprocess failed: "+str(tail[0])[:180])
    doctor=subprocess.run([sys.executable,"-m","hephaestus.cli.doctor",
        "--state-root",str(state_root),"--format","json"],env=env,text=True,capture_output=True,timeout=80)
    require(doctor.returncode==0 and json.loads(doctor.stdout).get("status") in {"ok","warning"},"Hephaestus doctor failed")
    replay=subprocess.run([sys.executable,"-m","hephaestus.cli.verify_replay",
        "--state-root",str(state_root),"--run-id","boot-"+rid,"--format","json"],env=env,text=True,capture_output=True,timeout=60)
    require(replay.returncode==0,"Hephaestus replay CLI failed")
    # Preserve a cold copy of JSON/inspection state, not active SQLite.
    snapshot_dir=base/"console-state-cold"
    if not snapshot_dir.exists():
        shutil.copytree(state_root,snapshot_dir,symlinks=False)
    require(snapshot_dir.is_dir() and not snapshot_dir.is_symlink(),
            "Cold console-state backup is not a directory")
    status("APP_IMPORTS_AND_DEMO_OK",doctor=json.loads(doctor.stdout).get("status"),demo_run="boot-"+rid)

    # These five archives are the real originally trained role adapters,
    # not placeholder model manifests. Verify their immutable registry hashes.
    from hephaestus.providers.models.role_stack import load_certified_role_model_stack
    from safetensors import safe_open
    registry=load_certified_role_model_stack(code/"configs/models/hephaestus_role_model_stack_v1.json")
    count=0;entries={}
    for role,spec in registry.roles.items():
        key=spec.adapter.s3_key
        require(key.startswith("hephaestus/"),"Unexpected source adapter key")
        rel=key[len("hephaestus/"):]
        path=EXPORT/"evidence"/rel
        require(path.is_file() and not path.is_symlink(),"Trained role adapter missing on Global")
        digest,size=hash_file(path)
        require((digest,size)==(spec.adapter.sha256,spec.adapter.bytes),"Trained role archive checksum mismatch")
        with tarfile.open(path,"r:gz") as tar:
            cfg=tar.extractfile("adapter/adapter_config.json")
            member=tar.extractfile("adapter/adapter_model.safetensors")
            require(cfg is not None and member is not None,"Incomplete selected-adapter archive")
            config=json.loads(cfg.read())
            require(config.get("peft_type")=="LORA","Expected LoRA adapter")
            adapter_bytes=member.read()
            # Write to local disk and actually materialize a tensor, not
            # merely checking a file's extension or a manifest.
            temp=pathlib.Path("/tmp")/("hephaestus-"+role+"-adapter.safetensors")
            temp.write_bytes(adapter_bytes)
            try:
                with safe_open(str(temp),framework="np",device="cpu") as tensors:
                    names=list(tensors.keys())
                    require(names,"Adapter safetensors contains no tensor")
                    selected=None
                    for n in names:
                        sl=tensors.get_slice(n)
                        shape=sl.get_shape()
                        numel=1
                        for v in shape:numel*=v
                        if numel<=1000000:
                            selected=n;break
                    require(selected is not None,"No reasonably small adapter tensor available")
                    tensor=tensors.get_tensor(selected)
                    require(tensor.size>0,"Adapter tensor load produced empty weights")
                entries[role]={"archive_sha256":digest,"tensor_count":len(names),"one_tensor_loaded":True}
                count+=1
            finally:temp.unlink(missing_ok=True)
    require(count==5,"Not all five model adapters loaded")
    status("FIVE_REAL_ROLE_ADAPTERS_LOADED",loaded_roles=count,role_names=sorted(entries))

    # No SQLite execution from the object-backed Global or network FUSE.
    # Fresh local SQLite is used for health smoke. Save a consistent *cold*
    # SQLite backup to the regional durable volume for later controlled restore.
    local=pathlib.Path("/tmp/hephaestus-running-state")
    local.mkdir(parents=True,exist_ok=True)
    sqlite_path=local/"infrastructure.sqlite3"
    infra_env=env.copy()
    infra_env.update({
        "HEPHAESTUS_STATE_ROOT":str(local),
        "HEPHAESTUS_ARTIFACT_ROOT":str(EXPORT),
        "HEPHAESTUS_SQLITE_PATH":str(sqlite_path),
        "HEPHAESTUS_QUEUE_BACKEND":"sqlite",
        "HEPHAESTUS_DATABASE_BACKEND":"sqlite",
        "HEPHAESTUS_WORKER_ID":"runpod-boot-"+rid
    })
    mig=subprocess.run([sys.executable,"-m","hephaestus.infrastructure.cli","migrate","--json"],env=infra_env,capture_output=True,text=True,timeout=80)
    require(mig.returncode==0,"Hephaestus SQLite infrastructure migrate failed")
    health=subprocess.run([sys.executable,"-m","hephaestus.infrastructure.cli","health","--json"],env=infra_env,capture_output=True,text=True,timeout=80)
    require(health.returncode==0,"Hephaestus infrastructure health failed")
    sqlite3.connect(str(sqlite_path)).close()
    backup=base/"infrastructure-cold-backup.sqlite3"
    with sqlite3.connect(str(sqlite_path)) as a,sqlite3.connect(str(backup)) as b:
        a.backup(b)
    with sqlite3.connect(str(backup)) as db:
        require(db.execute("PRAGMA integrity_check").fetchone()[0]=="ok","Cold SQLite backup failed integrity")
    snapshot_sha,_=hash_file(backup)
    status("LOCAL_SQLITE_READY_AND_COLD_BACKED_UP",backup_sha256=snapshot_sha)

    console=subprocess.Popen([sys.executable,"-m","hephaestus.app.console","--state-root",str(state_root),
        "--host","127.0.0.1","--port","8765"],env=infra_env,stdout=(base/"console.stdout.log").open("ab"),stderr=subprocess.STDOUT)
    response=None
    for _ in range(24):
        if console.poll() is not None:break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8765/healthz",timeout=3) as r:
                response=r.read(5000)
            if response is not None:break
        except Exception:
            time.sleep(1)
    require(response is not None and console.poll() is None,"Hephaestus console did not boot")
    status("READY",console="http://127.0.0.1:8765",console_pid=console.pid,
        model_scope="five original LoRA adapters materialized; no base-model CUDA inference",
        source_of_truth="GitHub pinned main; HF-derived Global artifacts",
        runtime_state="local SQLite, cold backup on regional volume")
    # Keep the inspection console available within this Pod until it is stopped.
    # Do not expose the mutation-capable console on a public ingress.
    while console.poll() is None:
        time.sleep(30)
    status("CONSOLE_EXITED",return_code=console.returncode)
    raise RuntimeError("Inspection console process exited")

if __name__=="__main__":
    try:main()
    except Exception as exc:
        # never reveal HF/S3/GitHub secrets in a logged traceback
        print("HEPHAESTUS_BOOT_FAILED="+type(exc).__name__+": "+str(exc)[:200],flush=True)
        try:
            import boto3
            run=os.environ.get("BOOT_RUN_ID","")
            if run.isdecimal():
                client=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
                  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
                  aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
                client.put_object(Bucket=REGIONAL_ID,Key="hephaestus/boot-status/"+run+".json",
                  Body=json.dumps({"schema":"hephaestus.boot-smoke.v1","run_id":run,"phase":"FAILED",
                      "error_class":type(exc).__name__,"error_summary":str(exc)[:120]},sort_keys=True).encode(),
                  ContentType="application/json")
        except Exception:
            print("HEPHAESTUS_BOOT_FAILURE_RECEIPT_UNAVAILABLE=true",flush=True)
        raise SystemExit(1)
