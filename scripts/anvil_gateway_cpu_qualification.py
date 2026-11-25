"""Exercise the real launch argv, authenticated HTTP, and exact setup on CPU Docker.

No RunPod credential, allocation, model loading, or scientific inference.
"""
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import secrets
import subprocess
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hephaestus.providers.runpod.anvil_live_startup import gateway_launch

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
LOCK_HASH = "01dd1ff5af4241ac9b6f97f02009dcf21f30836fb3fa1687576eeece96d93c92"


def command(*argv):
    return subprocess.check_output(argv, cwd=ROOT, text=True).strip()


def request(base, token, path, raw=None):
    req = Request(base + path, data=raw, headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/octet-stream"},
        method="GET" if raw is None else "POST")
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read(2*1024*1024+1))
    except HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read(4097))
        except (ValueError, OSError):
            return exc.code, None


def marker_report(name):
    proc = subprocess.run(["docker","logs",name], text=True, capture_output=True)
    return [x.split("=",1)[1] for x in (proc.stdout + proc.stderr).splitlines()
            if x in {"ANVIL_GATEWAY_EVENT=bootstrap_entered",
                     "ANVIL_GATEWAY_EVENT=env_validated","ANVIL_GATEWAY_EVENT=listener_bound"}
            or x.startswith("ANVIL_GATEWAY_EVENT=bootstrap_failed:") and x.split(":")[-1].isidentifier()]


def trial(mode, source, archive, lock_hash):
    entry,cmd = gateway_launch(source)
    token=secrets.token_hex(32)
    name="anvil-gateway-"+mode
    # Local Docker can express only the first ENTRYPOINT argument separately;
    # the final process argv is identical to the explicit provider contract.
    flags = (["--entrypoint",entry[0]] if mode=="explicit" else
             ["--entrypoint",""] if mode=="original-cleared" else [])
    args = entry[1:]+cmd if mode=="explicit" else entry+cmd
    env = {"ANVIL_LIVE_TOKEN":token,"ANVIL_DEADLINE_UTC":
           (datetime.now(timezone.utc)+timedelta(minutes=12)).isoformat(),
           "ANVIL_SOURCE_SHA256":sha256(archive).hexdigest(),"ANVIL_PACKAGE_LOCK_SHA256":lock_hash}
    result={"mode":mode,"ready":False,"setup":False,"credential_free_fixture":True}
    try:
        # Capture startup output rather than leaking environment/commands.
        command("docker","run","-d","--name",name,"-p","127.0.0.1::8787",
                *[x for k,v in env.items() for x in ("-e",k+"="+v)],*flags,IMAGE,*args)
        address=command("docker","port",name,"8787/tcp").splitlines()[0]
        base="http://"+address
        end=monotonic()+35
        while monotonic()<end:
            try:
                if request(base,token,"/health")== (200,{"status":"ready"}):
                    result["ready"]=True
                    break
            except (URLError, OSError, ValueError):
                pass
            if command("docker","inspect","-f","{{.State.Running}}",name)!="true":
                break
            sleep(1)
        if mode=="explicit":
            assert result["ready"], "explicit bootstrap health failed"
            assert request(base,"0"*64,"/health")==(403,{"status":"denied"})
            assert request(base,token,"/unknown")==(404,{"status":"unknown"})
            assert request(base,token,"/execute/qualify",b"{}")[0]==400
            assert request(base,token,"/execute/setup",archive)==(202,{"status":"consumed"})
            assert request(base,token,"/execute/setup",archive)==(409,{"status":"consumed"})
            end=monotonic()+620
            while monotonic()<end:
                status,doc=request(base,token,"/result/setup")
                assert status==200
                if doc.get("status")=="complete":
                    assert doc["result"]=={"source_sha256":env["ANVIL_SOURCE_SHA256"],"lock_sha256":lock_hash}
                    result["setup"]=True
                    break
                assert doc.get("status")=="running", "exact gateway setup failed: "+str(doc.get("error_type","unknown"))
                sleep(3)
            assert result["setup"], "exact setup not complete"
            result["startup_markers"]=marker_report(name)
            assert result["startup_markers"][:3]==["bootstrap_entered","env_validated","listener_bound"]
        else:
            result["startup_markers"]=marker_report(name)
    except Exception as exc:
        result["error_type"]=type(exc).__name__
        if mode=="explicit":
            print("ANVIL_GATEWAY_CPU_TRIAL="+json.dumps(result,sort_keys=True),flush=True)
            raise
    finally:
        subprocess.run(["docker","rm","-f",name],capture_output=True)
    print("ANVIL_GATEWAY_CPU_TRIAL="+json.dumps(result,sort_keys=True),flush=True)


def main():
    source=(ROOT/"scripts/anvil_live_pod_gateway.py").read_bytes()
    lock=(ROOT/"requirements-anvil-live.lock").read_bytes()
    assert sha256(lock).hexdigest()==LOCK_HASH
    tar=subprocess.check_output(["git","archive","--format=tar","HEAD","src","scripts","configs/models",
        "docs/evidence/role-model-stack-v1/certification.json","requirements-anvil-live.lock"],cwd=ROOT)
    archive=gzip.compress(tar,mtime=0)
    assert len(tar)<=64*1024**2 and len(archive)<=16*1024**2
    # Reconstruct the old gateway exactly rather than attributing new markers to it.
    old=subprocess.check_output(["git","show","c280e97b5e917a638429bbb2889403290020767f:scripts/anvil_live_pod_gateway.py"],cwd=ROOT)
    trial("original-cleared",old,archive,LOCK_HASH)
    trial("original-default",old,archive,LOCK_HASH)
    trial("explicit",source,archive,LOCK_HASH)
    print("ANVIL_EXACT_IMAGE_GATEWAY_CPU_QUALIFIED=true",flush=True)


if __name__=="__main__":
    main()
