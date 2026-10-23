"""Add S3-visible container startup stages to an otherwise frozen D pod shell."""
from __future__ import annotations
import re

BEGIN="""python -m pip install --disable-pip-version-check -q 'boto3>=1.35,<2'
persist_bootstrap() {
  write_bootstrap "$1"
  STAGE="$1" python - <<'PY'
import json, os, time, boto3
from botocore.config import Config
stage=os.environ['STAGE']
allowed={'container_started','base_packages_install_started',
         'repo_checkout_started','repo_checkout_complete',
         'environment_check_started','environment_install_started',
         'environment_ready','pack_build_started','runner_started','bootstrap_failed'}
if stage not in allowed:raise RuntimeError('unrecognized bootstrap stage')
record={'stage':stage,'timestamp_unix':time.time(),'remote_s3_bootstrap':True}
boto3.client('s3',
    endpoint_url=os.environ['RUNPOD_S3_ENDPOINT_URL'],
    region_name=os.environ['RUNPOD_DATACENTER_ID'],
    aws_access_key_id=os.environ['RUNPOD_S3_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['RUNPOD_S3_SECRET_ACCESS_KEY'],
    config=Config(retries={'mode':'standard','max_attempts':3})
).put_object(
    Bucket=os.environ['RUNPOD_NETWORK_VOLUME_ID'],
    Key=os.environ['HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY'],
    Body=json.dumps(record,sort_keys=True).encode()
)
# Save append-only stage receipt so the last successful stage survives ERR trap.
key=os.environ['HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY']
boto3.client('s3',
    endpoint_url=os.environ['RUNPOD_S3_ENDPOINT_URL'],
    region_name=os.environ['RUNPOD_DATACENTER_ID'],
    aws_access_key_id=os.environ['RUNPOD_S3_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['RUNPOD_S3_SECRET_ACCESS_KEY'],
    config=Config(retries={'mode':'standard','max_attempts':3})
).put_object(
    Bucket=os.environ['RUNPOD_NETWORK_VOLUME_ID'],
    Key=key.rsplit('/',1)[0]+'/events/'+str(time.time_ns())+'-'+stage+'.json',
    Body=json.dumps(record,sort_keys=True).encode()
)
print('D_CONTRAST_S3_BOOTSTRAP '+json.dumps(record,sort_keys=True),flush=True)
PY
}
"""

def instrument(command):
    if not isinstance(command,list) or len(command)!=3 or command[:2]!=["bash","-lc"]:
        raise ValueError("unexpected pod startup command")
    shell=command[2]
    if not isinstance(shell,str):
        raise ValueError("invalid container shell")
    first="write_bootstrap container_started"
    if shell.count(first)!=1 or "write_bootstrap() {" not in shell:
        raise ValueError("bootstrap template is unrecognized")
    stages=re.findall(r"(?m)^write_bootstrap ([a-z_]+)$",shell)
    if len(stages)<5 or stages[0]!="container_started" or stages[-1]!="runner_started":
        raise ValueError("unexpected number/order of bootstrap stages")
    patched=re.sub(r"(?m)^write_bootstrap ([a-z_]+)$",r"persist_bootstrap \1",shell)
    patched=patched.replace("persist_bootstrap container_started",
                            BEGIN+"persist_bootstrap container_started",1)
    old_trap="trap 'write_bootstrap bootstrap_failed' ERR"
    new_trap="trap 'failure_code=$?; trap - ERR; persist_bootstrap bootstrap_failed || true; exit $failure_code' ERR"
    if patched.count(old_trap)!=1:
        raise ValueError("existing bootstrap error trap changed")
    patched=patched.replace(old_trap,new_trap,1)
    if re.search(r"(?m)^write_bootstrap [a-z_]+$",patched):
        raise ValueError("bootstrap stage was not persisted")
    return ["bash","-lc",patched]
