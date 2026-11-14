"""Source-pinned, one-shot off-Pod RunPod Global scientific GPU controller.

Must run only in a separately reviewed, credentialed, one-time GitHub Actions
workflow AFTER the independent watchdog has published a fresh heartbeat.
No training, protected evaluation or model promotion authorization exists.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import time
from zipfile import ZipFile

from .prometheus_global_supervision import (
    ALLOWED_CENTERS, GLOBAL_ID, GPU_ID, IMAGE, MAX_LEASE_SECONDS,
    MAX_PRICE_HOUR, RESERVE_USD, S3_BUCKET, SOURCE_D, SOURCE_PARENT,
    api, authenticate_sources, balance, campaign_prefix, canonical,
    check_heartbeat, delete_exact_pod, get_bytes, iso_now,
    matching_pods, pod_name, put_once, read_json, s3_client, sha,
    verify_manifest, verify_pod,
)

PARENT_ARTIFACT = (13233709387, "granite-original-parent-dev-aggregate.json", SOURCE_PARENT)
D_ARTIFACT = (13217803440, "granite-d-readonly-rescore.json", SOURCE_D)
MAX_CODE_BUNDLE = 16 * 1024 * 1024
MIN_LEASE = 1800


def _download_source_artifact(artifact_id: int, filename: str, expected_hash: str) -> bytes:
    import requests
    token = os.environ["GH_TOKEN"]
    url = ("https://api.github.com/repos/kys928/Hephaestus/actions/artifacts/"
           + str(artifact_id) + "/zip")
    response = requests.get(url, headers={"Authorization": "Bearer " + token,
                                           "Accept": "application/vnd.github+json"},
                            timeout=65, allow_redirects=True)
    response.raise_for_status()
    if not response.url.startswith("https://"):
        raise PermissionError("artifact download left TLS")
    if len(response.content) > 2 * 1024 * 1024:
        raise PermissionError("historical archive exceeds expected small evidence export")
    with ZipFile(BytesIO(response.content)) as bundle:
        if bundle.namelist() != [filename]:
            raise PermissionError("historical evidence archive member set changed")
        source = bundle.read(filename)
    if sha(source) != expected_hash:
        raise PermissionError("historical GitHub Actions report SHA mismatch")
    return source


def _source_tarball(expected_source_sha: str) -> bytes:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_source_sha:
        raise PermissionError("GitHub source checkout drift from pinned source SHA")
    tar = subprocess.check_output(["git", "archive", "--format=tar.gz",
                                   "--prefix=source/", "HEAD"])
    if not tar or len(tar) > MAX_CODE_BUNDLE:
        raise PermissionError("pinned research source archive absent/oversized")
    return tar


def _pod_command(lease: int) -> list[str]:
    if not MIN_LEASE <= lease <= MAX_LEASE_SECONDS:
        raise ValueError("unsafe paid GPU lease")
    # No unpinned remote script or unverified GitHub branch is executed.
    # Worker first retrieves the exact source tarball and hashes it.
    boot = r"""set -Eeuo pipefail
umask 077
python -m pip install -q 'boto3>=1.35,<2'
python - <<'PY'
import os,hashlib,pathlib,tarfile,boto3
s3=boto3.client('s3',endpoint_url='https://s3api-eu-cz-1.runpod.io/',
    region_name='EU-CZ-1',aws_access_key_id=os.environ['RUNPOD_S3_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['RUNPOD_S3_SECRET_ACCESS_KEY'])
prefix=os.environ['PROMETHEUS_GLOBAL_PREFIX']
raw=s3.get_object(Bucket='cviwpryzao',Key=prefix+'/source/code.tar.gz')['Body'].read(16777217)
if hashlib.sha256(raw).hexdigest()!=os.environ['PROMETHEUS_CODE_SHA256'] or len(raw)>16777216:
    raise PermissionError('Prometheus pinned code bundle mismatch')
target=pathlib.Path('/tmp/prometheus-extracted')
target.mkdir(mode=0o700,parents=True,exist_ok=True)
archive=pathlib.Path('/tmp/prometheus-code.tar.gz')
archive.write_bytes(raw)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    if len(members)>5000:raise PermissionError('Source archive member limit')
    for m in members:
        path=pathlib.PurePosixPath(m.name)
        if (not path.parts or path.parts[0]!='source' or '..' in path.parts
                or path.is_absolute() or not (m.isdir() or m.isfile())):
            raise PermissionError('Unsafe pinned code tar member')
    tar.extractall(target,filter='data')
PY
cd /tmp/prometheus-extracted/source
test -f pyproject.toml
python -m pip install -q -e '.[role-inference,peft-training,s3]'
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0
timeout --signal=TERM --kill-after=30s __TIMEOUT__ python -u -m hephaestus.development.prometheus_global_envelope
"""
    return ["bash", "-lc", boot.replace("__TIMEOUT__", str(lease - 90))]


def _credit_lease() -> tuple[int, Decimal, Decimal]:
    credit, spending = balance()
    # One bounded GPU and other running Pods share the same prepaid wallet.
    remainder = credit - RESERVE_USD
    seconds = int((remainder * Decimal(3600) / (MAX_PRICE_HOUR + spending)).to_integral_value(
        rounding=ROUND_FLOOR)) if remainder > 0 else 0
    lease = min(MAX_LEASE_SECONDS, seconds)
    if lease < MIN_LEASE:
        raise PermissionError("prepaid credit insufficient for one supervised science window")
    return lease, credit, spending


def _verify_global_volume():
    doc = api("GET", "/global-volumes/" + GLOBAL_ID)
    volume = doc.get("globalVolume", doc.get("data", doc)) if isinstance(doc, dict) else {}
    if volume.get("id") != GLOBAL_ID:
        raise PermissionError("Global Volume identity not verified by live provider")


def _verify_worker_science(s3, prefix: str, manifest: dict) -> dict:
    if (manifest.get("worker_exit_code") != 0
            or manifest.get("source_parent_sha256") != SOURCE_PARENT
            or manifest.get("source_D_sha256") != SOURCE_D
            or manifest.get("global_volume_id") != GLOBAL_ID):
        raise PermissionError("GPU worker did not complete authenticated original science")
    names = {x["path"] for x in manifest["files"]}
    if not {"output/cycle/campaign-result.json",
            "output/cycle/decision/scientific-decision.json"} <= names:
        raise PermissionError("genuine scientific-decision files missing from durable manifest")
    if not any(n.startswith("output/cycle/decision/inference-receipts/") for n in names):
        raise PermissionError("original role inference receipts not durably retained")
    result = json.loads(get_bytes(
        s3, prefix + "/result/files/output/cycle/campaign-result.json", 256 * 1024))
    if (result.get("schema") != "prometheus-actual-gpu-cycle-result.v1"
            or result.get("model_inference_executed") is not True
            or result.get("new_candidate_trained") is not False
            or result.get("independent_certification") is not False
            or result.get("production_promotion") is not False):
        raise PermissionError("scientific completion claim violates original campaign boundary")
    return result


def launch(campaign: str, source_sha: str, *, clock=time) -> dict:
    prefix = campaign_prefix(campaign, source_sha)
    s3 = s3_client()
    check_heartbeat(s3, prefix)
    _verify_global_volume()
    if matching_pods(prefix):
        raise PermissionError("campaign Pod already allocated; refuse duplicate")
    if read_json(s3, prefix + "/launch-intent.json") is not None:
        raise PermissionError("one-shot campaign launch intent already consumed")
    lease, credit, spending = _credit_lease()
    parent = _download_source_artifact(*PARENT_ARTIFACT)
    historical_d = _download_source_artifact(*D_ARTIFACT)
    authenticate_sources(parent, historical_d)
    code = _source_tarball(source_sha)
    code_hash = sha(code)
    # Immutable staging: preserve each historical report byte-for-byte.
    put_once(s3, prefix + "/source/original-parent-dev.json", parent)
    put_once(s3, prefix + "/source/historical-d-dev.json", historical_d)
    put_once(s3, prefix + "/source/code.tar.gz", code)
    intent = {"schema": "prometheus-global-launch-intent.v1",
              "source_sha": source_sha, "prefix": prefix, "code_sha256": code_hash,
              "historical_parent_sha256": SOURCE_PARENT, "historical_D_sha256": SOURCE_D,
              "global_volume_id": GLOBAL_ID, "gpu": GPU_ID, "image": IMAGE,
              "max_lease_seconds": lease, "prepaid_usd": str(credit),
              "existing_spend_per_hour_usd": str(spending),
              "max_gpu_hourly_usd": str(MAX_PRICE_HOUR),
              "autonomous_training_allowed": False, "protected_holdout_allowed": False,
              "production_promotion_allowed": False, "recorded_at": iso_now()}
    check_heartbeat(s3, prefix)
    put_once(s3, prefix + "/launch-intent.json", canonical(intent))
    body = {
        "name": pod_name(prefix), "gpu": {"id": GPU_ID, "count": 1},
        "cloud": "SECURE", "dataCenterIds": sorted(ALLOWED_CENTERS),
        "image": IMAGE, "disk": 120,
        "mounts": {"global": [{"volumeId": GLOBAL_ID, "path": "/workspace-global"}]},
        "cmd": _pod_command(lease),
        "env": {
            "RUNPOD_GLOBAL_VOLUME_ID": GLOBAL_ID,
            "RUNPOD_S3_ACCESS_KEY_ID": os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
            "RUNPOD_S3_SECRET_ACCESS_KEY": os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "PROMETHEUS_GLOBAL_PREFIX": prefix,
            "PROMETHEUS_GLOBAL_CAMPAIGN": campaign,
            "PROMETHEUS_SOURCE_SHA": source_sha,
            "PROMETHEUS_CODE_SHA256": code_hash,
            "PROMETHEUS_WORKER_SECONDS": str(lease - 300),
        },
    }
    check_heartbeat(s3, prefix)
    pod = None
    scientific_result = None
    try:
        try:
            pod = api("POST", "/pods", body)
        except Exception:
            # A timeout during Pod creation may hide a successful allocation.
            matches = matching_pods(prefix)
            if not matches:
                raise
            pod = matches[0]
        if not isinstance(pod, dict) or not pod.get("id"):
            raise RuntimeError("RunPod Pod create did not return usable identity")
        # Cross-check provider reality; never trust the requested mount alone.
        pod = api("GET", "/pods/" + pod["id"])
        verify_pod(pod, prefix)
        cost = Decimal(str(pod.get("costPerHr", pod.get("cost", "-1"))))
        if not cost.is_finite() or not (0 < cost <= MAX_PRICE_HOUR):
            raise PermissionError("actual provider hourly charge exceeded exact approved hardware bound")
        put_once(s3, prefix + "/allocation.json", canonical({
            "schema": "prometheus-global-allocation.v1",
            "pod_id": pod["id"], "pod_name": pod_name(prefix),
            "datacenter_id": pod["dataCenterId"],
            "observed_cost_per_hour": str(cost),
            "lease_seconds": lease, "source_sha": source_sha,
            "global_volume_id": GLOBAL_ID, "created_at": iso_now(),
        }))
        print("PROMETHEUS_GLOBAL_POD_ALLOCATED=" + pod["id"], flush=True)
        deadline = clock.monotonic() + lease
        while clock.monotonic() < deadline:
            done = read_json(s3, prefix + "/status/worker-exit.json")
            if done:
                if done.get("schema") != "prometheus-global-worker-exit.v1" or done.get("prefix") != prefix:
                    raise PermissionError("worker completion identity mismatch")
                manifest = verify_manifest(s3, prefix)
                if done.get("manifest_sha256") != sha(canonical(manifest)):
                    raise PermissionError("worker manifest reference mismatch")
                scientific_result = _verify_worker_science(s3, prefix, manifest)
                break
            current = matching_pods(prefix)
            if not current:
                raise RuntimeError("Pod disappeared before durable scientific completion")
            clock.sleep(30)
        if scientific_result is None:
            raise TimeoutError("no independently verified GPU inference receipt within safe lease")
        return scientific_result
    finally:
        if pod is not None:
            # Independent watchdog is also responsible for this exact Pod.
            for attempt in range(4):
                current = matching_pods(prefix)
                if not current:
                    break
                delete_exact_pod(current[0], prefix)
                clock.sleep(3)
            if matching_pods(prefix):
                raise RuntimeError("paid Pod termination not independently observed")
            print("PROMETHEUS_GLOBAL_EXACT_POD_ABSENT=true", flush=True)


def main() -> int:
    for required in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID",
                     "RUNPOD_S3_SECRET_ACCESS_KEY", "GH_TOKEN", "HF_TOKEN"):
        if not os.getenv(required):
            raise PermissionError("authenticated launch secret missing: " + required)
    if os.getenv("PROMETHEUS_GLOBAL_APPROVED") != "single-bounded-science-20261114":
        raise PermissionError("one-shot explicit operator approval missing")
    campaign = os.environ["PROMETHEUS_GLOBAL_CAMPAIGN"]
    source_sha = os.environ["PROMETHEUS_SOURCE_SHA"]
    result = launch(campaign, source_sha)
    print("PROMETHEUS_GLOBAL_AUTHENTIC_SCIENCE_COMPLETED=true", flush=True)
    print("PROMETHEUS_GLOBAL_ADMITTED_ACTION=" + str(result.get("scientific_action")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
