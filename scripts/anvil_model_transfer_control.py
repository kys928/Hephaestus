"""Once-per-phase CPU model staging with independent, exact resource cleanup."""
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen
import zlib

import anvil_global_audit_control as audit
from anvil_planner_auxiliary import PLANNER_PREFIX
from anvil_stage_priority_models import MANIFEST_REF
from hephaestus.storage.verified import canonical_bytes

audit.LEASE_MINUTES = 52
audit.WATCHDOG_FILE = "anvil-model-transfer-watchdog.yml"
STAGING = "/workspace-global/anvil-priority-models-v1-" + MANIFEST_REF[:16]
REPORT = "ANVIL_MODEL_TRANSFER_REPORT="
ROOT = Path(__file__).resolve().parents[1]


def worker_command(phase):
    source = (ROOT / "scripts/anvil_stage_priority_models.py").read_text()
    definitions = (ROOT / "scripts/anvil_planner_auxiliary.py").read_text().split(
        "GIT_BLOBS = ", 1,
    )[1].split("\n\n", 1)[0]
    manifest = (ROOT / "configs/models/anvil_priority_model_files.json").read_bytes()
    if sha256(manifest).hexdigest() != MANIFEST_REF:
        raise ValueError("unexpected embedded model manifest")
    payload = source.split('if __name__ == "__main__":', 1)[0].replace(
        "from anvil_planner_auxiliary import GIT_BLOBS, ROOT",
        "ROOT = Path('/tmp/anvil-pinned-model-manifest')\nGIT_BLOBS = " + definitions,
    )
    planner = f"/workspace-source/{PLANNER_PREFIX.rstrip('/')}"
    args = [STAGING] + (["--planner-blobs", planner] if phase == "stage" else [])
    payload += f"""
import sys
from time import sleep
try:
    if not all(os.path.ismount(path) for path in ('/workspace-global', '/workspace-source')):
        raise PermissionError("expected two real mounts")
    put = ROOT / "configs/models/anvil_priority_model_files.json"
    put.parent.mkdir(parents=True, exist_ok=False)
    put.write_bytes({manifest!r})
    sys.argv[1:] = {args!r}
    main()
    print({REPORT!r} + json.dumps({{"status": "complete", "phase": {phase!r}, "manifest": MANIFEST_REF}}), flush=True)
except Exception as e:
    print({REPORT!r} + json.dumps({{"status": "failed", "phase": {phase!r}, "type": type(e).__name__, "detail": str(e)[:180]}}), flush=True)
sleep(120)
"""
    code = b64encode(zlib.compress(payload.encode(), level=9)).decode()
    return ["python", "-u", "-c", f"import base64,zlib;exec(zlib.decompress(base64.b64decode('{code}')))"]


def read_logs(pod_id):
    request = Request(f"https://api.runpod.io/v2/pods/{pod_id}/logs?source=container&tail=80", headers={
        "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
        "Accept": "text/event-stream",
    })
    with urlopen(request, timeout=20) as response:
        while raw := response.readline(65_537):
            if len(raw) > 65_536 or not raw.startswith(b"data:"):
                continue
            row = json.loads(raw[5:])
            line = row.get("line", "")
            if row.get("source") == "container":
                yield line


def control(provider, phase, name, marker, base):
    deadline = audit.active_watchdog(marker)
    audit.audited_cpu_price(provider)
    status, volume = provider._request("GET", "global-volumes/" + audit.GLOBAL)
    if status != 200 or volume.get("id") != audit.GLOBAL:
        raise PermissionError("existing Global Volume identity could not be authenticated")
    print("GLOBAL_ADDITIONAL_STORAGE_MAX_USD_MONTH_AT_0_09_PER_DECIMAL_GB=2.3043", flush=True)
    if audit.matches(provider, name):
        raise RuntimeError("exact operation already allocated")
    body = {
        "name": base, "cpu": {"id": audit.SKU, "vcpuCount": audit.VCPUS}, "cloud": "SECURE",
        "dataCenterIds": [audit.DC], "image": "python:3.12-slim", "disk": 5,
        "mounts": audit.MOUNTS, "ports": [], "entrypoint": [],
        "cmd": worker_command(phase), "env": {"ANVIL_AUDIT_SHA": marker},
    }
    begin = monotonic()
    try:
        pod = provider._create_pod(body, operation_id=marker)
        print("ANVIL_EXACT_CPU_POD_ID=" + pod["id"], flush=True)
        live = provider.get_pod(pod["id"])
        audit.assert_pod_identity(live, name, marker)
        cpu = live.get("cpu") or {}
        rate = audit.usd(live.get("costPerHr", live.get("cost")))
        if (cpu.get("id") != audit.SKU or cpu.get("vcpuCount") != audit.VCPUS or live.get("gpu")
                or live.get("dataCenterId") != audit.DC or not 0 < rate <= Decimal("0.14")):
            raise PermissionError("accepted Pod compute or price differs from authorization")
        print(f"ANVIL_CPU_ACCEPTED_USD_HOUR={rate}", flush=True)
        seen = set()
        while datetime.now(timezone.utc) < deadline:
            try:
                for line in read_logs(pod["id"]):
                    if datetime.now(timezone.utc) >= deadline:
                        raise TimeoutError("independent teardown deadline reached")
                    if line.startswith("{") and line not in seen:
                        seen.add(line)
                        print("ANVIL_FILE_RECEIPT=" + line, flush=True)
                    if line.startswith(REPORT):
                        info = json.loads(line[len(REPORT):])
                        print(REPORT + json.dumps(info, sort_keys=True), flush=True)
                        if info.get("status") == "complete" and info.get("phase") == phase and info.get("manifest") == MANIFEST_REF:
                            return
                        raise RuntimeError("model worker failed closed")
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                print("ANVIL_LOG_TRANSPORT=" + type(exc).__name__, flush=True)
            sleep(5)
        raise TimeoutError("model transfer reached watchdog deadline")
    finally:
        audit.cleanup(provider, name, marker)
        print(f"ANVIL_CONTROLLER_SECONDS={monotonic()-begin:.1f}", flush=True)


def main():
    mode, phase = sys.argv[1:]
    marker = os.environ["GITHUB_SHA"]
    if (mode not in {"prime", "watchdog", "control"} or phase not in {"stage", "readback"}
            or os.environ.get("GITHUB_REPOSITORY") != "kys928/Hephaestus"
            or os.environ.get("GITHUB_REF") != "refs/heads/feature/anvil-integration"
            or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch("[a-f0-9]{40}", marker)):
        raise PermissionError("requires original exact staging workflow")
    provider = audit.adapter()
    operation = sha256(canonical_bytes([provider.config.api_base_url, audit.REGIONAL, marker])).hexdigest()
    base = "hephaestus-anvil-model-" + phase
    name = base + "-" + operation[:32]
    if mode == "prime":
        audit.prime_watchdog(provider, name)
    elif mode == "watchdog":
        audit.watchdog(provider, name, marker)
    else:
        control(provider, phase, name, marker, base)


if __name__ == "__main__":
    main()
