"""Bounded CPU readback controller and separately hosted exact-Pod watchdog."""
from __future__ import annotations

import argparse
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing, usd
from hephaestus.providers.runpod.execution import RunPodApiError, RunPodExecutionAdapter
from hephaestus.storage.verified import canonical_bytes

GLOBAL = "cmw51pjsk000007l9a0mv1jga"
REGIONAL = "cviwpryzao"
DC = "EU-CZ-1"
SKU = "cpu3c"
VCPUS = 2
MAX_RATE = Decimal("0.15")
RESERVE = Decimal("2")
LEASE_MINUTES = 12
WATCHDOG_FILE = "anvil-global-audit-watchdog.yml"
MOUNTS = {
    "global": [{"volumeId": GLOBAL, "path": "/workspace-global"}],
    "network": [{"volumeId": REGIONAL, "path": "/workspace-source"}],
}
GITHUB = "https://api.github.com/repos/kys928/Hephaestus"


def adapter():
    config = RunPodConfig.from_env({
        "RUNPOD_DATACENTER_ID": DC, "RUNPOD_NETWORK_VOLUME_ID": REGIONAL,
        "RUNPOD_S3_ENDPOINT_URL": "https://s3api-eu-cz-1.runpod.io",
        "RUNPOD_API_BASE_URL": "https://api.runpod.io/v2",
    })
    root = Path("/tmp") / ("anvil-cpu-journal-" + os.environ["GITHUB_RUN_ID"])
    root.mkdir(mode=0o700, exist_ok=True)
    return RunPodExecutionAdapter(config, EnvironmentSecretsProvider(), state_root=root)


def read_json(url, token, limit=65536):
    req = Request(url, headers={"Authorization": "Bearer " + token} if token else {})
    with urlopen(req, timeout=15) as response:
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("scoped JSON response exceeds size ceiling")
        return json.loads(raw)


def matches(provider, name):
    return [p for p in provider.list_pods() if p.get("name") == name]


def assert_pod_identity(pod, name, marker):
    if (
        pod.get("name") != name or pod.get("mounts") != MOUNTS
        or not isinstance(pod.get("env"), dict)
        or pod["env"].get("ANVIL_AUDIT_SHA") != marker
        or not re.fullmatch(r"[A-Za-z0-9]{4,40}", str(pod.get("id", "")))
    ):
        raise RuntimeError("provider Pod failed exact audit identity")


def cleanup(provider, name, marker):
    for attempt in range(3):
        rows = matches(provider, name)
        if not rows:
            print("AUDIT_EXACT_POD_ABSENCE_CONFIRMED=true", flush=True)
            return
        for row in rows:
            pod = provider.get_pod(row["id"])
            assert_pod_identity(pod, name, marker)
            provider.delete_pod(pod["id"])
            try:
                provider.get_pod(pod["id"])
            except RunPodApiError as exc:
                if exc.status_code != 404:
                    raise
        if attempt < 2:
            sleep(5)
    raise RuntimeError("provider did not confirm exact Pod absence")


def prime_watchdog(provider, name):
    if matches(provider, name):
        raise RuntimeError("audit resource already exists before watchdog start")
    print("WATCHDOG_READY=true", flush=True)


def watchdog(provider, name, marker):
    started = monotonic()
    seen = False
    try:
        while monotonic() - started < (LEASE_MINUTES + 2) * 60:
            pods = matches(provider, name)
            if len(pods) > 1:
                raise RuntimeError("duplicate exact audit resources")
            if pods:
                seen = True
                pod = provider.get_pod(pods[0]["id"])
                assert_pod_identity(pod, name, marker)
                cpu = pod.get("cpu") or {}
                if cpu.get("id") != SKU or cpu.get("vcpuCount") != VCPUS or pod.get("gpu"):
                    raise RuntimeError("unexpected compute in independent CPU audit fence")
                rate = pod.get("costPerHr", pod.get("cost"))
                if rate is not None and usd(rate) > MAX_RATE:
                    raise RuntimeError("provider price exceeded independent ceiling")
            elif seen:
                print("WATCHDOG_CONFIRMED_EXACT_POD_ABSENCE=true", flush=True)
                return
            sleep(15)
    finally:
        cleanup(provider, name, marker)


def active_watchdog(marker):
    token = os.environ["GITHUB_TOKEN"]
    deadline = monotonic() + 150
    while monotonic() < deadline:
        result = read_json(f"{GITHUB}/actions/workflows/{WATCHDOG_FILE}/runs?head_sha={marker}&per_page=5", token)
        for run in result["workflow_runs"]:
            if run["head_sha"] != marker or run["status"] != "in_progress" or run["run_attempt"] != 1:
                continue
            result_jobs = read_json(f"{GITHUB}/actions/runs/{run['id']}/jobs", token)
            for job in result_jobs["jobs"]:
                if job["name"] != "watchdog" or job["status"] != "in_progress":
                    continue
                by_name = {step["name"]: step for step in job["steps"]}
                if (
                    by_name.get("Prime exact audit watchdog", {}).get("conclusion") == "success"
                    and by_name.get("Watch for exact audit Pod", {}).get("status") == "in_progress"
                ):
                    started = datetime.fromisoformat(job["started_at"].replace("Z", "+00:00"))
                    elapsed = datetime.now(timezone.utc) - started
                    if timedelta(seconds=0) <= elapsed <= timedelta(minutes=5):
                        print("INDEPENDENT_WATCHDOG_RUN=" + str(run["id"]), flush=True)
                        return started + timedelta(minutes=LEASE_MINUTES)
        sleep(6)
    raise RuntimeError("independent exact-Pod watchdog is not active")


def audited_cpu_price(provider):
    provider_status, catalog = provider._request("GET", "catalog/cpus?include=AVAILABILITY&product=POD")
    cpus = catalog.get("cpus", []) if isinstance(catalog, dict) else catalog
    candidates = [cpu for cpu in cpus if cpu.get("id") == SKU]
    if provider_status != 200 or len(candidates) != 1:
        raise RuntimeError("exact CPU catalog quote unavailable")
    cpu = candidates[0]
    datacenters = [item["id"] if isinstance(item, dict) else item for item in cpu["dataCenters"]]
    if DC not in datacenters:
        raise RuntimeError("approved datacenter does not offer the exact CPU SKU")
    hour = usd(cpu["price"]["securePerVcpu"]) * VCPUS
    # Quote CPU plus the disk placement allowance pre-allocation.
    total = hour + Decimal("0.01")
    account = ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"]).account()
    maximum_account_draw = (total + account.already_spending_usd_per_hour) * Decimal(LEASE_MINUTES) / 60
    if not 0 < hour or total > MAX_RATE or account.autopay_enabled or account.credit_balance_usd < RESERVE + maximum_account_draw:
        raise PermissionError("authenticated CPU price, prepaid reserve or rate ceiling failed")
    print(f"CPU_QUOTED_USD_HOUR={hour} MAX_AUDIT_USD_HOUR={MAX_RATE} MAX_MINUTES={LEASE_MINUTES}", flush=True)


def stream_report(pod_id, deadline):
    url = f"https://api.runpod.io/v2/pods/{pod_id}/logs?source=container&tail=40"
    request = Request(url, headers={
        "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
        "Accept": "text/event-stream",
    })
    with urlopen(request, timeout=18) as response:
        while monotonic() < deadline:
            raw = response.readline(65_537)
            if not raw:
                return
            if len(raw) > 65_536 or not raw.startswith(b"data:"):
                continue
            entry = json.loads(raw[5:])
            line = entry.get("line", "")
            if entry.get("source") != "container" or not line.startswith("GLOBAL_AUDIT_REPORT="):
                continue
            yield json.loads(line.removeprefix("GLOBAL_AUDIT_REPORT="))


def audit(provider, name, marker, base_name):
    deadline = active_watchdog(marker)
    audited_cpu_price(provider)
    if matches(provider, name):
        raise RuntimeError("already allocated an audit Pod")
    if datetime.now(timezone.utc) + timedelta(minutes=5) >= deadline:
        raise RuntimeError("independent watchdog lease too close to expiry")
    worker = Path(__file__).with_name("anvil_global_filesystem_audit.py").read_bytes()
    code = b64encode(zlib.compress(worker, level=9)).decode()
    registry_raw = json.loads(Path("configs/models/hephaestus_role_model_stack_v1.json").read_bytes())
    registry = {"roles": {
        role: {k: spec[k] for k in ("model_id", "revision", "adapter")}
        for role, spec in registry_raw["roles"].items()
    }}
    body = {
        "name": base_name, "cpu": {"id": SKU, "vcpuCount": VCPUS}, "cloud": "SECURE",
        "dataCenterIds": [DC], "image": "python:3.12-slim", "disk": 5,
        "mounts": MOUNTS, "entrypoint": [],
        "cmd": ["python", "-c", f"import base64,zlib;exec(zlib.decompress(base64.b64decode('{code}')))"],
        "env": {"ANVIL_AUDIT_SHA": marker,
                "ANVIL_ROLE_REGISTRY": json.dumps(registry, separators=(",", ":"))},
    }
    try:
        created = provider._create_pod(body, operation_id=marker)
        print("AUDIT_CPU_POD_ID=" + created["id"], flush=True)
        live = provider.get_pod(created["id"])
        assert_pod_identity(live, name, marker)
        cpu = live.get("cpu") or {}
        reported_rate = live.get("costPerHr", live.get("cost"))
        if (
            cpu.get("id") != SKU or cpu.get("vcpuCount") != VCPUS or live.get("gpu")
            or live.get("dataCenterId") != DC or not 0 < usd(reported_rate) <= MAX_RATE
        ):
            raise RuntimeError("accepted CPU price or placement differs from scope")
        print("AUDIT_ACCEPTED_USD_HOUR=" + str(reported_rate), flush=True)
        report_seen = False
        previous = None
        transport_deadline = monotonic() + 150
        while datetime.now(timezone.utc) < deadline:
            try:
                for result in stream_report(created["id"], monotonic() + 22):
                    if result.get("schema") != "hephaestus.anvil-global-readback.v1":
                        raise RuntimeError("wrong worker report schema")
                    report_seen = True
                    if result != previous:
                        print("GLOBAL_AUDIT_REPORT=" + json.dumps(result, sort_keys=True), flush=True)
                        previous = result
                    if result["status"] == "complete":
                        return
                    if result["status"] == "failed":
                        raise RuntimeError("mounted audit failed")
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                code = getattr(exc, "code", None)
                print(f"AUDIT_REPORT_TRANSPORT={type(exc).__name__},status={code}", flush=True)
            if not report_seen and monotonic() >= transport_deadline:
                raise RuntimeError("paid audit stopped after the authenticated log stream failed to start")
            sleep(10)
        raise TimeoutError("CPU audit reached watchdog deadline")
    finally:
        cleanup(provider, name, marker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prime", "watchdog", "audit"))
    mode = parser.parse_args().mode
    marker = os.environ["GITHUB_SHA"]
    if (
        not re.fullmatch(r"[a-f0-9]{40}", marker)
        or os.environ["GITHUB_REPOSITORY"] != "kys928/Hephaestus"
        or os.environ["GITHUB_REF"] != "refs/heads/feature/anvil-integration"
        or os.environ["GITHUB_RUN_ATTEMPT"] != "1"
    ):
        raise PermissionError("fresh exact Anvil audit workflow required")
    provider = adapter()
    key = sha256(canonical_bytes([provider.config.api_base_url, REGIONAL, marker])).hexdigest()
    base_name = "hephaestus-anvil-cpu-audit"
    name = base_name + "-" + key[:32]
    if mode == "prime":
        prime_watchdog(provider, name)
    elif mode == "watchdog":
        watchdog(provider, name, marker)
    else:
        audit(provider, name, marker, base_name)


if __name__ == "__main__":
    main()
