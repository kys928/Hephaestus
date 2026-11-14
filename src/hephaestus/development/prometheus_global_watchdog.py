"""Independent RunPod lifecycle watchdog; deploy in a SEPARATE Actions job.

A cancelled launcher cannot cancel this worker. It writes an authenticated
readiness and heartbeat into private S3 BEFORE any Pod may be created.
Only an exact-name, exact-GPU, exact-Global-mount Pod may be terminated.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import time

from .prometheus_global_supervision import (
    MAX_LEASE_SECONDS, campaign_prefix, canonical, delete_exact_pod,
    iso_now, matching_pods, pod_name, put_once, put_progress, read_json,
    s3_client, verify_manifest,
)

START_GRACE_SECONDS = 900
DEADLINE_GRACE_SECONDS = 900


def watch(campaign: str, source_sha: str, *, clock=time) -> str:
    prefix = campaign_prefix(campaign, source_sha)
    s3 = s3_client()
    name = pod_name(prefix)
    started = datetime.now(timezone.utc)
    hard_deadline = started + timedelta(seconds=MAX_LEASE_SECONDS + DEADLINE_GRACE_SECONDS)
    key = prefix + "/status/watchdog-ready.json"
    ready = {
        "schema": "prometheus-global-watchdog.v1",
        "prefix": prefix, "pod_name": name, "source_sha": source_sha,
        "started_at": started.isoformat(), "hard_deadline": hard_deadline.isoformat(),
        "separate_supervisor": True, "global_volume_only": True,
    }
    put_once(s3, key, canonical(ready))
    print("PROMETHEUS_GLOBAL_WATCHDOG_READY=true", flush=True)
    first = clock.monotonic()
    seen = False
    while datetime.now(timezone.utc) < hard_deadline:
        put_progress(s3, prefix + "/status/watchdog-heartbeat.json", {
            "schema": "prometheus-global-heartbeat.v1",
            "prefix": prefix, "observed_at": iso_now(),
        })
        matches = matching_pods(prefix)
        if matches:
            seen = True
            exit_doc = read_json(s3, prefix + "/status/worker-exit.json")
            if exit_doc:
                if (exit_doc.get("schema") != "prometheus-global-worker-exit.v1"
                        or exit_doc.get("prefix") != prefix):
                    raise PermissionError("worker-exit identity mismatch")
                try:
                    manifest = verify_manifest(s3, prefix)
                    verified = (exit_doc.get("manifest_sha256")
                                == __import__("hashlib").sha256(canonical(manifest)).hexdigest())
                except Exception:
                    verified = False
                delete_exact_pod(matches[0], prefix)
                put_once(s3, prefix + "/status/watchdog-termination.json", canonical({
                    "schema": "prometheus-global-watchdog-termination.v1",
                    "prefix": prefix, "reason": "worker_exit",
                    "durable_manifest_independently_verified": verified,
                    "pod_id": matches[0]["id"], "observed_at": iso_now(),
                }))
                print("PROMETHEUS_GLOBAL_EXACT_POD_TERMINATED=true", flush=True)
                return "terminated_after_worker"
        if not seen and clock.monotonic() - first >= START_GRACE_SECONDS:
            if not matching_pods(prefix):
                put_once(s3, prefix + "/status/watchdog-no-allocation.json",
                         canonical({"prefix": prefix, "observed_at": iso_now()}))
                print("PROMETHEUS_GLOBAL_NO_ALLOCATION=true", flush=True)
                return "no_allocation"
        clock.sleep(25)
    matches = matching_pods(prefix)
    for pod in matches:
        delete_exact_pod(pod, prefix)
    put_once(s3, prefix + "/status/watchdog-deadline.json", canonical({
        "schema": "prometheus-global-watchdog-deadline.v1",
        "prefix": prefix, "pod_count": len(matches), "observed_at": iso_now(),
    }))
    print("PROMETHEUS_GLOBAL_DEADLINE_TERMINATION_COUNT=" + str(len(matches)), flush=True)
    return "deadline"


def main() -> int:
    for secret in ("RUNPOD_API_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"):
        if not os.getenv(secret):
            raise PermissionError("watchdog cannot run without required separately scoped credentials")
    watch(os.environ["PROMETHEUS_GLOBAL_CAMPAIGN"], os.environ["PROMETHEUS_SOURCE_SHA"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
