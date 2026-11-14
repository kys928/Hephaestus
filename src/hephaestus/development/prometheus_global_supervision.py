"""Bounded, source-pinned Prometheus Global Pod supervision primitives.

Privileged caller is a separately reviewed one-shot GitHub Actions runner.
This module cannot bypass RunPod/S3 authorization and never auto-promotes a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

GLOBAL_ID = "cmw51pjsk000007l9a0mv1jga"
S3_BUCKET = "cviwpryzao"
S3_REGION = "EU-CZ-1"
S3_ENDPOINT = "https://s3api-eu-cz-1.runpod.io/"
GPU_ID = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
ALLOWED_CENTERS = frozenset({"EU-CZ-1", "EU-NL-1", "EU-FR-1", "EU-RO-1", "EU-SE-1"})
SOURCE_PARENT = "2d1ffa37a9e14939c5c0e9c3365a5fe0a4220a37b502678a1f72f5def366378f"
SOURCE_D = "7398eeed835422151678e4e1a79ee4c0d8ead4daf9dba61dd4333aa9acfbf953"
MAX_PRICE_HOUR = Decimal("3.50")
MAX_LEASE_SECONDS = 5400
RESERVE_USD = Decimal("1.50")
HEARTBEAT_AGE_SECONDS = 90


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def campaign_prefix(campaign: str, source_sha: str) -> str:
    if not re.fullmatch(r"prometheus-unbound-global-[0-9]{8}-[0-9]{2}", campaign):
        raise ValueError("campaign must be a fresh and explicitly scoped identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("full immutable source SHA required")
    return "hephaestus/scientific/prometheus/unbound-global-v1/" + campaign + "/" + source_sha


def pod_name(prefix: str) -> str:
    return "prometheus-global-" + sha(prefix.encode())[:24]


def s3_client():
    import boto3
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=S3_REGION,
                        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])


def get_bytes(s3, key: str, maximum: int = 64 * 1024 * 1024) -> bytes:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    body = obj["Body"]
    try:
        data = body.read(maximum + 1)
    finally:
        body.close()
    if len(data) > maximum:
        raise ValueError("S3 object exceeds admitted read limit")
    return data


def get_optional(s3, key: str) -> bytes | None:
    try:
        return get_bytes(s3, key)
    except Exception as exc:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def put_once(s3, key: str, payload: bytes) -> str:
    if get_optional(s3, key) is not None:
        raise FileExistsError("immutable campaign S3 key already exists: " + key)
    digest = sha(payload)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=payload)
    if get_bytes(s3, key, len(payload) + 1) != payload:
        raise IOError("S3 immutable readback mismatch")
    return digest


def put_progress(s3, key: str, obj: dict) -> None:
    blob = canonical(obj)
    if len(blob) > 8192:
        raise ValueError("oversized progress marker")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=blob)


def read_json(s3, key: str) -> dict | None:
    raw = get_optional(s3, key)
    if raw is None:
        return None
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("invalid JSON status")
    return result


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_heartbeat(s3, prefix: str, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ready = read_json(s3, prefix + "/status/watchdog-ready.json")
    beat = read_json(s3, prefix + "/status/watchdog-heartbeat.json")
    if (not ready or not beat or ready.get("schema") != "prometheus-global-watchdog.v1"
            or beat.get("schema") != "prometheus-global-heartbeat.v1"
            or ready.get("prefix") != prefix or beat.get("prefix") != prefix
            or ready.get("pod_name") != pod_name(prefix)):
        raise PermissionError("independent watchdog not authenticated")
    expiry = datetime.fromisoformat(ready["hard_deadline"])
    observed = datetime.fromisoformat(beat["observed_at"])
    if expiry.tzinfo is None or observed.tzinfo is None:
        raise PermissionError("watchdog timestamps not timezone aware")
    age = (now - observed).total_seconds()
    if not -15 <= age <= HEARTBEAT_AGE_SECONDS:
        raise PermissionError("independent watchdog heartbeat missing/stale")
    if (expiry - now).total_seconds() < MAX_LEASE_SECONDS + 120:
        raise PermissionError("watchdog has insufficient remaining lease")
    return ready


def api(method: str, path: str, payload: dict | None = None):
    if method not in ("GET", "POST", "DELETE") or not path.startswith("/"):
        raise ValueError("unsupported RunPod operation")
    raw = None if payload is None else canonical(payload)
    req = urllib.request.Request("https://api.runpod.io/v2" + path,
                                 data=raw, method=method,
                                 headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as exc:
        if method == "DELETE" and exc.code == 404:
            return {}
        raise


def pods() -> list[dict]:
    result = api("GET", "/pods")
    rows = result if isinstance(result, list) else result.get("pods", result.get("data", []))
    if not isinstance(rows, list):
        raise ValueError("RunPod list shape invalid")
    return [p for p in rows if isinstance(p, dict)]


def matching_pods(prefix: str) -> list[dict]:
    name = pod_name(prefix)
    matches = [p for p in pods() if p.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError("duplicate paid Pod name; manual reconciliation required")
    return matches


def verify_pod(pod: dict, prefix: str) -> None:
    if (pod.get("name") != pod_name(prefix)
            or pod.get("cloud") != "SECURE"
            or pod.get("image") != IMAGE
            or pod.get("dataCenterId") not in ALLOWED_CENTERS
            or pod.get("mounts") != {"global": [{"volumeId": GLOBAL_ID, "path": "/workspace-global"}]}
            or not isinstance(pod.get("gpu"), dict)
            or pod["gpu"].get("id") != GPU_ID or pod["gpu"].get("count") != 1
            or not isinstance(pod.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{4,100}", pod["id"])):
        raise PermissionError("Pod identity/config differs from approved global campaign")


def delete_exact_pod(pod: dict, prefix: str) -> None:
    verify_pod(pod, prefix)
    api("DELETE", "/pods/" + pod["id"])


def balance() -> tuple[Decimal, Decimal]:
    query = {"query": "query PrometheusGlobalBalance { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}
    req = urllib.request.Request("https://api.runpod.io/graphql", data=canonical(query),
                                 headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=35) as reply:
        doc = json.load(reply)
    info = doc.get("data", {}).get("myself", {})
    if doc.get("errors") or info.get("isAutoPayEnabled") is not False:
        raise PermissionError("RunPod prepaid-only account status not verified")
    credit = Decimal(str(info["clientBalance"]))
    spending = Decimal(str(info["currentSpendPerHr"]))
    if not credit.is_finite() or not spending.is_finite() or credit < 0 or spending < 0:
        raise PermissionError("invalid authenticated RunPod balance/spend")
    return credit, spending


def authenticate_sources(parent: bytes, historical_d: bytes) -> None:
    if sha(parent) != SOURCE_PARENT or sha(historical_d) != SOURCE_D:
        raise PermissionError("historical research report SHA-256 mismatch")


def verify_manifest(s3, prefix: str, *, max_files: int = 1000) -> dict:
    manifest = read_json(s3, prefix + "/result/manifest.json")
    if not manifest or manifest.get("schema") != "prometheus-global-evidence-manifest.v1" or manifest.get("prefix") != prefix:
        raise PermissionError("durable GPU worker manifest missing or mismatched")
    items = manifest.get("files")
    if not isinstance(items, list) or len(items) > max_files or not items:
        raise PermissionError("invalid GPU receipt inventory")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise PermissionError("invalid evidence file receipt")
        rel = item.get("path")
        digest = item.get("sha256")
        size = item.get("bytes")
        if (not isinstance(rel, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{1,240}", rel)
                or rel.startswith("/") or ".." in Path(rel).parts or rel in seen
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int or size < 0 or size > 30 * 1024 * 1024):
            raise PermissionError("evidence receipt path/hash/size invalid")
        seen.add(rel)
        body = get_bytes(s3, prefix + "/result/files/" + rel, size)
        if len(body) != size or sha(body) != digest:
            raise PermissionError("durable S3 inference evidence readback invalid")
    return manifest
