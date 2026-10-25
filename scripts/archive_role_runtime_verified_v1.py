#!/usr/bin/env python3
"""Freeze *observed* Hephaestus runtime evidence; never invent missing receipts.

Only GitHub Actions artifact GET and RunPod S3 GET are performed. Local output
is for a GitHub Release, not a tracked source-code artifact. All input bytes
and role-level inference receipt hashes must match the pinned manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED

def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)

def run() -> None:
    import boto3
    manifest = json.loads(Path("milestones/role-runtime-verified-v1/manifest.json").read_text())
    require(manifest["schema"] == "role-runtime-research-freeze.v1", "manifest identity")
    require(manifest["certification"]["autonomous_production"] is False, "cannot certify research")
    require(manifest["execution"]["cpu"]["completed_staged_operations"] == 5, "invalid CPU completion count")
    require(manifest["execution"]["gpu"]["accepted_frozen_role_inferences"] == 5, "invalid GPU role count")
    target = Path("release-evidence")
    target.mkdir(exist_ok=True)
    cpu_json = gpu_json = None
    original = {}
    for kind in ("cpu", "gpu"):
        item = manifest["sources"][kind]
        download_dir = Path("downloaded") / kind
        require(download_dir.is_dir(), kind + ": no GitHub Actions extraction")
        actual = {p.name for p in download_dir.iterdir() if p.is_file()}
        require(actual == set(item["members"]), kind + ": downloaded source names changed")
        members = {}
        for name, ref in item["members"].items():
            raw = (download_dir / name).read_bytes()
            require("sha256:" + digest(raw) == ref, kind + "/" + name + ": file changed")
            members[name] = raw
        if kind == "cpu":
            roles = json.loads(members["cpu-research-roles.json"])
            staged = json.loads(members["cpu-staged-prefix.json"])
            peft = json.loads(members["cpu-real-peft-and-judge.json"])
            require(len(roles["inference"]) == 5 and not roles["full_45_operation_campaign_executed"], "CPU model claim")
            require(len(staged["all_45_operation_coverage_matrix"]) == 45, "CPU inventory incomplete")
            require(sum(bool(r["completion_marker"]) for r in staged["all_45_operation_coverage_matrix"]) == 5, "CPU stage count")
            require(peft["certification_state"] == "not_certified" and not peft["promotion_applied"], "CPU certification mismatch")
            cpu_json = staged
        else:
            gpu_json = json.loads(members["cuda_role_output.json"])
            require(gpu_json["status"] == "five_frozen_role_inferences_completed", "GPU role status mismatch")
            require(len(gpu_json["role_results"]) == 5, "GPU role count mismatch")
            require(all(row["status"] == "accepted_model_generation" for row in gpu_json["role_results"]), "GPU role failed")
            require(all(not row["suggestion_executed"] for row in gpu_json["role_results"]), "unexpected GPU mutation")
        # GitHub Actions download-artifact extracts the original ZIP. Repack
        # original verified member bytes; record the NEW ZIP SHA separately.
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=8) as archive:
            for name in sorted(members):
                archive.writestr(name, members[name])
        blob = buffer.getvalue()
        fname = kind + "-verified-json-evidence.zip"
        (target / fname).write_bytes(blob)
        original[fname] = {
            "sha256": "sha256:" + digest(blob),
            "bytes": len(blob), "format": "repackaged_original_JSON_member_bytes",
            "source_artifact_original_zip_sha256": item["zip_sha256"],
            "source_members_verified_sha256": item["members"],
        }
    require(gpu_json is not None and cpu_json is not None, "missing entire scientific source")
    cfg = ("RUNPOD_S3_ENDPOINT_URL", "RUNPOD_NETWORK_VOLUME_ID",
           "RUNPOD_DATACENTER_ID", "RUNPOD_S3_ACCESS_KEY_ID",
           "RUNPOD_S3_SECRET_ACCESS_KEY")
    for key in cfg:
        require(bool(os.environ.get(key)), "missing read-only volume access: " + key)
    client = boto3.client(
        "s3", endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
    )
    bucket = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
    base = manifest["sources"]["gpu"]["runpod_key_prefix"].rstrip("/")
    def remote_get(key: str, limit: int) -> bytes:
        response = client.get_object(Bucket=bucket, Key=key)
        obj = response["Body"]
        try:
            result = obj.read(limit + 1)
        finally:
            obj.close()
        require(len(result) <= limit, "RunPod remote object too large")
        return result
    terminal = remote_get(base + "/role_result.json", 100_000)
    require("sha256:" + digest(terminal) == manifest["sources"]["gpu"]["terminal_sha256"],
            "RunPod original terminal receipt differs from archived GitHub version")
    require(json.loads(terminal) == gpu_json, "RunPod and Actions GPU data disagree")
    index = {
        "schema": "role-runtime-research-evidence-archive.v1",
        "milestone": manifest["milestone"],
        "source_github_commit": os.environ["GITHUB_SHA"],
        "terminal_runpod_sha256": "sha256:" + digest(terminal),
        "inference_receipt_hashes": {},
        "full_45_operations_executed": False,
        "production_certified": False,
    }
    buf = io.BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED, compresslevel=8) as dest:
        dest.writestr("original-cuda-role-result.json", terminal)
        for role in gpu_json["role_results"]:
            ref = role["inference_receipt_ref"]
            require(ref.startswith("sha256:") and len(ref) == 71, "invalid role receipt ref")
            hexsum = ref.partition(":")[2]
            key = base + "/immutable-receipts/objects/sha256/" + hexsum[:2] + "/" + hexsum
            blob = remote_get(key, 5_000_000)
            require(digest(blob) == hexsum, role["role"] + ": actual immutable receipt mismatch")
            index["inference_receipt_hashes"][role["role"]] = ref
            dest.writestr("frozen-roles/" + role["role"] + ".json", blob)
    blob = buf.getvalue()
    (target / "original-cuda-five-role-inference-receipts.zip").write_bytes(blob)
    original["original-cuda-five-role-inference-receipts.zip"] = {
        "sha256": "sha256:" + digest(blob), "bytes": len(blob)
    }
    index["archives"] = original
    (target / "evidence-index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "milestone": manifest["milestone"], "gpu_role_receipts_verified": len(index["inference_receipt_hashes"]),
        "cpu_staged_steps_verified": 5, "archives": original,
        "certification": "not_certified",
    }, sort_keys=True))

if __name__ == "__main__":
    run()
