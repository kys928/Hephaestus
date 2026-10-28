#!/usr/bin/env python3
"""Retain verified original GPU science on private content-addressed RunPod S3.

The publisher cannot issue scientific certification; independent review must
download the object in a fresh environment and recompute all member hashes.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from cold_verify_original_tier_a_cuda_archive import verify_original_cuda_archive


def _sha_file(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(1024*1024),b""):
            h.update(piece)
    return "sha256:"+h.hexdigest()


def retain_private_original_cuda_archive(path, *, client, bucket):
    path=Path(path)
    digest=_sha_file(path)
    verified=verify_original_cuda_archive(path,expected_sha256=digest)
    if verified.get("verified_native_operations")!=45:
        raise PermissionError("private evidence upload requires authenticated original GPU science")
    key="hephaestus/scientific/original_cuda_45/"+digest[7:]+"/original_cuda_45.zip"
    try:
        existing=client.head_object(Bucket=bucket,Key=key)
    except Exception as exc:
        code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
        if code not in {"404","NoSuchKey","NotFound"}:
            raise PermissionError("unable to confirm private immutable S3 key absence") from exc
    else:
        if int(existing.get("ContentLength",-1))!=path.stat().st_size:
            raise PermissionError("private immutable GPU archive key already exists with different size")
        # Do not overwrite; re-read unchanged original object.
    if "existing" not in locals():
        client.upload_file(str(path),bucket,key)
    with tempfile.TemporaryDirectory() as tmp:
        cold=Path(tmp)/"independently_downloaded_original.zip"
        client.download_file(bucket,key,str(cold))
        if _sha_file(cold)!=digest:
            raise PermissionError("remote original GPU archive bytes differ after upload")
        readback=verify_original_cuda_archive(cold,expected_sha256=digest)
    if readback!=verified:
        raise PermissionError("cold remote storage verification differs")
    return {
        "schema":"private-original-cuda45-s3-retention.v1",
        "volume":bucket,"s3_key":key,"archive_sha256":digest,
        "bytes":path.stat().st_size,
        "cold_storage_verified_native_operations":45,
        "cold_storage_verified_original_cuda_role_inferences":5,
        "cold_storage_verified_cuda_optimizer_steps":96,
        "independent_offworker_review_required":True,
        "external_historical_certification":False,
        "issue_102_complete":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    if a.receipt.exists():
        raise FileExistsError("do not overwrite original private retention receipt")
    cfg=RunPodConfig.from_env()
    client=RunPodNetworkVolumeStorage.from_boto3(
        cfg,EnvironmentSecretsProvider()).client
    result=retain_private_original_cuda_archive(
        a.archive,client=client,bucket=cfg.network_volume_id)
    a.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
