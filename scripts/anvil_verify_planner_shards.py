"""Read and attest the four Planner blobs on a read-only cache mount."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import stat


CZECH_VOLUME = "cviwpryzao"
PLANNER_PREFIX = "hephaestus-cache/huggingface/hub/models--mistralai--Ministral-3-8B-Reasoning-2512/blobs/"


def validated_shards(manifest: dict) -> list[dict]:
    if manifest["revision"] != "81eaece1948f3875421d9a45bc55487d10e2d894":
        raise ValueError("Planner revision differs from certified registry")
    shards = manifest["shards"]
    if len(shards) != 4 or len({row["sha256"] for row in shards}) != 4 or any(
        not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        or type(row["bytes"]) is not int or row["bytes"] <= 0 for row in shards
    ) or sum(row["bytes"] for row in shards) != 17_836_123_192:
        raise ValueError("expected four distinct pinned Planner shards and exact byte ceiling")
    return shards


def verify_stream(stream, size: int, digest: str) -> None:
    hasher = hashlib.sha256()
    update_from_stream(stream, size, digest, hasher)
    if hasher.hexdigest() != digest:
        raise ValueError(f"incorrect stream SHA-256 for {digest}")


def update_from_stream(stream, size: int, digest: str, hasher) -> None:
    remaining = size
    while remaining:
        chunk = stream.read(min(8 * 1024 * 1024, remaining))
        if not chunk or len(chunk) > remaining:
            raise ValueError(f"incorrect stream length for {digest}")
        hasher.update(chunk)
        remaining -= len(chunk)


def verify(root: Path, manifest: dict) -> list[dict]:
    shards = validated_shards(manifest)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for row in shards:
            digest, size = row["sha256"], row["bytes"]
            fd = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                    raise ValueError(f"unexpected blob type or length for {digest}")
                verify_stream(stream, size, digest)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns
                ):
                    raise ValueError(f"blob changed or failed SHA-256 verification: {digest}")
    finally:
        os.close(root_fd)
    return shards


def verify_s3_ranges(client, address: dict, size: int, digest: str, range_bytes: int) -> None:
    hasher = hashlib.sha256()
    for first in range(0, size, range_bytes):
        last = min(first + range_bytes, size) - 1
        response = client.get_object(**address, Range=f"bytes={first}-{last}")
        length = last - first + 1
        with closing(response["Body"]) as stream:
            if (response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 206
                    or response.get("ContentRange") != f"bytes {first}-{last}/{size}"
                    or response.get("ContentLength") != length):
                raise ValueError(f"incorrect bounded range for {digest}")
            update_from_stream(stream, length, digest, hasher)
    if hasher.hexdigest() != digest:
        raise ValueError(f"incorrect stream SHA-256 for {digest}")


def verify_s3(client, manifest: dict, *, metadata_only: bool, chunked: bool = False) -> list[dict]:
    shards = validated_shards(manifest)
    for row in shards:
        digest, size = row["sha256"], row["bytes"]
        address = {"Bucket": CZECH_VOLUME, "Key": PLANNER_PREFIX + digest}
        head = client.head_object(**address)
        if head["ContentLength"] != size:
            raise ValueError(f"unexpected object length for {digest}")
        if metadata_only:
            continue
        verify_s3_ranges(client, address, size, digest, 64 * 1024 * 1024 if chunked else size)
    return shards


def czech_s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3", endpoint_url="https://s3api-eu-cz-1.runpod.io", region_name="EU-CZ-1",
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(connect_timeout=10, read_timeout=30, retries={"total_max_attempts": 1},
                      s3={"addressing_style": "path"}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blobs", help="local directory, czech-s3-head, czech-s3-stream, or czech-s3-chunked")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manifest_sha256")
    args = parser.parse_args()
    with args.manifest.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("reviewed manifest digest does not match")
    manifest = json.loads(raw)
    metadata_only = args.blobs == "czech-s3-head"
    chunked = args.blobs == "czech-s3-chunked"
    if metadata_only or chunked or args.blobs == "czech-s3-stream":
        checked = verify_s3(czech_s3_client(), manifest, metadata_only=metadata_only, chunked=chunked)
    else:
        checked = verify(Path(args.blobs), manifest)
    field = "metadata_checked" if metadata_only else "verified"
    print(json.dumps({"revision": manifest["revision"], "manifest_sha256": args.manifest_sha256,
                      field: checked}, sort_keys=True))


if __name__ == "__main__":
    main()
