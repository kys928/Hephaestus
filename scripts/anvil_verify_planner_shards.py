"""Read and attest the four Planner blobs on a read-only cache mount."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def verify(root: Path, manifest: dict) -> list[dict]:
    if manifest["revision"] != "81eaece1948f3875421d9a45bc55487d10e2d894":
        raise ValueError("Planner revision differs from certified registry")
    shards = manifest["shards"]
    if len(shards) != 4 or len({row["sha256"] for row in shards}) != 4 or any(
        not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        or type(row["bytes"]) is not int or row["bytes"] <= 0 for row in shards
    ) or sum(row["bytes"] for row in shards) != 17_836_123_192:
        raise ValueError("expected four distinct pinned Planner shards and exact byte ceiling")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for row in shards:
            digest, size = row["sha256"], row["bytes"]
            fd = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                    raise ValueError(f"unexpected blob type or length for {digest}")
                hasher = hashlib.sha256()
                remaining = size
                while remaining:
                    chunk = stream.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"truncated blob {digest}")
                    hasher.update(chunk)
                    remaining -= len(chunk)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns
                ) or hasher.hexdigest() != digest:
                    raise ValueError(f"blob changed or failed SHA-256 verification: {digest}")
    finally:
        os.close(root_fd)
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blobs", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manifest_sha256")
    args = parser.parse_args()
    with args.manifest.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("reviewed manifest digest does not match")
    manifest = json.loads(raw)
    verified = verify(args.blobs, manifest)
    print(json.dumps({"revision": manifest["revision"], "manifest_sha256": args.manifest_sha256,
                      "verified": verified}, sort_keys=True))


if __name__ == "__main__":
    main()
