"""Build or independently verify a fresh, canonical offline cache for the original pair."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
from urllib.request import urlopen

from anvil_planner_auxiliary import GIT_BLOBS, ROOT

MANIFEST_REF = "d5193ac0f3c77d2098b093237034c0af1acef898bb01364d331966626b1d0f9a"
TOTAL_BYTES = 25_602_848_544


@contextmanager
def open_regular(path: Path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("model source must be a regular file")
        yield stream


def check_stream(stream, item, target=None):
    remaining = item["bytes"]
    digest = hashlib.sha256()
    while remaining:
        data = stream.read(min(remaining, 8 << 20))
        if not data or len(data) > remaining:
            raise ValueError("model payload length differs from manifest")
        remaining -= len(data)
        digest.update(data)
        if target is not None:
            target.write(data)
    if stream.read(1) or digest.hexdigest() != item["sha256"]:
        raise ValueError("model payload differs from manifest")


def run(destination: Path, *, planner_blobs: Path | None):
    raw = (ROOT / "configs/models/anvil_priority_model_files.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_REF:
        raise ValueError("priority manifest changed")
    manifest = json.loads(raw)
    if sum(item["bytes"] for role in manifest.values() for item in role["files"].values()) != TOTAL_BYTES:
        raise ValueError("unexpected priority payload ceiling")
    staging = planner_blobs is not None
    if staging:
        planner_blobs = planner_blobs.resolve(strict=True)
        if destination.is_relative_to(planner_blobs) or planner_blobs.is_relative_to(destination):
            raise ValueError("source and destination must be separate")
        destination.mkdir(exist_ok=False)
    for role, info in manifest.items():
        snapshot = destination / ("models--" + info["model_id"].replace("/", "--")) / "snapshots" / info["revision"]
        if staging:
            snapshot.mkdir(parents=True, exist_ok=False)
        elif snapshot.resolve(strict=True) != snapshot:
            raise ValueError("snapshot path traverses a link")
        for name, item in info["files"].items():
            path = snapshot / name
            if not staging:
                with open_regular(path) as stream:
                    check_stream(stream, item)
            else:
                if role == "planner":
                    source = open_regular(planner_blobs / GIT_BLOBS.get(name, item["sha256"]))
                else:
                    source = urlopen(f"https://huggingface.co/{info['model_id']}/resolve/{info['revision']}/{name}",
                                     timeout=30)
                with source as stream, path.open("xb") as target:
                    check_stream(stream, item, target)
            print(json.dumps({"role": role, "file": name, **item,
                              "observation": "staged_input_verified" if staging else "destination_readback_verified"}),
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--planner-blobs", type=Path, help="enable staging and read the original mounted Czech blobs")
    args = parser.parse_args()
    # Reject path aliases before both staging and independent readback.
    destination = args.destination.absolute()
    if destination.resolve() != destination:
        parser.error("destination must be a canonical mount path")
    run(destination, planner_blobs=args.planner_blobs)


if __name__ == "__main__":
    main()
