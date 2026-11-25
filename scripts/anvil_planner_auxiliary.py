"""Verify pinned Czech Planner aliases and auxiliary payloads, without weight reads."""
import hashlib
import json
from pathlib import Path
import re
import sys

from anvil_verify_planner_shards import CZECH_VOLUME, PLANNER_PREFIX, czech_s3_client

ROOT = Path(__file__).resolve().parents[1]


def bounded_object(client, key, limit, allowed_lengths):
    address = {"Bucket": CZECH_VOLUME, "Key": key}
    head = client.head_object(**address)
    size = head["ContentLength"]
    if size not in allowed_lengths or not 0 < size <= limit:
        raise ValueError("unexpected provider object length")
    result = client.get_object(**address)
    body = result["Body"]
    try:
        if result["ContentLength"] != size:
            raise ValueError("provider changed object length")
        data = body.read(size)
        if len(data) != size:
            raise ValueError("truncated object")
        return data
    finally:
        body.close()


def verify(client, target):
    manifest = json.loads((ROOT / "configs/models/anvil_priority_model_files.json").read_bytes())["planner"]
    model_prefix = PLANNER_PREFIX.removesuffix("blobs/")
    files = manifest["files"]
    verified = []
    if sum(item["bytes"] for name, item in files.items() if not name.endswith(".safetensors")) != 34_245_682:
        raise ValueError("unexpected auxiliary payload ceiling")
    for name, item in sorted(files.items()):
        key = f"{model_prefix}snapshots/{manifest['revision']}/{name}"
        alias = bounded_object(client, key, 76, {52, 76}).decode("ascii")
        match = re.fullmatch(r"\.\./\.\./blobs/([a-f0-9]{40}|[a-f0-9]{64})", alias)
        if not match:
            raise ValueError(f"unsafe cache alias for {name}")
        blob = match.group(1)
        weights = name.endswith(".safetensors")
        if len(blob) == 64 and blob != item["sha256"]:
            raise ValueError(f"content address differs for {name}")
        if weights:
            if blob != item["sha256"]:
                raise ValueError(f"weight alias differs from existing verified receipt for {name}")
        else:
            data = bounded_object(client, PLANNER_PREFIX + blob, item["bytes"], {item["bytes"]})
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"auxiliary SHA-256 differs for {name}")
            (target / name).write_bytes(data)
        receipt = {"name": name, "blob": blob, "bytes": item["bytes"], "sha256": item["sha256"],
                   "evidence": "verified_alias_prior_weight_receipt" if weights else "verified_auxiliary_bytes"}
        verified.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    index = json.loads((target / "model.safetensors.index.json").read_bytes())
    if set(index["weight_map"].values()) != {name for name in files if name.endswith(".safetensors")}:
        raise ValueError("pinned index differs from the verified weight aliases")
    return verified


if __name__ == "__main__":
    path = Path(sys.argv[1])
    path.mkdir(exist_ok=False)
    verify(czech_s3_client(), path)
