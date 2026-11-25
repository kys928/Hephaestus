"""Verify pinned Czech Planner aliases and auxiliary payloads, without weight reads."""
import hashlib
import json
from pathlib import Path
import re
import sys

from anvil_verify_planner_shards import CZECH_VOLUME, PLANNER_PREFIX, czech_s3_client

ROOT = Path(__file__).resolve().parents[1]
# Canonical Git blob names from the retained Czech inventory. Independently
# pinned SHA-256 and exact byte length are still required for each read.
GIT_BLOBS = {
    "chat_template.jinja": "aaa7e7bf4ecc13b1c76bbce11269442857a8a726",
    "config.json": "03e03146fcd00c01a19dfeefa5cc2e69081ff31d",
    "generation_config.json": "add11cbc06647495098ee6dd5c9cbc96841a445a",
    "model.safetensors.index.json": "c2b0eca7d7b6eaf9f94917e48f678c947db5f4e1",
    "params.json": "ab8216614d16eafb153643fffe36992f9f454e88",
    "processor_config.json": "a37d728b12fd27ac60a437894bd51de83449bf30",
    "special_tokens_map.json": "1a339be8e293fc04bace0e5ec4ca86c7990a618d",
    "tokenizer_config.json": "a0eeb06d09b8a240ad7b9c8287955f7bc52033f4",
}


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
        blob = GIT_BLOBS.get(name, item["sha256"])
        weights = name.endswith(".safetensors")
        alias_size = client.head_object(Bucket=CZECH_VOLUME, Key=key)["ContentLength"]
        if alias_size in {52, 76}:
            alias = bounded_object(client, key, 76, {alias_size}).decode("ascii")
            match = re.fullmatch(r"\.\./\.\./blobs/([a-f0-9]{40}|[a-f0-9]{64})", alias)
            if not match or match.group(1) != blob:
                raise ValueError(f"unexpected cache alias for {name}")
            alias_observation = "pointer_verified"
        elif alias_size == item["bytes"]:
            alias_observation = "dereferenced_size_only"
        else:
            raise ValueError(f"unexpected alias length for {name}: {alias_size}")
        if not weights:
            data = bounded_object(client, PLANNER_PREFIX + blob, item["bytes"], {item["bytes"]})
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"auxiliary SHA-256 differs for {name}")
            (target / name).write_bytes(data)
        receipt = {"name": name, "blob": blob, "bytes": item["bytes"], "sha256": item["sha256"],
                   "alias_observation": alias_observation,
                   "evidence": "prior_weight_receipt" if weights else "verified_auxiliary_bytes"}
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
