"""Build only metadata for the first Anvil model pair, without reading LFS payloads."""
import hashlib
import json
from urllib.request import urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co"
AUX = frozenset({
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "tekken.json", "processor_config.json", "params.json",
})


def read(url, limit):
    with urlopen(url, timeout=20) as response:
        result = response.read(limit + 1)
    if len(result) > limit:
        raise ValueError("upstream response exceeds metadata budget")
    return result


def manifest():
    registry = json.loads((ROOT / "configs/models/hephaestus_role_model_stack_v1.json").read_bytes())
    output = {}
    for role, count in (("diagnosis", 2), ("planner", 4)):
        entry = registry["roles"][role]
        model, revision = entry["model_id"], entry["revision"]
        shard_names = {f"model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)}
        items = json.loads(read(f"{HF}/api/models/{model}/tree/{revision}", 1 << 20))
        files = {}
        for item in items:
            name = item["path"]
            if name not in AUX | shard_names:
                continue
            length = item["size"]
            if name in files or type(length) is not int or not 0 < length <= 6_000_000_000:
                raise ValueError("unexpected pinned file metadata")
            if item.get("lfs"):
                digest = item["lfs"]["oid"]
                if item["lfs"]["size"] != length:
                    raise ValueError("LFS length differs from tree length")
            else:
                if name in shard_names or length > 1 << 20:
                    raise ValueError("unsupported non-LFS payload")
                raw = read(f"{HF}/{model}/resolve/{revision}/{name}", length)
                if len(raw) != length:
                    raise ValueError("auxiliary length differs from pinned tree")
                digest = hashlib.sha256(raw).hexdigest()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid upstream digest")
            files[name] = {"sha256": digest, "bytes": length}
        if set(files) != AUX | shard_names:
            raise ValueError("incomplete upstream Transformers file set")
        output[role] = {"model_id": model, "revision": revision, "files": files,
                        "payload_bytes": sum(item["bytes"] for item in files.values())}
    return output


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=2, sort_keys=True))
