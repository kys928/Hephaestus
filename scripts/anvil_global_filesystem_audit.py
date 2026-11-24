"""Read-only mounted audit. Publish only aggregates and independent hashes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import stat

SOURCE_ID="cviwpryzao"
GLOBAL_ID="cmw51pjsk000007l9a0mv1jga"
SOURCE=Path("/workspace-source")
GLOBAL=Path("/workspace-global")
DEST=GLOBAL/"hephaestus/hf-restores/export-129d690ef9ae24ef141fbb0c"
EVIDENCE=DEST/"evidence"
ARCHIVE=GLOBAL/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
INVENTORY_KEY="hephaestus/recovery-audit/source-inventory-43256848721.json"
INVENTORY_SHA256="046dde9ebe9a712665081794586d502de5722f7f815a61567377414b680f5fb8"
ROOTS={"hephaestus/scientific/":"scientific","hephaestus/research/":"research","Hephaestus/artifacts/":"legacy/artifacts"}
BAD=set(".git .hg .svn .cache .local .venv venv __pycache__ node_modules site-packages tmp temp private protected holdout holdouts secrets secret credentials keys status locks worktrees state eval evals evaluation evaluations eval_pack eval_packs protected_evals certification certifications".split())
BAD_SUFFIX=(".lock",".log",".tmp",".sqlite",".sqlite3",".db",".pem",".key",".p12",".pfx",".zip",".tar",".tar.gz",".tgz",".7z",".pyc",".pyo",".pkl",".pickle")
BAD_BITS=("secret","credential","access_key","api_key","password","token_secret","worker.log","watchdog","heartbeat","private_key","account.json","holdout","certification","private_eval","protected_eval")
MODEL_EXT={".safetensors",".gguf",".onnx",".pt",".pth",".ckpt"}
MODEL_CONF=set("adapter_config.json config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json tokenizer.model adapter_model.bin pytorch_model.bin model.safetensors.index.json".split())
DATA_EXT={".jsonl",".parquet",".arrow",".csv",".tsv",".npy",".npz",".bin"}
NON_ART={".py",".sh",".bat",".ps1",".ipynb",".so",".o",".exe",".dll"}
BUF=4*1024*1024
def reject(p):
 n=p.name.lower()
 return (not n or n.startswith(".") or n in {".env","id_rsa","id_ed25519","known_hosts","authorized_keys","config.json.bak"} or n.endswith(BAD_SUFFIX)
         or any(x in n for x in BAD_BITS) or any(x.lower() in BAD for x in p.parts) or p.suffix.lower() in NON_ART)
def kind(p):
 parts=[x.lower() for x in p.parts];n=p.name.lower();s=p.suffix.lower()
 model=any(x in {"models","model","adapters","adapter","tokenizers","tokenizer","weights"} for x in parts)
 data=any(x in {"dataset","datasets","training_data","corpus","corpora","preprocessed","trainable"} for x in parts)
 if any(x in n for x in ("optimizer","scheduler","rng_state","trainer_state","scaler.pt")):return "skip"
 if n in MODEL_CONF and model:return "models"
 if s in MODEL_EXT:return "models" if model or s in {".safetensors",".gguf"} else "skip"
 if (data and s in DATA_EXT) or (n.startswith("trainable") and s in DATA_EXT) or (data and (n.endswith("manifest.json") or n.endswith("contract.json"))):return "datasets"
 return "evidence"
def sha(path):
 h=hashlib.sha256();n=0
 with path.open("rb") as f:
  while blob:=f.read(BUF):h.update(blob);n+=len(blob)
 return h.hexdigest(),n
def safe_file(base,path,expected_size):
 if not path.is_relative_to(base) or path==base:raise RuntimeError("Path outside mounted volume")
 try:
  resolved=path.resolve(strict=True)
 except (FileNotFoundError,RuntimeError) as exc:
  raise RuntimeError("Frozen regional source path is unavailable or broken") from exc
 if not resolved.is_relative_to(base) or resolved==base:
  raise RuntimeError("Regional symlink escapes the original Czech volume")
 # Resolve aliases ONLY to approved evidence paths, not private, mutable, or model data.
 key=resolved.relative_to(base).as_posix()
 approved=False
 for prefix,_ in ROOTS.items():
  if key.startswith(prefix):
   tail=PurePosixPath(key[len(prefix):])
   approved=not reject(tail) and kind(tail)=="evidence"
   break
 if not approved:raise RuntimeError("Regional symlink points outside eligible evidence scope")
 try:st=resolved.lstat()
 except FileNotFoundError:raise RuntimeError("Frozen regional source file disappeared or is missing")
 if not stat.S_ISREG(st.st_mode) or st.st_size!=expected_size:
  raise RuntimeError("Resolved regional source file size/type differs from S3 inventory")
 return resolved

def _files(root):
    def fail_unreadable(error):
        raise error

    rows = []
    for parent, directories, files in os.walk(root, followlinks=False, onerror=fail_unreadable):
        for name in files + [name for name in directories if (Path(parent) / name).is_symlink()]:
            path = Path(parent) / name
            metadata = path.lstat()
            rows.append((path.relative_to(root).as_posix(), metadata.st_size, stat.S_IFMT(metadata.st_mode)))
    return sorted(rows)


def _hash_rows(rows):
    return sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _sealed_file(parent, path):
    if path.resolve(strict=True) != path or not path.is_relative_to(parent) or not path.is_file():
        raise ValueError("unconfined or non-regular sealed file")
    return path


def _snapshot(path, cache_repo):
    def file(name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid snapshot member")
        resolved = (path / relative).resolve(strict=True)
        if not (resolved.is_relative_to(path) or resolved.is_relative_to(cache_repo / "blobs")):
            raise ValueError("snapshot member escaped the admitted model cache")
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise ValueError("missing or empty snapshot member")
        if not os.access(resolved, os.R_OK):
            raise PermissionError("unreadable snapshot member")
        return resolved

    config = file("config.json")
    config_data = json.loads(config.read_bytes())
    file("tokenizer_config.json")
    if "mistral" in str(config_data.get("model_type", "")).casefold():
        tokens = [entry.name for entry in path.iterdir() if entry.name.startswith("tekken") and entry.suffix == ".json"]
        if not tokens:
            raise FileNotFoundError("Mistral tokenizer not found")
        file(tokens[0])
    else:
        try:
            file("tokenizer.json")
        except FileNotFoundError:
            file("tokenizer.model")
    index = path / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        weights = json.loads(file(index.name).read_bytes())["weight_map"]
        if not isinstance(weights, dict) or not weights:
            raise ValueError("empty shard index")
        shards = sorted(set(weights.values()))
    else:
        shards = ["model.safetensors"]
    # Check readability and reasonable SafeTensors header boundaries without loading a model.
    for shard in shards:
        data = file(shard)
        with data.open("rb") as stream:
            header_bytes = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_bytes <= min(100_000_000, data.stat().st_size - 8):
            raise ValueError("invalid SafeTensors shard header")
    return len(shards)


def _models(rows, registry):
    result = {}
    for role, spec in sorted(registry["roles"].items()):
        adapter = spec["adapter"]
        digest = adapter["sha256"]
        mounted = GLOBAL / f"hephaestus/runtime-models/role-mastery-v1/{role}/selected-adapter-{digest[:24]}.tar.gz"
        try:
            actual, size = sha(_sealed_file(GLOBAL, mounted))
            archive = {"sha256": actual, "bytes": size, "matches": (actual, size) == (digest, adapter["bytes"])}
        except (OSError, ValueError) as exc:
            archive = {"matches": False, "error": type(exc).__name__}
        suffix = ("models--" + spec["model_id"].replace("/", "--"), "snapshots", spec["revision"])
        candidates = set()
        for relative, _, _ in rows:
            parts = PurePosixPath(relative).parts
            for n in range(max(0, len(parts) - 3)):
                if parts[n:n+3] == suffix:
                    candidates.add(GLOBAL.joinpath(*parts[:n+3]))
        found = []
        for candidate in sorted(candidates):
            try:
                if candidate.resolve(strict=True) != candidate:
                    raise ValueError("snapshot root escapes the admitted path")
                found.append({"shards": _snapshot(candidate, candidate.parents[1]), "structurally_complete": True})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                found.append({"structurally_complete": False, "error": type(exc).__name__})
        result[role] = {"adapter": archive, "base_revision": spec["revision"], "snapshot_candidates": found}
    return result


def _evidence():
    path = _sealed_file(SOURCE, SOURCE / INVENTORY_KEY)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("inventory exceeds metadata quota")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != INVENTORY_SHA256:
        raise ValueError("source migration inventory anchor differs")
    data = json.loads(raw)
    if data["source_volume_id"] != SOURCE_ID or data["schema"] != "hephaestus.original-s3-storage-inventory.v1":
        raise ValueError("frozen source inventory has wrong identity")
    rows = []
    for item in data["objects"]:
        key = item["key"]
        if "\\" in key or any(segment in {"", ".", ".."} for segment in key.rstrip("/").split("/")):
            raise ValueError("unsafe inventory key")
        for prefix, destination in ROOTS.items():
            if key.startswith(prefix) and not key.endswith("/"):
                tail = PurePosixPath(key[len(prefix):])
                if not reject(tail) and kind(tail) == "evidence":
                    rows.append((f"{destination}/{tail}", key, int(item["bytes"])))
                break
    if (
        len(rows) != 8224 or sum(row[2] for row in rows) != 607271450
        or len({row[0] for row in rows}) != len(rows)
    ):
        raise ValueError("candidate row identities differ from historical inventory")
    totals = {"verified": 0, "unresolved_alias": 0, "missing_source": 0, "missing_or_different_global": 0}
    parity = []

    def compare(row):
        relative, key, size = row
        source = SOURCE / key
        try:
            original = safe_file(SOURCE, source, size)
        except RuntimeError:
            aliases = [source, *(parent for parent in source.parents if parent.is_relative_to(SOURCE))]
            return "unresolved_alias" if any(path.is_symlink() for path in aliases) else "missing_source", None
        try:
            target = _sealed_file(GLOBAL, EVIDENCE / relative)
            original_hash, original_size = sha(original)
            if (original_hash, original_size) != sha(target) or original_size != size:
                raise ValueError("source and Global bytes differ")
        except (OSError, ValueError):
            return "missing_or_different_global", None
        return "verified", (relative, size, original_hash)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for state, record in pool.map(compare, sorted(rows)):
            totals[state] += 1
            if record is not None:
                parity.append(record)
    receipt = DEST / "S3_EVIDENCE_RECEIPT.json"
    try:
        receipt_digest, receipt_size = sha(_sealed_file(GLOBAL, receipt))
        receipt_evidence = {"sha256": receipt_digest, "bytes": receipt_size}
    except (OSError, ValueError):
        receipt_evidence = None
    return {**totals, "parity_manifest_sha256": _hash_rows(parity), "existing_receipt": receipt_evidence}


def run():
    report = {"status": "running", "schema": "hephaestus.anvil-global-readback.v1"}
    def publish():
        print("GLOBAL_AUDIT_REPORT=" + json.dumps(report, sort_keys=True), flush=True)

    publish()
    try:
        if not (GLOBAL.is_mount() and SOURCE.is_mount()) or GLOBAL.is_symlink() or SOURCE.is_symlink():
            raise RuntimeError("expected two independent provider mounts")
        rows = _files(GLOBAL)
        report["phase"] = "inventory"
        report["inventory"] = {
            "entries": len(rows), "regular_files": sum(t == stat.S_IFREG for _, _, t in rows),
            "bytes": sum(n for _, n, t in rows if t == stat.S_IFREG),
            "symlinks": sum(t == stat.S_IFLNK for _, _, t in rows),
            "manifest_sha256": _hash_rows(rows),
        }
        publish()
        report["phase"] = "specialists"
        report["specialists"] = _models(rows, json.loads(os.environ["ANVIL_ROLE_REGISTRY"]))
        publish()
        report["phase"] = "migration"
        try:
            report["migration"] = _evidence()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["migration"] = {"error": type(exc).__name__}
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__)
    publish()
    # The controller independently terminates the exact Pod. Keep evidence available until then.
    import time
    time.sleep(1800)


if __name__ == "__main__":
    run()
