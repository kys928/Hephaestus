"""Inspect effective distribution versions rather than shadowed OS metadata."""
import hashlib
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path


def effective_packages():
    names = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower()
             for d in metadata.distributions()}
    return {name: metadata.version(name) for name in sorted(names)}


def environment(lock_path):
    import torch
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import cryptography
    from transformers import AutoConfig, AutoTokenizer, Mistral3ForConditionalGeneration
    import peft, accelerate, mistral_common, safetensors, huggingface_hub
    packages = effective_packages()
    raw = Path(lock_path).read_bytes()
    for line in raw.decode().splitlines():
        match = re.fullmatch(r"([a-z0-9-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})", line)
        if not match or packages.get(match[1]) != match[2]:
            raise RuntimeError("effective locked distribution differs")
    if packages["cryptography"] != cryptography.__version__:
        raise RuntimeError("imported cryptography differs from effective metadata")
    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("pinned CUDA Torch differs")
    key = Ed25519PrivateKey.generate()
    key.public_key().verify(key.sign(b"qualification"), b"qualification")
    return {"schema":"anvil.image-dependency-qualification.v1", "python":sys.version.split()[0],
            "torch":torch.__version__, "cuda":torch.version.cuda, "packages":packages,
            "lock_sha256":hashlib.sha256(raw).hexdigest(),
            "cuda_execution":False}


if __name__ == "__main__":
    result = environment(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ANVIL_EFFECTIVE_PACKAGES="+str(len(result["packages"])),flush=True)
