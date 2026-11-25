"""Credential-free exact GPU worker. One signed capsule per stdin invocation.

The externally launched transport must pin the authority and runtime identity.
This entry point does not allocate, self-attest, access Neon, or grant retries.
"""
import json
import os
from pathlib import Path
import re
import sys

from anvil_stage_priority_models import MANIFEST_REF
from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_live import run_worker_capsule
from hephaestus.production.original_cuda_identity import build_original_role_reasoner, verified_original_roles
from hephaestus.providers.runpod.global_volume import GlobalVolumeConfig, global_role_archive_source

GLOBAL = "cmw51pjsk000007l9a0mv1jga"
REGISTRY = "sha256:b48e3bb7cdc7f397cbdfb99384786c0d4b9c81e5ef0d404df6575f5a34a01b7d"


def main():
    import torch

    if any(key in os.environ for key in ("ANVIL_POSTGRES_DSN", "RUNPOD_API_KEY",
                                         "RUNPOD_S3_SECRET_ACCESS_KEY", "RUNPOD_S3_ACCESS_KEY_ID", "HF_TOKEN", "GITHUB_TOKEN")):
        raise PermissionError("model worker must not receive control-plane credentials")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    pod = os.environ.get("RUNPOD_POD_ID", "")
    if (not re.fullmatch(r"[A-Za-z0-9]{4,40}", pod) or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA A40"
            or not torch.cuda.is_bf16_supported()):
        raise PermissionError("expected exact A40 CUDA worker")
    volume = GlobalVolumeConfig(GLOBAL)
    if not os.path.ismount(volume.mount_path):
        raise PermissionError("verified Global mount absent")
    cache = volume.mount_path / ("anvil-priority-models-v1-" + MANIFEST_REF[:16])
    if cache.resolve(strict=True) != cache:
        raise PermissionError("offline cache differs")
    root = Path("/tmp/anvil-live-worker")
    root.mkdir(mode=0o700, exist_ok=True)
    stack, _ = verified_original_roles(REGISTRY)
    archives = global_role_archive_source(volume, stack)
    authority = AnvilAuthority(os.environ["ANVIL_AUTHORITY_PUBLIC_HEX"], os.environ["ANVIL_AUTHORITY_REF"])
    raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("worker capsule exceeds ceiling")
    capsule = json.loads(raw)
    def reasoner(store):
        return build_original_role_reasoner(expected_registry_digest=REGISTRY, archive_source=archives,
            bucket=GLOBAL, model_cache=cache, adapter_cache=root / "adapters", artifact_store=store)
    result = run_worker_capsule(capsule, root, authority, pod_id=pod,
        runtime_ref=os.environ["ANVIL_RUNTIME_REF"], reasoner_factory=reasoner)
    print("ANVIL_LIVE_RECEIPT=" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit("Anvil worker failed (" + type(error).__name__ + "); no automatic retry")
