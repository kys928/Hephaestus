"""Load both original Anvil roles offline on an already-approved exact A40 Pod."""
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from anvil_stage_priority_models import MANIFEST_REF
from hephaestus.production.original_cuda_identity import build_original_role_reasoner, verified_original_roles
from hephaestus.providers.runpod.global_volume import GlobalVolumeConfig, global_role_archive_source
from hephaestus.storage.filesystem import FileSystemArtifactStore

GLOBAL = "cmw51pjsk000007l9a0mv1jga"
REGISTRY_REF = "sha256:b48e3bb7cdc7f397cbdfb99384786c0d4b9c81e5ef0d404df6575f5a34a01b7d"


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch

    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9]{4,40}", pod_id):
        raise PermissionError("expected exact RunPod runtime identity")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != "NVIDIA A40" or not torch.cuda.is_bf16_supported()):
        raise PermissionError("approved A40 CUDA environment differs")
    volume = GlobalVolumeConfig(GLOBAL)
    if not os.path.ismount(volume.mount_path):
        raise PermissionError("approved Global Volume mount absent")
    cache = volume.mount_path / ("anvil-priority-models-v1-" + MANIFEST_REF[:16])
    if cache.resolve(strict=True) != cache:
        raise PermissionError("model cache traverses a link")
    stack, _ = verified_original_roles(REGISTRY_REF)
    archives = global_role_archive_source(volume, stack)
    with TemporaryDirectory(prefix="anvil-offline-preflight-") as root:
        temp = Path(root)
        runtime = build_original_role_reasoner(
            expected_registry_digest=REGISTRY_REF, archive_source=archives, bucket=GLOBAL,
            model_cache=cache, adapter_cache=temp / "adapters",
            artifact_store=FileSystemArtifactStore(temp / "artifacts"),
        )
        try:
            for role in ("diagnosis", "planner"):
                spec = runtime.stack.resolve(role)
                torch.cuda.reset_peak_memory_stats(0)
                adapter = runtime.adapters.materialize(spec)
                runtime.generator._load(spec, runtime.contract.profile(role), adapter)
                receipt = {
                    "schema": "anvil.cuda-offline-load.v1", "pod_id": pod_id, "role": role,
                    "model_id": spec.model_id, "revision": spec.revision,
                    "adapter_sha256": spec.adapter.sha256,
                    "manifest_sha256": MANIFEST_REF, "cuda": torch.version.cuda,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "generation_count": 0,
                }
                print(json.dumps(receipt, sort_keys=True), flush=True)
                runtime.generator.close()
        finally:
            runtime.generator.close()


if __name__ == "__main__":
    main()
