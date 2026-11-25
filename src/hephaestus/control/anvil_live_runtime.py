"""Off-Pod verification of actual qualification receipts before slot signing."""
from hashlib import sha256
import re
from hephaestus.providers.runpod.anvil_hardware import GPU_NAMES, MIN_BYTES

from hephaestus.control.anvil_live import digest
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.storage.verified import canonical_bytes

REGISTRY = "sha256:b48e3bb7cdc7f397cbdfb99384786c0d4b9c81e5ef0d404df6575f5a34a01b7d"
MANIFEST = "d5193ac0f3c77d2098b093237034c0af1acef898bb01364d331966626b1d0f9a"


def verify_loaded_runtime(result, *, pod_id, source_sha256, lock_sha256, image_qualification, gpu_sku=None):
    """Require exact CUDA, model, adapter, package and source identities.

    This verifies the returned operational evidence, not hardware-level remote
    attestation or independent scientific certification.
    """
    if not re.fullmatch(r"[A-Za-z0-9]{4,40}", pod_id):
        raise PermissionError("invalid exact Pod identity")
    if any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in (source_sha256, lock_sha256)):
        raise ValueError("source and lock require exact digests")
    if (image_qualification.get("schema") != "anvil.image-dependency-qualification.v1"
            or image_qualification.get("lock_sha256") != lock_sha256
            or image_qualification.get("cuda_execution") is not False):
        raise PermissionError("missing exact CPU image qualification")
    expected_environment = {
        "python": image_qualification["python"], "torch": image_qualification["torch"],
        "cuda": image_qualification["cuda"],
        "packages_sha256": sha256(canonical_bytes(image_qualification["packages"])).hexdigest(),
    }
    runtime = result["runtime"]
    if (runtime.get("schema") != "anvil.qualified-runtime.v1"
            or runtime.get("pod_id") != pod_id or runtime.get("source_sha256") != source_sha256
            or runtime.get("lock_sha256") != lock_sha256 or result.get("runtime_ref") != digest(runtime)
            or not isinstance(runtime.get("loads"), list) or len(runtime["loads"]) != 2):
        raise PermissionError("runtime binding differs")
    stack, _ = verified_original_roles(REGISTRY)
    for role, row in zip(("diagnosis", "planner"), runtime["loads"]):
        model = stack.resolve(role)
        expected = {
            "schema": "anvil.cuda-offline-load.v1", "pod_id": pod_id, "role": role,
            "model_id": model.model_id, "revision": model.revision, "adapter_sha256": model.adapter.sha256,
            "manifest_sha256": MANIFEST, "cuda": expected_environment["cuda"],
            "environment": expected_environment, "generation_count": 0,
        }
        peak = row.get("peak_allocated_bytes")
        ceiling=48*1024**3
        if gpu_sku is not None:
            ceiling=row.get("vram_bytes")
            if (gpu_sku not in GPU_NAMES or row.get("gpu_sku")!=gpu_sku or row.get("gpu_name") not in GPU_NAMES[gpu_sku]
                    or type(ceiling) is not int or ceiling<MIN_BYTES or row.get("compute_capability") not in ([8,0],[8,6],[8,9])):
                raise PermissionError("selected hardware qualification differs")
            if any(row.get(k)!=runtime["loads"][0].get(k) for k in ("gpu_sku","gpu_name","vram_bytes","compute_capability")):
                raise PermissionError("load receipts disagree about selected hardware")
        if (any(row.get(k) != v for k,v in expected.items())
                or type(peak) is not int or not 0 < peak <= ceiling):
            raise PermissionError("exact model load or qualified environment differs")
    return result["runtime_ref"]
