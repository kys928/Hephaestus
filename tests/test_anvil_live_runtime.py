"""Synthetic load evidence tests; no GPU or model execution."""
import copy
from hashlib import sha256
import pytest
from hephaestus.control.anvil_live import digest
from hephaestus.control.anvil_live_runtime import verify_loaded_runtime, REGISTRY, MANIFEST
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.storage.verified import canonical_bytes


def fixture():
    stack, _ = verified_original_roles(REGISTRY)
    image = {"schema":"anvil.image-dependency-qualification.v1", "lock_sha256":"b"*64,
             "python":"3.12.0","torch":"2.8.0","cuda":"12.8","packages":{"torch":"2.8.0"},
             "cuda_execution":False}
    env = {k:image[k] for k in ("python","torch","cuda")}
    env["packages_sha256"] = sha256(canonical_bytes(image["packages"])).hexdigest()
    loads=[]
    for role in ("diagnosis","planner"):
        model=stack.resolve(role)
        loads.append({"schema":"anvil.cuda-offline-load.v1","pod_id":"fixture123","role":role,
            "model_id":model.model_id,"revision":model.revision,"adapter_sha256":model.adapter.sha256,
            "manifest_sha256":MANIFEST,"cuda":"12.8","environment":env,
            "generation_count":0,"peak_allocated_bytes":100})
    runtime={"schema":"anvil.qualified-runtime.v1","pod_id":"fixture123",
             "source_sha256":"a"*64,"lock_sha256":"b"*64,"loads":loads}
    return runtime, dict(pod_id="fixture123",source_sha256="a"*64,lock_sha256="b"*64,image_qualification=image)


def test_runtime_binding_and_mutations():
    runtime, args = fixture()
    assert verify_loaded_runtime({"runtime":runtime,"runtime_ref":digest(runtime)},**args)==digest(runtime)
    for field,value in [("adapter_sha256","0"*64),("generation_count",1),("peak_allocated_bytes",0),
                        ("environment",{}),("role","planner"),("pod_id","other123")]:
        bad=copy.deepcopy(runtime)
        bad["loads"][0][field]=value
        with pytest.raises(PermissionError):
            verify_loaded_runtime({"runtime":bad,"runtime_ref":digest(bad)},**args)
    with pytest.raises(PermissionError):
        verify_loaded_runtime({"runtime":runtime,"runtime_ref":"sha256:"+"0"*64},**args)
