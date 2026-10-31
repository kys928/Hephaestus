"""Process races, interrupted extraction and independent adapter identity checks."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import multiprocessing
import tarfile
import pytest

from hephaestus.llm.certified_runtime import VerifiedRoleAdapters, validate_adapter_base
from hephaestus.llm.role_contract import RoleMasteryContract
from test_certified_role_inference import _archive, _roles, FakeS3, CONTRACT


class SharedS3:
    def __init__(self, blob, gets): self.blob, self.gets = blob, gets
    def get_object(self, **kwargs):
        import io
        with self.gets.get_lock(): self.gets.value += 1
        return {"Body":io.BytesIO(self.blob)}


def _worker(root, blob, gets, barrier, queue):
    try:
        barrier.wait(timeout=5)
        result = VerifiedRoleAdapters(SharedS3(blob, gets), "volume", Path(root)).materialize(_roles(blob).resolve("judge"))
        queue.put(("ok", str(result), (result / "adapter_model.safetensors").read_bytes()))
    except BaseException as exc:
        queue.put(("error", type(exc).__name__, str(exc)))


def test_concurrent_processes_publish_one_complete_adapter_generation(tmp_path):
    ctx = multiprocessing.get_context("fork")
    gets, barrier, queue = ctx.Value("i", 0), ctx.Barrier(3), ctx.Queue()
    blob = _archive()
    workers = [ctx.Process(target=_worker, args=(str(tmp_path), blob, gets, barrier, queue)) for _ in range(3)]
    for worker in workers: worker.start()
    try:
        outputs = [queue.get(timeout=10) for _ in workers]
        assert all(x[0] == "ok" for x in outputs), outputs
        assert len({x[1] for x in outputs}) == 1
        assert all(x[2] == b"safetensors-placeholder-bytes" for x in outputs)
        assert gets.value == 1
    finally:
        for worker in workers:
            worker.join(5)
            if worker.is_alive(): worker.terminate(); worker.join(5)
    assert all(worker.exitcode == 0 for worker in workers)


def test_cache_receipt_cannot_vouch_for_modified_adapter_weights(tmp_path):
    blob = _archive()
    client = FakeS3(blob)
    cache = VerifiedRoleAdapters(client, "volume", tmp_path)
    spec = _roles(blob).resolve("judge")
    first = cache.materialize(spec)
    (first / "adapter_model.safetensors").write_bytes(b"altered-weights")
    receipt_path = first.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["adapter_model.safetensors"] = sha256(b"altered-weights").hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    repaired = cache.materialize(spec)
    assert repaired != first and first.exists()
    assert (repaired / "adapter_model.safetensors").read_bytes() == b"safetensors-placeholder-bytes"
    assert cache.materialize(spec) == repaired and len(client.gets) == 2


def test_interrupted_extraction_never_publishes_partial_adapter(tmp_path, monkeypatch):
    blob = _archive()
    cache = VerifiedRoleAdapters(FakeS3(blob), "volume", tmp_path)
    spec = _roles(blob).resolve("judge")
    original = tarfile.TarFile.extractfile
    def interrupted(self, member):
        if member.name.endswith("adapter_model.safetensors"):
            raise RuntimeError("interrupted halfway through extraction")
        return original(self, member)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache.materialize(spec)
    assert not list(tmp_path.rglob("current.json"))
    monkeypatch.setattr(tarfile.TarFile, "extractfile", original)
    result = cache.materialize(spec)
    assert (result / "adapter_config.json").exists()
    assert (result / "adapter_model.safetensors").exists()


def test_adapter_configuration_base_mismatch_is_rejected_before_generation(tmp_path):
    blob = _archive(base_name="unrelated/model")
    with pytest.raises(ValueError, match="base-model identity"):
        VerifiedRoleAdapters(FakeS3(blob), "volume", tmp_path).materialize(_roles(blob).resolve("judge"))


def test_adapter_configuration_revision_mismatch_is_rejected(tmp_path):
    blob = _archive(revision="a"*40)
    with pytest.raises(ValueError, match="base revision"):
        VerifiedRoleAdapters(FakeS3(blob), "volume", tmp_path).materialize(_roles(blob).resolve("judge"))


def test_original_hf_snapshot_path_and_repository_identity_are_both_valid():
    spec = _roles(_archive()).resolve("judge")
    slug = "models--" + spec.model_id.replace("/", "--")
    for name in (spec.model_id, f"/opt/original-training/model/hf_cache/{slug}/snapshots/{spec.revision}"):
        validate_adapter_base(spec, {"peft_type":"LORA", "base_model_name_or_path":name, "revision":None})
    with pytest.raises(ValueError):
        validate_adapter_base(spec, {"peft_type":"LORA", "base_model_name_or_path":f"/opt/{slug}/snapshots/{'a'*40}"})


def test_runtime_rejects_a_changed_base_revision_before_inference():
    spec = _roles(_archive()).resolve("judge")
    with pytest.raises(ValueError, match="base model identity/revision"):
        RoleMasteryContract.load(CONTRACT).validate_model(replace(spec, revision="a"*40))
