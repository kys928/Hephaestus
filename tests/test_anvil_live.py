"""CPU-only transport/receipt fixtures; never evidence of model inference."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_context import AnvilAdmission
from hephaestus.control.anvil_dispatch import AnvilDispatch
from hephaestus.control.anvil_live import (
    AnvilLiveController, publish, publish_frozen_inputs, reconstruct_worker_capsule,
)
from hephaestus.jobs import SQLiteJobQueue
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.schemas.anvil import ContributionOrigin
from hephaestus.state.anvil import AnvilState
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json

ROOT = Path(__file__).resolve().parents[1]


def test_first_live_bundle_and_rounds(tmp_path):
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    key = private.public_key().public_bytes_raw()
    authority = AnvilAuthority(key.hex(), "sha256:" + sha256(key).hexdigest())
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    retained = FileSystemArtifactStore(tmp_path / "retained")
    repository = SQLiteStateRepository(tmp_path / "state.sqlite")
    queue = SQLiteJobQueue(tmp_path / "jobs.sqlite")
    bundle = json.loads((ROOT / "configs/research/anvil_diagnosis_planner_first_live_v1.json").read_bytes())
    plan_ref = publish_frozen_inputs(bundle, artifacts)
    plan = bundle["plan"]["document"]
    material = frozenset(plan["material_refs"])
    stack, contract = verified_original_roles(plan["registry_ref"])
    scope = {k: plan[k] for k in ("investigation", "run_id", "lineage_id")}

    def make_dispatch(trusted):
        board = AnvilState(repository, lambda origin: origin.inference_ref in trusted,
                           lambda origin, ref: ref in material)
        admission = AnvilAdmission(board, artifacts, plan["registry_ref"], material, trusted, repository)
        return AnvilDispatch(admission, queue, plan_ref)

    def sign(doc):
        return publish(artifacts, {"document": doc, "signature_hex": private.sign(canonical_bytes(doc)).hex()})

    dispatch = make_dispatch(frozenset())
    board = dispatch.admission._board
    model = stack.resolve("diagnosis")
    origin = ContributionOrigin(**scope, role="diagnosis", inference_ref=plan_ref,
        context_frontier=bundle["expected_initial_frontier"], idempotency_key="material",
        model_id=model.model_id, model_revision=model.revision, adapter_ref="sha256:" + model.adapter.sha256)
    for ref in material:
        dispatch.admission.admit_material(origin, ref)
    controller = AnvilLiveController(dispatch, repository, authority, make_dispatch)
    runtime_ref = "sha256:" + "f" * 64
    pod = "fixture-a40"
    with pytest.raises(PermissionError, match="expected slot"):
        controller.prepare_slot(2)
    histories, checkpoints = [], []
    for slot in range(4):
        prepared = controller.prepare_slot(slot)
        checkpoints.append(prepared.checkpoint_ref)
        grant = sign({"schema": "anvil.execution-grant.v1", "prepared_ref": prepared.prepared_ref,
            "pod_id": pod, "runtime_ref": runtime_ref, "device": "cuda:0", "seed": 271828,
            "max_new_tokens": 512, "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()})
        capsule_ref = controller.reserve_capsule(prepared, grant_ref=grant, pod_id=pod, runtime_ref=runtime_ref)
        with pytest.raises(PermissionError, match="already consumed"):
            controller.reserve_capsule(prepared, grant_ref=grant, pod_id=pod, runtime_ref=runtime_ref)
        capsule = read_verified_json(artifacts, capsule_ref)
        with pytest.raises(PermissionError, match="scope"):
            reconstruct_worker_capsule(capsule, tmp_path / "bad-worker", authority,
                                       pod_id="other", runtime_ref=runtime_ref)
        worker, _ = reconstruct_worker_capsule(capsule, tmp_path / "worker", authority,
                                              pod_id=pod, runtime_ref=runtime_ref)
        p = read_verified_json(artifacts, prepared.prepared_ref)
        payload = read_verified_json(artifacts, p["dispatch_payload_ref"])
        task = worker._prepare(payload)
        context = json.loads(task.situation)
        histories.append(context)
        assert len(context["advisory_history"]) == (0 if slot < 2 else 2)
        assert len(task.evidence) == (2 if slot < 2 else 3)
        if slot < 2:
            assert plan["material_refs"][2] not in {e.ref for e in task.evidence}
        spec = stack.resolve(task.role)
        vocabulary = contract.profile(task.role)["vocabulary"]
        answer = {k: vocabulary[k][0] for k in ("decision", "action", "primary_variable")}
        answer.update(confidence=0.5, uncertainties=["Fixture only"], evidence_refs=[task.evidence[0].ref],
                      rationale="Synthetic contract test, not model reasoning.")
        raw = canonical_bytes(answer).decode()
        prompt = contract.prompt(task)
        receipt = dict(schema="role-inference.v1", status="accepted", role=task.role, run_id=task.run_id,
            lineage_id=task.lineage_id, stage_name=task.stage_name, raw_output=raw, answer=answer,
            output_sha256=sha256(raw.encode()).hexdigest(), prompt=prompt,
            prompt_sha256=sha256(prompt.encode()).hexdigest(), model_id=spec.model_id,
            base_revision=spec.revision, adapter_sha256=spec.adapter.sha256, adapter_source=spec.adapter.s3_key,
            tokenizer_id=spec.model_id, tokenizer_revision=spec.revision, evidence_refs=[e.ref for e in task.evidence],
            generator="PinnedTransformersRoleGenerator", seed=271828, max_new_tokens=512,
            execution={"device": "cuda:0", "gpu_name": "NVIDIA A40", "cuda_version": "fixture",
                       "generated_token_ids": [1, 2]})
        receipt_ref = publish(artifacts, receipt)
        retention_ref = publish(artifacts, {"schema": "anvil.receipt-retention.v1",
            "receipt_ref": receipt_ref, "capsule_ref": capsule_ref, "independent_readback": True, "fixture": True})
        attestation = sign({"schema": "anvil.inference-attestation.v1", "status": "verified",
            "prepared_ref": prepared.prepared_ref, "receipt_ref": receipt_ref, "runtime_ref": runtime_ref,
            "plan_ref": plan_ref, "pod_id": pod, "retention_ref": retention_ref})
        if slot == 0:
            with pytest.raises(PermissionError, match="independent"):
                controller.admit_slot(slot=slot, receipt_ref=receipt_ref, attestation_ref=attestation,
                                      retention_ref=retention_ref, retained_artifacts=artifacts)
        publish(retained, receipt, receipt_ref)
        publish(retained, capsule, capsule_ref)
        result = controller.admit_slot(slot=slot, receipt_ref=receipt_ref, attestation_ref=attestation,
                                      retention_ref=retention_ref, retained_artifacts=retained)
        assert result.sequence == slot + 1
        assert controller.admit_slot(slot=slot, receipt_ref=receipt_ref, attestation_ref=attestation,
            retention_ref=retention_ref, retained_artifacts=retained) == result
    assert histories[0] == histories[1]
    assert histories[2] == histories[3]
    assert checkpoints[0] == checkpoints[1] and checkpoints[2] == checkpoints[3]
    assert checkpoints[0] != checkpoints[2]
    assert len(board.replay(plan["investigation"]).contributions) == 4
    assert len(repository.all("anvil.live-reservations")) == 4
    assert len(repository.all("anvil.live-admissions")) == 4


def test_frozen_bundle_rejects_changed_text(tmp_path):
    bundle = json.loads((ROOT / "configs/research/anvil_diagnosis_planner_first_live_v1.json").read_bytes())
    bundle["materials"][0]["document"]["text"] += " Changed."
    with pytest.raises(ValueError, match="frozen material"):
        publish_frozen_inputs(bundle, FileSystemArtifactStore(tmp_path / "artifacts"))
