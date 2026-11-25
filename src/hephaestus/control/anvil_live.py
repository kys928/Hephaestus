"""Bounded first-live orchestration; no provider allocation or model credentials here.

The controller owns preparation and admission. A named worker receives only a
hash-checked round snapshot and a signed slot grant. Independent retention and
review must succeed before a real output becomes shared knowledge.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

from hephaestus.control.anvil_attestation import AnvilAuthority
from hephaestus.control.anvil_context import AnvilAdmission
from hephaestus.control.anvil_dispatch import AnvilDispatch
from hephaestus.control.anvil_execution import _require_open, admit_attested, execute_prepared
from hephaestus.jobs import SQLiteJobQueue
from hephaestus.jobs.models import JobStatus
from hephaestus.llm.inference_receipts import verify_inference_receipt
from hephaestus.production.original_cuda_identity import verified_original_roles
from hephaestus.schemas.anvil import ContributionOrigin
from hephaestus.state.anvil import AnvilState
from hephaestus.state.execution_ownership import execution_owner
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes, read_verified_json

ROLES = ("diagnosis", "planner", "diagnosis", "planner")
SCOPE_KEYS = ("investigation", "run_id", "lineage_id")


def digest(value):
    return "sha256:" + sha256(canonical_bytes(value)).hexdigest()


def publish(store, doc, ref=None):
    actual = digest(doc)
    if ref is not None and ref != actual:
        raise ValueError("frozen artifact digest differs")
    return store.put_bytes(canonical_bytes(doc), expected_hash=actual,
                           media_type="application/json").artifact_ref


def publish_frozen_inputs(bundle, artifacts):
    """Validate and publish the checked-in, non-authorizing input bundle."""
    if (bundle.get("schema") != "anvil.live-inputs.v1"
            or bundle.get("execution_authorized") is not False):
        raise ValueError("expected frozen non-authorizing input bundle")
    plan = bundle["plan"]["document"]
    rows = bundle["materials"]
    scope = {key: plan[key] for key in SCOPE_KEYS}
    if (plan.get("schema") != "anvil.dispatch.v1" or len(rows) != 3
            or [row["ref"] for row in rows] != plan["material_refs"]
            or [row["document"]["kind"] for row in rows] != ["objective", "evidence", "evidence"]
            or bundle["expected_initial_frontier"] != digest(["anvil.snapshot.v2", scope["investigation"]])):
        raise ValueError("frozen round structure differs")
    verified_original_roles(plan["registry_ref"])
    for row in rows:
        doc = row["document"]
        if (doc.get("schema") != "anvil.material.v1"
                or any(doc.get(key) != value for key, value in scope.items())
                or not isinstance(doc.get("text"), str) or not 1 <= len(doc["text"]) <= 500
                or digest(doc) != row["ref"]):
            raise ValueError("invalid frozen material")
    if digest(plan) != bundle["plan"]["ref"]:
        raise ValueError("invalid frozen plan")
    for row in rows:
        publish(artifacts, row["document"], row["ref"])
    return publish(artifacts, plan, bundle["plan"]["ref"])


@dataclass(frozen=True, slots=True)
class PreparedLiveSlot:
    slot: int
    prepared_ref: str
    checkpoint_ref: str
    frontier: str


@dataclass(slots=True)
class AnvilLiveController:
    dispatch: AnvilDispatch
    repository: SQLiteStateRepository
    authority: AnvilAuthority
    # Rebind the same composition with only the independently reviewed receipt.
    verified_dispatch: Callable[[frozenset[str]], AnvilDispatch]

    @property
    def artifacts(self):
        return self.dispatch.admission.artifacts

    def _round(self, slot):
        if type(slot) is not int or not 0 <= slot < 4:
            raise ValueError("outside four-slot preregistration")
        plan = self.dispatch._plan()
        board = self.dispatch.admission._board
        snapshot = board.replay(plan["investigation"])
        if (len(snapshot.contributions) != slot
                or any(row.origin.role != ROLES[i] or any(
                    getattr(row.origin, key) != plan[key] for key in SCOPE_KEYS
                ) for i, row in enumerate(snapshot.contributions))):
            raise PermissionError("live accepted frontier differs from the expected slot")
        if len({r.origin.inference_ref for r in snapshot.contributions}) != slot:
            raise PermissionError("duplicate accepted inference receipt")
        sequence = 0 if slot < 2 else 2
        historical = board.read_snapshot(plan["investigation"], sequence=sequence)
        key = digest([self.dispatch.plan_ref, "round", sequence])
        previous = [r for r in self.repository.all("anvil.live-rounds") if r["key"] == key]
        if previous:
            checkpoint_ref = previous[0]["checkpoint_ref"]
        else:
            if slot != sequence:
                raise PermissionError("round checkpoint was not retained before first slot")
            checkpoint_ref = board.checkpoint(plan["investigation"], self.artifacts)
            self.repository.append("anvil.live-rounds", {
                "key": key, "checkpoint_ref": checkpoint_ref,
                "frontier": historical.frontier, "plan_ref": self.dispatch.plan_ref,
            }, operation_id=key)
        checkpoint = read_verified_json(self.artifacts, checkpoint_ref)
        if (checkpoint.get("schema") != "anvil.checkpoint.v2"
                or checkpoint.get("investigation") != plan["investigation"]
                or len(checkpoint["records"]) != sequence
                or checkpoint["frontier"] != historical.frontier):
            raise PermissionError("retained round checkpoint differs")
        AnvilState._replay_records(plan["investigation"], checkpoint["records"], historical.frontier)
        return plan, historical, checkpoint_ref

    def prepare_slot(self, slot, owner_id="anvil-offpod-controller"):
        with execution_owner(self.repository, "anvil-live-controller", self.dispatch.plan_ref):
            _require_open(self.repository, self.dispatch.plan_ref)
            plan, snapshot, checkpoint_ref = self._round(slot)
            stack, _ = verified_original_roles(plan["registry_ref"])
            role = ROLES[slot]
            model = stack.resolve(role)
            origin = ContributionOrigin(**{k: plan[k] for k in SCOPE_KEYS},
                role=role, inference_ref=self.dispatch.plan_ref, context_frontier=snapshot.frontier,
                idempotency_key=f"live-slot-{slot}", model_id=model.model_id,
                model_revision=model.revision, adapter_ref="sha256:" + model.adapter.sha256)
            refs = tuple(plan["material_refs"][:2] if slot < 2 else plan["material_refs"])
            job = self.dispatch.submit(owner_id=owner_id, origin=origin, slot=slot,
                                       sequence=len(snapshot.contributions), material_refs=refs)
            if job.status == JobStatus.QUEUED:
                job = self.dispatch.worker(owner_id).run_once()
            if job is None or job.status != JobStatus.SUCCEEDED or not job.result_ref:
                raise RuntimeError("durable preparation did not succeed; no compute authorized")
            prepared = read_verified_json(self.artifacts, job.result_ref)
            payload = read_verified_json(self.artifacts, prepared["dispatch_payload_ref"])
            if payload.get("slot") != slot or payload.get("plan_ref") != self.dispatch.plan_ref:
                raise PermissionError("prepared worker result belongs to another slot")
            return PreparedLiveSlot(slot, job.result_ref, checkpoint_ref, snapshot.frontier)

    def reserve_capsule(self, prepared, *, grant_ref, pod_id, runtime_ref):
        """Consume one external dispatch before transport; never replay uncertain slots."""
        with execution_owner(self.repository, "anvil-live-controller", self.dispatch.plan_ref):
            _require_open(self.repository, self.dispatch.plan_ref)
            plan, snapshot, checkpoint = self._round(prepared.slot)
            if (checkpoint, snapshot.frontier) != (prepared.checkpoint_ref, prepared.frontier):
                raise PermissionError("prepared round is no longer the retained round")
            self.authority.verify_execution(self.artifacts, grant_ref, prepared_ref=prepared.prepared_ref,
                                            pod_id=pod_id, runtime_ref=runtime_ref)
            raw = read_verified_json(self.artifacts, prepared.prepared_ref)
            payload = read_verified_json(self.artifacts, raw["dispatch_payload_ref"])
            if payload.get("slot") != prepared.slot or payload.get("plan_ref") != self.dispatch.plan_ref:
                raise PermissionError("prepared task scope differs")
            self.dispatch._prepare(payload)
            refs = [self.dispatch.plan_ref, prepared.prepared_ref, raw["dispatch_payload_ref"],
                    checkpoint, grant_ref, *payload["material_refs"]]
            capsule = {
                "schema": "anvil.worker-capsule.v1", "plan_ref": self.dispatch.plan_ref,
                "prepared_ref": prepared.prepared_ref, "checkpoint_ref": checkpoint,
                "grant_ref": grant_ref, "pod_id": pod_id, "runtime_ref": runtime_ref,
                "slot": prepared.slot, "frontier": prepared.frontier,
                "objects": [{"ref": ref, "document": read_verified_json(self.artifacts, ref)}
                            for ref in refs],
            }
            capsule_ref = publish(self.artifacts, capsule)
            key = digest([self.dispatch.plan_ref, "external-slot", prepared.slot])
            if any(r["key"] == key for r in self.repository.all("anvil.live-reservations")):
                raise PermissionError("external slot already consumed; reconcile without rerun")
            self.repository.append("anvil.live-reservations", {
                "key": key, "plan_ref": self.dispatch.plan_ref, "slot": prepared.slot,
                "capsule_ref": capsule_ref, "prepared_ref": prepared.prepared_ref,
                "runtime_ref": runtime_ref, "pod_id": pod_id, "grant_ref": grant_ref,
            }, operation_id=key)
            return capsule_ref

    def admit_slot(self, *, slot, receipt_ref, attestation_ref, retention_ref, retained_artifacts, gpu_name="NVIDIA A40"):
        """A separately reviewed, independently retained receipt is the only admission path."""
        with execution_owner(self.repository, "anvil-live-controller", self.dispatch.plan_ref):
            _require_open(self.repository, self.dispatch.plan_ref)
            plan = self.dispatch._plan()
            current = self.dispatch.admission._board.replay(plan["investigation"])
            # Reconcile an acknowledged database commit after local journal loss.
            if len(current.contributions) == slot:
                self._round(slot)
            elif (len(current.contributions) != slot + 1
                  or current.contributions[slot].origin.inference_ref != receipt_ref
                  or current.contributions[slot].origin.role != ROLES[slot]):
                raise PermissionError("live frontier does not permit this admission")
            key = digest([self.dispatch.plan_ref, "external-slot", slot])
            rows = [r for r in self.repository.all("anvil.live-reservations") if r["key"] == key]
            if len(rows) != 1:
                raise PermissionError("no unique consumed external slot")
            row = rows[0]
            if retained_artifacts is self.artifacts:
                raise PermissionError("independent receipt retention required")
            self.authority.verify(self.artifacts, attestation_ref,
                schema="anvil.inference-attestation.v1", expected={
                    "prepared_ref": row["prepared_ref"], "receipt_ref": receipt_ref,
                    "runtime_ref": row["runtime_ref"], "plan_ref": self.dispatch.plan_ref,
                    "pod_id": row["pod_id"], "retention_ref": retention_ref, "status": "verified",
                })
            retained = read_verified_json(retained_artifacts, receipt_ref)
            receipt = read_verified_json(self.artifacts, receipt_ref)
            retention = read_verified_json(self.artifacts, retention_ref)
            if (retained != receipt or retention.get("schema") != "anvil.receipt-retention.v1"
                    or retention.get("receipt_ref") != receipt_ref
                    or retention.get("capsule_ref") != row["capsule_ref"]
                    or read_verified_json(retained_artifacts, row["capsule_ref"]) !=
                       read_verified_json(self.artifacts, row["capsule_ref"])
                    or retention.get("independent_readback") is not True):
                raise PermissionError("independent retained receipt differs")
            _, contract = verified_original_roles(plan["registry_ref"])
            verify_inference_receipt(self.artifacts, receipt_ref, contract)
            execution = receipt.get("execution", {})
            tokens = execution.get("generated_token_ids")
            if (receipt.get("status") != "accepted" or receipt.get("fixture") is not None
                    or receipt.get("role") != ROLES[slot]
                    or receipt.get("generator") != "PinnedTransformersRoleGenerator"
                    or receipt.get("seed") != 271828 or receipt.get("max_new_tokens") != 512
                    or execution.get("device") != "cuda:0" or execution.get("gpu_name") != gpu_name
                    or not execution.get("cuda_version") or not isinstance(tokens, list)
                    or not 1 <= len(tokens) <= 512 or any(type(t) is not int or t < 0 for t in tokens)):
                raise PermissionError("receipt is not an accepted bounded CUDA observation")
            dispatch = self.verified_dispatch(frozenset({receipt_ref}))
            if (dispatch.plan_ref != self.dispatch.plan_ref or dispatch.admission.artifacts is not self.artifacts
                    or dispatch.admission._board.replay(plan["investigation"]) !=
                    self.dispatch.admission._board.replay(plan["investigation"])):
                raise PermissionError("reviewed admission binding differs")
            accepted = admit_attested(dispatch, prepared_ref=row["prepared_ref"], receipt_ref=receipt_ref,
                runtime_ref=row["runtime_ref"], attestation_ref=attestation_ref,
                authority=self.authority, repository=self.repository)
            self.repository.append("anvil.live-admissions", {
                "plan_ref": self.dispatch.plan_ref, "slot": slot, "receipt_ref": receipt_ref,
                "attestation_ref": attestation_ref, "retention_ref": retention_ref,
                "contribution_id": accepted.contribution_id, "frontier": accepted.frontier,
            }, operation_id=digest([self.dispatch.plan_ref, "external-admission", slot]))
            return accepted


def reconstruct_worker_capsule(capsule, root: Path, authority, *, pod_id, runtime_ref):
    """Restore only the signed round context locally; no hosted database login."""
    if (capsule.get("schema") != "anvil.worker-capsule.v1"
            or capsule.get("pod_id") != pod_id or capsule.get("runtime_ref") != runtime_ref
            or type(capsule.get("slot")) is not int or not 0 <= capsule["slot"] < 4
            or not isinstance(capsule.get("objects"), list) or len(capsule["objects"]) > 8):
        raise PermissionError("worker capsule scope differs")
    artifacts = FileSystemArtifactStore(root / "artifacts")
    refs = set()
    for item in capsule["objects"]:
        if item["ref"] in refs:
            raise ValueError("duplicate capsule object")
        refs.add(publish(artifacts, item["document"], item["ref"]))
    authority.verify_execution(artifacts, capsule["grant_ref"], prepared_ref=capsule["prepared_ref"],
                                pod_id=pod_id, runtime_ref=runtime_ref)
    plan = read_verified_json(artifacts, capsule["plan_ref"])
    prepared = read_verified_json(artifacts, capsule["prepared_ref"])
    payload = read_verified_json(artifacts, prepared["dispatch_payload_ref"])
    expected = {capsule["plan_ref"], capsule["prepared_ref"], prepared["dispatch_payload_ref"],
                capsule["checkpoint_ref"], capsule["grant_ref"], *payload["material_refs"]}
    if (refs != expected or payload["slot"] != capsule["slot"] or payload["plan_ref"] != capsule["plan_ref"]
            or payload["origin"]["context_frontier"] != capsule["frontier"]):
        raise PermissionError("capsule includes unexpected or mismatched context")
    repository = SQLiteStateRepository(root / "worker-reservations.sqlite")
    board = AnvilState(repository, lambda _: False, lambda *_: False)
    restored = board.restore(plan["investigation"], artifacts, capsule["checkpoint_ref"],
                             expected_frontier=capsule["frontier"])
    if len(restored.contributions) != payload["sequence"]:
        raise PermissionError("capsule checkpoint is not the exact round")
    admission = AnvilAdmission(board, artifacts, plan["registry_ref"],
                               frozenset(payload["material_refs"]), frozenset())
    dispatch = AnvilDispatch(admission, SQLiteJobQueue(root / "worker-jobs.sqlite"), capsule["plan_ref"])
    if canonical_bytes(prepared["task"]) != canonical_bytes(asdict(dispatch._prepare(payload))):
        raise PermissionError("worker task reconstruction differs")
    return dispatch, repository


def run_worker_capsule(capsule, root, authority, *, pod_id, runtime_ref, reasoner_factory):
    dispatch, reservations = reconstruct_worker_capsule(capsule, root, authority,
                                                       pod_id=pod_id, runtime_ref=runtime_ref)
    reasoner = reasoner_factory(dispatch.admission.artifacts)
    try:
        observation = execute_prepared(dispatch, prepared_ref=capsule["prepared_ref"],
            grant_ref=capsule["grant_ref"], runtime_ref=runtime_ref, authority=authority,
            reasoner=reasoner, reservations=reservations)
        ref = observation.inference_receipt_ref
    except ValueError as error:
        ref = getattr(error, "rejected_inference_receipt_ref", None)
        if not ref:
            raise
    finally:
        reasoner.generator.close()
    return {"receipt_ref": ref, "document": read_verified_json(dispatch.admission.artifacts, ref)}
