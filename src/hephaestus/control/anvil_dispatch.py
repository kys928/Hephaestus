"""Prepare frozen specialist inputs through the existing durable worker."""

from dataclasses import asdict, dataclass

from hephaestus.control.anvil_context import AnvilAdmission
from hephaestus.jobs import DurableWorker, JobRecord, SQLiteJobQueue
from hephaestus.jobs.durable_worker import JobExecutionContext
from hephaestus.schemas.anvil import ContributionOrigin, require_digest
from hephaestus.storage.verified import canonical_bytes, read_verified_json


_JOB_PREFIX = "anvil_prepare_"
_ROLES = ("diagnosis", "planner", "diagnosis", "planner")


@dataclass(slots=True)
class AnvilDispatch:
    admission: AnvilAdmission
    queue: SQLiteJobQueue
    plan_ref: str

    @property
    def _job_type(self) -> str:
        require_digest(self.plan_ref)
        return _JOB_PREFIX + self.plan_ref[7:47]

    def _plan(self) -> dict:
        plan = read_verified_json(self.admission.artifacts, self.plan_ref, limit=8192)
        if (plan.get("schema") != "anvil.dispatch.v1"
                or plan.get("registry_ref") != self.admission.registry_ref
                or not all(isinstance(plan.get(key), str) and plan[key]
                           for key in ("investigation", "run_id", "lineage_id"))
                or not isinstance(plan.get("material_refs"), list)
                or len(plan["material_refs"]) != 3
                or len(set(plan["material_refs"])) != 3):
            raise ValueError("invalid independently bound Anvil dispatch plan")
        for ref in plan["material_refs"]:
            require_digest(ref)
        return plan

    def _prepare(self, payload: dict):
        plan = self._plan()
        if payload.get("plan_ref") != self.plan_ref:
            raise PermissionError("dispatch uses a different investigation plan")
        origin = ContributionOrigin(**payload["origin"])
        slot = payload["slot"]
        if (type(slot) is not int or not 0 <= slot < len(_ROLES) or origin.role != _ROLES[slot]
                or (origin.investigation, origin.run_id, origin.lineage_id)
                != (plan["investigation"], plan["run_id"], plan["lineage_id"])):
            raise PermissionError("dispatch scope or specialist slot differs from the plan")
        round_two = slot >= 2
        expected_refs = sorted(plan["material_refs"] if round_two else plan["material_refs"][:2])
        if payload["sequence"] != (2 if round_two else 0) or payload["material_refs"] != expected_refs:
            raise PermissionError("dispatch inputs differ from the pinned investigation round")
        return self.admission.prepare(origin, payload["sequence"], tuple(expected_refs))

    def _publish(self, document: dict) -> str:
        return self.admission.artifacts.put_bytes(
            canonical_bytes(document), media_type="application/json",
        ).artifact_ref

    def submit(self, *, owner_id: str, origin: ContributionOrigin, slot: int, sequence: int,
               material_refs: tuple[str, ...]) -> JobRecord:
        payload = {"plan_ref": self.plan_ref, "origin": asdict(origin), "slot": slot,
                   "sequence": sequence, "material_refs": sorted(material_refs)}
        self._prepare(payload)
        return self.queue.submit(
            self._publish(payload), owner_id, run_id=origin.run_id,
            idempotency_key=f"{self._job_type}:{slot}", job_type=self._job_type, maximum_attempts=1,
        )

    def worker(self, worker_id: str) -> DurableWorker:
        require_digest(self.plan_ref)
        return DurableWorker(worker_id, self.queue, self._handle, job_type=self._job_type)

    def _handle(self, job: JobRecord, context: JobExecutionContext) -> str:
        context.raise_if_cancelled()
        payload = read_verified_json(self.admission.artifacts, job.payload_ref, limit=8192)
        task = self._prepare(payload)
        if job.job_type != self._job_type or job.run_id != task.run_id:
            raise ValueError("durable dispatch does not match the prepared task")
        context.raise_if_cancelled()
        return self._publish({
            "schema": "anvil.prepared-role-task.v1", "task": asdict(task),
            "dispatch_payload_ref": job.payload_ref, "execution_authorized": False,
        })
