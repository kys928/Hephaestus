"""Durable execution of admitted source measurements through the existing worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from hephaestus.jobs import DurableWorker, JobRecord, SQLiteJobQueue
from hephaestus.jobs.durable_worker import JobExecutionContext
from hephaestus.research.source_evidence import SourceReader
from hephaestus.schemas.research_objective import ResearchAdmissionError, ReviewedSourceJob
from hephaestus.storage.verified import canonical_bytes, read_verified_json

if TYPE_CHECKING:
    from hephaestus.production.composition import ProductionRuntime

_JOB_TYPE = "reviewed_research_sources"


@dataclass(slots=True)
class ReviewedSourceJobs:
    runtime: ProductionRuntime
    queue: SQLiteJobQueue
    source_reader: SourceReader
    catalog_path: Path
    catalog_ref: str
    eval_pack_ref: str

    def submit(self, intake_ref: str, *, run_id: str, lineage_id: str, owner_id: str) -> JobRecord:
        payload = ReviewedSourceJob(intake_ref, run_id, lineage_id, self.catalog_ref, self.eval_pack_ref)
        intake = read_verified_json(self.runtime.artifact_store, intake_ref)
        if (intake.get("schema") != "hephaestus-research-objective-intake.v1"
                or intake.get("run_id") != run_id or intake.get("lineage_id") != lineage_id
                or intake.get("reviewed_catalog_ref") != self.catalog_ref
                or intake.get("execution_authorized") is not False):
            raise ResearchAdmissionError("queued objective differs from the reviewed intake")
        raw = canonical_bytes(asdict(payload))
        ref = self.runtime.artifact_store.put_bytes(raw, media_type="application/json").artifact_ref
        key = _JOB_TYPE + ":" + sha256(raw).hexdigest()
        return self.queue.submit(ref, owner_id, run_id=run_id, idempotency_key=key, job_type=_JOB_TYPE)

    def worker(self, worker_id: str) -> DurableWorker:
        return DurableWorker(worker_id, self.queue, self._handle, job_type=_JOB_TYPE)

    def _handle(self, job: JobRecord, context: JobExecutionContext) -> str:
        context.raise_if_cancelled()
        payload = ReviewedSourceJob(**read_verified_json(self.runtime.artifact_store, job.payload_ref))
        if (job.job_type != _JOB_TYPE or job.run_id != payload.run_id
                or payload.catalog_ref != self.catalog_ref or payload.eval_pack_ref != self.eval_pack_ref):
            raise ResearchAdmissionError("worker configuration differs from the admitted job")
        result = self.runtime.observe_reviewed_research_sources(
            payload.intake_ref, run_id=payload.run_id, lineage_id=payload.lineage_id,
            catalog_path=self.catalog_path, independently_pinned_catalog_ref=self.catalog_ref,
            eval_pack_ref=self.eval_pack_ref, source_reader=self.source_reader,
        )
        context.raise_if_cancelled()
        return result
