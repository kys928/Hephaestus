"""Route bounded natural-language research objectives through the EXISTING job queue.

This is an evaluation/review job lane, not a replacement campaign scheduler.
Only content-addressed objective envelopes may enter execution; the handler
never accepts arbitrary code, shell commands, model identifiers or source keys
from a queued message.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from hephaestus.jobs.models import JobRecord
from hephaestus.jobs.queue import JobQueue
from hephaestus.jobs.durable_worker import JobExecutionContext
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json

from .autonomous_study import (
    EvaluationOnlyResearchJob, ResearchObjective, CertifiedDiagnosisObserver,
)
from .constrained_repair import (
    RepairBoundaryError, OfflineDockerPytestRunner, GitHubReviewBranchPublisher,
)
from .self_repair_controller import AutonomousDiagnosisRepairEpisode, LoopbackOllamaPatchModel
from .live_runtime import AdmittedCudaResearch


OBJECTIVE_SCHEMA="hephaestus-research-objective-job.v1"


def objective_kind(question: str) -> str:
    if not isinstance(question,str) or not 16<=len(question.strip())<=800:
        raise RepairBoundaryError("research goal text must be bounded")
    lowered=question.casefold()
    if "diagnos" not in lowered:
        raise RepairBoundaryError("no supported research objective route")
    if any(word in lowered for word in ("repair","fix")) and any(
            word in lowered for word in ("regression","code","probe","implementation")):
        return "diagnosis_code_repair"
    # Normal experimental study; its own objective gate validates details.
    ResearchObjective.from_text(question)
    return "diagnosis_evaluation_only"


def submit_research_objective(
    *, question: str, run_id: str, lineage_id: str,
    artifacts: ArtifactStore, queue: JobQueue, owner_id: str = "research-controller",
) -> JobRecord:
    kind=objective_kind(question)
    if not run_id or not lineage_id or not owner_id:
        raise RepairBoundaryError("queued scientific objective missing workflow identity")
    envelope={
        "schema":OBJECTIVE_SCHEMA,"kind":kind,
        "question":question.strip(),"run_id":run_id,
        "lineage_id":lineage_id,
        "policy":"registered-capabilities-only.v1",
        "external_spending_authorized":False,
        "model_replacement_authorized":False,
        "automatic_merge_authorized":False,
    }
    ref=artifacts.put_bytes(canonical_bytes(envelope),media_type="application/json").artifact_ref
    key="research:"+sha256(canonical_bytes([ref,owner_id,run_id,lineage_id])).hexdigest()
    return queue.submit(ref,owner_id,run_id=run_id,experiment_id=None,
                        idempotency_key=key)


@dataclass(slots=True)
class ResearchObjectiveJobHandler:
    """A real DurableWorker handler with explicit externally built capabilities."""

    artifacts: ArtifactStore
    research: EvaluationOnlyResearchJob | AdmittedCudaResearch | None = None
    repair: AutonomousDiagnosisRepairEpisode | None = None
    trusted_checkout: Path | None = None
    production_mode: bool = True

    def __call__(self, job: JobRecord, context: JobExecutionContext) -> str:
        context.raise_if_cancelled()
        envelope=read_verified_json(self.artifacts,job.payload_ref)
        if (envelope.get("schema")!=OBJECTIVE_SCHEMA
                or envelope.get("policy")!="registered-capabilities-only.v1"
                or envelope.get("external_spending_authorized") is not False
                or envelope.get("automatic_merge_authorized") is not False
                or envelope.get("model_replacement_authorized") is not False
                or envelope.get("run_id")!=job.run_id
                or not isinstance(envelope.get("lineage_id"),str)
                or not envelope["lineage_id"]):
            raise RepairBoundaryError("queued objective identity or policy scope invalid")
        question=envelope.get("question")
        if objective_kind(question)!=envelope.get("kind"):
            raise RepairBoundaryError("queued objective route changed after submission")
        if envelope["kind"]=="diagnosis_evaluation_only":
            if self.research is None:
                raise RepairBoundaryError("real bounded evaluation capability not configured")
            if self.production_mode:
                if (type(self.research) is not AdmittedCudaResearch
                        or type(self.research.job.observer) is not CertifiedDiagnosisObserver):
                    raise RepairBoundaryError(
                        "production evaluation requires signed live CUDA study and original frozen observer")
            result=self.research.run(question,run_id=job.run_id,lineage_id=envelope["lineage_id"])
        elif envelope["kind"]=="diagnosis_code_repair":
            if self.repair is None or self.trusted_checkout is None:
                raise RepairBoundaryError("real restricted repair capability not configured")
            if self.production_mode and (
                    type(self.repair.runner) is not OfflineDockerPytestRunner
                    or type(self.repair.proposer.model) is not LoopbackOllamaPatchModel
                    or type(self.repair.publisher) is not GitHubReviewBranchPublisher):
                raise RepairBoundaryError("production repair requires offline Docker, local coder and review-only GitHub PR provider")
            result=self.repair.run(question,checkout=self.trusted_checkout)
        else:
            raise RepairBoundaryError("unsupported queued research mode")
        context.raise_if_cancelled()
        return self.artifacts.put_bytes(canonical_bytes(result),media_type="application/json").artifact_ref
