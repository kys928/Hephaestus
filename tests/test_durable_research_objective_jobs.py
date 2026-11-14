"""Real SQLite durable worker dispatch of sealed, bounded natural-language goals."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hephaestus.jobs import DurableWorker, SQLiteJobQueue
from hephaestus.jobs.models import JobStatus
from hephaestus.research.durable_jobs import (
    ResearchObjectiveJobHandler, submit_research_objective, objective_kind,
)
from hephaestus.research.constrained_repair import RepairBoundaryError
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import read_verified_json, canonical_bytes


@dataclass
class DeclaredStudyTestDouble:
    calls: list[tuple[str,str,str]]
    def run(self,question,*,run_id,lineage_id):
        self.calls.append((question,run_id,lineage_id))
        return {"schema":"explicit-fake-unit-study","run_id":run_id,
                "lineage_id":lineage_id,"training_performed":False}


@dataclass
class DeclaredRepairTestDouble:
    calls: list[tuple[str,str]]
    def run(self,question,*,checkout):
        self.calls.append((question,str(checkout)))
        return {"schema":"explicit-fake-unit-repair","status":"no_verified_regression",
                "model_invocations":0,"production_mutated":False}


def _setup(tmp_path):
    artifacts=FileSystemArtifactStore(tmp_path/"artifacts")
    queue=SQLiteJobQueue(tmp_path/"state.db")
    return artifacts,queue


def test_real_durable_worker_executes_frozen_study_without_training(tmp_path):
    artifacts,queue=_setup(tmp_path)
    study=DeclaredStudyTestDouble([])
    handler=ResearchObjectiveJobHandler(artifacts,research=study)
    question="Investigate Diagnostician reliability and evidence coverage"
    first=submit_research_objective(question=question,run_id="study-a",
        lineage_id="science-a",artifacts=artifacts,queue=queue)
    duplicate=submit_research_objective(question=question,run_id="study-a",
        lineage_id="science-a",artifacts=artifacts,queue=queue)
    assert duplicate.job_id==first.job_id
    worker=DurableWorker("research-worker",queue,handler,lease_seconds=20,
                         heartbeat_interval_seconds=1)
    done=worker.run_once()
    assert done.status is JobStatus.SUCCEEDED
    assert done.result_ref.startswith("sha256:")
    evidence=read_verified_json(artifacts,done.result_ref)
    assert evidence["schema"]=="explicit-fake-unit-study"
    assert evidence["training_performed"] is False
    assert len(study.calls)==1
    assert worker.run_once() is None


def test_real_durable_worker_routes_registered_code_repair_without_switching_roles(tmp_path):
    artifacts,queue=_setup(tmp_path)
    repair=DeclaredRepairTestDouble([])
    handler=ResearchObjectiveJobHandler(artifacts,repair=repair,
                                        trusted_checkout=tmp_path/"trusted-code")
    question="Investigate and repair Diagnostician probe code regression"
    job=submit_research_objective(question=question,run_id="repair-a",
        lineage_id="science-a",artifacts=artifacts,queue=queue)
    assert read_verified_json(artifacts,job.payload_ref)["kind"]=="diagnosis_code_repair"
    done=DurableWorker("research-worker",queue,handler,lease_seconds=20,
                       heartbeat_interval_seconds=1).run_once()
    assert done.status is JobStatus.SUCCEEDED
    assert len(repair.calls)==1
    assert repair.calls[0][1]==str(tmp_path/"trusted-code")
    assert read_verified_json(artifacts,done.result_ref)["production_mutated"] is False


def test_missing_live_observer_blocks_instead_of_simulating_research(tmp_path):
    artifacts,queue=_setup(tmp_path)
    job=submit_research_objective(question="Investigate Diagnostician reliability and coverage",
           run_id="study-missing",lineage_id="science-a",artifacts=artifacts,queue=queue)
    result=DurableWorker("research-worker",queue,ResearchObjectiveJobHandler(artifacts),
                         lease_seconds=20,heartbeat_interval_seconds=1).run_once()
    assert result.status in {JobStatus.FAILED,JobStatus.DEAD_LETTER}
    assert result.result_ref is None


def test_unknown_objective_never_enters_queue(tmp_path):
    artifacts,queue=_setup(tmp_path)
    with pytest.raises(RepairBoundaryError):
        submit_research_objective(question="Modify the entire frozen evaluation policy",
            run_id="blocked",lineage_id="scope",artifacts=artifacts,queue=queue)
    assert queue.lease_next("worker",20) is None


def test_forged_job_envelope_does_not_become_model_spending_authority(tmp_path):
    artifacts,queue=_setup(tmp_path)
    question="Investigate Diagnostician reliability and evidence coverage"
    ref=artifacts.put_bytes(canonical_bytes({
        "schema":"hephaestus-research-objective-job.v1",
        "kind":"diagnosis_evaluation_only",
        "question":question,"run_id":"forged","lineage_id":"science",
        "policy":"registered-capabilities-only.v1",
        "external_spending_authorized":True,
        "automatic_merge_authorized":False,
        "model_replacement_authorized":False,
    })).artifact_ref
    queue.submit(ref,"attacker",run_id="forged")
    study=DeclaredStudyTestDouble([])
    result=DurableWorker("research-worker",queue,ResearchObjectiveJobHandler(
        artifacts,research=study),lease_seconds=20,heartbeat_interval_seconds=1).run_once()
    assert result.status in {JobStatus.FAILED,JobStatus.DEAD_LETTER}
    assert study.calls==[]
