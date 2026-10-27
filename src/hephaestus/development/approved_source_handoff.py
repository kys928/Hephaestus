"""Strict connector to real staged **approved** data, not a shortcut gate.

This bridge does not approve, download or preprocess sources. It invokes the
existing production StagedExperimentFinalizer._data verification, which
rehydrates exact workflow receipts, checks the operator's matching approval,
and rehashes materialized trainable data, contract, and processing evidence.

Until a source really passes that gate, only original-development data can
feed the separately labeled noncertified CPU PEFT research runner.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from hephaestus.control.staged_state import StagedOperationRequest
from hephaestus.production.staged_finalization import StagedExperimentFinalizer
from hephaestus.storage.base import StateRepository, ArtifactStore
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService


@dataclass(slots=True)
class ApprovedSourceCpuPeftHandoff:
    repository: StateRepository
    artifacts: ArtifactStore
    lifecycle: TransformersTrainingLifecycleService

    def verify(self, request: StagedOperationRequest) -> dict[str, object]:
        if request.substep != "training_input_binding":
            raise PermissionError("approved source can bind only at production training_input_binding")
        finalizer=StagedExperimentFinalizer(
            repository=self.repository, artifacts=self.artifacts,
            training=self.lifecycle, local_models_only=True,
        )
        # This uses the *production* chain: policy selection -> immutable
        # source discovery -> scoped approval -> actual acquisition receipt ->
        # preprocessor contract -> exact byte hashes -> tokenizer compatibility.
        decision, factory, evidence, processed, processing, contract, tokenizer, grant = finalizer._data(request)
        if not grant or not str(grant).strip():
            raise PermissionError("verified operator selection/acquisition approval required")
        return {
            "schema":"hephaestus-verified-approved-source-peft-handoff.v1",
            "run_id":request.run_id,
            "selection_decision_id":decision.decision_id,
            "approved_source_sha256":factory["processed_content_hash"],
            "approved_data_file":str(processed),
            "trainable_data_contract_file":str(contract),
            "processing_evidence_file":str(processing),
            "tokenizer_id":tokenizer["tokenizer_ref"],
            "selection_approval_ref":str(grant),
            "data_factory_schema":factory["schema"],
            "ready_for_independently_governed_training_finalizer":True,
            "training_launched":False,
            "production_approval_synthesized":False,
        }
