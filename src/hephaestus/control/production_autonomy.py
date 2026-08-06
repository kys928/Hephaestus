"""Final composition helpers for the production-autonomy continuation wave."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from hephaestus.control.autonomous_experiment import AutonomousExperimentCoordinator
from hephaestus.data.preprocessing import DataFactoryResult
from hephaestus.schemas.discovery_contract import ModelCandidate
from hephaestus.schemas.experiment_contract import ExperimentProposal


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class ProductionAutonomyCoordinator(AutonomousExperimentCoordinator):
    """Coordinator with backend-correct bindings for the reviewed real lifecycle.

    The original coordinator remains backward compatible with the local fixture.
    This subclass closes the integration mismatch between data-factory artifacts
    and the strict `TransformersTrainingLifecycleService` launch contract.
    """

    def bind_training_inputs(
        self,
        proposal: ExperimentProposal,
        data_result: DataFactoryResult,
        model_candidate: ModelCandidate,
        *,
        training_recipe_ref: str,
        max_steps: int,
        learning_rate: float | None = None,
        backend_id: str = "local_fixture",
        tokenizer_revision: str | None = None,
        training_mode: str = "full_finetune",
        optimizer: str = "adamw",
        scheduler: str = "constant",
        hidden_size: int | None = None,
        vocabulary_size: int | None = None,
        special_token_ids: Mapping[str, int] | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        backend_options: Mapping[str, object] | None = None,
    ) -> ExperimentProposal:
        if backend_id != "transformers_causal_lm":
            return super().bind_training_inputs(
                proposal,
                data_result,
                model_candidate,
                training_recipe_ref=training_recipe_ref,
                max_steps=max_steps,
                learning_rate=0.05 if learning_rate is None else learning_rate,
                backend_id=backend_id,
            )

        options = dict(backend_options or {})
        contract_ref = Path(data_result.artifact_dir) / "trainable_data_contract.json"
        evidence_ref = Path(
            str(data_result.processing_evidence.get("processing_evidence_ref") or "")
        )
        processed_ref = Path(
            str(data_result.processing_evidence.get("processed_dataset_ref") or "")
        )
        if not contract_ref.is_file():
            raise ValueError("real training binding requires a persisted trainable-data contract")
        if not evidence_ref.is_file():
            raise ValueError("real training binding requires persisted processing evidence")
        if not processed_ref.is_file():
            raise ValueError("real training binding requires the processed JSONL artifact")

        tokenizer_id = str(
            data_result.manifest.tokenizer_ref
            or model_candidate.tokenizer_ref
            or options.get("tokenizer_id")
            or ""
        )
        resolved_tokenizer_revision = str(
            tokenizer_revision
            or model_candidate.metadata.get("tokenizer_revision")
            or model_candidate.compatibility.get("tokenizer_revision")
            or model_candidate.revision
            or ""
        )
        resolved_hidden_size = int(
            hidden_size
            or model_candidate.compatibility.get("hidden_size")
            or model_candidate.runtime_requirements.get("hidden_size")
            or 0
        )
        resolved_vocabulary_size = int(
            vocabulary_size
            or model_candidate.compatibility.get("vocabulary_size")
            or model_candidate.metadata.get("vocabulary_size")
            or 0
        )
        resolved_special_tokens = dict(
            special_token_ids
            or (
                model_candidate.compatibility.get("special_token_ids")
                if isinstance(model_candidate.compatibility.get("special_token_ids"), dict)
                else {}
            )
        )
        approval_evidence = proposal.metadata.get("approval_evidence", {})
        approval_refs = (
            [str(value) for value in approval_evidence.values() if str(value).strip()]
            if isinstance(approval_evidence, dict)
            else []
        )
        provenance_ref = str(
            options.get("provenance_ref")
            or model_candidate.metadata.get("provenance_ref")
            or (model_candidate.evidence_refs[0] if model_candidate.evidence_refs else "")
        )
        constraints: dict[str, object] = {
            **proposal.training_constraints,
            **options,
            "backend_id": backend_id,
            "model_id": model_candidate.model_id,
            "model_revision": model_candidate.revision or "",
            "architecture_family": model_candidate.architecture_family or "",
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": resolved_tokenizer_revision,
            "training_mode": training_mode,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "training_recipe_ref": training_recipe_ref,
            "trainable_data_contract_ref": str(contract_ref),
            "trainable_data_contract_hash": _content_hash(contract_ref),
            "processed_dataset_ref": str(processed_ref),
            "processed_dataset_hash": data_result.processed_content_hash,
            "processing_evidence_ref": str(evidence_ref),
            "processing_evidence_hash": _content_hash(evidence_ref),
            "max_steps": max_steps,
            "learning_rate": 5e-5 if learning_rate is None else learning_rate,
            "parameter_count": int(model_candidate.parameter_count or 0),
            "context_length": int(model_candidate.context_length or 0),
            "hidden_size": resolved_hidden_size,
            "vocabulary_size": resolved_vocabulary_size,
            "special_token_ids": resolved_special_tokens,
            "device": device,
            "dtype": dtype,
            "trust_remote_code": False,
            "license": model_candidate.license or "",
            "provenance_ref": provenance_ref,
            "approval_refs": approval_refs,
        }
        bound = replace(
            proposal,
            training_constraints=constraints,
            metadata={
                **proposal.metadata,
                "training_binding": "transformers_causal_lm.v1",
                "processed_dataset_identity": data_result.dataset_identity,
                "model_candidate_id": model_candidate.candidate_id,
            },
        )
        self._record("experiment_training_binding", bound.to_dict())
        return bound


__all__ = ["ProductionAutonomyCoordinator"]
