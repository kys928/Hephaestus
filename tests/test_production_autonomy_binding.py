from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from hephaestus.control import (
    ApprovalAwareDatasetSelectionService,
    GuardedTrainingLifecycleService,
    InMemoryIntegrationRecordSink,
    ProductionAutonomyCoordinator,
)
from hephaestus.data.registry import DatasetProviderRegistry
from hephaestus.planning import ClosedLoopExperimentPlanner
from hephaestus.providers.models import DeterministicModelSelectionService
from hephaestus.schemas.discovery_contract import ModelCandidate
from hephaestus.schemas.experiment_contract import ExperimentProposal


class _NoopTraining:
    def launch(self, proposal):
        raise AssertionError("not used")

    def status(self, run_id):
        raise AssertionError("not used")

    def control(self, request):
        raise AssertionError("not used")


class _NoopEvaluation:
    def compare(self, proposal, runs):
        raise AssertionError("not used")


def _coordinator() -> ProductionAutonomyCoordinator:
    return ProductionAutonomyCoordinator(
        diagnosis_service=None,  # type: ignore[arg-type]
        planner=ClosedLoopExperimentPlanner(),
        dataset_registry=DatasetProviderRegistry(),
        dataset_selector=ApprovalAwareDatasetSelectionService(),
        model_providers={},
        model_selector=DeterministicModelSelectionService(),
        training_service=GuardedTrainingLifecycleService(_NoopTraining()),  # type: ignore[arg-type]
        evaluation_service=_NoopEvaluation(),  # type: ignore[arg-type]
        record_sink=InMemoryIntegrationRecordSink(),
    )


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_training_binding_matches_strict_lifecycle_contract(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "factory"
    artifact_dir.mkdir()
    processed = artifact_dir / "trainable.jsonl"
    processed.write_text(json.dumps({"text": "hello world", "token_count": 2}) + "\n", encoding="utf-8")
    evidence = artifact_dir / "processing_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "processed_dataset_ref": str(processed),
                "processed_content_hash": _sha(processed),
                "wrapper": {"template": "<|prompt|>{prompt}<|target|>{target}"},
                "prompt_target_boundary": {"status": "explicit"},
                "tokenizer_compatibility": {
                    "compatible": True,
                    "tokenizer_ref": str(tmp_path / "tokenizer"),
                },
            }
        ),
        encoding="utf-8",
    )
    contract = artifact_dir / "trainable_data_contract.json"
    contract.write_text(
        json.dumps(
            {
                "contract_id": "contract-1",
                "run_id": "run-1",
                "manifest_id": "manifest-1",
                "processed_dataset_ref": str(processed),
                "schema_version": "trainable-data.v1",
                "min_tokens": 1,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    data_result = SimpleNamespace(
        artifact_dir=artifact_dir,
        processed_content_hash=_sha(processed),
        dataset_identity="dataset@revision+content",
        manifest=SimpleNamespace(tokenizer_ref=str(tokenizer)),
        processing_evidence={
            "processing_evidence_ref": str(evidence),
            "processed_dataset_ref": str(processed),
        },
    )
    candidate = ModelCandidate(
        candidate_id="model-candidate-1",
        provider_id="local",
        model_id=str(model),
        revision="sha256:" + "a" * 64,
        architecture_family="gpt2",
        parameter_count=1000,
        context_length=128,
        tokenizer_ref=str(tokenizer),
        license="apache-2.0",
        compatibility={
            "tokenizer_revision": "sha256:" + "b" * 64,
            "hidden_size": 32,
            "vocabulary_size": 256,
            "special_token_ids": {"eos_token_id": 1, "pad_token_id": 0},
        },
        evidence_refs=["evidence://model-provenance"],
    )
    proposal = ExperimentProposal(
        experiment_id="experiment-1",
        run_id="run-1",
        lineage_id="lineage-1",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-1",
        intervention_id="intervention-1",
        primary_variable="learning_rate",
        status="approved",
        metadata={"approval_evidence": {"model": "approval://model/1"}},
    )

    bound = _coordinator().bind_training_inputs(
        proposal,
        data_result,  # type: ignore[arg-type]
        candidate,
        training_recipe_ref="recipe://tiny-causal-lm",
        max_steps=2,
        backend_id="transformers_causal_lm",
    )
    constraints = bound.training_constraints
    required = {
        "backend_id",
        "model_id",
        "model_revision",
        "architecture_family",
        "tokenizer_id",
        "tokenizer_revision",
        "training_mode",
        "optimizer",
        "scheduler",
        "trainable_data_contract_ref",
        "trainable_data_contract_hash",
        "processed_dataset_ref",
        "processed_dataset_hash",
        "processing_evidence_ref",
        "processing_evidence_hash",
    }
    assert required <= set(constraints)
    assert constraints["backend_id"] == "transformers_causal_lm"
    assert constraints["trainable_data_contract_hash"] == _sha(contract)
    assert constraints["processed_dataset_hash"] == _sha(processed)
    assert constraints["processing_evidence_hash"] == _sha(evidence)
    assert constraints["learning_rate"] == 5e-5
    assert constraints["trust_remote_code"] is False
    assert constraints["approval_refs"] == ["approval://model/1"]
    assert bound.metadata["training_binding"] == "transformers_causal_lm.v1"


def test_fixture_binding_remains_backward_compatible(tmp_path: Path) -> None:
    processed = tmp_path / "trainable.jsonl"
    processed.write_text("{}\n", encoding="utf-8")
    data_result = SimpleNamespace(
        processing_evidence={"processed_dataset_ref": str(processed)},
        processed_content_hash=_sha(processed),
        manifest=SimpleNamespace(tokenizer_ref="tokenizer://fixture"),
    )
    candidate = ModelCandidate(
        candidate_id="fixture-model",
        provider_id="fixture",
        model_id="fixture/model",
        revision="v1",
        architecture_family="fixture",
    )
    proposal = ExperimentProposal(
        experiment_id="experiment-fixture",
        run_id="run-fixture",
        lineage_id="lineage-fixture",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-fixture",
        intervention_id="intervention-fixture",
        primary_variable="learning_rate",
    )
    bound = _coordinator().bind_training_inputs(
        proposal,
        data_result,  # type: ignore[arg-type]
        candidate,
        training_recipe_ref="recipe://fixture",
        max_steps=3,
    )
    assert bound.training_constraints["backend_id"] == "local_fixture"
    assert bound.training_constraints["data_contract_ref"] == str(processed)
    assert bound.training_constraints["learning_rate"] == 0.05
