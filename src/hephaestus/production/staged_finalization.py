"""Deterministic scientific proposal finalization from immutable selected inputs.

This is the narrow missing bridge between Planner discovery, local Data Factory
outputs and the existing PEFT TrainingLifecycleService. It never chooses a model
or dataset, invents approvals, modifies benchmark evidence or launches a job.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
)
from hephaestus.schemas.discovery_contract import (
    ModelCandidate, ModelSelectionDecision, DatasetSelectionDecision,
)
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import read_verified_json
from hephaestus.training.peft_support import validate_recipe

from .staged_records import verified_workflow_payload


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verified_file(path: Path, expected_hash: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha(path) != expected_hash:
        raise ValueError(f"verified training input file missing or mutated: {path.name}")


@dataclass(slots=True)
class StagedExperimentFinalizer:
    repository: StateRepository
    artifacts: ArtifactStore
    training: object
    # V1 intentionally only admits locally pinned model/tokenizer bytes. Remote
    # HF downloads require the dedicated operator-approved acquisition flow.
    local_models_only: bool = True
    approved_recipe_refs: frozenset[str] = field(default_factory=frozenset)
    approved_baseline_refs: frozenset[str] = field(default_factory=frozenset)

    def _load(self, request: StagedOperationRequest, substep: str, kind: str):
        return verified_workflow_payload(
            self.repository, self.artifacts, request, substep=substep, kind=kind)

    def _approved(self, request: StagedOperationRequest, *, phase: str, subject: str,
                  required: set[str]) -> str | None:
        if not required:
            return None
        data = self._load(request, phase, "approval_decision")
        if (data.get("status") != "approved" or
                data.get("subject_ref") != subject or
                not data.get("approval_ref") or
                not required.issubset(set(data.get("requirements") or []))):
            raise PermissionError("staged operator approval is absent, unscoped or stale")
        return str(data["approval_ref"])

    def _model(self, request: StagedOperationRequest):
        decision_data = self._load(request, "model_selection", "model_selection_decision")
        discovery = self._load(request, "model_discovery", "model_discovery_result")
        selection = ModelSelectionDecision.from_dict(decision_data)
        if (selection.status != "selected" or not selection.selected_candidate_id or
                discovery.get("request_id") != selection.request_id or
                discovery.get("status") != "discovered"):
            raise ValueError("no verified policy-selected model")
        matches = [ModelCandidate(**data) for data in discovery.get("candidates", [])
                   if data.get("candidate_id") == selection.selected_candidate_id]
        if len(matches) != 1:
            raise ValueError("selected model not present uniquely in verified discovery")
        model = matches[0]
        if model.risk_signals or not model.license or model.missing_metadata:
            raise ValueError("selected model has unresolved licensing or risk")
        # For this initial complete integration, model and tokenizer must be
        # already present at pinned content-addressed local paths.
        if self.local_models_only:
            if (not model.model_id or not model.tokenizer_ref or
                    not Path(model.model_id).is_dir() or
                    not Path(model.tokenizer_ref).is_dir()):
                raise ValueError("local model/tokenizer not materialized and verified")
            if directory_content_identity(model.model_id) != model.revision:
                raise ValueError("selected model bytes differ from pinned revision")
        approvals = set(selection.required_approvals)
        model_approval = self._approved(
            request, phase="model_approval_gate",
            subject=selection.decision_id, required=approvals)
        return selection, model, model_approval

    def _data(self, request: StagedOperationRequest):
        decision_data = self._load(request, "dataset_selection", "dataset_selection_decision")
        selection = DatasetSelectionDecision.from_dict(decision_data)
        if selection.status != "selected" or len(selection.selected_candidate_ids) != 1:
            raise ValueError("training requires exactly one policy-selected dataset")
        approval = self._approved(
            request, phase="acquisition_approval_gate",
            subject=selection.decision_id, required={"dataset_acquisition"})
        receipt = self._load(request, "dataset_acquisition", "acquisition_receipt")
        factory = self._load(request, "manifest_production", "data_factory_result")
        if (receipt.get("selection_decision_id") != selection.decision_id or
                receipt.get("candidate_id") != selection.selected_candidate_ids[0] or
                receipt.get("approval_refs") != [approval] or
                factory.get("schema") != "staged-data-factory.v1"):
            raise ValueError("Data Factory is not bound to the approved selected dataset")
        evidence = factory.get("processing_evidence")
        contract = factory.get("trainable_data_contract")
        if not isinstance(evidence, dict) or not isinstance(contract, dict):
            raise ValueError("data preprocessing/contract evidence missing")
        processed = Path(str(factory.get("processed_dataset_ref") or ""))
        processing = Path(str(factory.get("processing_evidence_ref") or ""))
        contract_path = processing.parent / "trainable_data_contract.json"
        _verified_file(processed, str(factory["processed_content_hash"]))
        if (contract.get("processed_dataset_ref") != str(processed) or
                evidence.get("processed_dataset_ref") != str(processed) or
                evidence.get("processed_content_hash") != factory["processed_content_hash"] or
                evidence.get("approval_refs") != [approval] or
                contract.get("run_id") != request.run_id):
            raise ValueError("trainable contract does not match approved source")
        if (processing.is_symlink() or contract_path.is_symlink() or
                processing != Path(str(evidence.get("processing_evidence_ref") or "")) or
                json.loads(processing.read_text(encoding="utf-8")) != evidence or
                json.loads(contract_path.read_text(encoding="utf-8")) != contract):
            raise ValueError("local processed evidence or contract changed")
        tokenizer = evidence.get("tokenizer_compatibility")
        if (not isinstance(tokenizer, dict) or
                tokenizer.get("status") != "checked" or
                tokenizer.get("compatible") is not True):
            raise ValueError("tokenizer compatibility not independently checked")
        return selection, factory, evidence, processed, processing, contract_path, tokenizer, approval

    def finalize(self, request: StagedOperationRequest) -> ExperimentProposal:
        original = ExperimentProposal(**self._load(
            request, "experiment_proposal", "experiment_proposal"))
        if (original.run_id != request.run_id or
                original.lineage_id != request.lineage_id or
                original.stage_name != request.stage_name or
                not original.baseline_ref or not original.experiment_id):
            raise ValueError("Planner proposal is not owned by this campaign")
        if original.baseline_ref not in self.approved_baseline_refs:
            raise PermissionError("baseline is not pinned by independent campaign policy")
        dsel, data, evidence, processed, processing, contract, tokenizer, dagrant = self._data(request)
        msel, model, mgrant = self._model(request)
        if (original.dataset_selection_id not in (None, dsel.decision_id) or
                original.model_selection_id not in (None, msel.decision_id)):
            raise ValueError("Planner originally selected different scientific candidates")
        requirements = set(original.required_approvals)
        valid = {"dataset_acquisition"}
        if mgrant:
            valid.update(msel.required_approvals)
        if not requirements.issubset(valid):
            raise PermissionError("Planner requests additional unverified approval")
        constraints = dict(original.training_constraints)
        required_recipe_fields = (
            "training_mode", "peft_config", "training_recipe_ref", "seed", "max_steps",
            "optimizer", "scheduler", "tokenizer_revision", "context_length",
            "vocabulary_size", "special_token_ids",
        )
        if any(constraints.get(name) in (None, "", {}) for name in required_recipe_fields):
            raise ValueError("Planner training recipe is incomplete")
        if constraints["training_mode"] not in {"lora_finetune", "lora_continuation"}:
            raise ValueError("this finalizer requires a PEFT intervention")
        recipe_ref = constraints["training_recipe_ref"]
        if (not isinstance(recipe_ref, str) or recipe_ref not in self.approved_recipe_refs
                or not recipe_ref.startswith("sha256:")):
            raise PermissionError("PEFT recipe must be independently pinned by operator policy")
        approved_recipe = read_verified_json(self.artifacts, recipe_ref)
        if (approved_recipe.get("schema") != "approved-peft-recipe.v1"
                or approved_recipe.get("training_mode") != constraints["training_mode"]
                or approved_recipe.get("peft_config") != constraints["peft_config"]
                or approved_recipe.get("max_steps") != constraints["max_steps"]
                or approved_recipe.get("model_selection_id") != msel.decision_id):
            raise ValueError("training recipe differs from approved immutable policy")
        if type(constraints["seed"]) is not int or constraints["seed"] < 0:
            raise ValueError("training seed must be explicitly pinned")
        if type(constraints["max_steps"]) is not int or constraints["max_steps"] <= 0:
            raise ValueError("training steps must be explicitly bounded")
        tokenizer_id = model.tokenizer_ref
        if tokenizer.get("tokenizer_ref") != tokenizer_id:
            raise ValueError("processing checked another tokenizer")
        if self.local_models_only and directory_content_identity(tokenizer_id) != constraints["tokenizer_revision"]:
            raise ValueError("pinned tokenizer bytes mismatch")
        for key, expected in {
            "model_id": model.model_id, "model_revision": model.revision,
            "tokenizer_id": tokenizer_id, "architecture_family": model.architecture_family,
        }.items():
            if not expected or constraints.get(key) not in (None, expected):
                raise ValueError(f"Planner {key} contradicts selected immutable model")
        validate_recipe(dict(constraints["peft_config"]),
                        model_id=model.model_id, revision=model.revision,
                        mode=constraints["training_mode"])
        completed = {
            **constraints,
            "backend_id": "transformers_causal_lm",
            "model_id": model.model_id,
            "model_revision": model.revision,
            "architecture_family": model.architecture_family,
            "tokenizer_id": tokenizer_id,
            "trainable_data_contract_ref": str(contract),
            "trainable_data_contract_hash": _sha(contract),
            "processed_dataset_ref": str(processed),
            "processed_dataset_hash": _sha(processed),
            "processing_evidence_ref": str(processing),
            "processing_evidence_hash": _sha(processing),
            # Old integration aliases remain for their own policy checks.
            "data_contract_ref": str(processed),
            "data_contract_hash": _sha(processed).removeprefix("sha256:"),
        }
        metadata = dict(original.metadata)
        metadata.pop("unresolved_discovery", None)
        metadata["finalized_from_verified_staged_records"] = {
            "dataset_selection_id": dsel.decision_id,
            "model_selection_id": msel.decision_id,
            "dataset_approval_ref": dagrant,
            "model_approval_ref": mgrant,
            "trainable_contract_hash": _sha(contract),
            "processing_evidence_hash": _sha(processing),
        }
        return replace(original, dataset_selection_id=dsel.decision_id,
                       model_selection_id=msel.decision_id,
                       required_approvals=[], status="approved",
                       training_constraints=completed, metadata=metadata)

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep != "training_input_binding":
            return self.training.execute(request)
        try:
            proposal = self.finalize(request)
            # Reuse the existing training adapter's guards, never authorize a
            # launch from a new unchecked validation pathway.
            self.training._check_launchable(proposal)
            return StagedOperationResult(
                records=(StagedOutputRecord(
                    "experiment_training_binding", proposal.to_dict()),),
                metadata={"experiment_id": proposal.experiment_id,
                          "training_binding_verified": True,
                          "dataset_selection_id": proposal.dataset_selection_id,
                          "model_selection_id": proposal.model_selection_id},
            )
        except (ValueError, PermissionError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            return StagedOperationResult(
                status="blocked", resumable=False,
                blocking_issues=(f"scientific_training_finalization_failed:{type(exc).__name__}",),
            )
