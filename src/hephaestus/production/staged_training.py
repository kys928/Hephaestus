"""Durable scientific-proposal -> real PEFT lifecycle -> verified checkpoint handoff.

A narrow adapter for *existing* services, not another scheduler. The stage
must consume the immutable, workflow-owned scientific proposal rather than
a caller-supplied in-memory object. Missing discovery/data/approval evidence
blocks; it is NEVER synthesized. RunPod/workload launch remains behind the
existing guarded TrainingLifecycleService and the operator's configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from hephaestus.control.staged_state import (
    StagedOperationRequest, StagedOperationResult, StagedOutputRecord,
)
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes
from .staged_records import verified_workflow_payload
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest


_LAUNCHABLE = frozenset({"ready", "approved"})
_IN_FLIGHT = frozenset({"preparing", "queued", "running", "resuming", "interrupted"})
_NO_RUN_CODES = frozenset({"run_not_found"})
_SUBSTEPS = frozenset({
    "training_input_binding", "launch_configuration", "lifecycle_launch",
    "training_status_poll", "checkpoint_resolution",
})


@dataclass(slots=True)
class StagedTrainingLifecycleAdapter:
    """Use real persistent scientific artifacts and the configured lifecycle.

    This adapter is opt-in for the production stage slots. It does not start a
    subprocess until the workflow reaches lifecycle_launch *and* execute is
    explicitly enabled. Its launch id is fixed to the scientific run id.
    """

    repository: StateRepository
    artifacts: ArtifactStore
    lifecycle: object
    execute_enabled: bool = False

    def _record(self, request: StagedOperationRequest, *, step: str,
                kind: str) -> dict[str, object]:
        return verified_workflow_payload(self.repository, self.artifacts, request,
                                         substep=step, kind=kind)

    def _proposal(self, request: StagedOperationRequest, *,
                  finalized: bool = False) -> ExperimentProposal:
        step = "training_input_binding" if finalized else "experiment_proposal"
        kind = "experiment_training_binding" if finalized else "experiment_proposal"
        data = self._record(request, step=step, kind=kind)
        proposal = ExperimentProposal(**data)
        if (proposal.run_id != request.run_id
                or proposal.lineage_id != request.lineage_id
                or proposal.stage_name != request.stage_name
                or not proposal.experiment_id
                or not proposal.intervention_id
                or not proposal.primary_variable):
            raise ValueError("experiment proposal does not belong to this workflow")
        return proposal

    @staticmethod
    def _check_launchable(proposal: ExperimentProposal) -> None:
        if proposal.status not in _LAUNCHABLE:
            raise ValueError(f"experiment is not approved/ready: {proposal.status}")
        if proposal.metadata.get("unresolved_discovery"):
            raise ValueError("dataset/model discovery is unresolved; cannot train")
        if proposal.required_approvals:
            raise ValueError("training approvals require independently verified binding")
        # The real HF lifecycle verifies these and the dataset bytes in depth.
        # Fail early instead of attempting to launch a run that has no data.
        constraints = proposal.training_constraints
        for name in ("model_id", "model_revision", "data_contract_ref",
                     "training_recipe_ref", "max_steps"):
            if constraints.get(name) in (None, ""):
                raise ValueError(f"training input {name} is missing")
        steps = constraints["max_steps"]
        if type(steps) is not int or steps <= 0:
            raise ValueError("training maximum steps must be a positive integer")
        if not proposal.baseline_ref:
            raise ValueError("experiment has no baseline checkpoint/run identity")
        seed = constraints.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("training seed must be an explicit nonnegative integer")

    @staticmethod
    def _result(handle: TrainingRunHandle, *, polling: bool) -> StagedOperationResult:
        status = handle.status
        if polling and status in _IN_FLIGHT:
            staged_status, resumable = "interrupted", True
        elif status == "completed" or (not polling and status in _IN_FLIGHT):
            staged_status, resumable = "completed", False
        elif status in {"failed", "cancelled"}:
            staged_status, resumable = "terminal_failure", False
        else:
            staged_status, resumable = "terminal_failure", False
        problems = tuple(item.code for item in handle.issues if item.blocking)
        if staged_status == "terminal_failure" and not problems:
            problems = (f"training_{status or 'unknown'}",)
        return StagedOperationResult(
            status=staged_status, resumable=resumable,
            records=(StagedOutputRecord("training_run_handle", handle.to_dict()),),
            blocking_issues=problems,
            metadata={
                "training_status": status,
                "training_run_handle": handle.to_dict(),
                "experiment_id": handle.experiment_id,
                "run_id": handle.run_id,
                "checkpoint_ref": handle.checkpoint_refs[-1] if handle.checkpoint_refs else None,
            },
        )

    def _current_or_launch(self, request: StagedOperationRequest,
                           proposal: ExperimentProposal) -> TrainingRunHandle:
        current = self.lifecycle.status(request.run_id)
        if current.status == "failed" and current.issues and all(
            item.code in _NO_RUN_CODES for item in current.issues
        ):
            return self.lifecycle.launch(proposal)
        if current.run_id != request.run_id:
            raise ValueError("training lifecycle returned an unrelated run")
        if current.experiment_id != proposal.experiment_id:
            raise ValueError("existing training job belongs to a different experiment")
        return current

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep not in _SUBSTEPS:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=(f"unsupported_training_stage:{request.substep}",))
        try:
            if request.substep == "training_input_binding":
                proposal = self._proposal(request)
                self._check_launchable(proposal)
                return StagedOperationResult(
                    records=(StagedOutputRecord("experiment_training_binding", proposal.to_dict()),),
                    metadata={"experiment_id": proposal.experiment_id,
                              "training_binding_verified": True},
                )
            if request.substep == "launch_configuration":
                proposal = self._proposal(request, finalized=True)
                self._check_launchable(proposal)
                return StagedOperationResult(
                    records=(StagedOutputRecord("training_launch_configuration", {
                        "run_id": proposal.run_id,
                        "experiment_id": proposal.experiment_id,
                        "execution_enabled": self.execute_enabled,
                        "training_constraints": proposal.training_constraints,
                    }),),
                    metadata={"launch_configuration_verified": True,
                              "run_id": proposal.run_id,
                              "config_identity": "sha256:" + hashlib.sha256(
                                  canonical_bytes(proposal.training_constraints)).hexdigest(),
                              "seed_identity": "train-seed:" + str(
                                  proposal.training_constraints["seed"])},
                )
            if request.substep == "lifecycle_launch":
                proposal = self._proposal(request, finalized=True)
                self._check_launchable(proposal)
                prior = request.prior_outputs.get("training_engineer.launch_configuration", {})
                if prior.get("launch_configuration_verified") is not True:
                    raise ValueError("verified launch configuration is missing")
                if not self.execute_enabled:
                    return StagedOperationResult(
                        status="approval_pending", resumable=True,
                        blocking_issues=("training_execution_disabled",),
                    )
                return self._result(self._current_or_launch(request, proposal), polling=False)
            if request.substep == "training_status_poll":
                launch = request.prior_outputs.get("training_engineer.lifecycle_launch", {})
                if launch.get("run_id") != request.run_id:
                    raise ValueError("completed training launch receipt missing or mismatched")
                handle = self.lifecycle.status(request.run_id)
                if handle.experiment_id != launch.get("experiment_id"):
                    raise ValueError("training poll changed experiment identity")
                return self._result(handle, polling=True)
            # checkpoint_resolution is performed ONLY from a completed poll;
            # no evaluator can infer a final checkpoint from a launch receipt.
            poll = request.prior_outputs.get("runtime_monitor.training_status_poll", {})
            if poll.get("training_status") != "completed":
                raise ValueError("checkpoint resolution requires completed training")
            handle = self.lifecycle.status(request.run_id)
            if (handle.status != "completed" or not handle.checkpoint_refs
                    or handle.experiment_id != poll.get("experiment_id")
                    or handle.checkpoint_refs[-1] != poll.get("checkpoint_ref")):
                raise ValueError("final checkpoint differs from completed training receipt")
            checkpoint = handle.checkpoint_refs[-1]
            verified, manifest_hash = validate_checkpoint_manifest(Path(checkpoint))
            if not verified:
                raise ValueError(f"training checkpoint integrity failed: {manifest_hash}")
            return StagedOperationResult(
                records=(StagedOutputRecord("checkpoint_resolution", {
                    "checkpoint_ref": checkpoint, "manifest_hash": manifest_hash,
                    "run_id": request.run_id, "experiment_id": handle.experiment_id,
                }),),
                metadata={"checkpoint_ref": checkpoint, "checkpoint_manifest_hash": manifest_hash,
                          "experiment_id": handle.experiment_id},
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return StagedOperationResult(status="blocked", resumable=False,
                blocking_issues=(f"training_handoff_invalid:{type(exc).__name__}",))


@dataclass(slots=True)
class StagedCheckpointAndEvaluation:
    """Give checkpoint_resolution to real training; keep existing evaluator."""

    training: StagedTrainingLifecycleAdapter
    evaluator: object

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep == "checkpoint_resolution":
            return self.training.execute(request)
        return self.evaluator.execute(request)
