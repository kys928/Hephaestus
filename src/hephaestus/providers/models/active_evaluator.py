"""Separately authorized Evaluator activation, health proof and rollback.

An applied *checkpoint promotion* is insufficient. Routing changes require a
second exact-scope operator approval plus verified finalized PEFT checkpoint.
CPU test probes can only create simulated activations; production resolver
refuses to route through them. The original five-role registry is immutable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.llm.inference_receipts import RoleGeneration
from hephaestus.llm.role_contract import RoleEvidence, RoleMasteryContract, RoleTask
from hephaestus.policy.action_authorization import ActionAuthorizer
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.production.state import ACTION_EXECUTIONS
from hephaestus.providers.models.role_stack import (
    AdapterArtifact, CertifiedRoleModel, CertifiedRoleModelStack,
)
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.single_writer import single_writer
from hephaestus.storage.base import ArtifactStore, StateRepository
from hephaestus.storage.verified import canonical_bytes
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.training.peft_support import validate_adapter


ACTIVE_ROLE_EVENTS = "active_evaluator_events"


@dataclass(slots=True)
class EvaluatorRoleActivation:
    state_root: Path
    repository: StateRepository
    artifacts: ArtifactStore
    stack: CertifiedRoleModelStack
    contract: RoleMasteryContract
    policy: ApprovalPolicy
    # Only isolated tests explicitly opt into simulated routing. Default
    # production resolution refuses a CPU-simulated active role.
    allow_simulated_for_tests: bool = False

    def _valid_checkpoint(self, checkpoint: str) -> tuple[CertifiedRoleModel, Path, str, str]:
        original = self.stack.resolve("evaluator")
        folder = Path(checkpoint)
        valid, manifest_hash = validate_checkpoint_manifest(folder)
        if not valid:
            raise ValueError(f"finalized candidate checkpoint failed integrity: {manifest_hash}")
        adapter = folder / "model"
        validate_adapter(adapter, model_id=original.model_id, revision=original.revision)
        directory_hash = directory_content_identity(adapter)
        bytes_total = sum(p.stat().st_size for p in adapter.rglob("*") if p.is_file())
        if bytes_total <= 0:
            raise ValueError("candidate PEFT adapter is empty")
        spec = replace(original, adapter=AdapterArtifact(
            s3_key="local-finalized:" + checkpoint,
            sha256=directory_hash.removeprefix("sha256:"), bytes=bytes_total,
        ))
        return spec, adapter, manifest_hash, directory_hash

    def _events(self) -> list[dict]:
        return [r for r in self.repository.all(ACTIVE_ROLE_EVENTS) if r.get("role") == "evaluator"]

    def _current(self) -> dict | None:
        for row in reversed(self._events()):
            if row.get("status") in ("active_verified", "active_simulated",
                                     "rolled_back_verified", "rolled_back_simulated"):
                return row
        return None

    def resolve(self) -> tuple[CertifiedRoleModel, Path | None]:
        """Read active routing, independently reverify bytes, otherwise original."""
        current = self._current()
        original = self.stack.resolve("evaluator")
        if current is None or not current.get("checkpoint_ref"):
            return original, None
        if current.get("status", "").endswith("_simulated") and not self.allow_simulated_for_tests:
            raise PermissionError("simulated role activation cannot route production inference")
        spec, directory, manifest_hash, adapter_hash = self._valid_checkpoint(str(current["checkpoint_ref"]))
        if (current.get("checkpoint_manifest_hash") != manifest_hash
                or current.get("adapter_tree_hash") != adapter_hash
                or current.get("model_id") != original.model_id
                or current.get("revision") != original.revision):
            raise ValueError("active Evaluator identity or checkpoint bytes changed")
        return spec, directory

    def _authorize(self, *, action: str, approval_ref: str, run_id: str,
                   lineage_id: str, checkpoint: str | None) -> dict:
        lineage = LineageStore(self.state_root).get_current(lineage_id)
        if lineage is None:
            raise PermissionError("activation lineage does not exist")
        return ActionAuthorizer(self.state_root, self.policy, self.artifacts).verify(
            approval_ref=approval_ref, action=action, run_id=run_id,
            current=lineage, checkpoint_ref=checkpoint, child_lineage_id=None,
        )

    def activate(self, *, lineage_id: str, run_id: str, checkpoint_ref: str,
                 approval_ref: str, generator, simulated: bool = False) -> dict:
        """Health-check an already-promoted checkpoint before publishing routing.

        'simulated' is never a production activation. No global model flags
        or protected registry files are modified.
        """
        with single_writer(self.state_root):
            spec, adapter, manifest_hash, tree_hash = self._valid_checkpoint(checkpoint_ref)
            lineage = LineageStore(self.state_root).get_current(lineage_id)
            if (lineage is None or lineage.get("certified_stable_checkpoint_ref") != checkpoint_ref
                    or lineage.get("best_checkpoint_ref") != checkpoint_ref):
                raise PermissionError("candidate was not certified and promoted in this lineage")
            approvals = self._authorize(
                action="activate_role_adapter", approval_ref=approval_ref,
                run_id=run_id, lineage_id=lineage_id, checkpoint=checkpoint_ref,
            )
            promotions = [
                row for row in self.repository.all(ACTION_EXECUTIONS)
                if row.get("lineage_id") == lineage_id
                and row.get("checkpoint_ref") == checkpoint_ref
                and row.get("effective_action") == "promote_checkpoint"
                and row.get("status") == "applied"
                and row.get("certification_state") == "certification_passed"
                and isinstance(row.get("authorization"), dict)
                and row["authorization"].get("approval_ref")
            ]
            if not promotions:
                raise PermissionError("missing authoritative applied certified promotion")
            operation_id = "evaluator-activation-" + hashlib.sha256(canonical_bytes(
                [lineage_id, run_id, checkpoint_ref, approval_ref]
            )).hexdigest()[:32]
            for row in self._events():
                if row.get("operation_id") == operation_id and row.get("status") in (
                        "active_verified", "active_simulated"):
                    self.resolve()
                    return row
            self.contract.validate_model(spec)
            previous = self._current()
            # Durable intent before inference. A crash before publication
            # leaves the old adapter active. Retrying is safe for routing.
            intent = {
                "operation_id": operation_id, "role": "evaluator",
                "status": "prepared", "run_id": run_id, "lineage_id": lineage_id,
                "checkpoint_ref": checkpoint_ref,
                "checkpoint_manifest_hash": manifest_hash, "adapter_tree_hash": tree_hash,
                "model_id": spec.model_id, "revision": spec.revision,
                "approval_ref": approvals["approval_ref"],
                "promotion_execution_id": promotions[-1]["execution_id"],
                "previous_operation_id": previous["operation_id"] if previous else None,
            }
            if not any(row.get("operation_id") == operation_id for row in self._events()):
                self.repository.append(ACTIVE_ROLE_EVENTS, intent)
            if not simulated:
                from hephaestus.llm.certified_runtime import PinnedTransformersRoleGenerator
                if not isinstance(generator, PinnedTransformersRoleGenerator):
                    raise PermissionError("real activation requires actual Transformers GPU role generator")
            source_ref = "checkpoint-manifest:" + manifest_hash.removeprefix("sha256:")
            prompt = self.contract.prompt(RoleTask(
                "evaluator", "Perform a non-mutating Evaluator routing health assessment. "
                "Reference the manifest evidence; do not certify a new model.",
                (RoleEvidence(source_ref, "Verified finalized checkpoint manifest " + manifest_hash),),
                run_id, lineage_id, str(lineage["stage_name"]),
            ))
            try:
                result = generator.generate(
                    spec=spec, profile=self.contract.profile("evaluator"),
                    adapter_path=adapter, prompt=prompt, max_new_tokens=384, seed=271828,
                )
                if not isinstance(result, RoleGeneration) or result.finish_reason in ("timeout", "length"):
                    raise ValueError("Evaluator health probe did not produce a complete role generation")
                self.contract.parse("evaluator", result.text, {source_ref})
                if not simulated and not result.metadata.get("gpu_name"):
                    raise ValueError("real Evaluator role probe has no measured GPU provenance")
            except Exception:
                self.repository.append(ACTIVE_ROLE_EVENTS, {**intent, "status": "probe_failed"})
                raise
            proof = self.artifacts.put_bytes(canonical_bytes({
                "schema": "evaluator-activation-probe.v1", "role": "evaluator",
                "checkpoint_ref": checkpoint_ref, "checkpoint_manifest_hash": manifest_hash,
                "adapter_tree_hash": tree_hash, "prompt": prompt, "raw_output": result.text,
                "generator": type(generator).__name__, "simulation": simulated,
                "execution": dict(result.metadata), "finish_reason": result.finish_reason,
            }), media_type="application/json")
            final = {**intent,
                     "status": "active_simulated" if simulated else "active_verified",
                     "health_probe_ref": proof.artifact_ref}
            self.repository.append(ACTIVE_ROLE_EVENTS, final)
            return final

    def rollback(self, *, lineage_id: str, run_id: str, approval_ref: str,
                 simulated: bool = False) -> dict:
        """Restore immediately preceding active adapter, or frozen original."""
        with single_writer(self.state_root):
            current = self._current()
            if current is None:
                raise ValueError("no active Evaluator activation to roll back")
            self._authorize(
                action="rollback_active_role", approval_ref=approval_ref,
                run_id=run_id, lineage_id=lineage_id, checkpoint=None,
            )
            if current["lineage_id"] != lineage_id:
                raise PermissionError("active Evaluator belongs to another lineage")
            previous_id = current.get("previous_operation_id")
            prior = next((r for r in reversed(self._events())
                          if r.get("operation_id") == previous_id and
                          r.get("status") in ("active_verified", "active_simulated")), None)
            if previous_id and prior is None:
                raise RuntimeError("original active routing target is unavailable")
            if prior:
                self._valid_checkpoint(str(prior["checkpoint_ref"]))
            operation_id = "evaluator-rollback-" + hashlib.sha256(canonical_bytes(
                [lineage_id, run_id, approval_ref, current["operation_id"]]
            )).hexdigest()[:32]
            existing = next((r for r in reversed(self._events())
                             if r.get("operation_id") == operation_id and
                             r.get("status", "").startswith("rolled_back")), None)
            if existing:
                return existing
            if current.get("status", "").endswith("_simulated") and not simulated:
                raise PermissionError("simulated activation must not be rolled back as production")
            row = {
                "operation_id": operation_id, "role": "evaluator",
                "status": "rolled_back_simulated" if simulated else "rolled_back_verified",
                "run_id": run_id, "lineage_id": lineage_id, "approval_ref": approval_ref,
                "checkpoint_ref": prior["checkpoint_ref"] if prior else None,
                "checkpoint_manifest_hash": prior["checkpoint_manifest_hash"] if prior else None,
                "adapter_tree_hash": prior["adapter_tree_hash"] if prior else None,
                "model_id": prior["model_id"] if prior else None,
                "revision": prior["revision"] if prior else None,
                "previous_operation_id": prior["previous_operation_id"] if prior else None,
                "rolled_back_operation_id": current["operation_id"],
            }
            self.repository.append(ACTIVE_ROLE_EVENTS, row)
            return row
