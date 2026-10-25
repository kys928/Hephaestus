#!/usr/bin/env python3
"""Attempt genuine frozen five-role model inference on a bounded CUDA Pod.

This is an unpromoted, approval-neutral inference smoke: it never executes a
model's suggested action, changes certified registry entries, trains any
production model, or self-certifies. It uses exact model revision and PEFT SHA
from the original frozen role stack. Failures are terminal evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.llm.certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from hephaestus.llm.role_contract import RoleMasteryContract, RoleTask, RoleEvidence
from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRegistry, EvidenceRecord,
)
from hephaestus.providers.models.role_stack import REQUIRED_ROLES, load_certified_role_model_stack
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


ROOT = Path(os.environ.get("HEPHAESTUS_ROLE_RUN_ROOT",
                           "/workspace/hephaestus/research/cuda-roles-20261025"))
VOLUME = Path("/workspace")
MAX_RUNTIME_SECONDS = 1050


class MountedReadOnlyObjectStore:
    """Read preverified S3 object equivalents from the mounted network volume.

    VerifiedRoleAdapters itself recomputes the SHA and checks archive metadata.
    The logical S3 Bucket is never used as a filesystem prefix.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()

    def get_object(self, *, Bucket: str, Key: str):
        if (not Key.startswith("hephaestus/scientific/v1/role_mastery/")
                or Key.startswith("/") or ".." in Path(Key).parts):
            raise PermissionError("non-governed S3 object path refused")
        path = self.root / Key
        if path.is_symlink() or not path.is_file() or self.root not in path.resolve().parents:
            raise PermissionError("mounted frozen role adapter missing or linked outside volume")
        return {"Body": path.open("rb")}


def save(doc):
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = ROOT / "role_result.partial.json"
    temp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(temp, ROOT / "role_result.json")


def _prefetch_one(model_id: str, revision: str, cache: Path):
    """Pin and download publicly accessible base weights *before* local-only loader.

    This preparation is not a revision change, fine-tune, certification, or
    promotion. Never accepts latest/main as a replacement.
    """
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        repo_id=model_id, revision=revision, cache_dir=str(cache),
        local_files_only=False, max_workers=4,
    ))
    if not snapshot.is_dir() or snapshot.name != revision:
        raise ValueError("Hugging Face pinned revision does not match local snapshot")
    return snapshot


def main() -> int:
    started = time.monotonic()
    report = {
        "schema": "hephaestus-real-five-role-cuda-inference.v1",
        "run_id": os.environ.get("HEPHAESTUS_ROLE_RUN_ID", "unknown"),
        "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress", "role_results": [],
        "production_model_promoted": False,
        "judge_action_executed": False,
        "granite_scientific_improvement_certified": False,
        "scientific_certificate_issued": False,
        "full_45_stage_corpus_campaign": False,
    }
    save(report)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("certified five-role inference requires exactly one CUDA GPU")
        stack = load_certified_role_model_stack(
            "configs/models/hephaestus_role_model_stack_v1.json")
        contract = RoleMasteryContract.load(
            "configs/models/hephaestus_role_mastery_runtime_v1.json")
        for role in REQUIRED_ROLES:
            contract.validate_model(stack.resolve(role))
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda"] = torch.version.cuda
        report["registered_role_count"] = len(REQUIRED_ROLES)
        evidence_store = FileSystemArtifactStore(ROOT / "immutable-receipts")
        observed = {
            "schema": "cuda-five-role-source-evidence.v1",
            "source": "frozen-certified-role-registry",
            "registry_sha256": "sha256:" + hashlib.sha256(
                Path("configs/models/hephaestus_role_model_stack_v1.json").read_bytes()).hexdigest(),
            "exact_base_revisions_checked": True,
            "independent_holdout_review": False,
            "approved_scientific_action": False,
        }
        source_ref = evidence_store.put_bytes(
            canonical_bytes(observed), media_type="application/json").artifact_ref
        task_ref = source_ref
        registry = EvidenceRegistry({
            task_ref: EvidenceRecord(
                task_ref, "frozen_runtime_registry", report["run_id"],
                "cuda-role-probe-lineage", sha256=task_ref.removeprefix("sha256:"),
                observed_sha256=task_ref.removeprefix("sha256:"),
            ),
        })
        role_cache = ROOT / "frozen-peft-cache"
        hf_cache = Path(os.environ.get("HEPHAESTUS_HF_CACHE_DIR", str(ROOT / "hf-cache")))
        reasoner = CertifiedRoleReasoner(
            stack=stack, contract=contract,
            adapters=VerifiedRoleAdapters(
                client=MountedReadOnlyObjectStore(VOLUME),
                bucket="cviwpryzao", cache_root=role_cache,
            ),
            generator=PinnedTransformersRoleGenerator(
                hf_cache_dir=hf_cache, device="cuda:0",
            ),
            artifact_store=evidence_store,
        )
        for role in REQUIRED_ROLES:
            spec = stack.resolve(role)
            result = {
                "role": role, "model_id": spec.model_id, "revision": spec.revision,
                "adapter_sha256": spec.adapter.sha256,
                "status": "in_progress",
            }
            report["role_results"].append(result)
            save(report)
            try:
                if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                    raise TimeoutError("bounded five-role scientific smoke budget exhausted")
                result["hf_pinned_snapshot"] = str(_prefetch_one(
                    spec.model_id, spec.revision, hf_cache))
                task = RoleTask(
                    role=role,
                    situation=(
                        "Read-only CUDA runtime smoke on the frozen role registry. "
                        "Do not assert a scientific improvement, certification, or "
                        "permission to act. Return a bounded, evidence-based proposal."
                    ),
                    evidence=(RoleEvidence(
                        task_ref,
                        "SHA-verified frozen five-role registry. "
                        "No independently reviewed training or holdout result has "
                        "been supplied; promotion is not authorized.",
                    ),),
                    run_id=report["run_id"], lineage_id="cuda-role-probe-lineage",
                    stage_name="smoke_test",
                )
                context = DeterministicRoleContext(
                    run_id=report["run_id"], lineage_id="cuda-role-probe-lineage",
                    stage_name="smoke_test", approval_status="none",
                    stage_allowed_actions=("request_recheck", "abort_run"),
                    verified_facts={
                        "artifact_hash_match": True, "provenance_valid": True,
                        "approval_valid": False,
                    },
                )
                observation = reasoner.propose(
                    task, registry, context=context,
                    seed=271828, max_new_tokens=192,
                )
                result.update({
                    "status": "accepted_model_generation",
                    "inference_receipt_ref": observation.inference_receipt_ref,
                    "output_sha256": observation.output_sha256,
                    "proposal_action": observation.answer.action,
                    "proposal_decision": observation.answer.decision,
                    "gate_allowed": observation.action_gate.allowed
                        if observation.action_gate is not None else None,
                    "suggestion_executed": False,
                })
            except Exception as exc:
                result.update(status="failed", error_type=type(exc).__name__,
                              error_message=str(exc)[:700])
                # Neither fake role outputs nor a backup noncertified model
                # may be substituted for a failed frozen model.
                report["status"] = "partial_failure"
                save(report)
                break
            finally:
                reasoner.generator.close()
            save(report)
        if len(report["role_results"]) == 5 and all(
                r["status"] == "accepted_model_generation" for r in report["role_results"]):
            report["status"] = "five_frozen_role_inferences_completed"
        elif report["status"] != "partial_failure":
            report["status"] = "partial_failure"
    except Exception as exc:
        report["status"] = "failed_before_model_execution"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)[:700]
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["runtime_seconds"] = round(time.monotonic()-started,2)
        save(report)
    print(json.dumps({key: report.get(key) for key in (
        "schema", "run_id", "status", "runtime_seconds", "gpu",
        "full_45_stage_corpus_campaign", "granite_scientific_improvement_certified",
    )}, sort_keys=True))
    return 0 if report["status"] == "five_frozen_role_inferences_completed" else 1


if __name__ == "__main__":
    sys.exit(main())
