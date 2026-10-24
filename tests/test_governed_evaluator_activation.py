"""Authorization and active-routing integration, explicitly simulated CPU health only.

Positive actions use synthetic *authoritative operator fixtures* in an isolated
temporary state root. They do not certify models or activate production routing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hephaestus.llm.certified_runtime import CertifiedRoleReasoner
from hephaestus.llm.inference_receipts import RoleGeneration
from hephaestus.llm.role_contract import RoleMasteryContract
from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.production.state import ACTION_EXECUTIONS
from hephaestus.providers.models.active_evaluator import (
    ACTIVE_ROLE_EVENTS, EvaluatorRoleActivation,
)
from hephaestus.providers.models.role_stack import load_certified_role_model_stack
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

from approval_fixtures import grant_for_lineage
from test_certified_role_inference import CONTRACT, STACK, _inputs, ControlledModelOutput, FakeS3, _archive, _roles


def _checkpoint(tmp_path, model_id, revision):
    folder = tmp_path / "checkpoints" / "candidate"
    adapter = folder / "model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model_id, "revision": revision,
    }))
    (adapter / "adapter_provenance.json").write_text(json.dumps({
        "base_model_id": model_id, "base_revision": revision,
        "recipe": {"rank": 8},
    }))
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-untrained-weights")
    paths = {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in adapter.rglob("*") if p.is_file()
    }
    digest = "sha256:" + hashlib.sha256(json.dumps(
        paths, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (folder / "checkpoint_manifest.json").write_text(json.dumps({
        "components": paths, "manifest_hash": digest, "partial_write": False,
    }))
    assert validate_checkpoint_manifest(folder) == (True, digest)
    return str(folder)


class SyntheticProbe:
    def __init__(self):
        self.calls = []

    def generate(self, *, spec, profile, adapter_path, prompt, max_new_tokens, seed):
        assert adapter_path.is_dir()
        self.calls.append((spec.role, str(adapter_path), spec.adapter.sha256))
        ref = next(part.split(": ", 1)[0][2:] for part in prompt.splitlines()
                   if part.startswith("- checkpoint-manifest:"))
        vocab = profile["vocabulary"]
        return RoleGeneration(json.dumps({
            "decision": vocab["decision"][0], "action": vocab["action"][0],
            "primary_variable": vocab["primary_variable"][0], "confidence": 0.5,
            "evidence_refs": [ref], "uncertainties": [],
            "rationale": "Verified finalized checkpoint is loadable in the simulation.",
        }), metadata={"synthetic_cpu_probe": True})


def _setup(tmp_path, *, grant=True):
    stack = load_certified_role_model_stack(STACK)
    spec = stack.resolve("evaluator")
    checkpoint = _checkpoint(tmp_path, spec.model_id, spec.revision)
    LineageStore(tmp_path).set_current(LineageState(
        lineage_id="L", stage_name="smoke_test", status="stable",
        trust_level="high", best_checkpoint_ref=checkpoint,
        certified_stable_checkpoint_ref=checkpoint,
    ).to_dict())
    repo = SQLiteStateRepository(tmp_path / "state.sqlite3")
    repo.append(ACTION_EXECUTIONS, {
        "execution_id": "authorized-promotion", "status": "applied",
        "lineage_id": "L", "run_id": "promotion-run",
        "checkpoint_ref": checkpoint, "effective_action": "promote_checkpoint",
        "certification_state": "certification_passed",
        "authorization": {"approval_ref": "synthetic-promoted-event"},
    })
    artifacts = FileSystemArtifactStore(tmp_path / "artifacts")
    if grant:
        grant_for_lineage(tmp_path, "L", "activation-run",
            "activate_role_adapter", checkpoint, "activation-grant")
    manager = EvaluatorRoleActivation(
        tmp_path, repo, artifacts, stack,
        RoleMasteryContract.load(CONTRACT), ApprovalPolicy(),
        allow_simulated_for_tests=True,
    )
    return manager, checkpoint


def test_synthetic_activation_routes_actual_reasoner_through_finalized_local_adapter(tmp_path):
    manager, checkpoint = _setup(tmp_path)
    probe = SyntheticProbe()
    result = manager.activate(
        lineage_id="L", run_id="activation-run", checkpoint_ref=checkpoint,
        approval_ref="activation-grant", generator=probe, simulated=True,
    )
    assert result["status"] == "active_simulated"
    assert len(probe.calls) == 1
    assert result["health_probe_ref"].startswith("sha256:")
    spec, adapter = manager.resolve()
    assert adapter == Path(checkpoint) / "model"
    assert spec.adapter.sha256 == directory_content_identity(adapter).removeprefix("sha256:")
    # Actual model-backed executor uses selected local adapter, not S3 fallback.
    model_task, registry, context = _inputs("evaluator")
    gen = ControlledModelOutput(manager.contract, model_task)
    baseline_blob = _archive()
    remote = FakeS3(baseline_blob)
    from hephaestus.llm.certified_runtime import VerifiedRoleAdapters
    reasoner = CertifiedRoleReasoner(_roles(baseline_blob), manager.contract,
        VerifiedRoleAdapters(remote, "unused", tmp_path / "cache"), gen, manager.artifacts,
        active_evaluator=manager)
    observation = reasoner.propose(model_task, registry, context=context)
    assert observation.role == "evaluator"
    assert gen.calls[0]["adapter"] == str(adapter)
    assert remote.gets == []
    assert observation.adapter_sha256 == spec.adapter.sha256


def test_activation_cannot_use_fake_grant_or_unpromoted_checkpoint(tmp_path):
    manager, checkpoint = _setup(tmp_path, grant=False)
    with pytest.raises(PermissionError):
        manager.activate(
            lineage_id="L", run_id="activation-run", checkpoint_ref=checkpoint,
            approval_ref="invented-grant", generator=SyntheticProbe(), simulated=True,
        )
    assert manager._current() is None
    manager.repository.append(ACTION_EXECUTIONS, {
        "execution_id":"bad", "status":"applied", "lineage_id":"L",
        "run_id":"promotion-run", "checkpoint_ref":"other",
        "effective_action":"promote_checkpoint", "certification_state":"certification_passed",
    })


def test_simulated_activation_refused_in_production_resolution(tmp_path):
    manager, checkpoint = _setup(tmp_path)
    manager.activate(lineage_id="L", run_id="activation-run", checkpoint_ref=checkpoint,
        approval_ref="activation-grant", generator=SyntheticProbe(), simulated=True)
    manager.allow_simulated_for_tests = False
    with pytest.raises(PermissionError, match="simulated"):
        manager.resolve()


def test_tampered_adapter_after_activation_fails_closed(tmp_path):
    manager, checkpoint = _setup(tmp_path)
    manager.activate(lineage_id="L", run_id="activation-run", checkpoint_ref=checkpoint,
        approval_ref="activation-grant", generator=SyntheticProbe(), simulated=True)
    (Path(checkpoint) / "model" / "adapter_model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        manager.resolve()


def test_probe_failure_keeps_frozen_original_as_active(tmp_path):
    manager, checkpoint = _setup(tmp_path)
    class MalformedProbe:
        def generate(self, **kwargs):
            return RoleGeneration("this-is-not-json", metadata={"synthetic_cpu_probe":True})
    with pytest.raises(ValueError):
        manager.activate(lineage_id="L", run_id="activation-run", checkpoint_ref=checkpoint,
            approval_ref="activation-grant", generator=MalformedProbe(), simulated=True)
    spec, path = manager.resolve()
    assert path is None and spec.adapter.sha256 == manager.stack.resolve("evaluator").adapter.sha256
    assert not any(r["status"].startswith("active") for r in manager.repository.all(ACTIVE_ROLE_EVENTS))


def test_rollback_requires_separate_authority_and_restores_frozen_role(tmp_path):
    manager, checkpoint = _setup(tmp_path)
    manager.activate(lineage_id="L", run_id="activation-run", checkpoint_ref=checkpoint,
        approval_ref="activation-grant", generator=SyntheticProbe(), simulated=True)
    with pytest.raises(PermissionError):
        manager.rollback(lineage_id="L", run_id="rollback-run", approval_ref="invented",
                         simulated=True)
    grant_for_lineage(tmp_path, "L", "rollback-run",
        "rollback_active_role", None, "rollback-grant")
    event = manager.rollback(lineage_id="L", run_id="rollback-run",
                             approval_ref="rollback-grant", simulated=True)
    assert event["status"] == "rolled_back_simulated"
    spec, path = manager.resolve()
    assert path is None and spec.adapter.sha256 == manager.stack.resolve("evaluator").adapter.sha256
    again = manager.rollback(lineage_id="L", run_id="rollback-run",
                             approval_ref="rollback-grant", simulated=True)
    assert again == event
