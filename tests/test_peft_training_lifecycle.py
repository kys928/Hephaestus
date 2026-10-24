"""Real tiny-model CPU optimization; all weights are created locally in tests."""
import importlib.util
import json
import time
from pathlib import Path

import pytest

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService, validate_checkpoint_manifest
from hephaestus.schemas.experiment_contract import TrainingControlRequest
from test_real_hf_training import _tiny_transformers_fixture, _proposal, _wait

pytestmark = [pytest.mark.real_cpu, pytest.mark.skipif(
    any(importlib.util.find_spec(p) is None for p in ("torch", "transformers", "peft")),
    reason="optional real CPU PEFT dependencies are unavailable")]


def recipe(**updates):
    return {"rank": 2, "alpha": 4.0, "dropout": 0.0, "target_modules": ["c_attn"], **updates}


def setup_proposal(root, name, **changes):
    root.mkdir()
    base, revision, count = _tiny_transformers_fixture(root)
    return _proposal(root, name, model_dir=base, model_revision=revision, parameter_count=count,
        vocabulary_size=9, training_mode="lora_finetune", peft_config=recipe(), **changes)


def test_real_lora_optimization_exports_verified_adapter_and_generates(tmp_path):
    proposal = setup_proposal(tmp_path / "data", "fresh-lora")
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    launched = service.launch(proposal)
    assert launched.status == "running", launched.to_dict()
    done = _wait(service, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert done.status == "completed", done.to_dict()
    checkpoint = Path(done.checkpoint_refs[0])
    assert validate_checkpoint_manifest(checkpoint)[0]
    assert (checkpoint / "model" / "adapter_model.safetensors").is_file()
    assert not (checkpoint / "model" / "model.safetensors").exists()
    provenance = json.loads((checkpoint / "model" / "adapter_provenance.json").read_text())
    assert provenance["base_revision"] == proposal.training_constraints["model_revision"]
    assert not provenance["cross_run_optimizer_reset"]
    import torch
    from safetensors.torch import load_file
    weights = load_file(str(checkpoint / "model" / "adapter_model.safetensors"))
    assert any(torch.count_nonzero(v) > 0 for k, v in weights.items() if "lora_B" in k)
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert len(state["optimizer_state"]["param_groups"][0]["params"]) == len(weights)
    from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
    from hephaestus.generation.models import GenerationTask
    loading = json.loads((checkpoint / "loading_instructions.json").read_text())
    generated = TransformersCausalLMGenerationBackend().generate_batch(
        [GenerationTask("test", "generation_probes", "hello world", 7)], run_id="generated",
        loading_instructions=loading, decoding_config={"max_new_tokens": 8, "temperature": 1.0})
    assert len(generated) == 1 and generated[0].output
    # A reconstructed controller consumes the finished child without relaunch.
    restored = TransformersTrainingLifecycleService(tmp_path / "runs").status(proposal.run_id)
    assert restored.status == "completed" and restored.checkpoint_refs == done.checkpoint_refs


def test_parent_continuation_is_hash_verified_and_truthfully_resets_optimizer(tmp_path):
    proposal = setup_proposal(tmp_path / "data", "parent-lora")
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    service.launch(proposal)
    parent = _wait(service, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert parent.status == "completed", parent.to_dict()
    adapter = Path(parent.checkpoint_refs[0]) / "model"
    proposal.run_id = "child-lora"
    proposal.experiment_id = "experiment-child"
    proposal.training_constraints.update(training_mode="lora_continuation", peft_config=recipe(
        parent_adapter_ref=str(adapter), parent_adapter_hash=directory_content_identity(adapter),
        continuation_kind="adapter_warm_start"))
    assert service.launch(proposal).status == "running"
    child = _wait(service, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert child.status == "completed", child.to_dict()
    provenance = json.loads((Path(child.checkpoint_refs[0]) / "model" / "adapter_provenance.json").read_text())
    assert provenance["cross_run_optimizer_reset"]
    proposal.run_id = "tampered-child"
    proposal.training_constraints["peft_config"]["parent_adapter_hash"] = "sha256:" + "0" * 64
    refused = service.launch(proposal)
    assert refused.status == "failed" and any(i.code == "invalid_peft_recipe" for i in refused.issues)


def test_interrupted_lora_resume_matches_uninterrupted_adapter_exactly(tmp_path):
    proposal = setup_proposal(tmp_path / "data", "uninterrupted", max_steps=6, checkpoint_every_steps=1,
                              step_delay_seconds=0.1)
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    service.launch(proposal)
    baseline = _wait(service, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert baseline.status == "completed", baseline.to_dict()
    proposal.run_id = "interrupted-lora"
    proposal.experiment_id = "interrupted-experiment"
    service.launch(proposal)
    events = tmp_path / "runs" / proposal.run_id / "metrics.jsonl"
    deadline = time.monotonic() + 60
    while not events.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert events.exists()
    service.control(TrainingControlRequest("interrupt", proposal.run_id, "interrupt", "test", "failure injection"))
    interrupted = _wait(service, proposal.run_id, {"interrupted", "failed"}, timeout=60)
    assert interrupted.status == "interrupted", interrupted.to_dict()
    # New lifecycle process, immutable prepared job and checkpoint state.
    resumed = TransformersTrainingLifecycleService(tmp_path / "runs")
    result = resumed.control(TrainingControlRequest("resume", proposal.run_id, "resume", "test", "recover"))
    assert result.status in {"running", "resuming"}, result.to_dict()
    completed = _wait(resumed, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert completed.status == "completed", completed.to_dict()
    from safetensors.torch import load_file
    import torch
    left = load_file(str(Path(baseline.checkpoint_refs[0]) / "model" / "adapter_model.safetensors"))
    right = load_file(str(Path(completed.checkpoint_refs[0]) / "model" / "adapter_model.safetensors"))
    assert left.keys() == right.keys()
    assert all(torch.equal(left[k], right[k]) for k in left)
