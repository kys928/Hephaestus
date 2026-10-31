"""F06/F07: real CPU optimization validates dose, final counters and gradients."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService
from test_real_hf_training import _tiny_transformers_fixture, _proposal, _wait

pytestmark = [pytest.mark.real_cpu, pytest.mark.skipif(
    any(importlib.util.find_spec(p) is None for p in ("torch", "transformers", "tokenizers")),
    reason="optional tiny CPU training dependencies unavailable")]


def proposal_at(root, run, **updates):
    root.mkdir()
    model, _, count = _tiny_transformers_fixture(root)
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config.update(resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    config_path.write_text(json.dumps(config))
    return _proposal(root, run, model_dir=model, model_revision=directory_content_identity(model),
                     parameter_count=count, vocabulary_size=9, **updates)


def run_to_end(tmp_path, proposal):
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    assert service.launch(proposal).status == "running"
    result = _wait(service, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert result.status == "completed", result.to_dict()
    import torch
    state = torch.load(Path(result.checkpoint_refs[0]) / "training_state.pt", weights_only=True)
    return result, state


def test_off_cadence_final_checkpoint_retains_true_cumulative_counters(tmp_path):
    proposal = proposal_at(tmp_path / "input", "off-cadence", max_steps=3, checkpoint_every_steps=2)
    result, state = run_to_end(tmp_path, proposal)
    summary = json.loads(Path(result.metrics_ref).read_text())
    assert state["step"] == 3
    assert state["examples_processed"] == summary["examples_processed"] == 3
    assert state["tokens_processed"] == summary["tokens_processed"] > 0


def test_one_epoch_three_batches_accumulation_two_has_no_wrapped_sample(tmp_path):
    proposal = proposal_at(tmp_path / "input", "one-epoch", max_steps=10,
                           max_epochs=1.0, gradient_accumulation_steps=2, checkpoint_every_steps=3)
    constraints = proposal.training_constraints
    texts = ["hello world tiny model", "tiny model learns hello", "world model learns tiny"]
    data = Path(constraints["processed_dataset_ref"])
    data.write_text("".join(json.dumps({"text": t, "record_kind": "text"}) + "\n" for t in texts))
    digest = "sha256:" + hashlib.sha256(data.read_bytes()).hexdigest()
    constraints["processed_dataset_hash"] = digest
    evidence_path = Path(constraints["processing_evidence_ref"])
    evidence = json.loads(evidence_path.read_text())
    evidence["processed_content_hash"] = digest
    evidence_path.write_text(json.dumps(evidence))
    constraints["processing_evidence_hash"] = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    result, state = run_to_end(tmp_path, proposal)
    assert state["step"] == 2 and state["examples_processed"] == 3
    assert state["batches_processed"] == 3
    # Independent two-window reference. The short final window divides by one,
    # so a sample-count-only repair cannot hide underweighted final gradients.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(constraints["model_id"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(constraints["tokenizer_id"], local_files_only=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    for window in ((0, 1), (2,)):
        optimizer.zero_grad(set_to_none=True)
        for index in window:
            ids = torch.tensor([tokenizer.encode(texts[index], add_special_tokens=False) + [1]])
            (model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids).loss / len(window)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    actual = AutoModelForCausalLM.from_pretrained(Path(result.checkpoint_refs[0]) / "model", local_files_only=True)
    assert all(torch.allclose(value, actual.state_dict()[key], atol=1e-7, rtol=1e-6)
               for key, value in model.state_dict().items())


def test_resume_at_completed_step_is_a_no_op_with_preserved_checkpoint(tmp_path):
    import time
    from hephaestus.schemas.experiment_contract import TrainingControlRequest
    proposal = proposal_at(tmp_path / "input", "final-interruption", max_steps=2,
                           checkpoint_every_steps=1, step_delay_seconds=0.5)
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    assert service.launch(proposal).status == "running"
    metrics = tmp_path / "runs" / proposal.run_id / "metrics.jsonl"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if metrics.exists() and any(json.loads(line)["step"] == 2 for line in metrics.read_text().splitlines()):
            break
        time.sleep(0.01)
    else:
        pytest.fail("worker did not reach final step before interruption deadline")
    service.control(TrainingControlRequest("interrupt", proposal.run_id, "interrupt", "test", "predeclared final-step fault"))
    interrupted = _wait(service, proposal.run_id, {"interrupted", "failed"}, timeout=60)
    assert interrupted.status == "interrupted", interrupted.to_dict()
    checkpoint = Path(interrupted.checkpoint_refs[0])
    before = hashlib.sha256((checkpoint / "training_state.pt").read_bytes()).hexdigest()
    restored = TransformersTrainingLifecycleService(tmp_path / "runs")
    resumed = restored.control(TrainingControlRequest("resume", proposal.run_id, "resume", "test", "finish without another update"))
    assert resumed.status == "resuming", resumed.to_dict()
    completed = _wait(restored, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert completed.status == "completed", completed.to_dict()
    assert hashlib.sha256((checkpoint / "training_state.pt").read_bytes()).hexdigest() == before
    summary = json.loads(Path(completed.metrics_ref).read_text())
    assert summary["optimizer_updates_this_session"] == 0
    assert summary["examples_processed"] == 2 and summary["finite"]
