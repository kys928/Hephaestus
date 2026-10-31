"""F08: conditioning and target identity survive preprocessing or fail closed."""
import pytest
import importlib.util

from hephaestus.data.chunking import chunk_records
from hephaestus.training.hf_worker import _tokenize_records


class Tokenizer:
    def encode(self, text, **kwargs):
        return [10 + ord(c) for c in text]


def config(context=1000):
    return {"context_length": context, "max_total_tokens": 10000, "ignored_label_token": -100,
            "special_token_ids": {"eos_token_id": 1}, "prompt_masking": "mask_prompt_for_prompt_target"}


def test_long_pair_is_rejected_instead_of_split_into_prompt_supervision():
    with pytest.raises(ValueError, match="prompt.target"):
        chunk_records(({"prompt": "one two three four five six", "target": "answer"},),
            chunk_size_tokens=4, min_tokens=1, prompt_target_template="Q: {prompt} A: {target}")


def test_custom_template_preserves_structure_and_masks_only_prompt():
    records = chunk_records(({"prompt": "ask", "target": "answer"},),
        chunk_size_tokens=64, min_tokens=1, prompt_target_template="Question: {prompt}\nAnswer: {target}!").records
    samples, summary = _tokenize_records(Tokenizer(), list(records), config(), {})
    prefix = "Question: ask\nAnswer: "
    assert summary["prompt_masked_samples"] == 1
    assert samples[0]["labels"][:len(prefix)] == [-100] * len(prefix)
    assert samples[0]["labels"][len(prefix):] == Tokenizer().encode("answer!") + [1]


def test_missing_structured_boundary_never_falls_back_to_prompt_labels():
    with pytest.raises(ValueError, match="structured"):
        _tokenize_records(Tokenizer(), [{"text": "prompt-only chunk", "record_kind": "prompt_target"}], config(), {})


def test_long_tokenized_target_cannot_be_silently_truncated():
    records = chunk_records(({"prompt": "ask", "target": "long answer"},),
        chunk_size_tokens=64, min_tokens=1, prompt_target_template="Q: {prompt} A: {target}").records
    with pytest.raises(ValueError, match="context"):
        _tokenize_records(Tokenizer(), list(records), config(context=8), {})


@pytest.mark.real_cpu
@pytest.mark.skipif(any(importlib.util.find_spec(p) is None for p in ("torch", "transformers", "tokenizers")),
                    reason="optional local CPU model dependencies are unavailable")
def test_actual_cpu_worker_consumes_structured_target_without_prompt_loss(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from test_real_hf_training import _tiny_transformers_fixture, _proposal, _wait
    from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService
    model, revision, count = _tiny_transformers_fixture(tmp_path)
    proposal = _proposal(tmp_path, "structured-objective", model_dir=model, model_revision=revision,
                         parameter_count=count, vocabulary_size=9)
    constraints = proposal.training_constraints
    template = "Q: {prompt}\nA: {target}"
    rows = chunk_records(({"prompt": "hello world", "target": "tiny model"},),
        chunk_size_tokens=64, min_tokens=1, prompt_target_template=template).records
    path = Path(constraints["processed_dataset_ref"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    constraints["processed_dataset_hash"] = digest
    evidence_path = Path(constraints["processing_evidence_ref"])
    evidence = json.loads(evidence_path.read_text())
    evidence["processed_content_hash"] = digest
    evidence["wrapper"]["template"] = template
    evidence_path.write_text(json.dumps(evidence))
    constraints["processing_evidence_hash"] = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    assert service.launch(proposal).status == "running"
    result = _wait(service, proposal.run_id, {"completed", "failed"})
    assert result.status == "completed", result.to_dict()
    summary = json.loads((tmp_path / "runs" / proposal.run_id / "tokenization_summary.json").read_text())
    assert summary["prompt_masked_samples"] == 1 and summary["truncated_samples"] == 0
