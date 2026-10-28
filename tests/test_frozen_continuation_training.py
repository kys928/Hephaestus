"""Exact token-loader checks; synthetic inputs do not clear a real dataset."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hephaestus.schemas.continuation_subset import ContinuationSubsetContract
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.training.continuation_tokens import FORMAT, KIND, encode_frozen_records, validate_token_records
from hephaestus.training.hf_lifecycle import TransformersTrainingLifecycleService
from hephaestus.training.hf_worker import _load_jsonl, _read_prepared_data, _tokenize_records


def digest(raw):
    return "sha256:" + sha256(raw).hexdigest()


def fixture(root):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"<pad>": 0, "<eos>": 1, "<unk>": 2, "alpha": 3, "beta": 4,
         "gamma": 5, "delta": 6, "epsilon": 7}, unk_token="<unk>"))
    raw = tokenizer.to_str().encode()
    original = root / "original-tokenizer.json"
    original.write_bytes(raw)
    content_sha = digest(b"synthetic reviewed paragraph")
    ids = [3 + i % 5 for i in range(224)]
    records = [{
        "record_kind": KIND, "source_row_index": 2,
        "source_content_sha256": content_sha, "window_start_token": start,
        "token_count": 160, "input_ids": ids[start:start + 160],
        "labels": [-100] * 128 + ids[start + 128:start + 160],
    } for start in (0, 64)]
    contract = ContinuationSubsetContract(
        dataset_id="fixture/continuation", revision="a" * 40,
        source_file="train.parquet", source_file_sha256=digest(b"source"),
        source_bytes=6, source_rows=3, tokenizer_ref=digest(b"artifact"),
        tokenizer_file_sha256=sha256(raw).hexdigest(), public_eval_pack_sha256=digest(b"public pack"),
        selection_commitment_sha256=digest(f"2\t{content_sha.removeprefix('sha256:')}\n".encode()),
        accepted_rows=1, eligible_tokens=224, unknown_tokens=0, windows=2,
    )
    evidence = {
        "data_format": FORMAT, "continuation_contract": contract.to_dict(),
        "original_tokenizer_path": str(original),
        "tokenizer_compatibility": {"status": "checked", "compatible": True, "tokenizer_ref": contract.tokenizer_ref,
            "tokenizer_file_sha256": contract.tokenizer_file_sha256},
        "wrapper": {"kind": "no_text_wrapper", "template": "token_ids:128_prefix+32_target"},
        "prompt_target_boundary": {"status": "explicit", "prefix_tokens": 128, "target_tokens": 32},
    }
    config = {"data_format": FORMAT, "context_length": 160, "max_total_tokens": 320,
        "ignored_label_token": -100, "eos_handling": "no_insertion", "truncation": "forbidden",
        "vocabulary_size": 8, "maximum_rows": 2}
    return tokenizer, records, evidence, config


class BackendOnlyTokenizer:
    def __init__(self, backend):
        self.backend_tokenizer = backend

    def encode(self, *args, **kwargs):
        raise AssertionError("frozen tokens must never be re-encoded")


def test_explicit_token_loader_preserves_all_ids_mask_and_length(tmp_path):
    tok, records, evidence, config = fixture(tmp_path)
    encoded, summary = _tokenize_records(BackendOnlyTokenizer(tok), records, config, evidence)
    assert encoded == [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in records]
    assert summary == {"source_samples": 2, "encoded_samples": 2, "dropped_samples": 0,
                       "truncated_samples": 0, "prompt_masked_samples": 2, "total_tokens": 320}
    assert all(len(row["input_ids"]) == 160 for row in encoded)


@pytest.mark.parametrize("mutate", [
    lambda rs: rs.pop(),
    lambda rs: rs[0]["input_ids"].__setitem__(0, True),
    lambda rs: rs[0]["input_ids"].__setitem__(0, 8),
    lambda rs: rs[0]["labels"].__setitem__(0, 3),
    lambda rs: rs[1].__setitem__("window_start_token", 128),
    lambda rs: rs[1].__setitem__("source_row_index", 1),
    lambda rs: rs[1].__setitem__("source_content_sha256", "sha256:" + "0" * 64),
    lambda rs: rs[1]["input_ids"].__setitem__(0, 3),
    lambda rs: rs[0].__setitem__("text", "retokenize me"),
    lambda rs: rs.reverse(),
])
def test_token_loader_rejects_tampered_identity_offsets_masks_or_ids(tmp_path, mutate):
    _, records, evidence, _ = fixture(tmp_path)
    mutate(records)
    with pytest.raises(ValueError):
        validate_token_records(records, evidence, vocabulary_size=8)


@pytest.mark.parametrize("updates", [
    {"context_length": 159}, {"max_total_tokens": 319}, {"eos_handling": "append_if_missing"},
    {"truncation": "right_to_context_length"}, {"ignored_label_token": 0},
    {"data_format": "trainable-data.v1"},
])
def test_token_loader_refuses_recipe_changes(tmp_path, updates):
    tok, records, evidence, config = fixture(tmp_path)
    config.update(updates)
    with pytest.raises(ValueError):
        _tokenize_records(BackendOnlyTokenizer(tok), records, config, evidence)


def test_original_bytes_and_loaded_backend_both_bound(tmp_path):
    tok, records, evidence, config = fixture(tmp_path)
    changed = tok.__class__.from_str(tok.to_str())
    changed.add_tokens(["new-vocabulary-token"])
    with pytest.raises(ValueError, match="loaded tokenizer"):
        encode_frozen_records(BackendOnlyTokenizer(changed), records, config, evidence)
    original = Path(evidence["original_tokenizer_path"])
    original.write_bytes(original.read_bytes() + b" ")
    with pytest.raises(ValueError, match="original frozen tokenizer"):
        encode_frozen_records(BackendOnlyTokenizer(tok), records, config, evidence)


def write_inputs(root, records, evidence, schema=FORMAT):
    source = root / "trainable.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    contract = root / "trainable_data_contract.json"
    contract.write_text(json.dumps(TrainableDataContract("c", "r", "m", str(source), schema, 160).to_dict()))
    evidence = deepcopy(evidence)
    evidence.update(processed_dataset_ref=str(source), processed_content_hash=digest(source.read_bytes()))
    ep = root / "processing_evidence.json"
    ep.write_text(json.dumps(evidence))
    compatibility = {"processed_dataset_ref": str(source), "processed_dataset_hash": digest(source.read_bytes()),
        "trainable_data_contract_ref": str(contract), "trainable_data_contract_hash": digest(contract.read_bytes()),
        "tokenizer_id": "/a/distinct/hf/loader/location"}
    constraints = {"processing_evidence_ref": str(ep), "processing_evidence_hash": digest(ep.read_bytes()),
                   "vocabulary_size": 8}
    return compatibility, constraints


def test_lifecycle_admits_explicit_schema_and_separate_tokenizer_artifact_identity(tmp_path):
    _, records, evidence, _ = fixture(tmp_path)
    compatibility, constraints = write_inputs(tmp_path, records, evidence)
    issues = []
    summary = TransformersTrainingLifecycleService(tmp_path / "runs")._validate_data(compatibility, constraints, issues)
    assert not issues
    assert summary["data_format"] == FORMAT and summary["declared_tokens"] == 320


def test_lifecycle_refuses_implicit_schema_fallback(tmp_path):
    _, records, evidence, _ = fixture(tmp_path)
    compatibility, constraints = write_inputs(tmp_path, records, evidence, "trainable-data.v1")
    issues = []
    TransformersTrainingLifecycleService(tmp_path / "runs")._validate_data(compatibility, constraints, issues)
    assert "processed_dataset_invalid" in {i.code for i in issues}
    with pytest.raises(ValueError):
        _load_jsonl(Path(compatibility["processed_dataset_ref"]), 2)


@pytest.mark.parametrize("item", ["processed_dataset_ref", "trainable_data_contract_ref", "processing_evidence_ref"])
def test_worker_rehashes_exact_inputs_at_consumption_including_resume(tmp_path, item):
    _, records, evidence, config = fixture(tmp_path)
    compatibility, constraints = write_inputs(tmp_path, records, evidence)
    config.update(constraints)
    read_evidence, read_records = _read_prepared_data(config, compatibility)
    assert read_records == records and read_evidence["data_format"] == FORMAT
    path = Path({**compatibility, **constraints}[item])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA changed"):
        _read_prepared_data(config, compatibility)


@pytest.mark.real_cpu
def test_real_cpu_worker_trains_exact_token_ids_and_exports_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hephaestus.backends.hf_causal_lm import directory_content_identity
    from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
    from test_real_hf_training import _proposal, _wait

    backend, records, evidence, _ = fixture(tmp_path)
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    # HF special-token registration is part of the *frozen* tokenizer here.
    original = Path(evidence["original_tokenizer_path"])
    original.write_text(tokenizer.backend_tokenizer.to_str())
    tokenizer_sha = sha256(original.read_bytes()).hexdigest()
    evidence["continuation_contract"]["tokenizer_file_sha256"] = tokenizer_sha
    evidence["tokenizer_compatibility"]["tokenizer_file_sha256"] = tokenizer_sha
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    config = transformers.GPT2Config(vocab_size=8, n_positions=160, n_ctx=160,
        n_embd=16, n_layer=1, n_head=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(27)
    model = transformers.AutoModelForCausalLM.from_config(config)
    # Causal shift supervises the 32 target IDs, never any of the 128 prefix IDs.
    inputs = torch.tensor([records[0]["input_ids"]])
    labels = torch.tensor([records[0]["labels"]])
    output = model(input_ids=inputs, labels=labels)
    shifted = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 8),
        labels[:, 1:].reshape(-1), ignore_index=-100)
    assert int((labels[:, 1:] != -100).sum()) == 32
    torch.testing.assert_close(output.loss, shifted)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    parameters = sum(p.numel() for p in model.parameters())
    proposal = _proposal(tmp_path, "exact-token-cpu", model_dir=model_dir,
        model_revision=directory_content_identity(model_dir), parameter_count=parameters,
        vocabulary_size=8, context_length=160, max_total_tokens=320,
        max_steps=2, checkpoint_every_steps=2)
    compatibility, constraints = write_inputs(tmp_path, records, evidence)
    proposal.training_constraints.update({k: v for k, v in compatibility.items() if k != "tokenizer_id"})
    proposal.training_constraints.update(constraints)
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    started = service.launch(proposal)
    assert started.status == "running", started.to_dict()
    finished = _wait(service, proposal.run_id, {"completed", "failed"}, timeout=60)
    assert finished.status == "completed", finished.to_dict()
    checkpoint = Path(finished.checkpoint_refs[-1])
    assert validate_checkpoint_manifest(checkpoint)[0]
    trained = transformers.AutoModelForCausalLM.from_pretrained(checkpoint / "model", local_files_only=True)
    assert any(not torch.equal(before[name], p) for name, p in trained.named_parameters())
    summary = json.loads((tmp_path / "runs" / proposal.run_id / "tokenization_summary.json").read_text())
    assert summary["total_tokens"] == 320
    assert summary["truncated_samples"] == 0
