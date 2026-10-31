#!/usr/bin/env python3
"""Precommitted CPU format ablation on ORIGINAL DEVELOPMENT tasks, no evaluation leak.

No model training or GPU, no changes to frozen benchmark prompts. Only aggregate
development metrics and SHA commitments are written; no raw outputs are published.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from hephaestus.data.generated_factory import generate_original_splits, generator_code_sha256
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.storage.verified import canonical_bytes
from hephaestus.llm.cpu_causal_research import MODEL_ID, MODEL_REVISION

PROTOCOL_SCHEMA = "cpu-format-ablation-protocol.v1"
EXPECTED_ARMS = (("raw16", "original_raw", 16), ("raw64", "original_raw", 64),
                 ("chat16", "pinned_instruct_chat", 16), ("chat64", "pinned_instruct_chat", 64))


def prompt_text(tokenizer, task, mode):
    if mode == "original_raw":
        return task + "\nAnswer: "
    if mode == "pinned_instruct_chat":
        if not tokenizer.chat_template:
            raise ValueError("pinned instruct tokenizer has no chat template; no unsafe fallback")
        return tokenizer.apply_chat_template([
            {"role": "system", "content": "Return one JSON object containing the answer, without code fences or explanation."},
            {"role": "user", "content": task},
        ], tokenize=False, add_generation_prompt=True)
    raise ValueError("unapproved prompt protocol")


def exact_result(output, target):
    try:
        parsed, expected = json.loads(output), json.loads(target)
    except (ValueError, TypeError):
        return False, False
    valid = isinstance(parsed, dict)
    correct = valid and parsed == expected and all(type(parsed[k]) is type(v) for k, v in expected.items())
    return valid, correct


def check_protocol(path):
    payload = json.loads(Path(path).read_bytes())
    if (payload.get("schema") != PROTOCOL_SCHEMA or payload.get("claim_scope") != "internal_development_diagnosis_only" or
            payload.get("model_id") != MODEL_ID or payload.get("model_revision") != MODEL_REVISION or
            payload.get("dataset_generator_seed") != 20261028 or payload.get("generation_seed") != 20261028 or
            payload.get("max_dev_examples") != 24 or payload.get("GPU_execution_allowed") is not False or
            payload.get("model_promotion_allowed") is not False):
        raise PermissionError("invalid or unscoped CPU development study")
    arms = payload.get("arms")
    if not isinstance(arms, list) or [(r.get("name"), r.get("prompt_mode"), r.get("max_new_tokens")) for r in arms] != list(EXPECTED_ARMS):
        raise PermissionError("unregistered change to controlled development arms")
    return payload


def run(protocol_path, output, cache_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    protocol = check_protocol(protocol_path)
    if output.exists():
        raise FileExistsError("refuse to overwrite original experiment results")
    splits, generation_report = generate_original_splits(GeneratedDataSpec(protocol["dataset_generator_seed"]))
    dev = {"schema": "original-task-examples.v1", "split": "development", "examples": splits["development"]}
    dev_hash = "sha256:" + sha256(canonical_bytes(dev)).hexdigest()
    if dev_hash != protocol["original_development_artifact_sha256"]:
        raise PermissionError("development split differs from original historical artifact")
    examples = splits["development"]
    if len(examples) != 24:
        raise ValueError("unexpected original development case count")
    torch.manual_seed(protocol["generation_seed"])
    torch.set_num_threads(2)
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir),
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "model.safetensors", "generation_config.json"]))
    if snapshot.name != MODEL_REVISION:
        raise PermissionError("model revision not pinned")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                                                 trust_remote_code=False, dtype=torch.float32).to("cpu")
    model.eval()
    weights_hash = "sha256:" + sha256((snapshot / "model.safetensors").read_bytes()).hexdigest()
    tokenizer_hash = "sha256:" + sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()
    measurements = {}
    for arm in protocol["arms"]:
        counts = Counter()
        cases = []
        for row in examples:
            prompt = prompt_text(tokenizer, row["prompt"], arm["prompt_mode"])
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if tokens.input_ids.shape[1] + arm["max_new_tokens"] > model.config.max_position_embeddings:
                raise ValueError("ablation cannot truncate frozen development prompt")
            with torch.inference_mode():
                generated = model.generate(**tokens, max_new_tokens=arm["max_new_tokens"],
                    do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(generated[0, tokens.input_ids.shape[1]:], skip_special_tokens=True).strip()
            valid, correct = exact_result(output_text, row["target"])
            format_kind = ("python_fence" if output_text.startswith("```python") else
                           "other_fence" if output_text.startswith("```") else
                           "json_opening" if output_text.startswith("{") else "other")
            counts[format_kind] += 1
            counts["json_valid"] += int(valid)
            counts["exact_correct"] += int(correct)
            cases.append({"sample_id": row["sample_id"],
                          "prompt_sha256": "sha256:" + sha256(prompt.encode()).hexdigest(),
                          "output_sha256": "sha256:" + sha256(output_text.encode()).hexdigest(),
                          "output_tokens": int(generated.shape[1] - tokens.input_ids.shape[1]),
                          "json_valid": valid, "exact_correct": correct})
        measurements[arm["name"]] = {"counts": dict(counts), "cases": cases}
    result = {"schema": "cpu-real-development-format-ablation.v1",
              "protocol_sha256": "sha256:" + sha256(canonical_bytes(protocol)).hexdigest(),
              "original_development_ref": dev_hash,
              "model_revision": MODEL_REVISION, "model_weights_sha256": weights_hash,
              "tokenizer_sha256": tokenizer_hash,
              "generator_code_sha256": generator_code_sha256(),
              "arms": measurements, "claim_scope": "development_diagnostic_only",
              "evaluation_holdout_touched": False, "model_training_executed": False,
              "GPU_executed": False, "certified": False, "promotion_authorized": False}
    output.write_bytes(canonical_bytes(result))
    print(json.dumps({"arms": {k: v["counts"] for k, v in measurements.items()},
                      "output": str(output)}, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, default=Path("configs/research/cpu_dev_format_ablation_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/hephaestus-model-cache"))
    args = p.parse_args()
    run(args.protocol, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
