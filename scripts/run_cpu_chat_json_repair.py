#!/usr/bin/env python3
"""Real CPU chat-aligned supervised JSON repair on frozen ORIGINAL DEVELOPMENT tasks.

This is a new prospective development experiment. Not the original 45-stage
scientific campaign, not independently certified, and not GPU authorization.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import time

from hephaestus.data.generated_factory import generate_original_splits, generator_code_sha256
from hephaestus.data.generated_validation import validate_original_example
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.storage.verified import canonical_bytes
from hephaestus.llm.cpu_causal_research import MODEL_ID, MODEL_REVISION

SCHEMA = "cpu-chat-json-repair-experiment.v1"
PRECOMMITTED = {
    "schema": SCHEMA, "experiment_id": "json-format-repair-development-20261028",
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "generator_seed": 20261028, "train_seed": 1117, "epoch_count": 1,
    "training_examples": 96, "development_examples": 24, "optimizer_steps": 96,
    "max_cpu_seconds": 1800, "learning_rate": 0.0005, "adapter_rank": 8,
    "adapter_alpha": 16, "max_new_tokens": 64,
    "train_target_suffix": "pinned_eos_token",
    "objective": "single_json_object_with_correct_answer",
    "baseline_protocol": "pinned_instruct_chat",
    "candidate_protocol": "pinned_instruct_chat",
    "selection_metric": "exact_correct_json_development",
    "claim_scope": "internal_development_only", "evaluation_holdout_allowed": False,
    "model_promotion_authorized": False, "paid_gpu_authorized": False,
}
DEV_COMMITMENT = "sha256:4c9d54866719545f9a54466cd6da7b9aed4b79f935160d61e42e0639675c3f1f"


def validate_protocol(path):
    parsed = json.loads(Path(path).read_bytes())
    # Explicit *new* independent CPU initialization seeds: identical frozen
    # source, development items, optimizer, model and budget.
    seed = parsed.get("train_seed")
    if type(seed) is not int or seed not in (1117, 2227, 3337):
        raise PermissionError("CPU replication training seed not precommitted")
    expected = dict(PRECOMMITTED)
    if seed != 1117:
        expected["train_seed"] = seed
        expected["experiment_id"] = "json-format-repair-development-20261028-seed-" + str(seed)
    if parsed != expected or any(type(parsed[k]) is not type(expected[k]) for k in expected):
        raise PermissionError("CPU repair protocol cannot be widened or changed without a new reviewed experiment")
    return parsed


def chat_prompt(tokenizer, task):
    if not tokenizer.chat_template:
        raise ValueError("pinned instruction tokenizer has no chat template")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Return only one valid JSON object with exactly the requested answer, with no markdown, code, or prose."},
        {"role": "user", "content": task}
    ], tokenize=False, add_generation_prompt=True)


def check_json(output, expected):
    try:
        actual, target = json.loads(output), json.loads(expected)
    except (TypeError, ValueError):
        return False, False
    valid = isinstance(actual, dict)
    return valid, valid and actual == target and all(type(actual[k]) is type(v) for k, v in target.items())


def score(model, tokenizer, dev, *, max_new_tokens, start, maximum_seconds):
    import torch
    model.eval()
    totals = Counter()
    family = {}
    examples = []
    for row in dev:
        if time.monotonic() - start > maximum_seconds:
            raise TimeoutError("CPU research wall-clock budget exceeded")
        prefix = chat_prompt(tokenizer, row["prompt"])
        tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        if tokens.input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("frozen development prompt would require truncation")
        with torch.inference_mode():
            generated = model.generate(**tokens, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(generated[0, tokens.input_ids.shape[1]:], skip_special_tokens=True).strip()
        valid, correct = check_json(output, row["target"])
        totals["valid_json"] += int(valid)
        totals["exact_correct"] += int(correct)
        kind = "python_fence" if output.startswith("```python") else "other_fence" if output.startswith("```") else "json_opening" if output.startswith("{") else "other"
        totals[kind] += 1
        fam = family.setdefault(row["family"], {"count": 0, "valid_json": 0, "exact_correct": 0})
        fam["count"] += 1
        fam["valid_json"] += int(valid)
        fam["exact_correct"] += int(correct)
        examples.append({"sample_id": row["sample_id"], "family": row["family"],
                         "output_sha256": "sha256:" + sha256(output.encode()).hexdigest(),
                         "valid_json": valid, "exact_correct": correct})
    return {"n": len(dev), "counts": dict(totals), "by_family": family, "examples": examples}


def run(protocol_path, output_dir, cache_dir):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    protocol = validate_protocol(protocol_path)
    if output_dir.exists():
        raise FileExistsError("research output directory must be new, never overwritten")
    output_dir.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(protocol["train_seed"])
    random.seed(protocol["train_seed"])
    splits, generation_report = generate_original_splits(GeneratedDataSpec(protocol["generator_seed"]))
    train, dev = splits["train"], splits["development"]
    if len(train) != protocol["training_examples"] or len(dev) != protocol["development_examples"]:
        raise ValueError("original generated dataset cardinality changed")
    dev_commit = "sha256:" + sha256(canonical_bytes({
        "schema": "original-task-examples.v1", "split": "development", "examples": dev})).hexdigest()
    if dev_commit != DEV_COMMITMENT:
        raise PermissionError("frozen original development examples changed")
    train_keys, dev_keys = {validate_original_example(r) for r in train}, {validate_original_example(r) for r in dev}
    if train_keys & dev_keys or len(train_keys) != len(train) or len(dev_keys) != len(dev):
        raise ValueError("training/development canonical overlap or duplicate")
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir),
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "model.safetensors", "generation_config.json"]))
    if snapshot.name != MODEL_REVISION:
        raise PermissionError("model revision changed")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise ValueError("pinned EOS token required")
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                                                 trust_remote_code=False, dtype=torch.float32).to("cpu")
    baseline = score(model, tokenizer, dev, max_new_tokens=protocol["max_new_tokens"],
                     start=started, maximum_seconds=protocol["max_cpu_seconds"])
    model = get_peft_model(model, LoraConfig(r=protocol["adapter_rank"],
        lora_alpha=protocol["adapter_alpha"], lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    first = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=protocol["learning_rate"])
    train_hash = "sha256:" + sha256(b"".join(canonical_bytes(r)+b"\n" for r in train)).hexdigest()
    # Each of 96 source examples is trained exactly once. The frozen task family
    # order is cyclic to avoid 24-example single-family blocks.
    by_family = {fam:[i for i, r in enumerate(train) if r["family"] == fam]
                 for fam in ("arithmetic", "json", "symbolic", "reasoning")}
    order = [index for i in range(24) for fam in by_family for index in [by_family[fam][i]]]
    if len(order) != 96 or set(order) != set(range(96)):
        raise ValueError("task balancing failed")
    losses = []
    for step, index in enumerate(order):
        if time.monotonic()-started > protocol["max_cpu_seconds"]:
            raise TimeoutError("CPU training budget exceeded; incomplete study, no promotion")
        row = train[index]
        prefix = chat_prompt(tokenizer, row["prompt"])
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        target_ids = tokenizer(row["target"]+tokenizer.eos_token, add_special_tokens=False).input_ids
        if not prefix_ids or not target_ids:
            raise ValueError("empty supervised prompt/target")
        ids = prefix_ids + target_ids
        if len(ids) > model.config.max_position_embeddings:
            raise ValueError("training example requires truncation")
        inputs = torch.tensor([ids], dtype=torch.long)
        labels = torch.tensor([[-100]*len(prefix_ids)+target_ids], dtype=torch.long)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels).loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite supervised JSON loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    changed = any(not torch.equal(a,b.detach()) for a,b in zip(first,trainable))
    if not changed or len(losses) != protocol["optimizer_steps"]:
        raise ValueError("missing actual model adaptation")
    candidate = score(model, tokenizer, dev, max_new_tokens=protocol["max_new_tokens"],
                      start=started, maximum_seconds=protocol["max_cpu_seconds"])
    if [r["sample_id"] for r in baseline["examples"]] != [r["sample_id"] for r in candidate["examples"]]:
        raise ValueError("development sample identities changed")
    adapter = output_dir/"adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    model_digest = "sha256:" + sha256((adapter/"adapter_model.safetensors").read_bytes()).hexdigest()
    report = {"schema": "cpu-chat-json-repair-result.v1",
      "claim_scope": "original_generated_development_only",
      "experiment_id": protocol["experiment_id"],
      "protocol_sha256": "sha256:"+sha256(canonical_bytes(protocol)).hexdigest(),
      "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
      "model_weights_sha256":"sha256:"+sha256((snapshot/"model.safetensors").read_bytes()).hexdigest(),
      "tokenizer_sha256":"sha256:"+sha256((snapshot/"tokenizer.json").read_bytes()).hexdigest(),
      "generator_sha256": generator_code_sha256(), "training_source_sha256":train_hash,
      "frozen_development_ref":dev_commit, "optimizer_steps":len(losses),
      "epoch_completed": True, "training_samples_seen":len(order),
      "training_seed":protocol["train_seed"], "weights_changed":changed,
      "losses":losses, "adapter_sha256": model_digest,
      "baseline":baseline, "candidate":candidate,
      "development_exact_correct_change": candidate["counts"].get("exact_correct",0)-baseline["counts"].get("exact_correct",0),
      "development_valid_json_change": candidate["counts"].get("valid_json",0)-baseline["counts"].get("valid_json",0),
      "evaluation_holdout_used": False, "GPU_used": False, "model_promoted":False,
      "external_certification":False, "frozen_five_cuda_roles_executed":False}
    (output_dir/"report.json").write_bytes(canonical_bytes(report))
    print(json.dumps({"before":baseline["counts"],"after":candidate["counts"],
                      "development_exact_change":report["development_exact_correct_change"],
                      "development_json_change":report["development_valid_json_change"],
                      "updates":len(losses), "adapter_sha256":model_digest},sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,default=Path("configs/research/cpu_chat_json_repair_v1.json"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,default=Path("/tmp/hephaestus-model-cache"))
    args = p.parse_args()
    run(args.protocol,args.output,args.cache_dir)


if __name__ == "__main__":
    main()
