"""Governed original-task JSON training records for the real native CPU spine.

Admitted original records and their earlier exact raw-token materialization are
always independently revalidated. This new recipe uses the already-authorized
*same* original source bytes, with an explicitly versioned chat/EOS transform.
It never reads development or evaluation targets for gradient optimization.
"""
from hashlib import sha256
from pathlib import Path
from hephaestus.data.generated_factory import generate_original_splits
from hephaestus.data.generated_validation import validate_original_example
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.storage.verified import canonical_bytes, read_verified_json

RECIPE = "chat_json_r8_full_epoch_96_v1"
SYSTEM = "Return only one valid JSON object with exactly the requested answer, with no markdown, code, or prose."
STEP_FAMILIES = ("arithmetic", "json", "symbolic", "reasoning")


def chat_prompt(tokenizer, task):
    if not tokenizer.chat_template:
        raise PermissionError("pinned instruction chat template is required")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task}
    ], tokenize=False, add_generation_prompt=True)


def exact_chat_records(ctx, admitted_raw_tokens):
    if ctx.profile.training_recipe != RECIPE:
        raise PermissionError("new chat training records require the reviewed v2 profile")
    policy = ctx.admission.policy()
    if (policy.maximum_training_steps != 96 or
            ctx.campaign_config.dataset_seed != 20261028 or
            ctx.campaign_config.cpu_seed != 1117):
        raise PermissionError("native original chat research identity/step budget mismatch")
    original, _ = generate_original_splits(GeneratedDataSpec(ctx.campaign_config.dataset_seed))
    train = original["train"]
    if len(train) != 96 or len(admitted_raw_tokens) != 96:
        raise PermissionError("complete original train split is required")
    review = read_verified_json(ctx.artifacts, ctx.review_ref)
    source_refs = review["source_file_hashes"]
    if set(source_refs) != {"original-tasks.jsonl"}:
        raise PermissionError("unexpected governed source layout")
    raw_source = b"".join(canonical_bytes(r) + b"\n" for r in train)
    expected_source_ref = "sha256:" + sha256(raw_source).hexdigest()
    if source_refs["original-tasks.jsonl"] != expected_source_ref:
        raise PermissionError("admitted original task source differs from approved bytes")
    records = []
    tokenizer = ctx.scorer.tokenizer
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise PermissionError("pinned tokenizer must provide an EOS token")
    for i, (row, old) in enumerate(zip(train, admitted_raw_tokens, strict=True)):
        validate_original_example(row)
        raw_digest = "sha256:" + sha256(canonical_bytes(row)).hexdigest()
        if old.get("source_index") != i or old.get("sample_sha256") != raw_digest:
            raise PermissionError("chat recipe source position differs from governed token record")
        prefix = tokenizer(chat_prompt(tokenizer, row["prompt"]), add_special_tokens=False).input_ids
        target = tokenizer(row["target"] + tokenizer.eos_token, add_special_tokens=False).input_ids
        if not prefix or not target or any(type(x) is not int for x in prefix+target):
            raise ValueError("invalid JSON task chat tokens")
        records.append({"sample_id": row["sample_id"], "family": row["family"],
                        "source_index": i, "source_sample_sha256": raw_digest,
                        "input_ids": prefix+target,
                        "labels": [-100]*len(prefix)+target,
                        "prefix_tokens": len(prefix)})
    by_family = {family: [i for i, row in enumerate(records) if row["family"] == family]
                 for family in STEP_FAMILIES}
    if any(len(indices) != 24 for indices in by_family.values()):
        raise ValueError("original task families are not balanced")
    order = [by_family[family][i] for i in range(24) for family in STEP_FAMILIES]
    if len(order) != 96 or set(order) != set(range(96)):
        raise ValueError("not one complete original task epoch")
    return records, order, "sha256:" + sha256(b"".join(
        canonical_bytes(row) + b"\n" for row in records)).hexdigest()
