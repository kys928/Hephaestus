#!/usr/bin/env python3
"""GPU smoke for one immutable external model against the frozen continuation blocker."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.scoring.behavioral import evaluate_behavioral_sample

FROZEN_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"
SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")
SHARED_CACHE = SCIENTIFIC_ROOT / "model_cache" / "huggingface"
EXPECTED_DECODING = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 96, "seeds": [11, 29, 47]}


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = required("HEPHAESTUS_CANDIDATE_MODEL_ID")
    revision = required("HEPHAESTUS_CANDIDATE_REVISION")
    expected_license = required("HEPHAESTUS_CANDIDATE_LICENSE").lower()
    proof_run_id = required("HEPHAESTUS_PROOF_RUN_ID")
    attempt = required("HEPHAESTUS_ATTEMPT")
    repo_sha = required("HEPHAESTUS_REPO_SHA")

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"candidate revision is not immutable: {revision}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    observed_revision = str(info.sha or "")
    observed_license = str(getattr(getattr(info, "card_data", None), "license", "") or "").lower()
    if observed_revision != revision:
        raise RuntimeError(f"revision drift: {observed_revision} != {revision}")
    if observed_license != expected_license:
        raise RuntimeError(f"license drift: {observed_license!r} != {expected_license!r}")

    snapshot = Path(snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=SHARED_CACHE,
        local_files_only=False,
        allow_patterns=[
            "config.json", "generation_config.json",
            "model.safetensors", "model-*.safetensors", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
            "chat_template.jinja", "*.jinja",
        ],
    )).resolve()

    executable = [
        p.relative_to(snapshot).as_posix()
        for p in snapshot.rglob("*")
        if p.is_file() and p.suffix.lower() in {".py", ".sh", ".exe", ".dll", ".so", ".dylib"}
    ]
    if executable:
        raise RuntimeError(f"remote executable/code files are not admitted: {executable}")

    components: dict[str, str] = {}
    byte_size = 0
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            relative = path.relative_to(snapshot).as_posix()
            components[relative] = sha_file(path)
            byte_size += path.stat().st_size
    if not components or not any(name.endswith(".safetensors") for name in components):
        raise RuntimeError("candidate snapshot lacks admitted safetensors")
    manifest_hash = "sha256:" + hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    pack = load_eval_pack("semantic_behavior_v1", config_dir=Path("configs"))
    if pack["frozen"] is not True or pack["content_hash_verified"] is not True or pack["content_hash"] != FROZEN_HASH:
        raise RuntimeError("frozen evaluation pack identity/integrity failed")
    normalized = pack["eval_pack"]
    decoding = normalized["decoding_config"]
    if decoding != EXPECTED_DECODING:
        raise RuntimeError(f"frozen decoding drift: {decoding}")
    continuation_tasks = normalized["continuation_prompts"]
    if len(continuation_tasks) != 1 or continuation_tasks[0]["task_id"] != "observatory_continuation":
        raise RuntimeError("unexpected frozen continuation task set")
    task = continuation_tasks[0]
    prompt = task["prompt"]
    if prompt != "The observatory lost power just as the storm arrived.":
        raise RuntimeError("frozen continuation prompt drift")

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("candidate tokenizer has no local chat template after immutable acquisition")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16
    )
    model.to("cuda")
    model.eval()

    samples: list[dict[str, object]] = []
    for repeat in (1, 2, 3):
        for seed in decoding["seeds"]:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            encoded = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": prompt}],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                continue_final_message=True,
            )
            rendered = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
            if rendered.count(prompt) != 1:
                raise RuntimeError(f"assistant-prefill serialization mutated/duplicated frozen prompt: {rendered!r}")
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=int(decoding["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[0, input_width:]
            output = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            score = evaluate_behavioral_sample(task, output, seed)
            samples.append({
                "repeat": repeat,
                "seed": seed,
                "output": output,
                "output_sha256": "sha256:" + hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "prompt_tokens": int(encoded["attention_mask"][0].sum().item()),
                "generated_tokens": int(continuation.shape[0]),
                "finish_reason": "eos_or_stop" if int(continuation.shape[0]) < int(decoding["max_new_tokens"]) else "length",
                "deterministic_passed": score.deterministic_passed,
                "failed_hard_checks": score.failed_hard_checks,
                "hard_checks": [check.to_dict() for check in score.checks if check.hard],
                "serialization": "tokenizer_chat_template:assistant_prefill:continue_final_message",
                "frozen_prompt_unmodified": True,
            })

    def hard_gate(name: str) -> bool:
        return all(
            next(check for check in sample["hard_checks"] if check["name"] == name)["passed"]
            for sample in samples
        )

    result = {
        "result_version": "frozen-candidate-continuation-smoke.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proof_run_id": proof_run_id,
        "attempt": attempt,
        "repo_sha": repo_sha,
        "status": "completed",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model_id": model_id,
        "revision": revision,
        "license": observed_license,
        "snapshot_path": str(snapshot),
        "snapshot_manifest_hash": manifest_hash,
        "snapshot_byte_size": byte_size,
        "snapshot_component_hashes": components,
        "frozen_eval_pack_hash": FROZEN_HASH,
        "decoding_config": decoding,
        "task_id": task["task_id"],
        "prompt": prompt,
        "serialization": "tokenizer_chat_template:assistant_prefill:continue_final_message",
        "sample_count": len(samples),
        "all_hard_gates_passed": all(sample["deterministic_passed"] for sample in samples),
        "hard_gate_summary": {
            "continuation_not_echo": hard_gate("continuation_not_echo"),
            "continuation_repetition": hard_gate("continuation_repetition"),
            "continuation_termination": hard_gate("continuation_termination"),
        },
        "samples": samples,
    }
    out = SCIENTIFIC_ROOT / "executions" / proof_run_id / f"attempt-{attempt}" / "driver_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.partial")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({
        "status": result["status"], "gpu": result["gpu"], "model_id": model_id,
        "sample_count": result["sample_count"], "all_hard_gates_passed": result["all_hard_gates_passed"],
        "hard_gate_summary": result["hard_gate_summary"], "snapshot_manifest_hash": manifest_hash,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
