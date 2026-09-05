#!/usr/bin/env python3
"""Second paid real-model promotion wave with task-aware frozen-pack execution.

The prior 7B attempt proved two concrete blockers without changing the frozen
semantic pack: Qwen2.5-7B-Instruct omitted the final period on the exact-match
probe and the positive-proof adapter incorrectly serialized continuation tasks
as user-chat turns. This adapter keeps the frozen pack/hash/decoding untouched,
uses causal serialization only for ``continuation_prompts``, and upgrades the
governed immutable Apache-2.0 candidate set.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Sequence

import run_positive_promotion_proof as proof
from hephaestus.generation.models import GeneratedText, GenerationTask

PRIOR_REJECTION_EVIDENCE = (
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-33957215257/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-33957215257/cycles/cycle-02/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-33959818398/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-33959818398/cycles/cycle-02/cycle_summary.json",
)

proof.CANDIDATES = (
    {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "license": "apache-2.0",
        "judge_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "judge_revision": "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_REJECTION_EVIDENCE),
    },
    {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "license": "apache-2.0",
        "judge_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "judge_revision": "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_REJECTION_EVIDENCE),
    },
)


class TaskAwarePinnedBackend(proof.PinnedChatTemplateBackend):
    """Preserve frozen task text while honoring generation-vs-continuation mode.

    No expected answers, punctuation, stop strings, logit constraints, or frozen
    task text are injected here. The only distinction comes from the already
    frozen task_kind field: continuation prompts are causal prefixes; all other
    tasks use the candidate tokenizer's native user-chat template.
    """

    backend_id = "pinned_task_aware_transformers"

    def generate_batch(
        self,
        tasks: Sequence[GenerationTask],
        *,
        run_id: str,
        loading_instructions: dict[str, object],
        decoding_config: dict[str, object],
    ) -> list[GeneratedText]:
        del run_id, loading_instructions
        self._load()
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        temperature = float(decoding_config.get("temperature", 0.0))
        top_p = float(decoding_config.get("top_p", 1.0))
        max_new_tokens = int(decoding_config.get("max_new_tokens", 0))
        if max_new_tokens <= 0:
            raise RuntimeError("frozen max_new_tokens must be positive")
        do_sample = temperature > 0.0

        outputs: list[GeneratedText] = []
        for task in tasks:
            random.seed(task.seed)
            torch.manual_seed(task.seed)
            torch.cuda.manual_seed_all(task.seed)

            if task.task_kind == "continuation_prompts":
                rendered = task.prompt
                serialization = "causal_prefix:frozen_continuation_prompt"
            else:
                chat_kwargs: dict[str, object] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                # Qwen3-8B defaults to reasoning mode. The frozen decoding pack is
                # greedy and the benchmark scores the answer surface, so use the
                # model's documented non-thinking chat mode without changing the
                # user prompt itself.
                if self.model_id == "Qwen/Qwen3-8B":
                    chat_kwargs["enable_thinking"] = False
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": task.prompt}],
                    **chat_kwargs,
                )
                serialization = "tokenizer_chat_template:user_only"
                if self.model_id == "Qwen/Qwen3-8B":
                    serialization += ":thinking_disabled"

            encoded = tokenizer(rendered, return_tensors="pt").to("cuda")
            kwargs: dict[str, object] = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            with torch.inference_mode():
                generated = model.generate(**encoded, **kwargs)
            input_width = int(encoded["input_ids"].shape[1])
            continuation = generated[0, input_width:]
            text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            if not text:
                text = "[empty generation]"
            outputs.append(
                GeneratedText(
                    output=text,
                    finish_reason="completed",
                    prompt_tokens=int(encoded["attention_mask"][0].sum().item()),
                    generated_tokens=int(continuation.shape[0]),
                    metadata={
                        "model_id": self.model_id,
                        "revision": self.revision,
                        "snapshot_manifest_hash": self.manifest_hash,
                        "prompt_serialization": serialization,
                        "seed": task.seed,
                        "task_kind": task.task_kind,
                        "frozen_prompt_unmodified": True,
                    },
                )
            )
        return outputs


def materialize_model_minimal(
    *,
    proof_root: Path,
    model_id: str,
    revision: str,
    expected_license: str,
) -> dict[str, object]:
    """Download only the Transformers safetensors/tokenizer runtime surface."""
    from huggingface_hub import HfApi, snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"model revision is not immutable: {model_id}@{revision}")
    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    observed_revision = str(info.sha or "")
    if observed_revision != revision:
        raise RuntimeError(f"Hub revision drift for {model_id}: {observed_revision} != {revision}")
    observed_license = str(getattr(getattr(info, "card_data", None), "license", "") or "").lower()
    if observed_license != expected_license.lower():
        raise RuntimeError(
            f"model license drift for {model_id}@{revision}: {observed_license!r} != {expected_license!r}"
        )

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=proof_root / "hf_cache",
            local_files_only=False,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    ).resolve()
    executable = [
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".sh", ".exe", ".dll", ".so", ".dylib"}
    ]
    if executable:
        raise RuntimeError(f"remote executable/code files are not admitted: {executable}")

    components: dict[str, str] = {}
    byte_size = 0
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        components[relative] = proof.sha_file(path)
        byte_size += path.stat().st_size
    if not components or not any(name.endswith(".safetensors") for name in components):
        raise RuntimeError(f"model snapshot has no admitted safetensors weights: {model_id}@{revision}")

    manifest_payload: dict[str, object] = {
        "manifest_version": "external-model-snapshot.v1",
        "acquisition_profile": "transformers_safetensors_minimal_v1",
        "model_id": model_id,
        "provider": "huggingface",
        "requested_revision": revision,
        "resolved_revision": observed_revision,
        "license": observed_license,
        "trust_remote_code": False,
        "snapshot_path": str(snapshot),
        "component_count": len(components),
        "byte_size": byte_size,
        "components": components,
    }
    manifest_payload["manifest_hash"] = proof.canonical_hash(components)
    manifest_path = proof_root / "model_manifests" / proof.slug(model_id) / revision / "snapshot_manifest.json"
    proof.atomic_json(manifest_path, manifest_payload)
    manifest_payload["manifest_ref"] = str(manifest_path)
    return manifest_payload


proof.PinnedChatTemplateBackend = TaskAwarePinnedBackend
proof.materialize_model = materialize_model_minimal

if __name__ == "__main__":
    raise SystemExit(proof.main())
