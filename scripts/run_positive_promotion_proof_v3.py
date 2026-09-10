#!/usr/bin/env python3
"""Third real-model promotion wave selected from V2 rejection evidence.

V2 showed deterministic, perfectly repeatable rejection rather than noisy
semantic uncertainty. Qwen2.5-7B failed instruction_exact and continuation
termination; Mistral-7B failed instruction_exact, brief length, and strict JSON.
This wave therefore changes only model identity to stronger permissively
licensed instruction-following candidates while retaining the frozen eval pack,
decoding settings, baseline, Judge, certification, and promotion policy.

Execution topology is deliberately separate from those scientific variables.
For V3, the 14B candidate is sharded in FP16 across exactly two 24GB CUDA GPUs
so the proof can run in a datacenter where a single >=48GB GPU is unavailable.
"""
from __future__ import annotations

import run_positive_promotion_proof_v2 as wave2

proof = wave2.proof

PRIOR_V2_REJECTION_EVIDENCE = (
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-01/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-01/certification_decision.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-02/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-02/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-02/certification_decision.json",
)

# Keep the existing independent reviewer fixed so candidate-model identity is
# the single scientific variable changed by this wave.
JUDGE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_REVISION = "582efe62d7cfafd242bffca71ecbde1bcecc1bcc"

V3_SHARD_GPU_COUNT = 2
V3_MAX_MEMORY_GIB_PER_GPU = 22


class ShardedV3ChatTemplateBackend(proof.PinnedChatTemplateBackend):
    """Load V3 candidate weights across two CUDA devices without quantization."""

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real candidate generation")
        if torch.cuda.device_count() < V3_SHARD_GPU_COUNT:
            raise RuntimeError(
                f"V3 sharded FP16 generation requires {V3_SHARD_GPU_COUNT} CUDA GPUs; "
                f"found {torch.cuda.device_count()}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.snapshot_path), local_files_only=True, trust_remote_code=False
        )
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(f"candidate tokenizer has no chat template: {self.model_id}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        max_memory = {
            index: f"{V3_MAX_MEMORY_GIB_PER_GPU}GiB"
            for index in range(V3_SHARD_GPU_COUNT)
        }
        model = AutoModelForCausalLM.from_pretrained(
            str(self.snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            device_map="balanced",
            max_memory=max_memory,
        )

        device_map = getattr(model, "hf_device_map", None)
        if not isinstance(device_map, dict) or not device_map:
            raise RuntimeError("V3 sharded model load produced no hf_device_map")

        cuda_indices: set[int] = set()
        forbidden_devices: set[str] = set()
        for placement in device_map.values():
            if isinstance(placement, int):
                cuda_indices.add(placement)
                continue
            value = str(placement)
            if value.startswith("cuda:"):
                cuda_indices.add(int(value.split(":", 1)[1]))
            elif value in {"cpu", "disk"}:
                forbidden_devices.add(value)

        if forbidden_devices:
            raise RuntimeError(
                f"V3 sharded FP16 load attempted forbidden offload: {sorted(forbidden_devices)}"
            )
        if not set(range(V3_SHARD_GPU_COUNT)).issubset(cuda_indices):
            raise RuntimeError(
                "V3 sharded FP16 load did not span both requested CUDA GPUs: "
                f"{sorted(cuda_indices)}"
            )

        embedding_device = model.get_input_embeddings().weight.device
        if embedding_device.type != "cuda" or embedding_device.index != 0:
            raise RuntimeError(
                "V3 sharded model input embeddings must remain on cuda:0 because the "
                f"frozen generation backend places inputs there; got {embedding_device}"
            )

        model.eval()
        self._tokenizer = tokenizer
        self._model = model


proof.PinnedChatTemplateBackend = ShardedV3ChatTemplateBackend

proof.CANDIDATES = (
    {
        "model_id": "microsoft/phi-4",
        "revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
        "license": "mit",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V2_REJECTION_EVIDENCE),
        "selection_reason": "14B permissive model explicitly aligned for precise instruction adherence; targets V2 exact-instruction/format failures",
    },
    {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "license": "apache-2.0",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V2_REJECTION_EVIDENCE),
        "selection_reason": "same Qwen2.5 instruction family as V2 but doubled scale, preserving architecture while testing whether strict adherence failures are capability-limited",
    },
)

if __name__ == "__main__":
    raise SystemExit(proof.main())
