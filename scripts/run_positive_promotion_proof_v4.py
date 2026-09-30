#!/usr/bin/env python3
"""Fourth real-model promotion wave selected from exact V3 gate evidence.

V3 showed both 14B candidates were materially better than the random-init
baseline, perfectly repeatable, and low-variance, but were rejected by strict
hard deterministic formatting/termination gates. Phi-4 failed continuation
termination and strict JSON; Qwen2.5-14B failed exact punctuation and strict
JSON typing/wrapping. V4 therefore changes candidate-model identity to models
selected specifically for instruction-following and structured-output behavior
while retaining the frozen eval pack, decoding, baseline, fixed independent
reviewer, Judge, certification, and promotion policy.

Execution routing is infrastructure-only. The original 2x RTX 3090 topology
proved unavailable in EU-CZ-1, so V4 may run on one live >=48GB CUDA GPU. The
candidate weights remain FP16, Transformers device_map="balanced" remains in
use, and quantization plus CPU/disk model offload remain forbidden.
"""
from __future__ import annotations

from hephaestus.generation.backends import GenerationBackendError

import run_positive_promotion_proof_v3 as wave3

proof = wave3.proof

PRIOR_V3_REJECTION_EVIDENCE = (
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/promotion_gate_report.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/certification_decision.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/promotion_gate_report.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/certification_decision.json",
)

# Keep the independent reviewer fixed so candidate-model identity remains the
# only scientific variable changed by V4.
JUDGE_MODEL_ID = wave3.JUDGE_MODEL_ID
JUDGE_REVISION = wave3.JUDGE_REVISION

GRANITE_REVISION = "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
OLMO_REVISION = "3a5c85baefbb1896a54d56fe2e76c0395627ddf4"

V4_GPU_COUNT = 1
V4_MAX_MEMORY_GIB_PER_GPU = 44


class AdaptiveV4ChatTemplateBackend(proof.PinnedChatTemplateBackend):
    """Load V4 candidate weights in FP16 on one >=48GB CUDA GPU."""

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real candidate generation")
        if torch.cuda.device_count() < V4_GPU_COUNT:
            raise RuntimeError(
                f"V4 FP16 generation requires {V4_GPU_COUNT} CUDA GPU; "
                f"found {torch.cuda.device_count()}"
            )

        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_gib < 44.0:
            raise RuntimeError(
                f"V4 single-GPU FP16 execution requires a >=48GB class GPU; found {total_gib:.2f} GiB"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.snapshot_path), local_files_only=True, trust_remote_code=False
        )
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(f"candidate tokenizer has no chat template: {self.model_id}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        # Accelerate treats max_memory as the complete set of placement targets.
        # Supplying a synthetic "cpu": "0GiB" target can fail device-map
        # inference before any candidate sample is produced. Keep the budget GPU-
        # only and retain the strict post-load verification below so CPU/disk
        # offload remains forbidden rather than merely discouraged.
        model = AutoModelForCausalLM.from_pretrained(
            str(self.snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            device_map="balanced",
            max_memory={0: f"{V4_MAX_MEMORY_GIB_PER_GPU}GiB"},
        )

        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            forbidden_devices: set[str] = set()
            cuda_indices: set[int] = set()
            unknown_devices: set[str] = set()
            for placement in device_map.values():
                if isinstance(placement, int):
                    cuda_indices.add(placement)
                    continue
                value = str(placement)
                if value == "cuda":
                    cuda_indices.add(0)
                elif value.startswith("cuda:"):
                    cuda_indices.add(int(value.split(":", 1)[1]))
                elif value in {"cpu", "disk"}:
                    forbidden_devices.add(value)
                else:
                    unknown_devices.add(value)

            if forbidden_devices:
                raise RuntimeError(
                    f"V4 FP16 load attempted forbidden offload: {sorted(forbidden_devices)}"
                )
            if unknown_devices:
                raise RuntimeError(
                    "V4 FP16 load produced unsupported device-map placements: "
                    f"{sorted(unknown_devices)}"
                )
            if cuda_indices != {0}:
                raise RuntimeError(
                    "V4 single-GPU FP16 load must remain entirely on cuda:0; "
                    f"observed CUDA placements {sorted(cuda_indices)}"
                )
        else:
            # Transformers/Accelerate may collapse a balanced map to a single
            # CUDA device without retaining hf_device_map when the whole model
            # fits. In that case verify residency from the loaded tensors
            # themselves instead of rejecting valid single-GPU placement.
            observed_devices = {
                str(tensor.device)
                for tensor in (*model.parameters(), *model.buffers())
            }
            forbidden_devices = {
                device
                for device in observed_devices
                if not (device == "cuda" or device == "cuda:0")
            }
            if forbidden_devices:
                raise RuntimeError(
                    "V4 FP16 load without hf_device_map must have every parameter/buffer "
                    "resident on cuda:0; observed forbidden placements "
                    f"{sorted(forbidden_devices)}"
                )
            if not observed_devices:
                raise RuntimeError(
                    "V4 FP16 model load produced neither hf_device_map nor inspectable model tensors"
                )

        embedding_device = model.get_input_embeddings().weight.device
        if embedding_device.type != "cuda" or embedding_device.index not in {None, 0}:
            raise RuntimeError(
                "V4 model input embeddings must remain on cuda:0 because the frozen generation "
                f"backend places inputs there; got {embedding_device}"
            )

        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def generate_batch(self, *args, **kwargs):
        """Preserve the exact V4 runtime exception in persisted generation evidence."""
        try:
            return super().generate_batch(*args, **kwargs)
        except GenerationBackendError:
            raise
        except Exception as exc:
            detail = str(exc) or repr(exc)
            raise GenerationBackendError(
                "v4_generation_backend_failed",
                f"V4 candidate generation backend failed: {type(exc).__name__}: {detail}",
                retryable=True,
            ) from exc


class StrictV4EvaluationGenerationService(proof.EvaluationGenerationService):
    """Fail closed when required generation evidence is incomplete.

    The shared generation service intentionally returns partial evidence on a
    provider/runtime failure. That is useful generally, but a promotion proof
    must not feed a zero/partial-sample candidate into comparison, Judge,
    certification, or promotion gates and accidentally turn missing evidence
    into a scientific rejection.
    """

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        expected = len(self.plan().tasks)
        observed = len(result.report.samples)
        if not result.report.completed or observed != expected:
            issues = "; ".join(
                f"{issue.code}: {issue.message}" for issue in result.report.issues
            ) or "no issue detail"
            raise RuntimeError(
                f"V4 generation incomplete for run {result.report.run_id}: "
                f"expected {expected} samples, got {observed}; "
                f"completion_status={result.report.completion_status}; {issues}"
            )
        return result


proof.PinnedChatTemplateBackend = AdaptiveV4ChatTemplateBackend
proof.EvaluationGenerationService = StrictV4EvaluationGenerationService

proof.CANDIDATES = (
    {
        "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "revision": GRANITE_REVISION,
        "license": "apache-2.0",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V3_REJECTION_EVIDENCE),
        "selection_reason": (
            "V3 failures were strict exact-format/termination failures rather than broad capability failures; "
            "Granite 3.3 is selected for instruction-following and structured-output behavior while fitting "
            "comfortably inside the approved FP16 no-offload execution envelope"
        ),
    },
    {
        "model_id": "allenai/OLMo-2-1124-13B-Instruct",
        "revision": OLMO_REVISION,
        "license": "apache-2.0",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V3_REJECTION_EVIDENCE),
        "selection_reason": (
            "OLMo 2 13B Instruct is post-trained with Tulu-3/DPO/RLVR and selected as an independent model "
            "family aimed at strong instruction following, including IFEval-like behavior, while remaining "
            "within the approved FP16 no-offload execution envelope"
        ),
    },
)

if __name__ == "__main__":
    raise SystemExit(proof.main())
