"""Inference adapters for deterministic evaluation generation."""

from __future__ import annotations

import importlib.util
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

from .models import GeneratedText, GenerationTask


class GenerationBackendError(RuntimeError):
    """Raised when an inference backend cannot safely generate evidence."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class GenerationBackend(Protocol):
    backend_id: str

    def generate_batch(
        self,
        tasks: Sequence[GenerationTask],
        *,
        run_id: str,
        loading_instructions: dict[str, object],
        decoding_config: dict[str, object],
    ) -> list[GeneratedText]: ...


@dataclass(slots=True)
class DeterministicFakeGenerationBackend:
    """Offline backend with run-aware deterministic output fixtures."""

    outputs: dict[tuple[str, str, int], str] = field(default_factory=dict)
    fallback: Callable[[str, GenerationTask], str] | None = None
    backend_id: str = "deterministic_fake_generation"
    calls: int = 0

    def generate_batch(
        self,
        tasks: Sequence[GenerationTask],
        *,
        run_id: str,
        loading_instructions: dict[str, object],
        decoding_config: dict[str, object],
    ) -> list[GeneratedText]:
        del loading_instructions, decoding_config
        self.calls += 1
        results: list[GeneratedText] = []
        for task in tasks:
            key = (run_id, task.task_id, task.seed)
            if key in self.outputs:
                output = self.outputs[key]
            elif self.fallback is not None:
                output = self.fallback(run_id, task)
            else:
                output = f"{task.task_id}:{task.seed}"
            results.append(GeneratedText(output=output, finish_reason="completed"))
        return results


@dataclass(slots=True)
class TransformersCausalLMGenerationBackend:
    """Lazy optional inference from a finalized local training checkpoint.

    Model and tokenizer files must already exist inside the same verified
    checkpoint. This adapter never enables remote code or implicit network access.
    """

    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    backend_id: str = "transformers_causal_lm_generation"

    @staticmethod
    def capability() -> dict[str, object]:
        missing = [
            package
            for package in ("torch", "transformers")
            if importlib.util.find_spec(package) is None
        ]
        return {
            "supported": not missing,
            "missing_packages": missing,
            "remote_code": False,
            "network_acquisition": False,
            "checkpoint_manifest_required": True,
        }

    def generate_batch(
        self,
        tasks: Sequence[GenerationTask],
        *,
        run_id: str,
        loading_instructions: dict[str, object],
        decoding_config: dict[str, object],
    ) -> list[GeneratedText]:
        del run_id
        capability = self.capability()
        if not capability["supported"]:
            raise GenerationBackendError(
                "generation_backend_unavailable",
                "Optional torch/transformers generation dependencies are unavailable.",
            )
        if bool(loading_instructions.get("trust_remote_code", False)):
            raise GenerationBackendError(
                "remote_code_forbidden", "Generation refuses trust_remote_code=true."
            )
        if str(loading_instructions.get("backend", "")) != "transformers_causal_lm":
            raise GenerationBackendError(
                "unsupported_checkpoint_backend",
                "Loading instructions are not for the Transformers causal-LM backend.",
            )
        model_ref = Path(str(loading_instructions.get("model_artifact_ref", ""))).resolve()
        tokenizer_ref = Path(
            str(loading_instructions.get("tokenizer_artifact_ref", ""))
        ).resolve()
        if not model_ref.is_dir() or not tokenizer_ref.is_dir():
            raise GenerationBackendError(
                "generation_artifact_missing", "Model or tokenizer artifact directory is missing."
            )
        checkpoint_root = model_ref.parent
        if tokenizer_ref.parent != checkpoint_root:
            raise GenerationBackendError(
                "generation_artifact_boundary_mismatch",
                "Model and tokenizer artifacts must belong to the same finalized checkpoint.",
            )
        valid, manifest_evidence = validate_checkpoint_manifest(checkpoint_root)
        if not valid:
            raise GenerationBackendError(
                "checkpoint_integrity_failure",
                f"Generation checkpoint verification failed: {manifest_evidence}",
            )

        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        if self.device == "cuda" and not torch.cuda.is_available():
            raise GenerationBackendError("cuda_unavailable", "CUDA was requested but is unavailable.")
        if self.device not in {"cpu", "cuda"}:
            raise GenerationBackendError("unsupported_device", f"Unsupported device: {self.device}")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.dtype not in dtype_map or (self.device == "cpu" and self.dtype != "float32"):
            raise GenerationBackendError("unsupported_dtype", f"Unsupported dtype: {self.dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_ref), local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_ref),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype_map[self.dtype],
        )
        model.to(self.device)
        model.eval()
        if tokenizer.pad_token_id is None:
            raise GenerationBackendError(
                "pad_token_missing", "Pinned tokenizer has no padding token."
            )

        temperature = float(decoding_config.get("temperature", 0.0))
        top_p = float(decoding_config.get("top_p", 1.0))
        max_new_tokens = int(decoding_config.get("max_new_tokens", 0))
        if max_new_tokens <= 0:
            raise GenerationBackendError(
                "invalid_decoding_settings", "max_new_tokens must be positive."
            )
        do_sample = temperature > 0.0
        results: list[GeneratedText] = []

        # Sampling requests with different seeds are isolated. Greedy requests may
        # batch safely because the seed does not affect token selection.
        for start in range(0, len(tasks), max(1, self.batch_size)):
            batch = list(tasks[start : start + max(1, self.batch_size)])
            if not batch:
                continue
            if len({task.seed for task in batch}) != 1 and do_sample:
                for task in batch:
                    results.extend(
                        self.generate_batch(
                            [task],
                            run_id="isolated",
                            loading_instructions=loading_instructions,
                            decoding_config=decoding_config,
                        )
                    )
                continue
            seed = batch[0].seed
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            encoded = tokenizer(
                [task.prompt for task in batch],
                padding=True,
                return_tensors="pt",
                truncation=False,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            generation_kwargs: dict[str, object] = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs)
            for row, task in enumerate(batch):
                continuation = generated[row, input_width:]
                output = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                results.append(
                    GeneratedText(
                        output=output,
                        finish_reason="completed",
                        prompt_tokens=int(encoded["attention_mask"][row].sum().item()),
                        generated_tokens=int(continuation.shape[0]),
                        metadata={
                            "seed": task.seed,
                            "device": self.device,
                            "dtype": self.dtype,
                            "checkpoint_manifest_hash": manifest_evidence,
                        },
                    )
                )
        return results


def load_generation_instructions(reference: str | Path) -> dict[str, object]:
    path = Path(reference)
    if not path.is_file():
        raise GenerationBackendError(
            "generation_handoff_missing", f"Generation handoff is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GenerationBackendError(
            "generation_handoff_invalid", "Generation handoff is unreadable."
        ) from exc
    if not isinstance(payload, dict):
        raise GenerationBackendError(
            "generation_handoff_invalid", "Generation handoff must be a JSON object."
        )
    return {str(key): value for key, value in payload.items()}
