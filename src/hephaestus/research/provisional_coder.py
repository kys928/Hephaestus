"""Temporary parent-only Coder model, without the certified Planner LoRA.

This role is intentionally *not* appended to the certified five-role registry,
and does not change Planner/Diagnostician behavior. It proposes patch JSON text
only; independent verification and the human review boundary remain mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
import gc
import re
import time

from hephaestus.production.original_cuda_identity import verified_original_roles

PARENT_ID="mistralai/Ministral-3-8B-Reasoning-2512"
PARENT_REVISION="81eaece1948f3875421d9a45bc55487d10e2d894"
_SHA=re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(slots=True)
class ProvisionalMinistralCoder:
    """Genuine offline pinned parent CUDA inference; no PEFT or Ollama.

    The model's *own* response never creates an approval, selects a test,
    changes a running file or declares itself certified. An operator must
    provision GPU and model snapshot under separately approved resource policy.
    """

    model_cache: Path
    independent_registry_sha256: str
    device: str="cuda:0"
    seed: int=271828
    maximum_new_tokens: int=768
    last_receipt: dict[str,object] | None=field(default=None,init=False,repr=False)
    _model: Any=field(default=None,init=False,repr=False)
    _backend: Any=field(default=None,init=False,repr=False)

    def __post_init__(self) -> None:
        if (not _SHA.fullmatch(self.independent_registry_sha256)
                or not self.model_cache.is_absolute()
                or self.model_cache.is_symlink()
                or self.device!="cuda:0"
                or type(self.seed) is not int or not 0<=self.seed<2**32
                or type(self.maximum_new_tokens) is not int
                or not 64<=self.maximum_new_tokens<=1024):
            raise PermissionError("provisional Coder requires pinned offline CUDA parent policy")
        # Existing certified five-role identity is READ ONLY. We use only
        # the Planner's immutable parent revision, never its trained adapter.
        roles,_=verified_original_roles(self.independent_registry_sha256)
        parent=roles.resolve("planner")
        if (parent.model_id!=PARENT_ID or parent.revision!=PARENT_REVISION):
            raise PermissionError("provisional Coder parent differs from frozen original registry")

    def close(self) -> None:
        self._model=None
        self._backend=None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        except ImportError as exc:
            raise RuntimeError("provisional Ministral Coder needs original role-inference runtime") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise PermissionError("original parent Coder requires exactly one CUDA GPU, never CPU fallback")
        # All weights must already be present locally. This call never pulls a
        # different latest model, an Ollama derivative or a Planner LoRA.
        snapshot=snapshot_download(repo_id=PARENT_ID,revision=PARENT_REVISION,
            cache_dir=str(self.model_cache),local_files_only=True,
            ignore_patterns=["consolidated.safetensors","*.gguf","*.pth"])
        try:
            backend=MistralCommonBackend.from_pretrained(snapshot)
            model=Mistral3ForConditionalGeneration.from_pretrained(
                snapshot,local_files_only=True,torch_dtype=torch.bfloat16,
                device_map={"":0},trust_remote_code=False)
            model.eval()
            if not all(str(parameter.device).startswith("cuda")
                       for parameter in model.parameters()):
                raise PermissionError("provisional Coder model not fully GPU resident")
            self._backend=backend
            self._model=model
        except Exception:
            self.close()
            raise

    def generate_patch_json(self, prompt: str) -> str:
        if not isinstance(prompt,str) or not 40<=len(prompt.encode())<=65_000:
            raise ValueError("Coder prompt exceeds accepted code-repair context")
        self.last_receipt=None
        self._load()
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        encoded=self._backend.apply_chat_template(
            [{"role":"user","content":prompt}],
            return_tensors="pt",return_dict=True)
        ids=encoded["input_ids"]
        prefix_tokens=ids.shape[-1]
        if prefix_tokens>4096:
            raise ValueError("Coder prompt token budget exceeded")
        inputs={key:obj.to(self.device) if hasattr(obj,"to") else obj
                for key,obj in encoded.items()}
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        deadline=time.monotonic()+90

        class Deadline(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return time.monotonic()>=deadline

        started=time.monotonic()
        pad=getattr(self._backend,"pad_token_id",None)
        if pad is None:
            pad=getattr(self._backend,"eos_token_id",None)
        with torch.inference_mode():
            output=self._model.generate(
                **inputs,do_sample=False,max_new_tokens=self.maximum_new_tokens,
                stopping_criteria=StoppingCriteriaList([Deadline()]),
                pad_token_id=pad)[0]
        if time.monotonic()>=deadline:
            raise TimeoutError("provisional Coder inference exceeded fixed generation deadline")
        token_ids=output[prefix_tokens:].tolist()
        raw=self._backend.decode(token_ids,skip_special_tokens=True)
        if not raw.strip() or len(raw)>16_000:
            raise ValueError("provisional Coder response invalid or exceeds bounds")
        self.last_receipt={
            "schema":"provisional-parent-coder-inference.v1",
            "source":"original-pinned-transformers-cuda-parent-no-lora",
            "model_id":PARENT_ID,
            "revision":PARENT_REVISION,
            "adapter_used":False,
            "device":"cuda:0",
            "seed":self.seed,
            "decoding":{"do_sample":False,
                        "max_new_tokens":self.maximum_new_tokens},
            "prompt_sha256":"sha256:"+sha256(prompt.encode()).hexdigest(),
            "output_sha256":"sha256:"+sha256(raw.encode()).hexdigest(),
            "generated_token_ids_sha256":"sha256:"+sha256(
                ",".join(str(x) for x in token_ids).encode()).hexdigest(),
            "input_tokens":prefix_tokens,
            "generated_tokens":len(token_ids),
            "elapsed_seconds":time.monotonic()-started,
            "certified_coder":False,
        }
        return raw
