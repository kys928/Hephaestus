"""Native full-precision model loading with explicit GPU residency evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_gpu_snapshot(snapshot: Path, spec: dict[str, Any]):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("model wave requires exactly one visible CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory / (1024 ** 3) < 44:
        raise RuntimeError("model wave requires a >=48GB-class GPU")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if config.model_type != spec["model_type"]:
        raise RuntimeError("runtime architecture differs from admission")
    if getattr(config, "quantization_config", None):
        raise RuntimeError("quantized model configuration is forbidden")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if not getattr(tokenizer, "chat_template", None) or tokenizer.eos_token_id is None:
        raise RuntimeError("runtime tokenizer lacks admitted template/EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    loaders = {"AutoModelForCausalLM": AutoModelForCausalLM,
               "AutoModelForImageTextToText": AutoModelForImageTextToText}
    model = loaders[spec["loader"]].from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        device_map="balanced", max_memory={0: "44GiB"},
    )
    devices = {str(t.device) for t in (*model.parameters(), *model.buffers())}
    if not devices or not devices <= {"cuda", "cuda:0"}:
        raise RuntimeError(f"forbidden model tensor residency: {sorted(devices)}")
    floating_dtypes = {str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    if floating_dtypes != {"torch.float16"}:
        raise RuntimeError(f"runtime weight dtype is not uniformly FP16: {sorted(floating_dtypes)}")
    if model.get_input_embeddings().weight.device.type != "cuda":
        raise RuntimeError("model input embeddings are not on CUDA")
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError("FP16 model materialization contains nonfinite parameters")
    model.eval()
    facts = {
        "gpu": props.name, "gpu_memory_bytes": props.total_memory,
        "gpu_count": torch.cuda.device_count(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "tensor_devices": sorted(devices), "weight_dtypes": sorted(floating_dtypes),
        "loader": spec["loader"], "cpu_disk_offload": False, "quantization": False,
        "native_load_verified": True, "generation_smoke_test_passed": False,
        "allocated_memory_bytes": torch.cuda.memory_allocated(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    return model, tokenizer, facts
