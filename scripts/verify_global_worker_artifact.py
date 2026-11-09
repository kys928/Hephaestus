#!/usr/bin/env python3
"""Read-only Global Volume artifact check and optional offline LoRA inference.

No API, Pod creation, downloads, storage modifications or promotion.
Run inside a newly mounted worker. For paired inference, point at an
already-local frozen base model, tokenizer and complete PEFT adapter directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from hephaestus.providers.runpod.global_volume import GlobalArtifactLibrary, GlobalVolumeConfig, _sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="sha256 receipt reference")
    parser.add_argument("--kind", required=True, choices=[
        "model", "adapter", "tokenizer", "dataset", "sealed_checkpoint", "scientific_evidence",
    ])
    parser.add_argument("--base-model", type=Path, help="already-local frozen base model directory")
    parser.add_argument("--adapter-dir", type=Path, help="already-local complete PEFT adapter directory")
    parser.add_argument("--tokenizer", type=Path, help="already-local tokenizer directory")
    parser.add_argument("--prompt", default="Hello.")
    args = parser.parse_args()

    config = GlobalVolumeConfig.from_env()
    content = GlobalArtifactLibrary(config).resolve(args.receipt, expected_kind=args.kind)
    print(json.dumps({
        "receipt_verified": args.receipt,
        "volume_id": config.volume_id,
        "file_sha256": _sha256(content)[0],
        "pod_id": os.getenv("RUNPOD_POD_ID", "not-set"),
        "datacenter_id": os.getenv("RUNPOD_DC_ID", "not-set"),
    }, sort_keys=True))

    if not args.base_model and not args.adapter_dir and not args.tokenizer:
        print("artifact-only verification successful; no model inference requested")
        return 0
    if not args.base_model or not args.adapter_dir or not args.tokenizer:
        raise SystemExit("paired inference requires --base-model, --adapter-dir and --tokenizer")
    if not all(p.is_dir() for p in (args.base_model, args.adapter_dir, args.tokenizer)):
        raise SystemExit("model/adapter/tokenizer must already exist locally")
    # A frozen base/adapter/tokenizer identity must be approved by the external
    # deterministic state owner before entering this opt-in smoke harness.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.base_model), local_files_only=True)
    model = PeftModel.from_pretrained(base, str(args.adapter_dir), is_trainable=False)
    model.eval()
    tokenized = tokenizer(args.prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**tokenized, max_new_tokens=8, do_sample=False)
    tokens = output[0].tolist()
    print(json.dumps({
        "offline_generation_completed": True,
        "output_token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "output_token_count": len(tokens),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
