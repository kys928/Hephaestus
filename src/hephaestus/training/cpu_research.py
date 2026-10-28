"""Actual bounded PEFT optimization of pinned pretrained CPU causal LM weights.

Exact governed tokens are revalidated at consumption. Each completed optimizer
step has a verified adapter+optimizer snapshot so restart cannot invent updates.
"""
from hashlib import sha256
from pathlib import Path
import json
import os
import time
from hephaestus.data.generated_factory import load_authorized_tokens
from hephaestus.policy.internal_research import put_record
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest


def hash_file(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def verify_cpu_checkpoint(result):
    checkpoint = Path(result["checkpoint_ref"])
    if (not validate_checkpoint_manifest(checkpoint)[0] or
            hash_file(checkpoint / "checkpoint_manifest.json") != result.get("checkpoint_manifest_sha256") or
            hash_file(checkpoint / "adapter_model.safetensors") != result.get("adapter_sha256")):
        raise ValueError("CPU checkpoint differs from immutable optimizer receipt")
    return checkpoint


def verify_snapshot(snapshot, hashes):
    for name, digest in hashes.items():
        if name not in {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt"}:
            raise PermissionError("unexpected CPU model file")
        if hash_file(snapshot / name) != digest:
            raise PermissionError("pinned CPU pretrained snapshot bytes changed")


class CpuExactTokenTrainer:
    def __init__(self, context):
        self.context = context

    def run(self, operation_id):
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel
        ctx = self.context
        cached = ctx.state.get(operation_id, ctx.materialization_ref)
        if cached:
            verify_cpu_checkpoint(cached)
            return cached
        ctx.check_authority()
        profile, policy = ctx.profile, ctx.admission.policy()
        verify_snapshot(ctx.snapshot, profile.snapshot_hashes)
        rows = load_authorized_tokens(ctx.admission, ctx.candidate, ctx.review_ref, ctx.materialization_ref, **ctx.scope)
        torch.set_num_threads(profile.cpu_threads)
        torch.manual_seed(profile.seed)
        model = AutoModelForCausalLM.from_pretrained(ctx.snapshot, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32).to("cpu")
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.0002)
        initial = [p.detach().clone() for p in trainable]
        steps, losses, consumed = 0, [], []
        root = ctx.root / "checkpoints"
        root.mkdir(exist_ok=True)
        # A previous committed step must be loaded rather than executed again.
        previous = ctx.repository.get_latest("cpu_research_optimizer_steps", "run_id", policy.run_id)
        if previous:
            receipt = read_verified_json(ctx.artifacts, previous["receipt_ref"])
            if (receipt["profile_ref"] != ctx.profile_ref or receipt["materialization_ref"] != ctx.materialization_ref):
                raise PermissionError("optimizer resume profile/data mismatch")
            prior_path = checkpoint = Path(receipt["checkpoint_ref"])
            verify_cpu_checkpoint(receipt)
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            state = load_file(str(prior_path / "adapter_model.safetensors"))
            from peft import get_peft_model_state_dict
            if set(state) != set(get_peft_model_state_dict(model)):
                raise ValueError("resumed adapter component set differs from exact model recipe")
            mismatch = set_peft_model_state_dict(model, state)
            if mismatch.unexpected_keys:
                raise ValueError("unexpected resumed adapter keys")
            optimizer.load_state_dict(torch.load(prior_path / "optimizer.pt", map_location="cpu", weights_only=True))
            steps, losses, consumed = receipt["completed_steps"], receipt["losses"], receipt["consumed_sample_indices"]
            torch.set_rng_state(torch.load(prior_path / "rng.pt", weights_only=True))
        started = time.process_time()
        for step in range(steps, profile.maximum_training_steps):
            ctx.check_deadline()
            index = step % len(rows)
            row = rows[index]
            inputs = torch.tensor([row["input_ids"]], dtype=torch.long)
            labels = torch.tensor([row["labels"]], dtype=torch.long)
            if inputs.shape[1] > int(model.config.max_position_embeddings):
                raise ValueError("CPU recipe cannot truncate authorized example")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite CPU training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
                raise ValueError("nonfinite CPU training gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            losses.append(float(loss.detach()))
            consumed.append(index)
            steps = step + 1
            checkpoint = root / f"step-{steps}"
            partial = root / f"step-{steps}.partial-{os.getpid()}"
            if partial.exists():
                raise FileExistsError("uncommitted partial checkpoint requires inspection")
            partial.mkdir()
            model.save_pretrained(partial, safe_serialization=True)
            torch.save(optimizer.state_dict(), partial / "optimizer.pt")
            torch.save(torch.get_rng_state(), partial / "rng.pt")
            components = {p.name: hash_file(p) for p in sorted(partial.iterdir()) if p.is_file()}
            manifest = {"schema": "checkpoint-manifest.v1", "partial_write": False,
                "components": components, "manifest_hash": "sha256:" + sha256(canonical_bytes(components)).hexdigest()}
            (partial / "checkpoint_manifest.json").write_bytes(canonical_bytes(manifest))
            if not validate_checkpoint_manifest(partial)[0]:
                raise ValueError("CPU checkpoint validation failed")
            if checkpoint.exists():
                # A crash after publication but before step commit is detected;
                # never count, overwrite or reconstruct it as original history.
                raise FileExistsError("uncommitted published checkpoint needs recovery inspection")
            partial.rename(checkpoint)
            receipt = {"schema": "cpu-actual-optimizer-step.v1", "profile_ref": ctx.profile_ref,
                "materialization_ref": ctx.materialization_ref, "checkpoint_ref": str(checkpoint),
                "checkpoint_manifest_sha256": hash_file(checkpoint / "checkpoint_manifest.json"),
                "adapter_sha256": hash_file(checkpoint / "adapter_model.safetensors"),
                "completed_steps": steps, "losses": list(losses), "gradient_norm": norm,
                "consumed_sample_indices": list(consumed), "run_id": policy.run_id,
                "cpu_seconds_this_process": time.process_time() - started}
            ref = put_record(ctx.artifacts, receipt)
            ctx.repository.append("cpu_research_optimizer_steps", {"run_id": policy.run_id,
                "completed_steps": steps, "receipt_ref": ref})
        changed = any(not torch.equal(before, after.detach()) for before, after in zip(initial, trainable))
        if not changed or steps != profile.maximum_training_steps:
            raise ValueError("no verified completed CPU optimizer update")
        result = {"schema": "cpu-research-training-result.v1", "run_id": policy.run_id,
            "experiment_id": policy.experiment_id, "backend_id": "actual_cpu_transformers_peft",
            "status": "completed", "checkpoint_ref": str(checkpoint),
            "checkpoint_manifest_sha256": hash_file(checkpoint / "checkpoint_manifest.json"),
            "adapter_sha256": hash_file(checkpoint / "adapter_model.safetensors"),
            "completed_optimizer_steps": steps, "weights_changed": changed, "losses": losses,
            "source_rows_admitted": len(rows), "consumed_sample_indices": consumed,
            "full_dataset_epoch_completed": False, "profile_ref": ctx.profile_ref,
            "external_certification": False, "checkpoint_refs": [str(checkpoint)]}
        ctx.state.put(operation_id, ctx.materialization_ref, result)
        return result
