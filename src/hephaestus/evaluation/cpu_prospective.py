"""Prospective original-task measurement. No historical benchmark claims."""
from .exact_json import SCORER_VERSION, score_prediction

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from hephaestus.policy.internal_research import put_record
from hephaestus.storage.verified import read_verified_json
from hephaestus.schemas.experiment_contract import ExperimentComparison
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.data.generated_validation import validate_original_example


class CpuProspectiveEvaluator:
    def __init__(self, context):
        self.context = context

    def generate(self, operation_id, *, checkpoint=None, repeat=0):
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        ctx = self.context
        protocol = ctx.admission.evaluation_store.require_frozen(ctx.scope["experiment_id"], ctx.profile_protocol)
        input_ref = put_record(ctx.artifacts, {"protocol_ref": ctx.profile_protocol, "scorer_version": SCORER_VERSION,
            "checkpoint": str(checkpoint) if checkpoint else "pinned-pretrained-baseline",
            "repeat": repeat, "profile_ref": ctx.profile_ref})
        cached = ctx.state.get(operation_id, input_ref)
        if cached:
            if checkpoint and not validate_checkpoint_manifest(Path(checkpoint))[0]:
                raise ValueError("candidate checkpoint changed before generation replay")
            return cached
        ctx.check_authority()
        rows = read_verified_json(ctx.artifacts, protocol["evaluation_ref"])["examples"]
        if checkpoint:
            if not validate_checkpoint_manifest(Path(checkpoint))[0]:
                raise ValueError("candidate checkpoint failed verification")
            model = AutoModelForCausalLM.from_pretrained(ctx.snapshot, local_files_only=True,
                trust_remote_code=False, dtype=torch.float32).to("cpu")
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
        else:
            model = ctx.scorer.model
        model.eval()
        torch.manual_seed(ctx.profile.seed)
        outputs = []
        for row in rows:
            ctx.check_deadline()
            validate_original_example(row)
            json_recipe = ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1"
            if json_recipe:
                from hephaestus.training.cpu_chat_json_recipe import chat_prompt
                prefix = chat_prompt(ctx.scorer.tokenizer, row["prompt"])
            else:
                prefix = row["prompt"] + "\nAnswer: "
            max_new_tokens = 64 if json_recipe else 16
            tokens = ctx.scorer.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
            if tokens.input_ids.shape[1] + max_new_tokens > int(model.config.max_position_embeddings):
                raise ValueError("prospective generation cannot truncate frozen prompt")
            # Teacher-forced reference-token NLL is a diagnostic, not an answer-correctness pass.
            # Preserve the same exact prompt/target segmentation as the governed loader.
            target_ids = ctx.scorer.tokenizer(row["target"], add_special_tokens=False).input_ids
            if not target_ids:
                raise ValueError("frozen evaluation target is empty")
            prefix_ids = tokens.input_ids[0].tolist()
            combined = torch.tensor([prefix_ids + target_ids], dtype=torch.long)
            masked = torch.tensor([[-100] * len(prefix_ids) + target_ids], dtype=torch.long)
            if combined.shape[1] > int(model.config.max_position_embeddings):
                raise ValueError("teacher-forced evaluation cannot truncate frozen target")
            with torch.inference_mode():
                generated = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=ctx.scorer.tokenizer.eos_token_id,
                    eos_token_id=ctx.scorer.tokenizer.eos_token_id)
                nll = float(model(input_ids=combined, attention_mask=torch.ones_like(combined), labels=masked).loss)
            if not torch.isfinite(torch.tensor(nll)):
                raise ValueError("nonfinite held-out target negative log-likelihood")
            text = ctx.scorer.tokenizer.decode(generated[0, tokens.input_ids.shape[1]:], skip_special_tokens=True).strip()
            valid, correct = False, False
            measurement = score_prediction(text, row["target"])
            valid, correct = measurement["valid_json"], measurement["exact_correct"]
            outputs.append({"sample_id": row["sample_id"], "family": row["family"],
                "prompt_sha256": "sha256:" + sha256(prefix.encode()).hexdigest(),
                "output": text, "output_sha256": "sha256:" + sha256(text.encode()).hexdigest(),
                "json_valid": valid, "correct": correct, "target_nll": nll})
        result = {"schema": "cpu-prospective-real-generation.v2", "scorer_version": SCORER_VERSION, "profile_ref": ctx.profile_ref,
            "protocol_ref": ctx.profile_protocol, "checkpoint_ref": str(checkpoint) if checkpoint else None,
            "generation_settings_id": ("greedy-chat64-json-eos-no-truncation.v2" if
                ctx.profile.training_recipe == "chat_json_r8_full_epoch_96_v1" else
                "greedy-max16-exact-prompt-no-truncation.v1"),
            "samples": outputs, "correct": sum(r["correct"] for r in outputs),
            "json_valid": sum(r["json_valid"] for r in outputs), "count": len(outputs),
            "family_counts": dict(Counter(r["family"] for r in outputs)),
            "mean_teacher_forced_target_nll": sum(r["target_nll"] for r in outputs) / len(outputs),
            "external_certification": False, "repeat_index": repeat}
        ctx.state.put(operation_id, input_ref, result)
        return result

    def compare(self, baseline, candidate):
        ctx = self.context
        if (baseline.get("scorer_version") != SCORER_VERSION or candidate.get("scorer_version") != SCORER_VERSION):
            raise ValueError("future comparisons require the same explicit scorer version")
        if (baseline["protocol_ref"] != candidate["protocol_ref"] or baseline["generation_settings_id"] != candidate["generation_settings_id"] or
                [r["sample_id"] for r in baseline["samples"]] != [r["sample_id"] for r in candidate["samples"]]):
            raise ValueError("prospective baseline/candidate protocol parity failed")
        regressions = [b["sample_id"] for b, c in zip(baseline["samples"], candidate["samples"], strict=True)
            if (b["correct"] and not c["correct"]) or (b["json_valid"] and not c["json_valid"])]
        delta = candidate["correct"] - baseline["correct"]
        outcome = "regressed" if regressions or delta < 0 else "improved" if delta > 0 else "equivalent_within_evidence"
        baseline_ref, candidate_ref = put_record(ctx.artifacts, baseline), put_record(ctx.artifacts, candidate)
        return ExperimentComparison("comparison-" + ctx.scope["run_id"], ctx.scope["experiment_id"],
            "pinned-cpu-baseline", [ctx.scope["run_id"]], evaluation_report_refs=[baseline_ref, candidate_ref],
            evidence_refs=[baseline_ref, candidate_ref, ctx.profile_protocol], primary_outcome=outcome,
            effect_summary={"correct_before": baseline["correct"], "correct_after": candidate["correct"],
                "json_before": baseline["json_valid"], "json_after": candidate["json_valid"],
                "cases": baseline["count"], "new_hard_regressions": regressions,
                "mean_target_nll_before": baseline["mean_teacher_forced_target_nll"],
                "mean_target_nll_after": candidate["mean_teacher_forced_target_nll"],
                "mean_target_nll_change": candidate["mean_teacher_forced_target_nll"] - baseline["mean_teacher_forced_target_nll"],
                "nll_is_diagnostic_not_promotion_authority": True,
                "missing_evidence": ["independent training-seed replication", "external review"],
                "repeatability": {"candidate_run_count": 1, "direction_consistency": 0.0}},
            deterministic_gate_status="failed" if regressions else "passed", variance_risk="unknown",
            recommendation="human_review_required" if delta > 0 else "retain_baseline",
            confidence=0.25, metadata={"claim_scope": "internal_prospective", "external_certification": False})
