"""Research-only causal-LM role proposal scoring using genuine pinned HF weights.

This is deliberately NOT CertifiedRoleReasoner or the frozen production
five-role model runtime. All candidate interpretations remain suggestions,
source evidence is validated before inference, and no policy action is
executed. A tiny LM choosing between admissible options cannot certify a
scientific claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
import json

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRegistry,
)
from hephaestus.llm.role_contract import RoleTask
from hephaestus.storage.verified import canonical_bytes

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SCOPE = "cpu-causal-lm-proposal-smoke-no-certification.v1"

# All entries are bounded admissible proposals, NOT observations, approval,
# diagnoses, or scientific truths. Importantly, "approved" is absent from
# Judge options until a separate independent approval/replay gate exists.
CANDIDATES = {
    "controller": (
        ("blocked", "request_recheck", "approval_status"),
        ("retry", "request_recheck", "artifact_availability"),
        ("noop", "noop", "stage_policy"),
    ),
    "diagnosis": (
        ("inconclusive", "collect_more_evidence", "evidence"),
        ("evaluation_integrity", "repair_evaluation", "evaluation_protocol"),
        ("data_coverage", "replace_or_mix_dataset", "dataset_mixture"),
    ),
    "planner": (
        ("collect_more_evidence", "request_recheck", "diagnostic_measurement"),
        ("repair_evaluation", "request_recheck", "evaluation_protocol"),
        ("replace_or_mix_dataset", "branch_new_experiment", "dataset_mixture"),
    ),
    "evaluator": (
        ("incomplete_evidence", "request_recheck", "metric_integrity"),
        ("no_material_change", "retain_baseline", "candidate_quality"),
        ("evaluation_invalid", "request_recheck", "evaluation_protocol"),
    ),
    "judge": (
        ("blocked", "reject_checkpoint", "evidence"),
        ("blocked", "abort_run", "stage_policy"),
        ("blocked", "continue_lineage_best", "deterministic_gate_status"),
    ),
}


@dataclass(slots=True)
class CpuCausalProposalScorer:
    """Genuine, pinned pretrained causal LM, never production certified."""

    tokenizer: object
    model: object
    artifact_store: object
    snapshot_dir: Path

    @classmethod
    def load(cls, artifact_store, *, cache_dir: Path):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        torch.set_num_threads(2)
        cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir),
            allow_patterns=(
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "merges.txt",
                "generation_config.json", "model.safetensors",
            ), local_files_only=False,
        ))
        if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
            raise ValueError("CPU research causal model does not match pinned revision")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.float32,
        )
        model.eval().to("cpu")
        return cls(tokenizer, model, artifact_store, snapshot)

    def _score(self, prefix: str, suffix: str) -> float:
        import torch
        before = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
        after = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if before.shape[1] == 0 or after.shape[1] == 0:
            raise ValueError("empty candidate text tokens")
        if before.shape[1] + after.shape[1] > 768:
            raise ValueError("CPU inference prompt exceeds bounded context")
        with torch.inference_mode():
            joint = torch.cat((before, after), dim=1)
            logprobs = torch.log_softmax(self.model(joint).logits[0], dim=-1)
            token_indices = after[0]
            selected = logprobs[
                torch.arange(before.shape[1]-1, joint.shape[1]-1), token_indices]
            value = float(selected.mean().item())
        if not math.isfinite(value):
            raise ValueError("nonfinite causal role score")
        return value

    def propose(self, task: RoleTask, registry: EvidenceRegistry,
                context: DeterministicRoleContext, *,
                admissible_candidates: tuple[tuple[str, str, str], ...] | None = None
                ) -> dict[str, object]:
        if task.role not in CANDIDATES:
            raise ValueError("unknown CPU role")
        if (not task.run_id or task.run_id != context.run_id
                or task.lineage_id != context.lineage_id
                or task.stage_name != context.stage_name):
            raise PermissionError("role inference workflow scope mismatch")
        if not task.evidence or len(task.evidence) > 16:
            raise ValueError("one or more bounded source refs required")
        refs = [item.ref for item in task.evidence]
        validated = registry.validate(refs, run_id=task.run_id,
                                      lineage_id=task.lineage_id)
        if not validated.passed or set(validated.valid_refs) != set(refs):
            raise PermissionError("CPU research model lacks verified source evidence")
        # The original source set has already been fully verified. This
        # small CPU model receives the two most recent bounded excerpts only.
        # Never imply that the LM read the other verified source bodies.
        presented = task.evidence[-2:]
        source_text = "\n".join(
            "SOURCE " + row.ref + " [FACT EXCERPT; ORIGINAL HASH VERIFIED]: " +
            " ".join(row.fact.split())[:80]
            for row in presented
        )
        prefix = (
            "Hephaestus scientific research assistant. This CPU model may only "
            "recommend a review; never claim a certificate or approval.\n"
            "Role: " + task.role + "\n"
            "Scientific problem (bounded excerpt): " + task.situation[:220] + "\n"
            "Verified source statements:\n" + source_text + "\n"
            "Choose the best next review proposal. Possible proposal:\n"
        )
        options = (tuple(CANDIDATES[task.role]) if admissible_candidates is None
                   else tuple(admissible_candidates))
        if (not options or len(set(options)) != len(options) or
                any(row not in CANDIDATES[task.role] for row in options)):
            raise PermissionError("CPU role proposals must be a nonempty bounded policy-admitted subset")
        scored = []
        for decision, action, variable in options:
            continuation = (
                "Decision: " + decision + "; Action: " + action +
                "; Primary variable: " + variable + "."
            )
            scored.append({
                "decision":decision, "action":action,
                "primary_variable":variable,
                "mean_conditional_log_likelihood":self._score(prefix, continuation),
            })
        ranked = sorted(scored, key=lambda x:x["mean_conditional_log_likelihood"], reverse=True)
        selected = ranked[0]
        identity = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "model_family": "actual_135M_pretrained_causal_lm",
            "model_backend": type(self.model).__name__,
            "model_weights_file_sha256": "sha256:" + sha256(
                (self.snapshot_dir / "model.safetensors").read_bytes()).hexdigest(),
        }
        receipt = {
            "schema":SCOPE, "role":task.role, "run_id":task.run_id,
            "lineage_id":task.lineage_id, "stage_name":task.stage_name,
            "prompt_sha256":"sha256:"+sha256(prefix.encode()).hexdigest(),
            "input_source_refs":refs,
            "context_evidence_refs":[item.ref for item in presented],
            "verified_sources_not_seen_by_model":[item.ref for item in task.evidence[:-2]],
            "candidates":scored, "selected_candidate":selected,
            "identity":identity,
            "advisory_only":True, "approval_granted":False,
            "action_executed":False, "production_role_certified":False,
            "scientific_quality_certified":False,
            "user_originating_hypotheses_not_verified_as_fact":True,
        }
        record = self.artifact_store.put_bytes(
            canonical_bytes(receipt), media_type="application/json")
        return {
            **receipt, "inference_receipt_ref":record.artifact_ref,
        }
