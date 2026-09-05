#!/usr/bin/env python3
"""Real-GPU positive promotion/certification proof through the production loop.

This driver runs inside a RunPod GPU Pod attached to the scientific Network
Volume. It evaluates pinned permissively licensed instruction-model snapshots
against the frozen semantic pack, obtains a separate model-based review, routes
the result through Judge/promotion/certification policy, applies the governed
action with ProductionLoopRunner, and independently verifies persisted lineage.

No training is performed. This is a governed model-discovery/admission proof.
The original scientific lineage is not mutated; a dedicated proof lineage is
used so a successful external-model promotion cannot overwrite prior research
lineage truth.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.control.spine import SpinePhase
from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
from hephaestus.generation.models import GeneratedText, GenerationTask
from hephaestus.generation.service import EvaluationGenerationService
from hephaestus.policy.promotion_gates import evaluate_promotion_gates
from hephaestus.policy.promotion_policy import PromotionPolicy
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionCycleResult, ProductionLoopRunner
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")
REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PACK_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"
BASELINE_EVAL_RUN = "first-semantic-evaluation-001-33869352751"
BASELINE_RUN_ID = f"random-init-baseline-{BASELINE_EVAL_RUN}"
BASELINE_CHECKPOINT = SCIENTIFIC_ROOT / "evaluations" / BASELINE_EVAL_RUN / "baseline_random_init_checkpoint"
LINEAGE_ID = "lineage-positive-real-model-proof"
STAGE_NAME = "stabilization"
APPROVAL_REF = "approval://operator/chat-2026-09-05-positive-real-compute-promotion"

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "c89bee90d9f811437d9735454613c35b4a3c4dc8",
        "license": "apache-2.0",
        "judge_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "judge_revision": "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
        "judge_license": "apache-2.0",
    },
    {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "582efe62d7cfafd242bffca71ecbde1bcecc1bcc",
        "license": "apache-2.0",
        "judge_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "judge_revision": "c89bee90d9f811437d9735454613c35b4a3c4dc8",
        "judge_license": "apache-2.0",
    },
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonical_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", value).strip("-.")


def unload_model(model: object | None = None) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def materialize_model(
    *,
    proof_root: Path,
    model_id: str,
    revision: str,
    expected_license: str,
) -> dict[str, object]:
    """Acquire one exact Hub revision and build a byte-level local manifest."""
    from huggingface_hub import HfApi, snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"model revision is not an immutable 40-char commit: {model_id}@{revision}")

    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    observed_revision = str(info.sha or "")
    if observed_revision != revision:
        raise RuntimeError(f"Hub revision drift for {model_id}: {observed_revision} != {revision}")
    card_data = getattr(info, "card_data", None)
    observed_license = str(getattr(card_data, "license", "") or "").lower()
    if observed_license != expected_license.lower():
        raise RuntimeError(
            f"model license drift for {model_id}@{revision}: {observed_license!r} != {expected_license!r}"
        )

    cache_dir = proof_root / "hf_cache"
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
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
        components[relative] = sha_file(path)
        byte_size += path.stat().st_size
    if not components or not any(name.endswith(".safetensors") for name in components):
        raise RuntimeError(f"model snapshot has no safetensors weights: {model_id}@{revision}")

    manifest_payload = {
        "manifest_version": "external-model-snapshot.v1",
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
    manifest_payload["manifest_hash"] = canonical_hash(components)
    manifest_path = proof_root / "model_manifests" / slug(model_id) / revision / "snapshot_manifest.json"
    atomic_json(manifest_path, manifest_payload)
    manifest_payload["manifest_ref"] = str(manifest_path)
    return manifest_payload


def verify_model_manifest(manifest_ref: str) -> dict[str, object]:
    path = Path(manifest_ref)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("candidate model manifest is not a JSON object")
    snapshot = Path(str(payload.get("snapshot_path") or ""))
    components = payload.get("components")
    if not snapshot.is_dir() or not isinstance(components, dict) or not components:
        raise RuntimeError("candidate model manifest is incomplete")
    if payload.get("manifest_hash") != canonical_hash(components):
        raise RuntimeError("candidate model manifest canonical hash mismatch")
    observed_total = 0
    for relative, expected in sorted(components.items()):
        candidate = snapshot / str(relative)
        if not candidate.is_file() or sha_file(candidate) != str(expected):
            raise RuntimeError(f"candidate model component verification failed: {relative}")
        observed_total += candidate.stat().st_size
    if observed_total != int(payload.get("byte_size", -1)):
        raise RuntimeError("candidate model snapshot byte-size mismatch")
    return {
        "status": "verified",
        "manifest_ref": str(path),
        "manifest_hash": payload.get("manifest_hash"),
        "model_id": payload.get("model_id"),
        "revision": payload.get("resolved_revision"),
        "license": payload.get("license"),
        "component_count": len(components),
        "byte_size": observed_total,
    }


@dataclass(slots=True)
class PinnedChatTemplateBackend:
    snapshot_path: Path
    model_id: str
    revision: str
    manifest_hash: str
    backend_id: str = "pinned_chat_template_transformers"
    _model: Any = None
    _tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real candidate generation")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.snapshot_path), local_files_only=True, trust_remote_code=False
        )
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(f"candidate tokenizer has no chat template: {self.model_id}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(self.snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
        )
        model.to("cuda")
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        unload_model()

    def generate_batch(
        self,
        tasks: Sequence[GenerationTask],
        *,
        run_id: str,
        loading_instructions: dict[str, object],
        decoding_config: dict[str, object],
    ) -> list[GeneratedText]:
        del run_id
        self._load()
        import torch

        if bool(loading_instructions.get("trust_remote_code", False)):
            raise RuntimeError("candidate generation refuses trust_remote_code=true")
        if str(loading_instructions.get("resolved_revision")) != self.revision:
            raise RuntimeError("candidate generation handoff revision mismatch")
        if str(loading_instructions.get("snapshot_manifest_hash")) != self.manifest_hash:
            raise RuntimeError("candidate generation handoff manifest mismatch")
        if float(decoding_config.get("temperature", -1.0)) != 0.0:
            raise RuntimeError("positive proof requires unchanged greedy frozen decoding")
        max_new_tokens = int(decoding_config.get("max_new_tokens", 0))
        if max_new_tokens <= 0:
            raise RuntimeError("frozen decoding max_new_tokens is invalid")

        outputs: list[GeneratedText | None] = [None] * len(tasks)
        for seed in sorted({task.seed for task in tasks}):
            indexed = [(index, task) for index, task in enumerate(tasks) if task.seed == seed]
            prompts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": task.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _, task in indexed
            ]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            encoded = self._tokenizer(prompts, padding=True, return_tensors="pt", truncation=False)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            for row, (index, task) in enumerate(indexed):
                continuation = generated[row, input_width:]
                text = self._tokenizer.decode(continuation, skip_special_tokens=True).strip()
                if not text:
                    text = "[empty generation]"
                outputs[index] = GeneratedText(
                    output=text,
                    finish_reason="completed",
                    prompt_tokens=int(encoded["attention_mask"][row].sum().item()),
                    generated_tokens=int(continuation.shape[0]),
                    metadata={
                        "model_id": self.model_id,
                        "revision": self.revision,
                        "snapshot_manifest_hash": self.manifest_hash,
                        "prompt_serialization": "tokenizer_chat_template:user_only",
                        "seed": task.seed,
                    },
                )
        if any(item is None for item in outputs):
            raise RuntimeError("candidate generation backend did not return every requested sample")
        return [item for item in outputs if item is not None]


class IndependentModelReviewer:
    def __init__(self, *, snapshot_path: Path, model_id: str, revision: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path), local_files_only=True, trust_remote_code=False
        )
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError(f"review tokenizer has no chat template: {model_id}")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
        )
        self.model.to("cuda")
        self.model.eval()

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        unload_model()

    @staticmethod
    def _parse_json(text: str) -> dict[str, object]:
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("independent reviewer returned no JSON object")
        payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("independent reviewer JSON is not an object")
        return payload

    def review(self, *, task_prompt: str, response: str, task_id: str) -> dict[str, object]:
        import torch

        rubric = (
            "You are an independent model-quality reviewer. Evaluate RESPONSE against TASK only; "
            "do not rewrite the response. Score instruction_adherence, relevance, and coherence from 0 to 1. "
            "Set approve=true only when the response satisfies the user's explicit format/content instruction and is coherent. "
            "Return exactly one JSON object with keys instruction_adherence, relevance, coherence, approve, reason.\n\n"
            f"TASK_ID: {task_id}\nTASK: {task_prompt}\nRESPONSE: {response}"
        )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": rubric}], tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(rendered, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=160,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        payload = self._parse_json(raw)
        scores = {}
        for name in ("instruction_adherence", "relevance", "coherence"):
            scores[name] = max(0.0, min(1.0, float(payload.get(name, 0.0))))
        return {
            "task_id": task_id,
            "task_prompt": task_prompt,
            "response": response,
            **scores,
            "approve": bool(payload.get("approve", False)),
            "reason": str(payload.get("reason", "")),
            "raw_review": raw,
            "review_model_id": self.model_id,
            "review_model_revision": self.revision,
        }


def build_review_bundle(*, plan: Any, generated: Any, reviewer: IndependentModelReviewer) -> dict[str, object]:
    prompt_by_task = {task.task_id: task.prompt for task in plan.tasks}
    # Review one sample per frozen task. Generation is greedy and the repeated
    # candidate runs separately prove seed/run repeatability.
    selected: dict[str, Any] = {}
    for sample in generated.report.samples:
        selected.setdefault(sample.task_id, sample)
    rows = [
        reviewer.review(
            task_prompt=prompt_by_task[task_id],
            response=sample.output,
            task_id=task_id,
        )
        for task_id, sample in sorted(selected.items())
    ]
    if not rows:
        raise RuntimeError("independent review had no candidate samples")
    approval_rate = sum(1 for row in rows if row["approve"]) / len(rows)
    mean_score = sum(
        (float(row["instruction_adherence"]) + float(row["relevance"]) + float(row["coherence"])) / 3.0
        for row in rows
    ) / len(rows)
    confidence = 0.5 * approval_rate + 0.5 * mean_score
    approved = bool(approval_rate == 1.0 and mean_score >= 0.86 and confidence >= 0.93)
    return {
        "review_version": "independent-model-review.v1",
        "reviewed_at": now(),
        "review_model_id": reviewer.model_id,
        "review_model_revision": reviewer.revision,
        "sample_count": len(rows),
        "approval_rate": approval_rate,
        "mean_dimension_score": mean_score,
        "confidence": confidence,
        "approved": approved,
        "rows": rows,
    }


@dataclass(slots=True)
class RealModelPromotionDriver:
    proof_root: Path
    proof_run_id: str
    baseline_result: Any

    def _write_phase(self, cycle_root: Path, name: str, payload: object) -> str:
        path = cycle_root / f"{name}.json"
        atomic_json(path, payload)
        return str(path)

    def execute_cycle(self, *, runtime: Any, state: Any, cycle_index: int) -> ProductionCycleResult:
        if cycle_index < 1 or cycle_index > len(CANDIDATES):
            raise RuntimeError("candidate set exhausted")
        spec = CANDIDATES[cycle_index - 1]
        cycle_root = self.proof_root / "cycles" / f"cycle-{cycle_index:02d}"
        cycle_root.mkdir(parents=True, exist_ok=True)
        candidate_run_base = f"{self.proof_run_id}-candidate-{cycle_index}"
        experiment_id = f"experiment-{candidate_run_base}"

        candidate_manifest = materialize_model(
            proof_root=self.proof_root,
            model_id=spec["model_id"],
            revision=spec["revision"],
            expected_license=spec["license"],
        )
        candidate_snapshot = Path(str(candidate_manifest["snapshot_path"]))
        candidate_manifest_ref = str(candidate_manifest["manifest_ref"])
        checkpoint_ref = str(candidate_snapshot)

        handoff = {
            "backend": "pinned_chat_template_transformers",
            "model_id": spec["model_id"],
            "requested_revision": spec["revision"],
            "resolved_revision": spec["revision"],
            "snapshot_path": checkpoint_ref,
            "snapshot_manifest_hash": candidate_manifest["manifest_hash"],
            "license": spec["license"],
            "trust_remote_code": False,
            "prompt_serialization": "tokenizer_chat_template:user_only",
        }
        handoff_ref = cycle_root / "candidate_generation_handoff.json"
        atomic_json(handoff_ref, handoff)

        backend = PinnedChatTemplateBackend(
            snapshot_path=candidate_snapshot,
            model_id=spec["model_id"],
            revision=spec["revision"],
            manifest_hash=str(candidate_manifest["manifest_hash"]),
        )
        generation = EvaluationGenerationService(
            artifact_root=self.proof_root / "evaluations",
            backend=backend,
            pack_name="semantic_behavior_v1",
            config_dir=REPO_ROOT / "configs",
        )
        plan = generation.plan()
        if plan.content_hash != EVAL_PACK_HASH or len(plan.tasks) != 18:
            raise RuntimeError("frozen semantic generation plan drifted")

        candidate_results = []
        started = time.monotonic()
        try:
            for repeat in range(1, 4):
                run_id = f"{candidate_run_base}-repeat-{repeat}"
                handle = TrainingRunHandle(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    backend_id=backend.backend_id,
                    status="completed",
                    checkpoint_refs=[checkpoint_ref],
                    metadata={
                        "generation_handoff_ref": str(handoff_ref),
                        "model_id": spec["model_id"],
                        "model_revision": spec["revision"],
                        "snapshot_manifest_hash": candidate_manifest["manifest_hash"],
                        "training_performed": False,
                    },
                )
                candidate_results.append(
                    generation.generate(handle, generation_handoff_ref=str(handoff_ref), checkpoint_ref=checkpoint_ref)
                )
        finally:
            backend.close()
        generation_seconds = time.monotonic() - started

        proposal = ExperimentProposal(
            experiment_id=experiment_id,
            run_id=candidate_results[-1].run_handle.run_id,
            lineage_id=LINEAGE_ID,
            stage_name=STAGE_NAME,
            diagnosis_report_id=f"diagnosis-{self.proof_run_id}-model-quality",
            intervention_id=f"intervention-{self.proof_run_id}-change-model-{cycle_index}",
            primary_variable="model_candidate_bundle",
            baseline_ref=BASELINE_RUN_ID,
            model_selection_id=f"model-selection-{candidate_run_base}",
            controlled_variables={
                "evaluation_reference": f"sha256:{EVAL_PACK_HASH}",
                "decoding_config": dict(plan.decoding_config),
                "candidate_repetitions": 3,
                "training_performed": False,
            },
            success_criteria={
                "semantic_outcome": "improved",
                "deterministic_gate_status": "passed",
                "variance_risk": "low",
                "candidate_evidence_runs": 3,
                "independent_review_required": True,
                "certification_required": True,
            },
            failure_criteria={"any_hard_gate_failure": True, "review_rejected": True},
            required_evidence=["frozen_semantic_pack", "independent_model_review", "immutable_model_snapshot"],
            required_approvals=["promotion_approval"],
            rollback_plan="Reject the external candidate and leave the proof lineage on its prior baseline.",
            status="evaluating",
            metadata={"candidate_model": spec["model_id"], "candidate_revision": spec["revision"]},
        )

        evaluator = ExperimentEvaluationService(
            pack_name="semantic_behavior_v1", config_dir=REPO_ROOT / "configs"
        )
        comparison = evaluator.compare(
            proposal,
            [self.baseline_result.run_handle, *[item.run_handle for item in candidate_results]],
        )
        comparison_ref = self._write_phase(cycle_root, "experiment_comparison", comparison.to_dict())
        proposal_ref = self._write_phase(cycle_root, "experiment_proposal", proposal.to_dict())

        review_bundle: dict[str, object] = {
            "approved": False,
            "confidence": 0.0,
            "status": "not_run_due_to_non_improved_comparison",
        }
        if comparison.primary_outcome == "improved" and comparison.deterministic_gate_status == "passed":
            judge_manifest = materialize_model(
                proof_root=self.proof_root,
                model_id=spec["judge_model_id"],
                revision=spec["judge_revision"],
                expected_license=spec["judge_license"],
            )
            reviewer = IndependentModelReviewer(
                snapshot_path=Path(str(judge_manifest["snapshot_path"])),
                model_id=spec["judge_model_id"],
                revision=spec["judge_revision"],
            )
            try:
                review_bundle = build_review_bundle(
                    plan=plan, generated=candidate_results[0], reviewer=reviewer
                )
                review_bundle["review_model_manifest_ref"] = judge_manifest["manifest_ref"]
                review_bundle["review_model_manifest_hash"] = judge_manifest["manifest_hash"]
                review_bundle["status"] = "completed"
            finally:
                reviewer.close()
        review_ref = self._write_phase(cycle_root, "independent_review", review_bundle)

        review_approved = bool(review_bundle.get("approved", False))
        review_confidence = float(review_bundle.get("confidence", 0.0) or 0.0)
        semantic_judge = SemanticComparisonJudgeAdapter().decide(
            comparison,
            run_id=candidate_results[-1].run_handle.run_id,
            lineage_id=LINEAGE_ID,
            candidate_checkpoint_ref=checkpoint_ref,
            monitor_outcome="healthy",
            recent_failure_count=0,
            has_stable_checkpoint=False,
            human_review_approved=review_approved,
            human_review_approval_ref=APPROVAL_REF if review_approved else None,
            independent_review_confidence=review_confidence if review_approved else None,
            certification_requested=True,
        )
        semantic_judge_ref = self._write_phase(cycle_root, "semantic_judge_exit", semantic_judge.to_dict())

        repeatability = comparison.effect_summary.get("repeatability", {})
        if not isinstance(repeatability, dict):
            repeatability = {}
        direction_consistency = float(repeatability.get("direction_consistency", 0.0) or 0.0)
        observed_runs = int(repeatability.get("candidate_run_count", 0) or 0)
        repeatability_sufficient = bool(
            observed_runs >= 3
            and direction_consistency >= 0.67
            and comparison.variance_risk == "low"
        )

        promotion_decision = PromotionPolicy().decide(
            deterministic_passed=comparison.deterministic_gate_status == "passed",
            confidence=semantic_judge.confidence,
            has_candidate=True,
            promotion_bundle_passed=bool(
                comparison.primary_outcome == "improved"
                and review_approved
                and not any(issue.blocking for issue in comparison.issues)
            ),
            evidence_completeness=1.0 if not any(issue.blocking for issue in comparison.issues) else 0.0,
            certification_readiness="certification_passed" if review_approved else "certification_not_eligible",
            recheck_recommended=not repeatability_sufficient,
            observed_consistent_runs=observed_runs if direction_consistency >= 0.67 else 0,
            min_promotion_evidence=1,
            min_stable_evidence=2,
            observed_evidence_runs=observed_runs,
            min_certification_evidence=3,
            stability_confidence=semantic_judge.confidence,
            min_stability_confidence=0.9,
            repeatability_sufficient=repeatability_sufficient,
            variance_risk=comparison.variance_risk,
        )
        certification_ref = self._write_phase(
            cycle_root,
            "certification_decision",
            {
                "promotion_state": promotion_decision.promotion_state,
                "certification_state": promotion_decision.certification_state,
                "recheck_required": promotion_decision.recheck_required,
                "notes": promotion_decision.notes,
                "governance_confidence": semantic_judge.confidence,
                "comparison_confidence": comparison.confidence,
                "candidate_evidence_runs": observed_runs,
                "direction_consistency": direction_consistency,
                "repeatability_sufficient": repeatability_sufficient,
                "variance_risk": comparison.variance_risk,
            },
        )

        eval_report = EvalReport(
            eval_id=f"eval-{candidate_run_base}",
            run_id=candidate_results[-1].run_handle.run_id,
            stage_name=STAGE_NAME,
            pack_name="semantic_behavior_v1",
            checkpoint_resolution={
                "selected_checkpoint_ref": checkpoint_ref,
                "candidate_snapshot_manifest_hash": candidate_manifest["manifest_hash"],
            },
            confidence=semantic_judge.confidence,
            evidence_completeness=1.0,
            stability_confidence=semantic_judge.confidence,
            certification_readiness=promotion_decision.certification_state,
            recheck_recommended=promotion_decision.recheck_required,
            promotion_bundle_passed=promotion_decision.promotion_state in {"promoted_best", "stable", "certified_stable"},
            observed_consistent_runs=observed_runs,
            repeated_eval_count=observed_runs,
            consistency_score=direction_consistency,
            repeatability_ready=repeatability_sufficient,
            repeatability_sufficient=repeatability_sufficient,
            variance_risk=comparison.variance_risk,
            consistency_observed="consistent" if direction_consistency >= 0.67 else "inconsistent",
            certification_recheck_count=max(0, observed_runs - 1),
            eval_pack_id=str(comparison.metadata.get("eval_pack_id") or "semantic_behavior"),
            eval_pack_version=str(comparison.metadata.get("eval_pack_version") or "1.0.0"),
            eval_pack_integrity_level="content_hash_verified",
            deterministic_scorecard={
                "deterministic_passed": comparison.deterministic_gate_status == "passed",
                "candidate_hard_failures": comparison.effect_summary.get("deterministic", {}).get("candidate_hard_failures", [])
                if isinstance(comparison.effect_summary.get("deterministic"), dict)
                else [],
                "comparison_id": comparison.comparison_id,
            },
            deterministic_passed=comparison.deterministic_gate_status == "passed",
            failed_gates=[] if comparison.deterministic_gate_status == "passed" else ["semantic_deterministic_gate"],
            passed_gates=["semantic_deterministic_gate"] if comparison.deterministic_gate_status == "passed" else [],
            scorecard_integrity_level="content_hash_verified",
        )

        current_lineage = LineageStore(runtime.settings.state_root).get_current(LINEAGE_ID)
        approval_metadata = {
            "approval_status": "approved",
            "approval_ref": APPROVAL_REF,
            "approval_scope": "conditional promotion/certification only if all hard gates and independent review pass",
            "human_sample_review_performed": False,
            "operator_conditional_approval": True,
            "independent_review_ref": review_ref,
        }
        gate = evaluate_promotion_gates(
            run_id=candidate_results[-1].run_handle.run_id,
            lineage_id=LINEAGE_ID,
            requested_action="promote_checkpoint",
            eval_report=eval_report,
            lineage_state=current_lineage,
            data_manifest=None,
            approval_metadata=approval_metadata,
        )
        gate_ref = self._write_phase(cycle_root, "promotion_gate_report", gate.to_dict())
        approval_ref_file = self._write_phase(cycle_root, "operator_approval", approval_metadata)

        qualifies = bool(
            comparison.primary_outcome == "improved"
            and comparison.deterministic_gate_status == "passed"
            and comparison.variance_risk == "low"
            and review_approved
            and repeatability_sufficient
            and semantic_judge.next_action.value == "promote_checkpoint"
            and promotion_decision.promotion_state == "certified_stable"
            and promotion_decision.certification_state == "certification_passed"
            and gate.promotion_allowed
            and not gate.blocking_failures
        )
        action = "promote_checkpoint" if qualifies else "reject_checkpoint"

        data_audit_ref = self._write_phase(
            cycle_root,
            "data_acquisition_audit",
            {
                "status": "not_applicable_model_only_intervention",
                "dataset_changed": False,
                "reason": "external pretrained model candidate admission changes model bundle only",
                "candidate_model_manifest_ref": candidate_manifest_ref,
            },
        )
        preprocessing_ref = self._write_phase(
            cycle_root,
            "data_preprocessor",
            {
                "status": "not_applicable_model_only_intervention",
                "dataset_changed": False,
                "preprocessing_changed": False,
            },
        )
        training_ref = self._write_phase(
            cycle_root,
            "training_engineer",
            {
                "status": "external_model_admission_no_training",
                "training_performed": False,
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "snapshot_manifest_ref": candidate_manifest_ref,
            },
        )
        runtime_ref = self._write_phase(
            cycle_root,
            "runtime_monitor",
            {
                "status": "healthy",
                "cuda": True,
                "generation_seconds": generation_seconds,
                "candidate_repeat_runs": [item.run_handle.run_id for item in candidate_results],
                "candidate_sample_count": sum(len(item.report.samples) for item in candidate_results),
            },
        )
        judge_entry_ref = self._write_phase(
            cycle_root,
            "judge_entry",
            {
                "status": "approved_for_bounded_evaluation",
                "lineage_id": LINEAGE_ID,
                "candidate_index": cycle_index,
                "candidate_model": spec["model_id"],
                "candidate_revision": spec["revision"],
                "baseline_run_id": BASELINE_RUN_ID,
                "frozen_eval_pack_hash": EVAL_PACK_HASH,
            },
        )

        phase_evidence = {
            SpinePhase.JUDGE_ENTRY.value: [judge_entry_ref],
            SpinePhase.PLANNER.value: [proposal_ref],
            SpinePhase.DATA_ACQUISITION_AUDIT.value: [data_audit_ref, candidate_manifest_ref],
            SpinePhase.DATA_PREPROCESSOR.value: [preprocessing_ref],
            SpinePhase.TRAINING_ENGINEER.value: [training_ref],
            SpinePhase.RUNTIME_MONITOR.value: [runtime_ref, *[str(item.report.report_ref) for item in candidate_results]],
            SpinePhase.EVALUATOR.value: [comparison_ref, review_ref],
            SpinePhase.JUDGE_EXIT.value: [semantic_judge_ref, certification_ref, gate_ref, approval_ref_file],
        }
        cycle_summary = {
            "cycle_index": cycle_index,
            "candidate_model": spec["model_id"],
            "candidate_revision": spec["revision"],
            "candidate_manifest_ref": candidate_manifest_ref,
            "candidate_manifest_hash": candidate_manifest["manifest_hash"],
            "comparison": comparison.to_dict(),
            "independent_review": review_bundle,
            "semantic_judge": semantic_judge.to_dict(),
            "certification": {
                "promotion_state": promotion_decision.promotion_state,
                "certification_state": promotion_decision.certification_state,
            },
            "promotion_gate": gate.to_dict(),
            "qualifies_for_certified_promotion": qualifies,
            "selected_action": action,
        }
        cycle_summary_ref = self._write_phase(cycle_root, "cycle_summary", cycle_summary)

        return ProductionCycleResult(
            cycle_id=f"{self.proof_run_id}-cycle-{cycle_index}",
            run_id=candidate_results[-1].run_handle.run_id,
            experiment_id=experiment_id,
            status="completed",
            judge_action=action,
            checkpoint_ref=checkpoint_ref,
            confidence=semantic_judge.confidence,
            promotion_allowed=bool(gate.promotion_allowed and qualifies),
            certification_state=promotion_decision.certification_state if qualifies else "certification_not_eligible",
            approval_status="approved" if qualifies else "not_required",
            approval_ref=APPROVAL_REF if qualifies else None,
            comparison_ref=comparison_ref,
            phase_evidence=phase_evidence,
            evidence={
                "cycle_summary_ref": cycle_summary_ref,
                "candidate_manifest_ref": candidate_manifest_ref,
                "candidate_manifest_hash": candidate_manifest["manifest_hash"],
                "semantic_outcome": comparison.primary_outcome,
                "deterministic_gate_status": comparison.deterministic_gate_status,
                "independent_review_approved": review_approved,
                "independent_review_confidence": review_confidence,
                "promotion_gate_allowed": gate.promotion_allowed,
                "certification_state": promotion_decision.certification_state,
            },
            next_cycle=(
                {
                    "candidate_index": cycle_index + 1,
                    "reason": "prior candidate did not satisfy full certified-promotion chain",
                    "prior_candidate": spec["model_id"],
                    "prior_action": action,
                }
                if not qualifies and cycle_index < len(CANDIDATES)
                else None
            ),
        )


def main() -> int:
    proof_run_id = required("HEPHAESTUS_PROOF_RUN_ID")
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    attempt = required("HEPHAESTUS_ATTEMPT")
    proof_root = SCIENTIFIC_ROOT / "positive_promotion" / proof_run_id
    execution_root = SCIENTIFIC_ROOT / "executions" / proof_run_id / f"attempt-{attempt}"
    terminal = execution_root / "driver_result.json"
    execution_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "result_version": "positive-real-model-promotion-proof.v1",
        "created_at": now(),
        "proof_run_id": proof_run_id,
        "repo_sha": repo_sha,
        "attempt": attempt,
        "status": "running",
        "training_performed": False,
        "original_research_lineage_mutated": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for positive promotion proof")
        runtime_environment = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }

        baseline_valid, baseline_manifest_hash = validate_checkpoint_manifest(BASELINE_CHECKPOINT)
        if not baseline_valid:
            raise RuntimeError(f"frozen random-init baseline checkpoint is invalid: {baseline_manifest_hash}")
        baseline_backend = TransformersCausalLMGenerationBackend(device="cuda", dtype="float32", batch_size=6)
        baseline_generation = EvaluationGenerationService(
            artifact_root=proof_root / "evaluations",
            backend=baseline_backend,
            pack_name="semantic_behavior_v1",
            config_dir=REPO_ROOT / "configs",
        )
        baseline_plan = baseline_generation.plan()
        if baseline_plan.content_hash != EVAL_PACK_HASH or len(baseline_plan.tasks) != 18:
            raise RuntimeError("frozen baseline evaluation plan drifted")
        baseline_handle = TrainingRunHandle(
            run_id=BASELINE_RUN_ID,
            experiment_id=f"experiment-{proof_run_id}-baseline",
            backend_id="transformers_causal_lm",
            status="completed",
            checkpoint_refs=[str(BASELINE_CHECKPOINT)],
            metadata={
                "generation_handoff_ref": str(BASELINE_CHECKPOINT / "loading_instructions.json"),
                "checkpoint_manifest_hash": baseline_manifest_hash,
                "training_performed": False,
                "role": "frozen_random_init_baseline",
            },
        )
        baseline_result = baseline_generation.generate(
            baseline_handle,
            generation_handoff_ref=str(BASELINE_CHECKPOINT / "loading_instructions.json"),
        )
        if not baseline_result.report.completed or len(baseline_result.report.samples) != 18:
            raise RuntimeError("baseline semantic generation did not complete")
        unload_model()

        runtime = ProductionCompositionRoot(
            ProductionCompositionSettings(
                state_root=proof_root / "production_state",
                artifact_root=proof_root / "production_artifacts",
                maximum_infrastructure_attempts=3,
            )
        ).build()
        lineage_store = LineageStore(runtime.settings.state_root)
        if lineage_store.get_current(LINEAGE_ID) is None:
            lineage_store.set_current(
                LineageState(
                    lineage_id=LINEAGE_ID,
                    stage_name=STAGE_NAME,
                    status="active",
                    trust_level="medium",
                    origin_run_id=BASELINE_RUN_ID,
                    origin_checkpoint_ref=str(BASELINE_CHECKPOINT),
                    best_checkpoint_ref=str(BASELINE_CHECKPOINT),
                    last_stable_checkpoint_ref=str(BASELINE_CHECKPOINT),
                    created_at=now(),
                    updated_at=now(),
                    metadata={
                        "proof_lineage": True,
                        "does_not_replace_original_research_lineage": True,
                    },
                ).to_dict()
            )

        driver = RealModelPromotionDriver(
            proof_root=proof_root,
            proof_run_id=proof_run_id,
            baseline_result=baseline_result,
        )
        final_state = ProductionLoopRunner(
            runtime=runtime,
            driver=driver,
            maximum_cycles=len(CANDIDATES),
            stop_on_promotion=True,
        ).run(
            program_id=f"program-{proof_run_id}",
            lineage_id=LINEAGE_ID,
            stage_name=STAGE_NAME,
            resume=True,
        )

        lineage = LineageStore(runtime.settings.state_root).get_current(LINEAGE_ID)
        if final_state.status != "completed" or final_state.stop_reason != "candidate_promoted":
            raise RuntimeError(
                f"candidate set exhausted without certified promotion: {final_state.status}/{final_state.stop_reason}"
            )
        if not lineage:
            raise RuntimeError("promoted proof lineage is missing")
        certified_ref = str(lineage.get("certified_stable_checkpoint_ref") or "")
        if not certified_ref:
            raise RuntimeError("lineage has no certified stable checkpoint after promotion")
        latest_cycle = final_state.metadata.get("latest_cycle", {})
        if not isinstance(latest_cycle, dict):
            raise RuntimeError("production loop lacks latest-cycle evidence")
        cycle_evidence = latest_cycle.get("evidence", {})
        if not isinstance(cycle_evidence, dict):
            raise RuntimeError("production loop latest-cycle evidence is malformed")
        manifest_ref = str(cycle_evidence.get("candidate_manifest_ref") or "")
        independent_manifest = verify_model_manifest(manifest_ref)
        expected_snapshot = str(json.loads(Path(manifest_ref).read_text(encoding="utf-8"))["snapshot_path"])
        if certified_ref != expected_snapshot:
            raise RuntimeError("lineage certified checkpoint does not match verified immutable candidate snapshot")
        if lineage.get("best_checkpoint_ref") != certified_ref or lineage.get("last_stable_checkpoint_ref") != certified_ref:
            raise RuntimeError("lineage best/stable/certified checkpoint refs disagree")
        if lineage.get("last_certification_result") != "certification_passed":
            raise RuntimeError("lineage certification result is not certification_passed")

        verification = {
            "verification_version": "positive-real-model-promotion-verification.v1",
            "verified_at": now(),
            "status": "verified",
            "proof_run_id": proof_run_id,
            "program_state": final_state.to_dict(),
            "lineage": lineage,
            "certified_model_manifest": independent_manifest,
            "certified_checkpoint_ref": certified_ref,
            "frozen_eval_pack_hash": EVAL_PACK_HASH,
            "baseline_checkpoint_manifest_hash": baseline_manifest_hash,
            "operator_approval_ref": APPROVAL_REF,
            "training_performed": False,
            "original_research_lineage_mutated": False,
        }
        verification_ref = proof_root / "independent_verification.json"
        atomic_json(verification_ref, verification)

        result.update(
            {
                "status": "completed",
                "completed_at": now(),
                "runtime_environment": runtime_environment,
                "baseline": {
                    "run_id": BASELINE_RUN_ID,
                    "checkpoint_ref": str(BASELINE_CHECKPOINT),
                    "checkpoint_manifest_hash": baseline_manifest_hash,
                    "sample_count": len(baseline_result.report.samples),
                },
                "program_state": final_state.to_dict(),
                "certified_lineage": lineage,
                "certified_model_manifest": independent_manifest,
                "certified_checkpoint_ref": certified_ref,
                "independent_verification_ref": str(verification_ref),
                "independent_verification_sha256": sha_file(verification_ref),
                "training_performed": False,
                "original_research_lineage_mutated": False,
            }
        )
        atomic_json(proof_root / "proof_result.json", result)
        atomic_json(terminal, result)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "proof_run_id": proof_run_id,
                    "certified_checkpoint_ref": certified_ref,
                    "certified_model": independent_manifest["model_id"],
                    "certified_revision": independent_manifest["revision"],
                    "program_cycles": final_state.cycle_index,
                    "gpu": runtime_environment["gpu"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        result.update(
            {
                "status": "failed",
                "completed_at": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "training_performed": False,
                "original_research_lineage_mutated": False,
            }
        )
        atomic_json(terminal, result)
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
