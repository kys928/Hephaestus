"""Real but NONCERTIFIED bounded CPU development cycle.

Use Hephaestus's existing frozen pack loader, deterministic behavioral scoring,
and Judge policy. Train a genuine byte-level GRU causal model with autograd on
original, reproducibly generated arithmetic records. This is a deliberately
limited integration experiment, NOT the 45-slot governed campaign, not an
HF data acquisition, and never a scientific certification or promotion.

No external source files, hosted inference APIs, protected evaluation data,
dataset acquisition approvals, synthetic approval records, or GPU allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.scoring.behavioral import evaluate_behavioral_sample

SCHEMA = "hephaestus-cpu-experimental-full-cycle.v1"
SOURCE_SCHEMA = "original-arithmetic-json-records.v1"
SEED = 1729
MAX_STEPS = 85
GENERATED_LICENSE_BASIS = (
    "Original deterministic arithmetic facts and strings generated entirely "
    "by this repository algorithm. No external examples or copyrighted corpus."
)
FROZEN_PACK_NAME = "semantic_behavior_v1"
EVAL_TASK_FIELDS = (
    "generation_probes", "continuation_prompts", "regression_prompts",
    "structure_tests", "repetition_checks", "length_termination_checks",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return "sha256:" + sha256(raw).hexdigest()


def save_json(path: Path, obj: object) -> str:
    raw = canonical(obj) + b"\n"
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    os.replace(tmp, path)
    return digest(raw)


def source_data() -> list[dict[str, Any]]:
    """Create genuine trainable examples without importing third-party rights.

    Values are distinct; the model must learn from gradients rather than a
    test double. Evaluation uses disjoint example *identities*, with a common
    intentionally shared instruction template. No benchmark-secrecy claim.
    """
    items = []
    for left in range(10):
        for right in range(10):
            prompt = f"Return JSON with integer sum for {left} plus {right}."
            target = json.dumps({"sum": left + right}, separators=(",", ":"))
            items.append({
                "id": f"addition-{left:02d}-{right:02d}",
                "prompt": prompt, "response": target,
                "operation": "addition", "left": left, "right": right,
            })
    return items


def isolate_data(items: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict]:
    """Predeclare reproducible split; reject exact and normalized pair overlap."""
    if len(items) != 100 or len({x["id"] for x in items}) != len(items):
        raise ValueError("unexpected source size or duplicate IDs")
    # Heldout identity selection chosen independently of model/optimizer
    # outcomes, using a deterministic SHA predicate rather than cherry-pick.
    ranked = sorted(items, key=lambda row: sha256(
        (f"{SEED}:{row['id']}").encode()).hexdigest())
    eval_rows = ranked[:20]
    train = ranked[20:]
    train_ids, eval_ids = ({x["id"] for x in group} for group in (train, eval_rows))
    if train_ids & eval_ids:
        raise ValueError("training and validation ID overlap")
    pair = lambda r: (" ".join(r["prompt"].casefold().split()),
                      " ".join(r["response"].casefold().split()))
    train_pairs = {pair(x) for x in train}
    eval_pairs = {pair(x) for x in eval_rows}
    if train_pairs & eval_pairs:
        raise ValueError("training and validation prompt/target contamination")
    checks = {
        "train_examples": len(train), "heldout_examples": len(eval_rows),
        "train_id_overlap": len(train_ids & eval_ids),
        "exact_prompt_target_overlap": len(train_pairs & eval_pairs),
        "template_overlap_expected": True,
        "public_development_evaluation_only": True,
        "semantic_contamination_not_certified": True,
    }
    return train, eval_rows, checks


def plan_evaluation(source: list[dict], heldout: list[dict],
                    frozen: dict[str, Any]) -> dict[str, Any]:
    """Automatically select checks from actual dataset schema *before* training."""
    schema = set(source[0])
    if not {"prompt", "response", "operation", "left", "right"} <= schema:
        raise ValueError("unsupported source schema; fail closed rather than guess")
    if set(r["operation"] for r in source) != {"addition"}:
        raise ValueError("unsupported operation")
    if any(json.loads(r["response"]) != {"sum": r["left"] + r["right"]} for r in source):
        raise ValueError("unverified ground-truth answers")
    if not frozen["content_hash_verified"] or not frozen["frozen"]:
        raise ValueError("original generic regression pack must be frozen and verified")
    tasks = [{
        "task_id": row["id"],
        "prompt": row["prompt"],
        "checks": [
            {"name": "valid_typed_json", "type": "json_object",
             "required_keys": ["sum"], "required_types": {"sum": "integer"},
             "dimension": "structured_format", "hard": True},
            {"name": "correct_sum", "type": "json_object",
             "required_keys": ["sum"], "expected_values": {"sum": row["left"] + row["right"]},
             "dimension": "answer_correctness", "hard": True},
            {"name": "length", "type": "max_words", "value": 6,
             "dimension": "bounded_length", "hard": True},
        ],
        "expected_sum": row["left"] + row["right"],
    } for row in heldout]
    return {
        "schema": "hephaestus-derived-task-evaluation-plan.v1",
        "source_type": SOURCE_SCHEMA,
        "task_family": "structured_json_arithmetic",
        "selection_rule": "actual_prompt_response_and_arithmetic_ground_truth_schema",
        "tasks": tasks,
        "training_seed": SEED,
        "target_metrics": ["structured_format", "answer_correctness", "bounded_length"],
        "regression_pack_name": FROZEN_PACK_NAME,
        "regression_pack_id": frozen["eval_pack_id"],
        "regression_pack_canonical_sha256": frozen["content_hash"],
        "regression_pack_unchanged": True,
        "created_before_training": True,
        "holdout_claimed": False,
        "certification_allowed": False,
    }


class ByteGRULM:
    def __init__(self):
        import torch
        self.net = torch.nn.ModuleDict({
            "embedding": torch.nn.Embedding(256, 24),
            "rnn": torch.nn.GRU(24, 48, batch_first=True),
            "head": torch.nn.Linear(48, 256),
        })

    def forward(self, ids):
        z = self.net["embedding"](ids)
        z, _ = self.net["rnn"](z)
        return self.net["head"](z)


def make_training_batch(rows: list[dict]):
    import torch
    # One causal stream per record; losses computed on assistant answer and
    # explicit end-of-answer delimiter, never on the instruction prefix.
    xs, labels = [], []
    for row in rows:
        prefix = ("User: " + row["prompt"] + "\nAssistant: ").encode("ascii")
        answer = (row["response"] + "\n").encode("ascii")
        data = list(prefix + answer)
        inputs = data[:-1]
        targets = data[1:]
        mask = [False] * (len(prefix) - 1) + [True] * len(answer)
        if len(inputs) != len(mask) or not mask[-1]:
            raise ValueError("causal assistant-target offset failed")
        xs.append(inputs)
        labels.append([token if enabled else -100
                       for token, enabled in zip(targets, mask)])
    max_length = max(map(len, xs))
    padded = [v + [0] * (max_length - len(v)) for v in xs]
    target_pad = [v + [-100] * (max_length - len(v)) for v in labels]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(target_pad, dtype=torch.long)


def generate(net: ByteGRULM, prompt: str, *, limit: int = 24) -> str:
    import torch
    prefix = ("User: " + prompt + "\nAssistant: ").encode("ascii")
    if limit > 64 or len(prefix) > 160:
        raise ValueError("bounded generation limit exceeded")
    result = bytearray()
    net.net.eval()
    with torch.inference_mode():
        x = torch.tensor([[int(v) for v in prefix]], dtype=torch.long)
        _, hidden = net.net["rnn"](net.net["embedding"](x))
        current = torch.tensor([[prefix[-1]]], dtype=torch.long)
        # Prefix has already been consumed. Use its last hidden state to
        # produce the first next token instead of feeding the last byte twice.
        for _ in range(limit):
            logits = net.net["head"](hidden[-1])
            token = int(logits.argmax(dim=-1).item())
            if token == 10:
                break
            if token < 32 or token > 126:
                # Keep real invalid bytes visible as escaped text; never
                # normalize into a valid-looking JSON answer.
                result.extend(f"\\x{token:02x}".encode("ascii"))
            else:
                result.append(token)
            embedding = net.net["embedding"](
                torch.tensor([[token]], dtype=torch.long))
            _, hidden = net.net["rnn"](embedding, hidden)
    return result.decode("ascii")


def run_generations(net: ByteGRULM, tasks: list[dict], *, seed: int) -> dict:
    import torch
    torch.manual_seed(seed)
    samples = []
    for task in tasks:
        output = generate(net, task["prompt"])
        score = evaluate_behavioral_sample(task, output, seed)
        samples.append({
            "task_id": task["task_id"], "seed": seed,
            "prompt_sha256": digest(task["prompt"].encode()),
            "output": output, "output_sha256": digest(output.encode()),
            "checks": score.to_dict(),
        })
    return {
        "sample_count": len(samples),
        "hard_pass_count": sum(s["checks"]["deterministic_passed"] for s in samples),
        "mean_score": sum(s["checks"]["overall_score"] for s in samples) / max(1, len(samples)),
        "samples": samples,
    }


def frozen_regressions(net: ByteGRULM, frozen: dict, *, seed: int) -> dict:
    pack = frozen["eval_pack"]
    tasks = []
    for field in EVAL_TASK_FIELDS:
        tasks.extend(pack.get(field, []))
    if len(tasks) < 6:
        raise ValueError("original evaluation tasks missing")
    result = run_generations(net, tasks, seed=seed)
    result["frozen_pack_hash"] = frozen["content_hash"]
    result["original_task_count"] = len(tasks)
    return result


def run_cycle(out: Path, *, steps: int = MAX_STEPS) -> dict:
    """Single resumed-state-free CPU experiment; fail closed at each stage."""
    import torch
    if not 1 <= steps <= MAX_STEPS:
        raise ValueError("invalid CPU training cap")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("cycle destination must be empty; never overwrite evidence")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen = load_eval_pack(FROZEN_PACK_NAME)
    if not frozen.get("content_hash_verified") or not frozen.get("frozen"):
        raise PermissionError("frozen reference pack is not verified")
    # Persist original content hash; do not edit configs/eval_packs.
    source = source_data()
    train, heldout, screening = isolate_data(source)
    source_ref = save_json(out / "source_provenance.json", {
        "schema": SOURCE_SCHEMA, "seed": SEED, "rows": len(source),
        "rights_basis": GENERATED_LICENSE_BASIS,
        "external_dataset_acquired": False,
        "operator_dataset_approval": "not_applicable_to_original_development_only_examples",
        "experimental_use_permitted": True,
        "production_use_authorized": False,
        "validation": screening,
        "dataset_sha256": digest(canonical(source)),
    })
    plan = plan_evaluation(source, heldout, frozen)
    plan_ref = save_json(out / "evaluation_plan.json", plan)
    # Input/label batch materialized strictly after the precommitted plan.
    x, y = make_training_batch(train)
    net = ByteGRULM()
    net.net.eval()
    baseline = run_generations(net, plan["tasks"], seed=SEED)
    baseline_reg = frozen_regressions(net, frozen, seed=SEED)
    base_ref = save_json(out / "baseline_evaluation.json", {
        "target": baseline, "frozen_regressions": baseline_reg,
        "evaluation_plan_ref": plan_ref,
    })
    # A real gradient-updated model, not mocked loss or predicted outputs.
    net.net.train()
    params_before = [p.detach().clone() for p in net.net.parameters()]
    optimizer = torch.optim.AdamW(net.net.parameters(), lr=0.012, weight_decay=0.0)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = net.forward(x)
        loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite real training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.net.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))
    delta = sum(float(torch.linalg.vector_norm(a.detach() - b).item())
                for a, b in zip(net.net.parameters(), params_before))
    if not math.isfinite(delta) or delta <= 0:
        raise ArithmeticError("optimizer performed no real weight update")
    checkpoint = out / "trained_cpu_gru_weights.pt"
    torch.save(net.net.state_dict(), checkpoint)
    saved_sha = digest(checkpoint.read_bytes())
    reload_net = ByteGRULM()
    reload_net.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True),
                                   strict=True)
    reload_net.net.eval()
    candidate = run_generations(reload_net, plan["tasks"], seed=SEED)
    candidate_reg = frozen_regressions(reload_net, frozen, seed=SEED)
    candidate_ref = save_json(out / "candidate_evaluation.json", {
        "target": candidate, "frozen_regressions": candidate_reg,
        "evaluation_plan_ref": plan_ref, "trained_weights_sha256": saved_sha,
    })
    if candidate_reg["frozen_pack_hash"] != baseline_reg["frozen_pack_hash"]:
        raise AssertionError("baseline/candidate frozen pack drifted")
    difference = candidate["mean_score"] - baseline["mean_score"]
    regression_delta = candidate_reg["mean_score"] - baseline_reg["mean_score"]
    hard_regression = candidate_reg["hard_pass_count"] < baseline_reg["hard_pass_count"]
    # A single development run never has enough evidence for promotion;
    # existing JudgePolicy chooses a finite, nonpromotion action.
    judge_action = JudgePolicy().decide_exit_action(
        deterministic_passed=not hard_regression,
        confidence=0.0, monitor_outcome="healthy",
        promotion_state=("rejected" if hard_regression else "inconclusive"),
        has_candidate_checkpoint=True,
    )
    if str(judge_action.value) not in {"reject_checkpoint", "continue_lineage_best"}:
        raise PermissionError("development cycle cannot issue promotion or approval")
    verdict = {
        "schema": "hephaestus-development-judge-decision.v1",
        "target_score_delta": difference,
        "frozen_regression_score_delta": regression_delta,
        "hard_frozen_regression": hard_regression,
        "target_evidence": "measured_model_text_on_precommitted_heldout_prompts",
        "result": ("regression_observed" if hard_regression else
                   "target_improved_noncertified" if difference > 0 else
                   "no_measured_target_improvement"),
        "judge_policy_action": judge_action.value,
        "recommendation": ("investigate_regression" if hard_regression else
                           "gather_repeatability_evidence" if difference > 0 else
                           "diagnose_undertraining_or_data"),
        "human_review_completed": False, "training_approved_for_production": False,
        "promotion_authorized": False, "scientific_certification": False,
        "independent_holdout_examined": False,
        "original_45_campaign_advanced": False,
    }
    judge_ref = save_json(out / "judge_verdict.json", verdict)
    manifest = {
        "schema": SCHEMA, "status": "completed_experimental_noncertified",
        "dataset_source_ref": source_ref,
        "evaluation_plan_ref": plan_ref,
        "baseline_eval_ref": base_ref, "candidate_eval_ref": candidate_ref,
        "judge_ref": judge_ref,
        "checkpoint_sha256": saved_sha,
        "optimizer": {"algorithm": "AdamW", "steps": steps,
                      "first_loss": losses[0], "last_loss": losses[-1],
                      "total_parameter_delta_l2_sum": delta, "seed": SEED},
        "original_frozen_eval_pack_sha256": frozen["content_hash"],
        "dataset_policy": {"source": "self_generated_no_third_party",
                           "external_data_access": False,
                           "rights_evidence": GENERATED_LICENSE_BASIS},
        "stage_receipts": {
            "source_discovery": source_ref, "screening": source_ref,
            "evaluation_planning_pretraining": plan_ref,
            "baseline_generation": base_ref,
            "real_optimization_checkpoint": saved_sha,
            "candidate_generation": candidate_ref,
            "evaluation_and_judge": judge_ref,
        },
        "no_dataset_acquisition_approval_forged": True,
        "original_campaign": "11/45_unchanged",
        "frozen_gpu_roles_used": False,
    }
    final_ref = save_json(out / "cycle_manifest.json", manifest)
    # Independent local replay verification over every recorded referenced file.
    replay = verify_cycle(out)
    replay_ref = save_json(out / "replay_verification.json", replay)
    return {"manifest": final_ref, "replay": replay_ref, "judge": verdict,
            "optimizer_steps": steps, "loss_before": losses[0],
            "loss_after": losses[-1], "weight_delta": delta,
            "target_delta": difference, "regression_delta": regression_delta,
            "checkpoint_sha256": saved_sha, "output": str(out),
            "development_only": True, "original_campaign_advanced": False}


def verify_cycle(root: Path) -> dict:
    report = json.loads((root / "cycle_manifest.json").read_text())
    mapping = {
        "dataset_source_ref": "source_provenance.json",
        "evaluation_plan_ref": "evaluation_plan.json",
        "baseline_eval_ref": "baseline_evaluation.json",
        "candidate_eval_ref": "candidate_evaluation.json",
        "judge_ref": "judge_verdict.json",
        "checkpoint_sha256": "trained_cpu_gru_weights.pt",
    }
    for key, filename in mapping.items():
        got = digest((root / filename).read_bytes())
        if got != report[key]:
            raise ValueError(f"cycle replay content mismatch: {filename}")
    plan = json.loads((root / "evaluation_plan.json").read_text())
    b = json.loads((root / "baseline_evaluation.json").read_text())
    c = json.loads((root / "candidate_evaluation.json").read_text())
    if (b["evaluation_plan_ref"] != report["evaluation_plan_ref"]
            or c["evaluation_plan_ref"] != report["evaluation_plan_ref"]
            or c["trained_weights_sha256"] != report["checkpoint_sha256"]
            or plan["regression_pack_canonical_sha256"] != report["original_frozen_eval_pack_sha256"]
            or b["frozen_regressions"]["frozen_pack_hash"]
                != c["frozen_regressions"]["frozen_pack_hash"]):
        raise ValueError("cycle replay frozen plan or source lineage mismatch")
    if report["original_campaign"] != "11/45_unchanged" or not report["no_dataset_acquisition_approval_forged"]:
        raise ValueError("cycle may not rewrite protected campaign evidence")
    return {"schema": "hephaestus-cpu-development-replay.v1",
            "valid": True, "verified_files": len(mapping),
            "candidate_checkpoint_verified": True,
            "baseline_candidate_frozen_pack_equal": True,
            "autonomous_development_only": True,
            "original_campaign_unmodified": True}
