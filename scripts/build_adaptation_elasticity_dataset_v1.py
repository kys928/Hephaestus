#!/usr/bin/env python3
"""Build the governed, deterministic training set for adaptation elasticity V1."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TOPOLOGY_SPEC = Path("configs/eval_packs/hephaestus_cognitive_topology_v1.json")
PROTOCOL = Path("configs/experiments/hephaestus_adaptation_elasticity_v1.json")
ROLE_RULES = {
    "diagnosis": "Diagnose from evidence without pretending certainty. Preserve missing/conflicting evidence and separate runtime/data/config failures from scientific regression.",
    "planner": "Propose but never execute. Change one primary variable at a time, avoid repeated dead ends, and prefer information gain when evidence is incomplete.",
    "evaluator": "Classify evidence honestly. Incomplete evidence is not rejection; hard deterministic regressions dominate flattering averages; protocol integrity is mandatory.",
    "judge": "Apply promotion and lineage policy conservatively. Hard regressions block promotion, incomplete evidence retains the safer lineage, and approval boundaries remain binding.",
    "controller": "Apply the action registry exactly. Auto-allowed actions can proceed; approval-required actions need approval; forbidden and unknown actions stay blocked.",
}
REQUIRED_KEYS = ["decision", "action", "primary_variable", "confidence", "evidence_refs", "uncertainties", "rationale"]


def training_user_prompt(example: dict[str, Any]) -> str:
    role = str(example["role"])
    evidence = "\n".join(f"- {row['ref']}: {row['fact']}" for row in example["evidence"])
    allowed = ", ".join(example["allowed_evidence_refs"])
    return f"""You are acting only as the Hephaestus {role.upper()} role.

ROLE BOUNDARY:
{ROLE_RULES[role]}

SITUATION:
{example['situation']}

EVIDENCE:
{evidence}

Return exactly one JSON object and nothing else. Do not use markdown or code fences.
The object must contain exactly these keys: {', '.join(REQUIRED_KEYS)}.
- decision: string
- action: string
- primary_variable: string
- confidence: JSON number from 0 to 1
- evidence_refs: JSON array of evidence ref strings; cite only evidence that materially supports the decision from: {allowed}
- uncertainties: JSON array of short strings; use [] when no material uncertainty remains
- rationale: one concise string; do not invent evidence

Do not reveal hidden reasoning. Give only the requested decision record."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _norm(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_norm(a)), set(_norm(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, " ".join(_norm(a)), " ".join(_norm(b))).ratio()


def _target(decision: str, action: str, primary: str, confidence: float, refs: list[str], uncertainties: list[str], rationale: str) -> dict[str, Any]:
    return {"decision": decision, "action": action, "primary_variable": primary, "confidence": confidence, "evidence_refs": refs, "uncertainties": uncertainties, "rationale": rationale}


def _example(role: str, family: str, variant: int, situation: str, facts: list[str], target: dict[str, Any]) -> dict[str, Any]:
    refs = [f"TR-{role[:1].upper()}-{family.upper()}-{variant}-{idx+1}" for idx in range(len(facts))]
    evidence = [{"ref": ref, "fact": fact} for ref, fact in zip(refs, facts)]
    target = dict(target)
    if target.get("evidence_refs") == ["FIRST"]:
        target["evidence_refs"] = refs[:1]
    elif target.get("evidence_refs") == ["FIRST2"]:
        target["evidence_refs"] = refs[:2]
    elif target.get("evidence_refs") == ["LAST"]:
        target["evidence_refs"] = refs[-1:]
    elif target.get("evidence_refs") == ["ALL"]:
        target["evidence_refs"] = refs
    return {"example_id": f"TR-{role.upper()}-{family.upper()}-{variant}", "role": role, "situation": situation, "evidence": evidence, "allowed_evidence_refs": refs, "target": target}


def _build_role(role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for v in range(1, 7):
        n = 100 + v
        if role == "diagnosis":
            families = [
                ("checksum", f"A materialized checkpoint shard fails its recorded SHA-256 during readback on attempt {v}; no semantic evaluation has started.", [f"Shard part-{v:02d} hash differs from its signed manifest.", "No candidate generations or scorecards exist yet."], _target("artifact_integrity", "repair_artifact", "artifact_integrity", 0.98, ["FIRST2"], [], "The verified byte mismatch is sufficient to classify this as artifact integrity, not model quality.")),
                ("storage", f"A remote worker loses its object-store connection after {n} seconds and writes only 7 of 20 required generation records.", ["Object-store requests return repeated transport timeouts.", "Only 7/20 required generation records are present."], _target("runtime_incomplete", "retry_runtime", "runtime", 0.94, ["FIRST2"], [], "The run is incomplete because execution/storage failed before required evidence was produced.")),
                ("provenance", f"A dataset bundle is byte-complete and training has not begun, but its source-license receipt is missing for source batch {v}.", ["Dataset bytes and manifest hashes match.", "Required source-license receipt is absent."], _target("provenance", "repair_provenance", "provenance", 0.96, ["LAST"], [], "The blocking defect is missing provenance evidence, not a scientific model regression.")),
                ("optimizer_state", f"Resume attempt {v} fails while loading optimizer state before the first update; model weights and tokenizer hashes still match the parent checkpoint.", ["Model and tokenizer manifests match the parent checkpoint.", "Optimizer state deserialization fails before any training step."], _target("checkpoint_integrity", "rollback_to_checkpoint", "checkpoint_lineage", 0.95, ["FIRST2"], [], "Failure occurs at checkpoint-state restoration before training can generate new scientific evidence.")),
                ("ambiguous_mix", f"After a corpus mixture refresh, quality falls by {v+3} points, but augmentation, order, and sampling weights also changed together and no ablation exists.", ["Several data variables changed in the same run.", "The score decrease is reproducible but no single-variable ablation exists."], _target("inconclusive", "collect_more_evidence", "evidence", 0.45, ["FIRST2"], ["causal variable is not isolated"], "The regression is real, but attribution is not supported because multiple data variables changed together.")),
                ("scheduler", f"Run {v} was configured for cosine decay but execution metadata shows a constant learning rate from step 0 through step {n}.", ["Approved recipe specifies cosine decay.", "Recorded optimizer learning rate remains constant for the observed steps."], _target("training_config", "repair_training_config", "scheduler_config", 0.97, ["FIRST2"], [], "Execution differs from the approved scheduler contract.")),
                ("mount", f"Worker {v} resolves the checkpoint path to an empty local mount while the network-volume manifest shows the checkpoint exists remotely.", ["Expected checkpoint files exist on the governed network volume.", "Worker mount path contains no checkpoint files."], _target("runtime_storage_mount", "repair_runtime", "storage_mount", 0.96, ["FIRST2"], [], "The evidence isolates a runtime mount/path failure rather than a model failure.")),
                ("metrics_gap", f"Training attempt {v} completes, but gradient norms and validation loss are absent while only a final scalar reward is available.", ["Required training-dynamics metrics are missing.", "A final reward exists but cannot localize a failure domain."], _target("inconclusive", "collect_more_evidence", "evidence", 0.35, ["FIRST2"], ["training dynamics are unobserved"], "The available scalar is insufficient to diagnose the training failure domain.")),
            ]
        elif role == "planner":
            families = [
                ("lr_instability", f"Diagnosis for branch {v} isolates learning-rate instability with 0.93 confidence; data, tokenizer, architecture, and evaluation are verified unchanged.", ["Gradient spikes begin immediately after the scheduled learning-rate peak.", "All non-optimizer contracts match the stable parent run."], _target("change_training_recipe", "branch_new_experiment", "learning_rate", 0.93, ["FIRST2"], [], "A bounded branch that changes only learning rate directly tests the diagnosed instability.")),
                ("dedup", f"Diagnosis finds duplicate-example concentration in shard group {v}; model/runtime evidence is healthy and no preprocessing change has yet been tested.", ["Near-duplicate rate is 38% in the failing shard group.", "Model loading and evaluation infrastructure are healthy."], _target("change_preprocessing", "branch_new_experiment", "deduplication", 0.91, ["FIRST2"], [], "Change only deduplication so the data-quality hypothesis receives a controlled test.")),
                ("missing_logs", f"Experiment {v} has a claimed regression, but the required generation artifact and runtime log are missing from persistent storage.", ["Comparison summary exists without underlying generations.", "Runtime log is missing."], _target("collect_more_evidence", "request_recheck", "evidence", 0.92, ["FIRST2"], [], "Do not alter training while the evidence needed to validate the claimed regression is missing.")),
                ("dead_data_mix", f"The exact 70/30 corpus mixture was attempted twice for lineage {v} and reproduced the same failure; an alternative approved 60/40 mixture has not been tested.", ["Two prior attempts with the exact 70/30 mixture failed the same gate.", "A separately approved 60/40 mixture is available."], _target("change_data_policy", "branch_new_experiment", "data_mixture", 0.86, ["FIRST2"], [], "Avoid the repeated dead end and test one approved mixture variable.")),
                ("throughput", f"Scientific outputs are stable, but dataloader starvation leaves GPU utilization below {40+v}% on worker {v}.", ["Scores and generations are unchanged across repeats.", "GPU traces show idle gaps waiting for batches."], _target("optimize_runtime", "branch_new_experiment", "dataloader_workers", 0.88, ["FIRST2"], [], "Use an infrastructure-only branch changing dataloader concurrency while keeping science frozen.")),
                ("grad_noise", f"Diagnosis attributes unstable updates to too-small effective batch size; learning rate and data are otherwise verified for run {v}.", ["Gradient variance is high at the current accumulation setting.", "No data, tokenizer, or evaluation drift is observed."], _target("change_training_recipe", "branch_new_experiment", "gradient_accumulation", 0.89, ["FIRST2"], [], "Change only effective batch size through accumulation to test the diagnosed update-noise hypothesis.")),
                ("serializer", f"A serving adapter for role {v} prepends explanatory prose before otherwise valid machine records; model outputs without the wrapper are valid JSON.", ["Direct model output is valid JSON.", "Only the serving serializer adds prose before the JSON object."], _target("change_preprocessing", "branch_new_experiment", "output_serializer", 0.96, ["FIRST2"], [], "The wrapper is isolated, so change only serialization rather than retraining the model.")),
                ("arch_family", f"Three controlled runs show a context-window bottleneck in architecture family A for workload {v}; family B is permissively licensed but not yet admitted.", ["Family A reproduces the same context truncation under three seeds.", "Family B exists but lacks Hephaestus admission evidence."], _target("admit_alternative_model", "request_recheck", "model_family", 0.82, ["FIRST2"], ["alternative model is not yet admitted"], "The next information-gain step is model admission, not execution of an unverified family.")),
            ]
        elif role == "evaluator":
            families = [
                ("flat", f"Candidate {v} completes every required sample with matching protocol hash; aggregate and deterministic metrics remain within 0.2 points of baseline.", ["All required samples are complete under the frozen protocol.", "Aggregate and deterministic metrics are statistically unchanged."], _target("no_material_change", "retain_baseline", "evaluation_result", 0.9, ["FIRST2"], [], "Complete evidence shows no material behavioral change.")),
                ("partial", f"Only {8+v}/24 required evaluation samples exist after worker interruption; the existing subset looks strong.", [f"Only {8+v}/24 required samples are present.", "Subset score is high but the required evidence set is incomplete."], _target("incomplete_evidence", "request_recheck", "evidence_completeness", 0.97, ["FIRST2"], [], "A partial subset cannot support promotion or rejection.")),
                ("variance", f"Candidate {v} has mean score 0.81, but repeat scores are 0.93, 0.80, and 0.70 under identical settings.", ["All samples are present.", "Repeat-to-repeat variance is large under identical settings."], _target("unstable_variance", "request_recheck", "repeatability", 0.92, ["FIRST2"], [], "The candidate is not repeatable enough for a stable comparison.")),
                ("safety_gate", f"Candidate {v} improves average quality to 0.91 but fails the frozen unsafe-action deterministic check in every repeat.", ["Aggregate quality improves materially.", "The frozen unsafe-action hard gate fails in all repeats."], _target("regression", "reject_candidate", "deterministic_gate", 0.99, ["FIRST2"], [], "The hard deterministic regression dominates the flattering average.")),
                ("clean_win", f"Candidate {v} completes all samples, passes every hard gate, improves aggregate score by 0.08, and repeat variance stays below the accepted bound.", ["All required samples and protocol hashes match.", "All hard gates pass; aggregate delta is +0.08; repeatability is within policy."], _target("improved", "advance_to_judge", "candidate_quality", 0.97, ["FIRST2"], [], "The complete governed comparison supports an improved classification.")),
                ("baseline_drift", f"Candidate {v} was scored against baseline checkpoint B2 while the frozen comparison contract requires baseline B1.", ["Candidate evidence is complete.", "Observed baseline identity B2 differs from required B1."], _target("invalid_comparison", "repair_evaluation", "baseline_identity", 0.98, ["FIRST2"], [], "The comparison is invalid because the reference baseline drifted.")),
                ("reward_conflict", f"Reward model score rises sharply for candidate {v}, but an exact schema hard gate fails on every seed.", ["Reward score rises by 0.22.", "Exact-schema hard gate fails in every deterministic repeat."], _target("regression", "reject_candidate", "deterministic_gate", 0.99, ["FIRST2"], [], "Deterministic failure cannot be overridden by the reward signal.")),
                ("review_missing", f"Candidate {v} passes deterministic checks, but the policy-required independent review artifact is absent.", ["Deterministic evidence is complete and passing.", "Required independent-review evidence is missing."], _target("incomplete_evidence", "request_recheck", "review_evidence", 0.95, ["FIRST2"], [], "Evaluation cannot become terminal while a required review artifact is missing.")),
            ]
        elif role == "judge":
            families = [
                ("promote", f"Candidate {v} has complete evidence, all hard gates pass, repeatability is sufficient, provenance is verified, and matching high-risk promotion approval is present.", ["Evaluator verdict is improved with all hard gates passing.", "Matching promotion approval and provenance evidence are verified."], _target("promote", "promote_checkpoint", "promotion_decision", 0.98, ["FIRST2"], [], "All scientific and governance gates required for promotion are satisfied.")),
                ("hard_fail", f"Candidate {v} has a high mean score but a frozen deterministic safety gate fails in all repeats.", ["Aggregate score is high.", "A frozen hard safety gate fails deterministically."], _target("reject", "reject_candidate", "deterministic_regression", 0.99, ["FIRST2"], [], "Hard deterministic regression blocks promotion regardless of aggregate quality.")),
                ("incomplete", f"Candidate {v} is missing one required repeat and therefore lacks a complete comparison.", ["Two of three required repeats exist.", "Policy requires all repeats before a terminal candidate decision."], _target("retain", "request_recheck", "evidence_completeness", 0.96, ["FIRST2"], [], "Retain the safer lineage until complete evidence exists.")),
                ("approval_missing", f"Candidate {v} passes science and provenance but no matching high-risk promotion approval record exists.", ["Scientific promotion gates pass.", "High-risk promotion approval is absent."], _target("blocked", "request_recheck", "approval", 0.99, ["FIRST2"], [], "Promotion remains blocked by the missing governance approval.")),
                ("provenance_missing", f"Candidate {v} passes quality gates but its training-data provenance manifest is missing.", ["Quality and repeatability gates pass.", "Required training-data provenance manifest is absent."], _target("blocked", "request_recheck", "provenance", 0.98, ["FIRST2"], [], "Certification cannot proceed without required provenance evidence.")),
                ("review_conflict", f"Candidate {v} passes deterministic gates, but two required reviewers disagree and policy marks unresolved review conflict as non-terminal.", ["Deterministic gates pass.", "Required reviewers disagree and conflict resolution is absent."], _target("retain", "request_recheck", "review_conflict", 0.9, ["FIRST2"], ["review conflict is unresolved"], "Retain the current lineage until the required conflicting evidence is resolved.")),
                ("rollback_invalid", f"Candidate {v} regresses and the proposed rollback target lacks a verified checkpoint manifest.", ["Candidate has a genuine regression.", "Proposed rollback checkpoint has no verified manifest."], _target("reject", "request_recheck", "rollback_target", 0.92, ["FIRST2"], ["safe rollback target is not yet verified"], "Reject the regressed candidate but do not mutate lineage to an unverified rollback target.")),
                ("variance_block", f"Candidate {v} is better on average, but variance exceeds the certification threshold and no additional repeat is available.", ["Mean candidate quality exceeds baseline.", "Repeatability threshold is violated."], _target("retain", "request_recheck", "repeatability", 0.95, ["FIRST2"], [], "Insufficient repeatability blocks a terminal promotion decision.")),
            ]
        else:
            action_sets = [
                ("read_memory", "", "allowed", 0.99),
                ("summarize_lineage", "", "allowed", 0.99),
                ("record_incident", "", "allowed", 0.99),
                ("modify_data_policy", "missing", "blocked", 0.99),
                ("change_stage", "approved", "allowed", 0.99),
                ("archive_lineage", "approved", "allowed", 0.99),
                ("delete_decision_history", "override_approved", "blocked", 1.0),
                ("totally_unknown_action", "approved", "blocked", 0.99),
            ]
            families = []
            for idx, (action_name, approval, decision, conf) in enumerate(action_sets, 1):
                approval_text = "No approval record is present." if approval == "missing" else (f"Approval status is {approval}." if approval else "The action is in the automatic boundary and needs no approval.")
                facts = [f"Requested action is {action_name}.", approval_text]
                families.append((f"boundary{idx}", f"Controller request {v} asks to execute `{action_name}` under the stated governance context.", facts, _target(decision, action_name, "action_boundary", conf, ["FIRST2"], [], "Apply the registered action boundary and approval state exactly.")))
        for family, situation, facts, target in families:
            rows.append(_example(role, family, v, situation, facts, target))
    return rows


def build_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ROLE_RULES:
        rows.extend(_build_role(role))
    return rows


def _case_text(case: dict[str, Any]) -> str:
    return str(case.get("situation", "")) + " " + " ".join(str(row.get("fact", "")) for row in case.get("evidence", []) if isinstance(row, dict))


def contamination_report(rows: list[dict[str, Any]], topology: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    max_j = float(protocol["dataset"]["max_normalized_token_jaccard"])
    max_s = float(protocol["dataset"]["max_sequence_similarity"])
    eval_cases = [case for case in topology["cases"] if isinstance(case, dict)]
    forbidden_tokens = {str(case.get("case_id")) for case in eval_cases}
    forbidden_tokens |= {str(ev.get("ref")) for case in eval_cases for ev in case.get("evidence", []) if isinstance(ev, dict)}
    violations: list[dict[str, Any]] = []
    maxima = {"jaccard": 0.0, "sequence_similarity": 0.0}
    for row in rows:
        text = _case_text(row)
        if any(token and token in text for token in forbidden_tokens):
            violations.append({"example_id": row["example_id"], "reason": "eval_identifier_present"})
        for case in eval_cases:
            reference = _case_text(case)
            j, s = _jaccard(text, reference), _sim(text, reference)
            maxima["jaccard"] = max(maxima["jaccard"], j)
            maxima["sequence_similarity"] = max(maxima["sequence_similarity"], s)
            if j > max_j or s > max_s:
                violations.append({"example_id": row["example_id"], "eval_case_id": case.get("case_id"), "jaccard": j, "sequence_similarity": s})
    return {"checker_version": "adaptation-elasticity-contamination.v1", "reference_protocol_id": topology["protocol_id"], "reference_protocol_sha256": hashlib.sha256(TOPOLOGY_SPEC.read_bytes()).hexdigest(), "example_count": len(rows), "thresholds": {"max_normalized_token_jaccard": max_j, "max_sequence_similarity": max_s}, "observed_maxima": maxima, "violations": violations, "passed": not violations}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="adaptation_elasticity_preflight")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topology = json.loads(TOPOLOGY_SPEC.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = build_dataset()
    expected = int(protocol["dataset"]["examples_per_role"])
    counts = {role: sum(1 for row in rows if row["role"] == role) for role in ROLE_RULES}
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"unexpected role counts: {counts}")
    report = contamination_report(rows, topology, protocol)
    if not report["passed"]:
        raise RuntimeError(f"training/eval contamination gate failed: {report['violations'][:5]}")
    dataset_path = out / "training.jsonl"
    dataset_raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()
    dataset_path.write_bytes(dataset_raw)
    report["dataset_sha256"] = _sha(dataset_raw)
    report["role_counts"] = counts
    (out / "contamination_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"dataset_id": protocol["dataset"]["generator_id"], "dataset_sha256": report["dataset_sha256"], "record_count": len(rows), "role_counts": counts, "source": protocol["dataset"]["source"], "approval_source": protocol["dataset"]["approval_source"], "contamination_passed": True, "frozen_eval_prompt_or_evidence_identifiers_present": False, "schema_keys": REQUIRED_KEYS}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ADAPTATION_DATASET_JSON " + json.dumps(manifest, sort_keys=True))
    print("ADAPTATION_CONTAMINATION_JSON " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
