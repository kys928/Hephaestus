#!/usr/bin/env python3
"""Run the frozen, non-mutating Hephaestus cognitive-topology cohort on one GPU."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from hephaestus.providers.models.admission import RUNTIME_PATTERNS, validate_identity
from hephaestus.providers.models.gpu_snapshot import load_gpu_snapshot

SPEC_PATH = Path(__file__).resolve().parents[1] / "configs/eval_packs/hephaestus_cognitive_topology_v1.json"
SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")
REQUIRED_KEYS = {"decision", "action", "primary_variable", "confidence", "evidence_refs", "uncertainties", "rationale"}

ROLE_RULES = {
    "diagnosis": "Diagnose from evidence without pretending certainty. Missing or conflicting evidence must remain explicit; do not confuse runtime incompleteness with scientific regression.",
    "planner": "Propose but never execute. Prefer one primary variable, avoid known dead ends, and collect evidence when diagnosis is not sufficient for an experiment-changing intervention.",
    "evaluator": "Classify evidence honestly. Zero/partial required samples are incomplete evidence, not model rejection. A frozen hard deterministic failure is a scientific regression even when aggregate score is high.",
    "judge": "Apply finite Judge semantics. Hard deterministic regression blocks promotion. Inconclusive evidence retains the safer lineage. Promotion is high-risk and requires matching approval evidence.",
    "controller": "Apply the action boundary exactly. Auto-allowed actions may proceed without approval; approval-required actions need approval; forbidden actions remain blocked even with override approval; unknown actions are not auto-allowed.",
}


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "--" for ch in value).strip("-.")


def _protocol() -> tuple[dict[str, Any], str]:
    raw = SPEC_PATH.read_bytes()
    return json.loads(raw), _sha256(raw)


def _paths() -> tuple[Path, Path]:
    proof_run_id = _required("HEPHAESTUS_PROOF_RUN_ID")
    attempt = _required("HEPHAESTUS_ATTEMPT")
    return (
        SCIENTIFIC_ROOT / "cognitive_topology" / proof_run_id,
        SCIENTIFIC_ROOT / "executions" / proof_run_id / f"attempt-{attempt}",
    )


def _load_admission(protocol_sha256: str) -> dict[str, Any]:
    repo_sha = _required("HEPHAESTUS_REPO_SHA")
    root = SCIENTIFIC_ROOT / "cognitive_topology" / "model_admission" / repo_sha
    admission = json.loads((root / "admission.json").read_text(encoding="utf-8"))
    requirements = (root / "requirements.txt").read_bytes()
    if admission.get("repo_sha") != repo_sha:
        raise RuntimeError("topology admission repository SHA mismatch")
    if admission.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("topology protocol hash differs from admitted protocol")
    if _sha256(requirements) != admission.get("requirements_sha256"):
        raise RuntimeError("topology dependency lock hash mismatch")
    if admission.get("lineage_mutation_allowed") is not False or admission.get("training_allowed") is not False or admission.get("promotion_allowed") is not False:
        raise RuntimeError("topology admission is not non-mutating")
    return admission


def _materialize(candidate: dict[str, Any], admission: dict[str, Any], proof_root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    validate_identity(candidate)
    info = HfApi().model_info(candidate["model_id"], revision=candidate["revision"], files_metadata=True)
    observed_license = str(getattr(getattr(info, "card_data", None), "license", "") or "").lower()
    if str(info.sha) != candidate["revision"] or observed_license != candidate["license"]:
        raise RuntimeError("immutable model revision/license drift during topology materialization")
    snapshot = Path(snapshot_download(
        repo_id=candidate["model_id"], revision=candidate["revision"], allow_patterns=RUNTIME_PATTERNS,
        cache_dir="/opt/hephaestus-model-materialization/hf_cache",
    )).resolve()
    components = {path.relative_to(snapshot).as_posix(): _sha_file(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    if not any(name.endswith(".safetensors") for name in components):
        raise RuntimeError("topology materialization lacks safetensors")
    admitted = next((row for row in admission["candidates"] if row["model_id"] == candidate["model_id"] and row["revision"] == candidate["revision"]), None)
    if admitted is None:
        raise RuntimeError("runtime candidate was not admitted")
    for name, expected in admitted["metadata"]["metadata_component_hashes"].items():
        if components.get(name) != expected:
            raise RuntimeError(f"admitted metadata drifted: {name}")
    manifest = {
        "manifest_version": "cognitive-topology-model-snapshot.v1",
        "model_id": candidate["model_id"], "revision": candidate["revision"], "license": observed_license,
        "snapshot_path": str(snapshot), "components": components,
        "manifest_hash": "sha256:" + _sha256(json.dumps(components, sort_keys=True, separators=(",", ":")).encode()),
        "byte_size": sum((snapshot / name).stat().st_size for name in components),
        "trust_remote_code": False, "cache_persistence": "ephemeral",
    }
    _atomic_json(proof_root / "model_manifests" / _slug(candidate["model_id"]) / "snapshot_manifest.json", manifest)
    return snapshot, manifest


def _prompt(spec: dict[str, Any], case: dict[str, Any]) -> str:
    schema = spec["response_schema"]
    evidence = "\n".join(f"- {row['ref']}: {row['fact']}" for row in case["evidence"])
    allowed = ", ".join(case["allowed_evidence_refs"])
    return f"""You are acting only as the Hephaestus {case['role'].upper()} role.

ROLE BOUNDARY:
{ROLE_RULES[case['role']]}

SITUATION:
{case['situation']}

EVIDENCE:
{evidence}

Return exactly one JSON object and nothing else. Do not use markdown or code fences.
The object must contain exactly these keys: {', '.join(schema['required_exact_keys'])}.
- decision: string
- action: string
- primary_variable: string
- confidence: JSON number from 0 to 1
- evidence_refs: JSON array of evidence ref strings. Cite only evidence that materially supports your decision. Relevant refs for scoring are a subset of: {allowed}
- uncertainties: JSON array of short strings; use [] when no material uncertainty remains
- rationale: one concise string; do not invent evidence

Do not reveal hidden reasoning. Give only the requested decision record."""


def _strict_parse(raw: str) -> tuple[dict[str, Any] | None, bool]:
    try:
        payload = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False
    if not isinstance(payload, dict) or set(payload) != REQUIRED_KEYS:
        return payload if isinstance(payload, dict) else None, False
    valid = (
        isinstance(payload["decision"], str)
        and isinstance(payload["action"], str)
        and isinstance(payload["primary_variable"], str)
        and isinstance(payload["confidence"], (int, float)) and not isinstance(payload["confidence"], bool)
        and 0.0 <= float(payload["confidence"]) <= 1.0
        and isinstance(payload["evidence_refs"], list) and all(isinstance(item, str) for item in payload["evidence_refs"])
        and isinstance(payload["uncertainties"], list) and all(isinstance(item, str) for item in payload["uncertainties"])
        and isinstance(payload["rationale"], str)
    )
    return payload, bool(valid)


def score_response(spec: dict[str, Any], case: dict[str, Any], raw: str) -> dict[str, Any]:
    payload, schema_ok = _strict_parse(raw)
    expected = case["expected"]
    components = {"schema": 1.0 if schema_ok else 0.0, "decision": 0.0, "action": 0.0, "primary_variable": 0.0,
                  "evidence_grounding": 0.0, "confidence_calibration": 0.0, "forbidden_claim_avoidance": 1.0}
    hallucination_rate = 1.0 if payload is None else 0.0
    if payload is not None:
        components["decision"] = float(payload.get("decision") == expected["decision"])
        components["action"] = float(payload.get("action") == expected["action"])
        components["primary_variable"] = float(payload.get("primary_variable") == expected["primary_variable"])
        confidence = payload.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            components["confidence_calibration"] = float(expected["confidence_min"] <= float(confidence) <= expected["confidence_max"])
        cited = payload.get("evidence_refs") if isinstance(payload.get("evidence_refs"), list) else []
        cited = [item for item in cited if isinstance(item, str)]
        allowed = set(map(str, case["allowed_evidence_refs"]))
        if cited:
            valid = sum(1 for ref in cited if ref in allowed)
            components["evidence_grounding"] = valid / len(cited)
            hallucination_rate = 1.0 - valid / len(cited)
        else:
            components["evidence_grounding"] = 0.0
            hallucination_rate = 0.0
    lowered = raw.casefold()
    hits = [claim for claim in case.get("forbidden_claims", []) if str(claim).casefold() in lowered]
    components["forbidden_claim_avoidance"] = 0.0 if hits else 1.0
    weights = spec["scoring"]
    quality = sum(float(weights[name]) * value for name, value in components.items())
    return {
        "quality": quality, "quality_100": quality * 100.0, "components": components,
        "schema_compliant": schema_ok, "hallucination_rate": hallucination_rate,
        "forbidden_claim_hits": hits, "parsed": payload,
    }


def _generate_one(model: Any, tokenizer: Any, prompt: str, *, seed: int, max_new_tokens: int) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class FirstTokenClock(StoppingCriteria):
        def __init__(self) -> None:
            self.first_time: float | None = None
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
            if self.first_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.first_time = time.perf_counter()
            return False

    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    clock = FirstTokenClock()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([clock]), return_dict_in_generate=True,
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    sequence = generated.sequences[0]
    continuation = sequence[input_width:]
    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    finish_reason = "eos" if len(continuation) and int(continuation[-1]) in eos_ids else "max_tokens" if len(continuation) >= max_new_tokens else "stopped"
    generated_tokens = int(continuation.shape[0])
    total = finished - started
    ttft = (clock.first_time - started) if clock.first_time is not None else total
    return {
        "output": text, "prompt_tokens": int(encoded["attention_mask"].sum().item()),
        "generated_tokens": generated_tokens, "finish_reason": finish_reason,
        "ttft_seconds": ttft, "total_latency_seconds": total,
        "tokens_per_second": generated_tokens / total if total > 0 else 0.0,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "allocated_vram_bytes_after": int(torch.cuda.memory_allocated()),
    }


def _warmup(model: Any, tokenizer: Any) -> None:
    import torch
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Return the single word READY."}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        model.generate(**encoded, do_sample=False, max_new_tokens=2, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()


def _mean(rows: list[float]) -> float:
    return statistics.mean(rows) if rows else 0.0


def _summarize_model(spec: dict[str, Any], candidate: dict[str, Any], samples: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    case_map = {case["case_id"]: case for case in spec["cases"]}
    roles: dict[str, Any] = {}
    for role in spec["roles"]:
        rows = [row for row in samples if row["role"] == role]
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row)
        exact_repeat = []
        decision_repeat = []
        for case_rows in by_case.values():
            output_hashes = {_sha256(row["output"].encode()) for row in case_rows}
            exact_repeat.append(float(len(output_hashes) == 1))
            triplets = {
                (str((row["score"].get("parsed") or {}).get("decision")), str((row["score"].get("parsed") or {}).get("action")), str((row["score"].get("parsed") or {}).get("primary_variable")))
                for row in case_rows
            }
            decision_repeat.append(float(len(triplets) == 1))
        role_cases = {case_id: _mean([r["score"]["quality"] for r in case_rows]) for case_id, case_rows in by_case.items()}
        irrelevant_robustness = []
        for pair_id in {case_map[cid].get("pair_id") for cid in by_case if case_map[cid].get("pair_id")}:
            pair_cases = [case for case in spec["cases"] if case.get("pair_id") == pair_id and case["role"] == role]
            base = next((case for case in pair_cases if case["condition"] not in {"irrelevant", "contradictory"}), None)
            irrelevant = next((case for case in pair_cases if case["condition"] == "irrelevant"), None)
            if base and irrelevant and base["case_id"] in role_cases and irrelevant["case_id"] in role_cases:
                irrelevant_robustness.append(max(0.0, 1.0 - abs(role_cases[irrelevant["case_id"]] - role_cases[base["case_id"]])))
        contradiction_rows = [row for row in rows if row["condition"] == "contradictory"]
        roles[role] = {
            "quality": _mean([row["score"]["quality"] for row in rows]),
            "quality_100": 100.0 * _mean([row["score"]["quality"] for row in rows]),
            "schema_compliance": _mean([float(row["score"]["schema_compliant"]) for row in rows]),
            "evidence_grounding": _mean([row["score"]["components"]["evidence_grounding"] for row in rows]),
            "confidence_calibration": _mean([row["score"]["components"]["confidence_calibration"] for row in rows]),
            "hallucination_rate": _mean([row["score"]["hallucination_rate"] for row in rows]),
            "exact_repeatability": _mean(exact_repeat), "decision_repeatability": _mean(decision_repeat),
            "mean_generated_tokens": _mean([row["generation"]["generated_tokens"] for row in rows]),
            "mean_ttft_seconds": _mean([row["generation"]["ttft_seconds"] for row in rows]),
            "mean_tokens_per_second": _mean([row["generation"]["tokens_per_second"] for row in rows]),
            "mean_total_latency_seconds": _mean([row["generation"]["total_latency_seconds"] for row in rows]),
            "peak_vram_bytes": max((row["generation"]["peak_vram_bytes"] for row in rows), default=0),
            "robustness_to_irrelevant_evidence": _mean(irrelevant_robustness),
            "contradictory_evidence_quality": _mean([row["score"]["quality"] for row in contradiction_rows]) if contradiction_rows else None,
            "sample_count": len(rows), "case_count": len(by_case)
        }
    all_rows = list(samples)
    return {
        "model_id": candidate["model_id"], "revision": candidate["revision"], "license": candidate["license"],
        "status": "complete", "sample_count": len(all_rows), "expected_sample_count": len(spec["cases"]) * int(spec["generation"]["repetitions"]),
        "overall_quality": _mean([row["score"]["quality"] for row in all_rows]),
        "overall_quality_100": 100.0 * _mean([row["score"]["quality"] for row in all_rows]),
        "roles": roles, "runtime": runtime,
    }


def _aggregate(spec: dict[str, Any], summaries: list[dict[str, Any]], protocol_sha256: str) -> dict[str, Any]:
    olmo = next(summary for summary in summaries if summary["model_id"] == "allenai/OLMo-2-1124-13B-Instruct")
    specialization = {}
    for role in spec["roles"]:
        ranked = sorted(((summary["roles"][role]["quality_100"], summary["model_id"]) for summary in summaries), reverse=True)
        best_score, best_model = ranked[0]
        olmo_score = olmo["roles"][role]["quality_100"]
        specialization[role] = {
            "best_model": best_model, "best_score": best_score, "olmo_score": olmo_score,
            "specialization_gap_best_minus_olmo": best_score - olmo_score,
            "ranking": [{"model_id": model_id, "score": score} for score, model_id in ranked]
        }
    return {
        "result_version": "hephaestus-cognitive-topology.v1", "protocol_id": spec["protocol_id"],
        "protocol_sha256": protocol_sha256, "scientific_variable": spec["scientific_variable"],
        "candidate_count": len(summaries), "case_count": len(spec["cases"]),
        "model_summaries": summaries, "specialization": specialization,
        "topology_decision_ready": len(summaries) == len(spec["candidates"]),
        "training_performed": False, "promotion_performed": False, "lineage_mutated": False
    }


def _unload(model: Any | None, tokenizer: Any | None) -> None:
    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def main() -> int:
    import torch
    spec, protocol_sha256 = _protocol()
    admission = _load_admission(protocol_sha256)
    proof_root, execution_root = _paths()
    proof_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(proof_root / "protocol_snapshot.json", spec)
    _atomic_json(proof_root / "run_manifest.json", {
        "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_sha256,
        "repo_sha": _required("HEPHAESTUS_REPO_SHA"), "proof_run_id": _required("HEPHAESTUS_PROOF_RUN_ID"),
        "attempt": _required("HEPHAESTUS_ATTEMPT"), "started_at_unix": time.time(),
        "candidate_order": [row["model_id"] for row in spec["candidates"]],
        "non_mutating": True, "training_performed": False
    })
    model_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_per_model = len(spec["cases"]) * int(spec["generation"]["repetitions"])

    for candidate in spec["candidates"]:
        model = tokenizer = None
        model_slug = _slug(candidate["model_id"])
        sample_path = proof_root / "samples" / f"{model_slug}.jsonl"
        try:
            snapshot, manifest = _materialize(candidate, admission, proof_root)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model, tokenizer, runtime = load_gpu_snapshot(snapshot, candidate)
            runtime.update({"model_id": candidate["model_id"], "revision": candidate["revision"], "manifest_hash": manifest["manifest_hash"]})
            _warmup(model, tokenizer)
            runtime["warmup_completed"] = True
            runtime["allocated_after_warmup_bytes"] = int(torch.cuda.memory_allocated())
            _atomic_json(proof_root / "runtime" / f"{model_slug}.json", runtime)
            samples: list[dict[str, Any]] = []
            for seed in spec["generation"]["seeds"]:
                for case in spec["cases"]:
                    prompt = _prompt(spec, case)
                    generated = _generate_one(model, tokenizer, prompt, seed=int(seed), max_new_tokens=int(spec["generation"]["max_new_tokens"]))
                    score = score_response(spec, case, generated["output"])
                    row = {
                        "model_id": candidate["model_id"], "revision": candidate["revision"], "case_id": case["case_id"],
                        "role": case["role"], "condition": case["condition"], "pair_id": case.get("pair_id"), "seed": int(seed),
                        "output": generated["output"], "generation": {key: value for key, value in generated.items() if key != "output"},
                        "score": score
                    }
                    samples.append(row)
                    _append_jsonl(sample_path, row)
            if len(samples) != expected_per_model:
                raise RuntimeError(f"incomplete topology evidence: {len(samples)} != {expected_per_model}")
            summary = _summarize_model(spec, candidate, samples, runtime)
            model_summaries.append(summary)
            _atomic_json(proof_root / "model_summaries" / f"{model_slug}.json", summary)
            print("TOPOLOGY_MODEL_COMPLETE_JSON " + json.dumps({"model_id": candidate["model_id"], "overall_quality_100": summary["overall_quality_100"], "roles": {role: data["quality_100"] for role, data in summary["roles"].items()}}, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"model_id": candidate["model_id"], "revision": candidate["revision"], "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            failures.append(failure)
            _atomic_json(proof_root / "runtime" / f"{model_slug}-failure.json", failure)
            print("TOPOLOGY_MODEL_FAILURE_JSON " + json.dumps({key: failure[key] for key in ("model_id", "error_type", "error")}, sort_keys=True), flush=True)
        finally:
            _unload(model, tokenizer)

    if failures or len(model_summaries) != len(spec["candidates"]):
        result = {
            "result_version": "hephaestus-cognitive-topology.v1", "status": "failed", "disposition": "incomplete_evidence",
            "protocol_sha256": protocol_sha256, "completed_models": len(model_summaries), "expected_models": len(spec["candidates"]),
            "failures": failures, "training_performed": False, "promotion_performed": False, "lineage_mutated": False
        }
        _atomic_json(proof_root / "cohort_result.json", result)
        _atomic_json(execution_root / "driver_result.json", result)
        return 2

    aggregate = _aggregate(spec, model_summaries, protocol_sha256)
    aggregate.update({"status": "completed", "disposition": "scientific_cohort_complete"})
    _atomic_json(proof_root / "cohort_result.json", aggregate)
    _atomic_json(execution_root / "driver_result.json", aggregate)
    print("COGNITIVE_TOPOLOGY_RESULT_JSON " + json.dumps({
        "status": aggregate["status"], "disposition": aggregate["disposition"],
        "protocol_sha256": protocol_sha256, "specialization": aggregate["specialization"]
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
