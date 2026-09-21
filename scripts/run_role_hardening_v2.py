#!/usr/bin/env python3
"""Continue certified role adapters on Phase II targeted hardening data.

Checkpoint selection uses only the new hardening development split. The original
role-mastery certification split and frozen 600-case post-training red-team are
final-only evidence and never influence checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))

import launch_first_bounded_scientific_training as storage
import run_cognitive_topology_v1 as topo
import run_diagnosis_foundation_bakeoff_v1 as modelio
import run_role_mastery_v1 as mastery_runner
import run_post_training_redteam_v1 as redteam_runner

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_role_hardening_v2.json"


def req(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def put_json(client: Any, key: str, obj: object) -> None:
    raw = (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    client.put_object(Bucket=storage.VOLUME_ID, Key=key, Body=raw)
    if storage.read_key(client, key) != raw:
        raise RuntimeError(f"S3 readback mismatch: {key}")


def put_jsonl(client: Any, key: str, rows: list[dict[str, Any]]) -> None:
    raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()
    client.put_object(Bucket=storage.VOLUME_ID, Key=key, Body=raw)
    if storage.read_key(client, key) != raw:
        raise RuntimeError(f"S3 readback mismatch: {key}")


def _import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_hardening_pack(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    builder = _import(ROOT / cfg["pack"]["builder_path"], "role_hardening_pack")
    pack = builder.build_pack()
    builder.validate(pack)
    observed = builder.canonical_sha256(pack)
    if observed != cfg["pack"]["canonical_sha256"]:
        raise RuntimeError(f"hardening pack hash drift: {observed}")
    return builder, pack


def role_prompt(pack: dict[str, Any], case: dict[str, Any]) -> str:
    role = case["role"]
    vocab = pack["contract_vocabulary"][role]
    evidence = "\n".join(f"- {x['ref']}: {x['fact']}" for x in case["evidence"])
    machine = case.get("verified_machine_facts") or {}
    machine_text = "\n".join(
        f"- {key} = {json.dumps(value, ensure_ascii=False, sort_keys=True)}"
        for key, value in sorted(machine.items())
    ) or "- none supplied"
    keys = ", ".join(pack["response_schema"]["required_exact_keys"])
    return f"""You are the Hephaestus {role.upper()} specialist.

NARROWED ROLE CONTRACT:
{pack['role_rules'][role]}

VERIFIED MACHINE FACTS:
These facts were already established by deterministic infrastructure. Treat them as
authoritative inputs. Do not spend reasoning on re-validating hashes, approvals,
identity, stage legality, admission, budget, idempotency, or provenance.
{machine_text}

SCIENTIFIC / GOVERNANCE SITUATION:
{case['situation']}

OBSERVATIONAL EVIDENCE:
{evidence}

Return exactly one JSON object and nothing else. No markdown.
The object must contain exactly these keys: {keys}.
Use only the role vocabulary:
- decision: {', '.join(vocab['decision'])}
- action: {', '.join(vocab['action'])}
- primary_variable: {', '.join(vocab['primary_variable'])}
- confidence: number in [0,1]
- evidence_refs: cite only material OBSERVATIONAL EVIDENCE references
- uncertainties: short JSON array
- rationale: one concise sentence about the interpretation that remains after deterministic facts

Do not invent evidence. Do not override verified machine facts. Do not perform another role's job. Do not reveal hidden reasoning."""


def training_example(
    tokenizer: Any,
    cand: dict[str, Any],
    pack: dict[str, Any],
    builder: Any,
    case: dict[str, Any],
    maxlen: int,
) -> dict[str, Any]:
    import torch

    user = role_prompt(pack, case)
    answer = json.dumps(builder.target(case), ensure_ascii=False, separators=(",", ":"))
    prompt_ids = mastery_runner.template_ids(
        tokenizer, cand, [{"role": "user", "content": user}], True
    )
    full_ids = mastery_runner.template_ids(
        tokenizer,
        cand,
        [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        False,
    )
    lcp = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        lcp += 1
    if lcp < 8:
        raise RuntimeError(f"chat template prefix mismatch {case['case_id']} lcp={lcp}")
    if len(full_ids) > maxlen:
        raise RuntimeError(f"training example {case['case_id']} length {len(full_ids)}>{maxlen}")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad, (list, tuple)):
        pad = pad[0] if pad else None
    if pad is None:
        raise RuntimeError("no pad/eos token id")
    ids = full_ids + [int(pad)] * (maxlen - len(full_ids))
    mask = [1] * len(full_ids) + [0] * (maxlen - len(full_ids))
    labels = [-100] * min(lcp, len(full_ids)) + full_ids[min(lcp, len(full_ids)) :]
    labels += [-100] * (maxlen - len(labels))
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.tensor([mask]),
        "labels": torch.tensor([labels]),
        "nonpad_tokens": len(full_ids),
        "supervised_tokens": sum(x != -100 for x in labels),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    skills: dict[str, Any] = {}
    for skill in sorted({str(row["skill"]) for row in rows}):
        subset = [row for row in rows if row["skill"] == skill]
        skills[skill] = {
            "sample_count": len(subset),
            "quality_100": mean([float(x["score"]["quality_100"]) for x in subset]),
            "schema_compliance": mean([float(x["score"]["schema_compliant"]) for x in subset]),
            "evidence_grounding": mean([float(x["score"]["components"]["evidence_grounding"]) for x in subset]),
            "confidence_calibration": mean([float(x["score"]["components"]["confidence_calibration"]) for x in subset]),
            "hallucination_rate": mean([float(x["score"]["hallucination_rate"]) for x in subset]),
            "exact_contract_pass_rate": mean([float(x["score"]["quality_100"] >= 99.999) for x in subset]),
        }
    return {
        "sample_count": len(rows),
        "quality_100": mean([float(x["score"]["quality_100"]) for x in rows]),
        "schema_compliance": mean([float(x["score"]["schema_compliant"]) for x in rows]),
        "evidence_grounding": mean([float(x["score"]["components"]["evidence_grounding"]) for x in rows]),
        "confidence_calibration": mean([float(x["score"]["components"]["confidence_calibration"]) for x in rows]),
        "hallucination_rate": mean([float(x["score"]["hallucination_rate"]) for x in rows]),
        "exact_contract_pass_rate": mean([float(x["score"]["quality_100"] >= 99.999) for x in rows]),
        "mean_latency_seconds": mean([float(x["generation"]["total_latency_seconds"]) for x in rows]),
        "mean_generated_tokens": mean([float(x["generation"]["generated_tokens"]) for x in rows]),
        "peak_eval_vram_bytes": max([int(x["generation"]["peak_vram_bytes"]) for x in rows], default=0),
        "skills": skills,
    }


def evaluate_mastery_style(
    *,
    model: Any,
    tokenizer: Any,
    cand: dict[str, Any],
    scoring_pack: dict[str, Any],
    cases: list[dict[str, Any]],
    prompt_fn: Callable[[dict[str, Any], dict[str, Any]], str],
    seed: int,
    max_new: int,
    deadline: float,
    client: Any,
    key: str,
    heartbeat: Any,
    phase: str,
) -> dict[str, Any]:
    rows = []
    model.eval()
    for i, case in enumerate(cases, 1):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"hardening hard wall during {phase}")
        generated = modelio.generate(
            model, tokenizer, cand, prompt_fn(scoring_pack, case), seed, max_new, deadline
        )
        raw = generated.pop("raw_output")
        normalized, extraction = modelio.extract_complete_json(raw)
        score = topo.score_response(scoring_pack, case, normalized)
        rows.append(
            {
                "case_id": case["case_id"],
                "root_case_id": case["root_case_id"],
                "skill": case["skill"],
                "phase": phase,
                "raw_output": raw,
                "output": normalized,
                "extraction": extraction,
                "generation": generated,
                "score": score,
            }
        )
        if i == 1 or i % 25 == 0 or i == len(cases):
            heartbeat(
                f"{phase}_evaluation",
                completed=i,
                total=len(cases),
                quality_so_far=summarize(rows)["quality_100"],
            )
    put_jsonl(client, key, rows)
    return summarize(rows)


def evaluate_redteam_once(
    *,
    model: Any,
    tokenizer: Any,
    cand: dict[str, Any],
    redteam_pack: dict[str, Any],
    cases: list[dict[str, Any]],
    seeds: list[int],
    max_new: int,
    deadline: float,
    client: Any,
    prefix: str,
    heartbeat: Any,
) -> dict[str, Any]:
    seed_rows: dict[int, list[dict[str, Any]]] = {}
    seed_summaries: dict[str, Any] = {}
    for seed in seeds:
        rows: list[dict[str, Any]] = []
        for i, case in enumerate(cases, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("hardening hard wall during frozen red-team acceptance")
            generated = modelio.generate(
                model,
                tokenizer,
                cand,
                redteam_runner.prompt(redteam_pack, case),
                seed,
                max_new,
                deadline,
            )
            raw = generated.pop("raw_output")
            projected = modelio.project(raw, cand)
            normalized, extraction = modelio.extract_complete_json(projected)
            score = redteam_runner.score_case(redteam_pack, case, raw, normalized)
            rows.append(
                {
                    "sample_version": "hephaestus-role-hardening-v2-redteam.v1",
                    "role": case["role"],
                    "condition": "hardened",
                    "seed": seed,
                    "case_id": case["case_id"],
                    "dimension": case["dimension"],
                    "case_kind": case["case_kind"],
                    "pair_id": case["pair_id"],
                    "pair_relation": case["pair_relation"],
                    "stressors": case["stressors"],
                    "prompt_variant": case["prompt_variant"],
                    "expected": case["expected"],
                    "required_evidence_refs": case["required_evidence_refs"],
                    "raw_output": raw,
                    "projected_output": projected,
                    "output": normalized,
                    "extraction": extraction,
                    "generation": generated,
                    "score": score,
                }
            )
            if i == 1 or i % 20 == 0 or i == len(cases):
                heartbeat(
                    "frozen_redteam_acceptance",
                    seed=seed,
                    completed=i,
                    total=len(cases),
                    quality_so_far=redteam_runner.subset_summary(rows)["quality_100"],
                )
        seed_rows[seed] = rows
        seed_summaries[str(seed)] = redteam_runner.summarize(rows)
        put_jsonl(client, f"{prefix}/frozen_redteam/seed-{seed}.jsonl", rows)
    all_rows = [row for rows in seed_rows.values() for row in rows]
    overall = redteam_runner.subset_summary(all_rows)
    overall["stochastic_consistency"] = redteam_runner.stochastic_consistency(seed_rows)
    overall["seed_summaries"] = seed_summaries
    overall["pair_metrics_by_seed"] = {
        str(seed): redteam_runner.pair_metrics(rows) for seed, rows in seed_rows.items()
    }
    return overall


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError("adapter archive path traversal detected")
        tf.extractall(destination)
    adapter_dir = destination / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError("source adapter archive lacks adapter_config.json")
    return adapter_dir


def download_source_adapter(client: Any, cand: dict[str, Any], root: Path) -> Path:
    meta = cand["source_adapter"]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source-adapter.tar.gz"
    client.download_file(storage.VOLUME_ID, meta["s3_key"], str(archive))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        raise RuntimeError(f"source adapter SHA-256 mismatch: {digest}")
    if archive.stat().st_size != int(meta["bytes"]):
        raise RuntimeError("source adapter byte-size mismatch")
    return safe_extract_tar(archive, root / "extracted")


def archive_adapter(path: Path, out: Path, client: Any, key: str) -> dict[str, Any]:
    with tarfile.open(out, "w:gz") as tf:
        tf.add(path, arcname="adapter")
    digest = hashlib.sha256()
    with out.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    client.upload_file(str(out), storage.VOLUME_ID, key)
    head = client.head_object(Bucket=storage.VOLUME_ID, Key=key)
    return {"s3_key": key, "sha256": digest.hexdigest(), "bytes": int(head["ContentLength"])}


def certification(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    min_skill = min((x["quality_100"] for x in summary["skills"].values()), default=0.0)
    gates = {
        "quality_100": summary["quality_100"] >= float(thresholds["quality_100"]),
        "schema": summary["schema_compliance"] >= float(thresholds["schema"]),
        "grounding": summary["evidence_grounding"] >= float(thresholds["grounding"]),
        "hallucination": summary["hallucination_rate"] <= float(thresholds["hallucination_max"]),
        "min_skill_quality": min_skill >= float(thresholds["min_skill_quality"]),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "thresholds": thresholds,
        "observed_min_skill_quality": min_skill,
    }


def set_training_mode(model: Any, enabled: bool) -> None:
    mastery_runner.set_training_mode(model, enabled)


def main() -> int:
    import torch
    from peft import PeftModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=os.environ.get("HEPHAESTUS_ROLE"))
    args = parser.parse_args()
    role = str(args.role or "").strip()

    cfg = json.loads(CFG_PATH.read_text())
    if role not in cfg["candidates"]:
        raise RuntimeError(f"unknown role {role}")
    if cfg["governance"]["training_authorized"] is not True:
        raise RuntimeError("hardening training is not authorized")
    if cfg["pack"]["frozen_redteam_optimization_use"] != "forbidden":
        raise RuntimeError("frozen red-team optimization boundary is not enforced")

    builder, hardening_pack = load_hardening_pack(cfg)
    mastery_builder = _import(ROOT / "scripts/build_role_mastery_v1.py", "mastery_pack_final_holdout")
    mastery_pack = mastery_builder.build_pack()
    mastery_builder.validate(mastery_pack)
    if mastery_builder.canonical_sha256(mastery_pack) != cfg["source_stack"]["original_role_mastery_pack_sha256"]:
        raise RuntimeError("original role-mastery certification pack identity drift")
    redteam_builder = _import(ROOT / "scripts/build_post_training_redteam_v1.py", "redteam_pack_final_holdout")
    redteam_pack = redteam_builder.build_pack()
    redteam_builder.validate(redteam_pack)
    if redteam_builder.canonical_sha256(redteam_pack) != cfg["source_stack"]["original_redteam_pack_sha256"]:
        raise RuntimeError("frozen red-team pack identity drift")

    cand = dict(cfg["candidates"][role])
    run_id = req("HEPHAESTUS_ROLE_HARDENING_RUN_ID")
    repo_sha = req("HEPHAESTUS_REPO_SHA")
    client = storage.s3_client()
    prefix = f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}/roles/{role}"
    deadline = time.monotonic() + int(cfg["execution"]["hard_wall_seconds"])
    root = Path("/opt/hephaestus-role-hardening-v2") / run_id / role
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "result_version": "hephaestus-role-hardening-v2.v1",
        "status": "running",
        "role": role,
        "run_id": run_id,
        "repo_sha": repo_sha,
        "pack_sha256": cfg["pack"]["canonical_sha256"],
        "frozen_redteam_pack_sha256": cfg["source_stack"]["original_redteam_pack_sha256"],
        "model_id": cand["model_id"],
        "revision": cand["revision"],
        "source_adapter": cand["source_adapter"],
        "production_promotion_performed": False,
        "production_certification_mutated": False,
        "automatic_role_dispatch_enabled": False,
    }

    def heartbeat(stage: str, **extra: Any) -> None:
        payload = {
            "run_id": run_id,
            "role": role,
            "stage": stage,
            "timestamp_unix": time.time(),
            "remaining_wall_seconds": max(0.0, deadline - time.monotonic()),
            **extra,
        }
        put_json(client, f"{prefix}/progress.json", payload)
        print("ROLE_HARDENING_V2_HEARTBEAT_JSON " + json.dumps(payload, sort_keys=True), flush=True)

    try:
        heartbeat("materializing_foundation")
        snapshot, _, manifest = modelio.materialize(cand, root / "foundation")
        put_json(client, f"{prefix}/foundation_manifest.json", manifest)
        base_model, tokenizer, runtime = modelio.load(cand, snapshot, None, cfg)
        put_json(client, f"{prefix}/runtime.json", runtime)

        heartbeat("verifying_source_adapter")
        source_adapter_dir = download_source_adapter(client, cand, root / "source-adapter")
        model = PeftModel.from_pretrained(base_model, str(source_adapter_dir), is_trainable=True)
        model.to("cuda")
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or not all("lora_" in name for name, _ in trainable):
            raise RuntimeError("continued adapter training exposed non-LoRA trainable parameters")
        if role in {"diagnosis", "planner"} and any("vision_tower" in name for name, _ in trainable):
            raise RuntimeError("vision tower became trainable")

        train_cases = hardening_pack["splits"][role]["train"]
        dev_cases = hardening_pack["splits"][role]["dev"]
        cert_cases = hardening_pack["splits"][role]["cert"]
        maxlen = int(cfg["training"]["max_sequence_length"])
        heartbeat("tokenizing_hardening_training", cases=len(train_cases))
        examples = [
            training_example(tokenizer, cand, hardening_pack, builder, case, maxlen)
            for case in train_cases
        ]

        steps = int(cfg["training"]["optimizer_steps_by_role"][role])
        accum = int(cfg["training"]["gradient_accumulation_steps"])
        if steps * accum != len(examples):
            raise RuntimeError(f"coverage mismatch steps*acc={steps * accum} train={len(examples)}")
        optimizer = torch.optim.AdamW(
            [p for _, p in trainable],
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        order = list(range(len(examples)))
        random.Random(int(cfg["training"]["seed"])).shuffle(order)
        checkpoints = {int(x) for x in cfg["training"]["dev_checkpoints_by_role"][role]}
        cursor = 0
        losses: list[float] = []
        supervised = 0
        nonpad = 0
        dev_records: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        train_start = time.perf_counter()
        set_training_mode(model, True)

        for step in range(1, steps + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("hardening hard wall during training")
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accum):
                example = examples[order[cursor]]
                cursor += 1
                batch = {
                    key: value.to("cuda", non_blocking=True)
                    for key, value in example.items()
                    if key in {"input_ids", "attention_mask", "labels"}
                }
                output = model(**batch, use_cache=False)
                loss = output.loss / accum
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite hardening loss at step {step}")
                loss.backward()
                step_loss += float(loss.detach().cpu()) * accum
                supervised += int(example["supervised_tokens"])
                nonpad += int(example["nonpad_tokens"])
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], float(cfg["training"]["max_grad_norm"])
            )
            optimizer.step()
            losses.append(step_loss / accum)
            if step == 1 or step % 25 == 0 or step == steps:
                heartbeat("training", optimizer_step=step, optimizer_steps=steps, loss=losses[-1])

            if step in checkpoints:
                checkpoint = root / "checkpoints" / f"step-{step:04d}"
                mastery_runner.save_adapter(model, checkpoint)
                set_training_mode(model, False)
                dev = evaluate_mastery_style(
                    model=model,
                    tokenizer=tokenizer,
                    cand=cand,
                    scoring_pack=hardening_pack,
                    cases=dev_cases,
                    prompt_fn=role_prompt,
                    seed=int(cfg["evaluation"]["seed"]),
                    max_new=int(cfg["evaluation"]["max_new_tokens_by_role"][role]),
                    deadline=deadline,
                    client=client,
                    key=f"{prefix}/hardening_dev/step-{step:04d}.jsonl",
                    heartbeat=heartbeat,
                    phase=f"hardening_dev_step_{step}",
                )
                record = {"step": step, "summary": dev, "adapter_dir": str(checkpoint)}
                dev_records.append(record)
                min_skill = min((x["quality_100"] for x in dev["skills"].values()), default=0.0)
                rank = (
                    dev["quality_100"],
                    min_skill,
                    dev["evidence_grounding"],
                    -dev["hallucination_rate"],
                    dev["exact_contract_pass_rate"],
                )
                if best is None or rank > best["rank"]:
                    best = {"rank": rank, **record}
                set_training_mode(model, True)

        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        if cursor != len(examples):
            raise RuntimeError("not all hardening examples were consumed exactly once")
        if best is None:
            raise RuntimeError("no hardening dev checkpoint selected")

        set_training_mode(model, False)
        selected_step = int(best["step"])
        if selected_step != steps:
            base_model = model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = PeftModel.from_pretrained(base_model, best["adapter_dir"], is_trainable=False)
            model.to("cuda")
            model.eval()

        heartbeat("hardening_certification")
        hardening_cert = evaluate_mastery_style(
            model=model,
            tokenizer=tokenizer,
            cand=cand,
            scoring_pack=hardening_pack,
            cases=cert_cases,
            prompt_fn=role_prompt,
            seed=int(cfg["evaluation"]["seed"]),
            max_new=int(cfg["evaluation"]["max_new_tokens_by_role"][role]),
            deadline=deadline,
            client=client,
            key=f"{prefix}/hardening_certification.jsonl",
            heartbeat=heartbeat,
            phase="hardening_certification",
        )
        hardening_cert_result = certification(
            hardening_cert, cfg["evaluation"]["hardening_cert_thresholds"][role]
        )

        heartbeat("original_certification_regression")
        original_cert = evaluate_mastery_style(
            model=model,
            tokenizer=tokenizer,
            cand=cand,
            scoring_pack=mastery_pack,
            cases=mastery_pack["splits"][role]["cert"],
            prompt_fn=mastery_runner.role_prompt,
            seed=int(cfg["evaluation"]["seed"]),
            max_new=int(cfg["evaluation"]["max_new_tokens_by_role"][role]),
            deadline=deadline,
            client=client,
            key=f"{prefix}/original_certification_regression.jsonl",
            heartbeat=heartbeat,
            phase="original_certification_regression",
        )
        reg = cfg["evaluation"]["original_cert_regression"]
        original_regression_checks = {
            "quality_preserved": original_cert["quality_100"]
            >= float(cand["original_certification_quality_100"]) - float(reg["maximum_quality_drop_points"]),
            "schema": original_cert["schema_compliance"] >= float(reg["minimum_schema_rate"]),
            "hallucination": original_cert["hallucination_rate"] <= float(reg["maximum_hallucination_rate"]),
        }
        original_regression = {
            "passed": all(original_regression_checks.values()),
            "checks": original_regression_checks,
            "baseline_quality_100": cand["original_certification_quality_100"],
            "observed": original_cert,
        }

        heartbeat("frozen_redteam_final_only")
        redteam_acceptance = evaluate_redteam_once(
            model=model,
            tokenizer=tokenizer,
            cand=cand,
            redteam_pack=redteam_pack,
            cases=redteam_pack["cases"][role],
            seeds=[int(x) for x in cfg["evaluation"]["adapted_redteam_seeds"]],
            max_new=int(cfg["evaluation"]["max_new_tokens_by_role"][role]),
            deadline=deadline,
            client=client,
            prefix=prefix,
            heartbeat=heartbeat,
        )
        acceptance_cfg = cfg["evaluation"]["frozen_redteam_acceptance"]
        redteam_improvement = (
            redteam_acceptance["quality_100"] - float(cand["original_redteam_quality_100"])
        )
        redteam_checks = {
            "minimum_absolute_improvement": redteam_improvement
            >= float(acceptance_cfg["minimum_absolute_improvement_points"]),
            "quality_floor": redteam_acceptance["quality_100"]
            >= float(acceptance_cfg["role_quality_floor"][role]),
        }
        redteam_result = {
            "passed": all(redteam_checks.values()),
            "checks": redteam_checks,
            "baseline_quality_100": cand["original_redteam_quality_100"],
            "quality_improvement_points": redteam_improvement,
            "observed": redteam_acceptance,
            "used_for_checkpoint_selection": False,
            "evaluation_count_after_selection": 1,
        }

        selected_path = Path(best["adapter_dir"])
        artifact = archive_adapter(
            selected_path,
            root / "selected-hardened-adapter.tar.gz",
            client,
            f"{prefix}/selected-hardened-adapter.tar.gz",
        )
        overall_passed = bool(
            hardening_cert_result["passed"]
            and original_regression["passed"]
            and redteam_result["passed"]
        )
        result.update(
            {
                "status": "completed",
                "selected_dev_step": selected_step,
                "dev_checkpoints": [
                    {"step": x["step"], "summary": x["summary"]} for x in dev_records
                ],
                "training": {
                    "optimizer_steps": steps,
                    "training_cases_consumed": cursor,
                    "supervised_token_updates": supervised,
                    "nonpad_token_updates": nonpad,
                    "trainable_parameter_count": sum(p.numel() for _, p in trainable),
                    "training_seconds": train_seconds,
                    "peak_training_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "loss_first": losses[0],
                    "loss_last": losses[-1],
                    "loss_mean": sum(losses) / len(losses),
                    "continued_from_certified_adapter": True,
                },
                "hardening_certification_summary": hardening_cert,
                "hardening_certification": hardening_cert_result,
                "original_certification_regression": original_regression,
                "frozen_redteam_acceptance": redteam_result,
                "hardening_accepted": overall_passed,
                "selected_adapter": artifact,
                "completed_at_unix": time.time(),
            }
        )
        put_json(client, f"{prefix}/result.json", result)
        heartbeat(
            "complete",
            hardening_accepted=overall_passed,
            hardening_quality_100=hardening_cert["quality_100"],
            original_cert_quality_100=original_cert["quality_100"],
            frozen_redteam_quality_100=redteam_acceptance["quality_100"],
            frozen_redteam_improvement_points=redteam_improvement,
            selected_dev_step=selected_step,
        )
        print("ROLE_HARDENING_V2_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at_unix": time.time(),
            }
        )
        put_json(client, f"{prefix}/result.json", result)
        print("ROLE_HARDENING_V2_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        raise
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
