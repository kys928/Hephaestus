#!/usr/bin/env python3
"""Resolve the first inconclusive post-failure diagnosis with measured probes.

This command is read-only with respect to model/data artifacts. It reads the
exact failed run from the RunPod Network Volume through S3, gathers targeted
diagnostic evidence, reruns EvidenceBasedDiagnosisService, and only then hands
the resulting DiagnosisReport to ClosedLoopExperimentPlanner. It never launches
training or applies a Planner/Judge action.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import boto3
from botocore.config import Config

from hephaestus.diagnosis import EvidenceBasedDiagnosisService, PostFailureDiagnosticProbe
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN = "first-bounded-scientific-training-001-33866198758"
EVAL_RUN = "first-semantic-evaluation-001-33869352751"
LINEAGE = "lineage-first-scientific"
STAGE = "smoke_test"
CHECKPOINT_HASH = "sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3"
MODEL_ID = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
TOKENIZER_ID = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
PROCESSED_DATA = "sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c"
TRAINING_CONFIG_SHA = "sha256:fd0296a5bedfc059522d4a7d6acc0f71f8cb4b50181a830bd331fb15f815003f"
EVAL_PACK_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"
PREVIOUS_DIAGNOSIS_REPORT_SHA = "57c5c0b4a31b5b7b033964ebad7178aa1fa342a9fa3fd15bf1759a87d648bece"
PREVIOUS_DIAGNOSIS_INPUT_SHA = "ff4a5be479695d1c595ea5e0af92111f2452b2049891cefa4342e9fef28cf455"

RUN_PREFIX = f"hephaestus/scientific/v1/runs/{TRAIN_RUN}"
BINDING_PREFIX = f"hephaestus/scientific/v1/runtime_bindings/{TRAIN_RUN}"
METRICS_KEY = f"{RUN_PREFIX}/metrics.jsonl"
CONFIG_KEY = f"{RUN_PREFIX}/normalized_training_config.json"
DATASET_KEY = f"{BINDING_PREFIX}/dataset/trainable.jsonl"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 4, "mode": "standard"}),
    )


def get_bytes(client: Any, bucket: str, key: str) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def get_json(client: Any, bucket: str, key: str) -> dict[str, object]:
    payload = json.loads(get_bytes(client, bucket, key).decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"S3 JSON has invalid shape: {key}")
    return payload


def get_jsonl(client: Any, bucket: str, key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in get_bytes(client, bucket, key).decode("utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if not isinstance(item, dict):
            raise RuntimeError(f"S3 JSONL row has invalid shape: {key}")
        rows.append(item)
    return rows


def stream_dataset_rows(client: Any, bucket: str, key: str) -> Iterable[dict[str, object]]:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        for raw in body.iter_lines(chunk_size=1024 * 1024):
            if not raw:
                continue
            item = json.loads(raw.decode("utf-8"))
            if not isinstance(item, dict):
                raise RuntimeError("processed dataset row is not an object")
            yield item
    finally:
        body.close()


def verify_frozen_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    previous_dir = ROOT / "docs/evidence/first-post-failure-diagnosis-001-33871942379"
    report_path = previous_dir / "report.json"
    input_path = previous_dir / "input.json"
    if sha_file(report_path) != PREVIOUS_DIAGNOSIS_REPORT_SHA:
        raise RuntimeError("previous frozen diagnosis report drifted")
    if sha_file(input_path) != PREVIOUS_DIAGNOSIS_INPUT_SHA:
        raise RuntimeError("previous frozen diagnosis input drifted")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prior_input = json.loads(input_path.read_text(encoding="utf-8"))
    if report.get("status") != "inconclusive" or report.get("confidence") != 0.0:
        raise RuntimeError("expected exact prior inconclusive diagnosis")

    eval_pack_path = ROOT / "configs/eval_packs/semantic_behavior_v1.yaml"
    eval_pack = json.loads(eval_pack_path.read_text(encoding="utf-8"))
    if eval_pack.get("content_hash") != EVAL_PACK_HASH or eval_pack.get("frozen") is not True:
        raise RuntimeError("frozen semantic_behavior_v1 identity drifted")
    return report, prior_input, eval_pack


def enrich_for_planner(report: Any, config: dict[str, object]) -> None:
    report.metadata.update(
        {
            "baseline_ref": f"run://{TRAIN_RUN}",
            "baseline_justification": (
                "The failed run is the immutable matched baseline. A follow-up must retain the same "
                "random initialization, tokenizer, evaluation pack, decoding settings, and recipe "
                "except for the single Planner-selected primary variable."
            ),
            "baseline_quality": 0.95,
            "random_seed": config.get("seed", 1729),
            "eval_pack_ref": f"sha256:{EVAL_PACK_HASH}",
            "decoding_ref": "generation-settings-2ba1dd1322d04d4925c36ac4",
            "architecture_ref": MODEL_ID,
            "tokenizer_ref": TOKENIZER_ID,
            "dataset_manifest_ref": PROCESSED_DATA,
            "training_recipe_ref": config.get("config_fingerprint", "first-bounded-scientific-training.v2"),
            "lineage_trust_level": "verified",
            "new_evidence_refs": [METRICS_KEY, DATASET_KEY],
            "capability_targets": [
                "instruction adherence",
                "structured JSON response",
                "bounded response length",
                "termination",
                "non-repetition",
                "causal-LM continuation",
            ],
            "required_languages": ["en"],
            "required_domains": ["general language", "instruction following"],
            "required_formats": ["causal_lm_text", "instruction_response", "structured_json"],
            "training_constraints": {
                "seed": config.get("seed"),
                "model_revision": MODEL_ID,
                "tokenizer_revision": TOKENIZER_ID,
                "max_steps": config.get("max_steps"),
                "learning_rate": config.get("learning_rate"),
                "warmup_steps": config.get("warmup_steps"),
                "scheduler": config.get("scheduler"),
                "batch_size": config.get("batch_size"),
                "context_length": config.get("context_length"),
            },
        }
    )


def main() -> int:
    previous_report, prior_input, eval_pack = verify_frozen_inputs()
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    client = s3_client()

    metrics_raw = get_bytes(client, bucket, METRICS_KEY)
    metrics = []
    for line in metrics_raw.decode("utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise RuntimeError("metrics.jsonl contains a non-object row")
            metrics.append(item)
    config_raw = get_bytes(client, bucket, CONFIG_KEY)
    if sha_bytes(config_raw) != TRAINING_CONFIG_SHA:
        raise RuntimeError(f"normalized training config drift: {sha_bytes(config_raw)}")
    config = json.loads(config_raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("training config has invalid shape")

    dataset_head = client.head_object(Bucket=bucket, Key=DATASET_KEY)
    dataset_bytes = int(dataset_head.get("ContentLength", -1))
    # Full byte identity is checked while streaming the same object below through a second pass.
    dataset_raw = get_bytes(client, bucket, DATASET_KEY)
    if sha_bytes(dataset_raw) != PROCESSED_DATA:
        raise RuntimeError("processed dataset content identity drifted")
    dataset_rows = []
    for raw in dataset_raw.decode("utf-8").splitlines():
        if raw.strip():
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise RuntimeError("processed dataset contains non-object row")
            dataset_rows.append(item)

    probe = PostFailureDiagnosticProbe().run(
        metrics=metrics,
        training_config=config,
        dataset_rows=dataset_rows,
        eval_pack=eval_pack,
        metrics_source_ref=f"runpod-s3://{bucket}/{METRICS_KEY}",
        dataset_source_ref=f"runpod-s3://{bucket}/{DATASET_KEY}",
    )

    prior_request = prior_input.get("request")
    if not isinstance(prior_request, dict):
        raise RuntimeError("previous frozen diagnosis input has no request")
    prior_records = prior_request.get("observed_failures")
    if not isinstance(prior_records, list) or not all(isinstance(row, dict) for row in prior_records):
        raise RuntimeError("previous diagnosis observed_failures are invalid")

    request = DiagnosisRequest(
        request_id="diagnose-first-semantic-regression-targeted-probes-001",
        run_id=TRAIN_RUN,
        lineage_id=LINEAGE,
        stage_name=STAGE,
        observed_failures=[*prior_records, *probe.evidence_records],
        requested_by="post_failure_targeted_probe_handoff",
    )
    diagnosis = EvidenceBasedDiagnosisService().diagnose(request)
    enrich_for_planner(diagnosis, config)

    planner = ClosedLoopExperimentPlanner()
    interventions = list(planner.propose_interventions(diagnosis))
    if not interventions:
        raise RuntimeError("planner produced no intervention")
    selected = interventions[0]
    dataset_request, model_request = planner.create_discovery_requests(diagnosis, selected)
    experiment = None
    blocked_on = None
    if dataset_request is None and model_request is None:
        experiment = planner.propose_experiment(diagnosis, selected, None, None)
    elif dataset_request is not None:
        blocked_on = "dataset_discovery_and_selection"
    elif model_request is not None:
        blocked_on = "model_discovery_and_selection"

    leading = next(
        (item for item in diagnosis.hypotheses if item.hypothesis_id == diagnosis.leading_hypothesis_id),
        None,
    )
    result = {
        "result_version": "first-targeted-post-failure-diagnostics.v1",
        "upstream": {
            "training_run_id": TRAIN_RUN,
            "evaluation_run_id": EVAL_RUN,
            "previous_diagnosis_report_id": previous_report.get("report_id"),
            "previous_diagnosis_status": previous_report.get("status"),
            "checkpoint_manifest_hash": CHECKPOINT_HASH,
            "model_identity": MODEL_ID,
            "tokenizer_identity": TOKENIZER_ID,
            "processed_dataset_identity": PROCESSED_DATA,
            "eval_pack_content_hash": EVAL_PACK_HASH,
        },
        "s3_evidence": {
            "metrics_key": METRICS_KEY,
            "metrics_sha256": sha_bytes(metrics_raw),
            "metric_points": len(metrics),
            "training_config_key": CONFIG_KEY,
            "training_config_sha256": sha_bytes(config_raw),
            "dataset_key": DATASET_KEY,
            "dataset_sha256": PROCESSED_DATA,
            "dataset_bytes": dataset_bytes,
            "dataset_rows": len(dataset_rows),
        },
        "probe": {
            "evidence_records": probe.evidence_records,
            "measurements": probe.measurements,
            "unresolved_questions": probe.unresolved_questions,
        },
        "diagnosis": diagnosis.to_dict(),
        "leading_diagnosis": None if leading is None else leading.to_dict(),
        "planner": {
            "ranked_interventions": [item.to_dict() for item in interventions],
            "selected_intervention": selected.to_dict(),
            "one_primary_variable": selected.primary_variable,
            "dataset_search_request": None if dataset_request is None else dataset_request.to_dict(),
            "model_search_request": None if model_request is None else model_request.to_dict(),
            "experiment_proposal": None if experiment is None else experiment.to_dict(),
            "proposal_blocked_on": blocked_on,
            "planner_executed_intervention": False,
            "training_launched": False,
        },
    }

    atomic_json(Path("first_targeted_post_failure_diagnostics.json"), result)
    atomic_json(Path("first_targeted_post_failure_probe.json"), result["probe"])
    atomic_json(Path("first_targeted_post_failure_diagnosis.json"), diagnosis.to_dict())
    atomic_json(Path("first_targeted_post_failure_planner.json"), result["planner"])

    print(
        json.dumps(
            {
                "diagnosis_status": diagnosis.status,
                "leading_domain": None if leading is None else leading.failure_domain,
                "diagnosis_confidence": diagnosis.confidence,
                "ranked_intervention_count": len(interventions),
                "selected_intervention_kind": selected.intervention_kind,
                "selected_primary_variable": selected.primary_variable,
                "proposal_blocked_on": blocked_on,
                "training_launched": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
