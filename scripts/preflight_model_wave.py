#!/usr/bin/env python3
"""Verify admission and prior evidence before any model-wave GPU allocation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from hephaestus.providers.models.admission import inspect_immutable_model
from hephaestus.providers.models.selection import DeterministicModelSelectionService
from hephaestus.schemas.discovery_contract import ModelSearchRequest

import launch_positive_promotion_proof as launcher

OUT = Path("model_wave_preflight")


def main() -> int:
    spec = json.loads(Path("configs/models/promotion_wave_v5.json").read_text())
    OUT.mkdir(exist_ok=True)
    s3 = launcher.base.s3_client()
    prefix = f"{launcher.SCIENTIFIC_PREFIX}/model_admission/{os.environ['GITHUB_SHA']}"
    records = []
    for candidate in spec["candidates"]:
        record = inspect_immutable_model(candidate, OUT / "metadata_cache")
        records.append(record)
        print("MODEL_ADMISSION_JSON " + json.dumps(record.to_dict(), sort_keys=True))
    request = ModelSearchRequest(
        request_id="v5-metadata-preflight", diagnosis_report_id="v4-deterministic-rejections",
        problem_statement="Exact instruction, structured output and bounded termination",
        task_requirements=["causal_lm", "chat"], license_allowlist=["apache-2.0", "mit"],
        runtime_constraints={"backend": "pinned_chat_template_transformers", "max_memory_gb": 48},
        evidence_refs=[row["summary_key"] for row in spec["prior_evidence"]["cycles"]],
    )
    decision = DeterministicModelSelectionService().select(request, records)
    if decision.rejected_candidates or len(decision.ranked_candidate_ids) != len(records):
        raise RuntimeError(f"candidate admission blocked: {decision.to_dict()}")
    prior = []
    for row in spec["prior_evidence"]["cycles"]:
        raw = s3.get_object(Bucket=launcher.VOLUME_ID, Key=row["summary_key"])["Body"].read()
        if hashlib.sha256(raw).hexdigest() != row["summary_sha256"]:
            raise RuntimeError("prior V4 scientific evidence hash mismatch")
        cycle = json.loads(raw)
        if cycle["qualifies_for_certified_promotion"] or cycle["selected_action"] != "reject_checkpoint":
            raise RuntimeError("prior V4 disposition differs from the audited record")
        prior.append({"key": row["summary_key"], "sha256": row["summary_sha256"]})
    packages = ["transformers", "accelerate", "tokenizers", "safetensors", "huggingface-hub"]
    locked = {name: importlib.metadata.version(name) for name in packages}
    lock = "".join(f"{name}=={version}\n" for name, version in locked.items())
    admission = {
        "admission_version": "immutable-model-wave-admission.v1",
        "repo_sha": os.environ["GITHUB_SHA"], "wave_id": spec["wave_id"],
        "status": "metadata_admitted_runtime_pending",
        "protocol": spec["protocol"], "candidates": [c.to_dict() for c in records],
        "selection_decision": decision.to_dict(), "research_order": [c["model_id"] for c in spec["candidates"]],
        "prior_evidence": prior, "runtime_packages": locked,
        "requirements_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "gpu_generation_tested": False,
    }
    for name, raw in [("admission.json", (json.dumps(admission, indent=2, sort_keys=True) + "\n").encode()),
                      ("requirements.txt", lock.encode())]:
        path = OUT / name
        path.write_bytes(raw)
        key = prefix + "/" + name
        s3.put_object(Bucket=launcher.VOLUME_ID, Key=key, Body=raw)
        observed = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read()
        if observed != raw:
            raise RuntimeError(f"admission persistence readback mismatch: {key}")
        print("ADMISSION_PERSISTED_JSON " + json.dumps(
            {"key": key, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}))
    # Read only public runtime facts from prior execution logs.
    prior_run = "positive-real-model-promotion-001-" + spec["prior_evidence"]["run_id"]
    key = f"{launcher.SCIENTIFIC_PREFIX}/executions/{prior_run}/attempt-1/pod_runtime.log"
    log = s3.get_object(Bucket=launcher.VOLUME_ID, Key=key)["Body"].read().decode("utf-8", "replace")
    for line in log.splitlines():
        if line.startswith('{"torch":') or line.startswith("Successfully installed "):
            print("PRIOR_RUNTIME " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
