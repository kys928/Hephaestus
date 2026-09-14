#!/usr/bin/env python3
"""Admitted V5 identities through the existing untouched frozen scientific spine."""
from __future__ import annotations

import importlib.metadata
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path

from hephaestus.providers.models.admission import RUNTIME_PATTERNS, validate_identity
from hephaestus.providers.models.gpu_snapshot import load_gpu_snapshot
from hephaestus.providers.models.generation_observation import GenerationObserver
import run_positive_promotion_proof_v4 as wave4

proof = wave4.proof
SPEC = json.loads((proof.REPO_ROOT / "configs/models/promotion_wave_v5.json").read_text())
ADMISSION_ROOT = proof.SCIENTIFIC_ROOT / "model_admission" / os.environ.get("HEPHAESTUS_REPO_SHA", "unset")
ORIGINAL_ATOMIC_JSON = proof.atomic_json


def root() -> Path:
    return proof.SCIENTIFIC_ROOT / "positive_promotion" / proof.required("HEPHAESTUS_PROOF_RUN_ID")


def diagnostic(stage, exc, **context):
    record = {"stage": stage, "at": proof.now(), "error_type": type(exc).__name__,
              "error": str(exc), "traceback": traceback.format_exc(), **context}
    try:
        import torch
        record["cuda"] = {"available": torch.cuda.is_available(),
                          "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                          "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0}
    finally:
        ORIGINAL_ATOMIC_JSON(root() / "runtime" / (proof.slug(stage) + "-failure.json"), record)


def admission():
    record = json.loads((ADMISSION_ROOT / "admission.json").read_text())
    if record["repo_sha"] != proof.required("HEPHAESTUS_REPO_SHA"):
        raise RuntimeError("admission repository revision mismatch")
    if record["protocol"] != SPEC["protocol"]:
        raise RuntimeError("admission scientific protocol mismatch")
    for package, expected in record["runtime_packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"runtime package drift: {package}")
    return record


def materialize_model(*, proof_root, model_id, revision, expected_license):
    from huggingface_hub import HfApi, snapshot_download
    validate_identity({"model_id": model_id, "revision": revision, "license": expected_license})
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    license_id = str(getattr(info.card_data, "license", "") or "").lower()
    if info.sha != revision or license_id != expected_license:
        raise RuntimeError("immutable revision/license drift during materialization")
    snapshot = Path(snapshot_download(
        repo_id=model_id, revision=revision, allow_patterns=RUNTIME_PATTERNS,
        cache_dir="/opt/hephaestus-model-materialization/hf_cache")).resolve()
    components = {p.relative_to(snapshot).as_posix(): proof.sha_file(p)
                  for p in sorted(snapshot.rglob("*")) if p.is_file()}
    if not any(name.endswith(".safetensors") for name in components):
        raise RuntimeError("materialized snapshot lacks safetensors")
    if any(Path(name).suffix.lower() in {".py", ".sh", ".so", ".dll", ".exe"} for name in components):
        raise RuntimeError("executable remote files in admitted snapshot")
    upstream_hashes = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        expected = getattr(lfs, "sha256", None)
        if expected and sibling.rfilename in components:
            if components[sibling.rfilename] != "sha256:" + expected:
                raise RuntimeError(f"Hub LFS SHA256 mismatch: {sibling.rfilename}")
            upstream_hashes[sibling.rfilename] = "sha256:" + expected
    records = admission()["candidates"]
    admitted = next((c for c in records if c["model_id"] == model_id and c["revision"] == revision), None)
    if admitted is not None:
        for name, expected in admitted["metadata"]["metadata_component_hashes"].items():
            if components.get(name) != expected:
                raise RuntimeError(f"admitted metadata bytes drifted: {name}")
    elif model_id != SPEC["reviewer"]["model_id"] or revision != SPEC["reviewer"]["revision"]:
        raise RuntimeError("model was not admitted and is not the fixed independent reviewer")
    payload = {
        "manifest_version": "external-model-snapshot.v1", "model_id": model_id,
        "provider": "huggingface", "requested_revision": revision, "resolved_revision": info.sha,
        "license": license_id, "trust_remote_code": False, "snapshot_path": str(snapshot),
        "component_count": len(components),
        "byte_size": sum((snapshot / name).stat().st_size for name in components),
        "components": components, "manifest_hash": proof.canonical_hash(components),
        "upstream_lfs_sha256_verified": upstream_hashes,
        "reproducible_artifact_ref": f"hf://models/{model_id}@{revision}",
        "cache_persistence": "ephemeral", "admission_ref": str(ADMISSION_ROOT / "admission.json"),
    }
    path = proof_root / "model_manifests" / proof.slug(model_id) / revision / "snapshot_manifest.json"
    ORIGINAL_ATOMIC_JSON(path, payload)
    payload["manifest_ref"] = str(path)
    return payload


class AdmittedV5Backend(wave4.AdaptiveV4ChatTemplateBackend):
    def _load(self):
        if self._model is not None:
            return
        spec = next(c for c in SPEC["candidates"]
                    if c["model_id"] == self.model_id and c["revision"] == self.revision)
        try:
            model, self._tokenizer, self.runtime_facts = load_gpu_snapshot(self.snapshot_path, spec)
            self.runtime_facts.update({"model_id": self.model_id, "revision": self.revision,
                                       "manifest_hash": self.manifest_hash})
            self._model = GenerationObserver(model)
            ORIGINAL_ATOMIC_JSON(root() / "runtime" / (proof.slug(self.model_id) + ".json"), self.runtime_facts)
        except Exception as exc:
            diagnostic("model-load", exc, model_id=self.model_id, revision=self.revision)
            raise

    def generate_batch(self, tasks, **kwargs):
        try:
            self._load()
            self._model.observations.clear()
            outputs = super().generate_batch(tasks, **kwargs)
            order = [i for seed in sorted({t.seed for t in tasks})
                     for i, t in enumerate(tasks) if t.seed == seed]
            observed = self._model.observations
            if len(observed) != len(outputs):
                raise RuntimeError("passive generation observation count mismatch")
            for index, fact in zip(order, observed, strict=True):
                outputs[index] = replace(outputs[index], metadata={**outputs[index].metadata, **fact})
            self.runtime_facts["generation_smoke_test_passed"] = True
            self.runtime_facts["last_run_id"] = kwargs["run_id"]
            self.runtime_facts["last_sample_count"] = len(outputs)
            ORIGINAL_ATOMIC_JSON(root() / "runtime" / (proof.slug(self.model_id) + ".json"), self.runtime_facts)
            return outputs
        except Exception as exc:
            diagnostic("generation", exc, model_id=self.model_id, revision=self.revision,
                       run_id=kwargs.get("run_id"))
            raise


class EvidencePreservingDriver(proof.RealModelPromotionDriver):
    def execute_cycle(self, **kwargs):
        try:
            return super().execute_cycle(**kwargs)
        except Exception as exc:
            diagnostic("cycle-" + str(kwargs["cycle_index"]), exc)
            raise


def defer_terminal(path, payload):
    # The launcher may tear the Pod down as soon as driver_result.json appears.
    # Publish only after V5 classification and persistence finish.
    path = Path(path)
    if path.name == "driver_result.json":
        path = path.with_name("driver_core_result.json")
    ORIGINAL_ATOMIC_JSON(path, payload)


def main():
    execution_root = proof.SCIENTIFIC_ROOT / "executions" / proof.required("HEPHAESTUS_PROOF_RUN_ID") / ("attempt-" + proof.required("HEPHAESTUS_ATTEMPT"))
    try:
        admission()
        if SPEC["protocol"]["content_hash"] != proof.EVAL_PACK_HASH:
            raise RuntimeError("V5 frozen evaluation hash differs from the existing proof")
        code = proof.main()
        result = json.loads((execution_root / "driver_core_result.json").read_text())
        cycles = [json.loads(p.read_text()) for p in sorted((root() / "cycles").glob("cycle-*/cycle_summary.json"))]
        reports = [json.loads(p.read_text()) for p in (root() / "evaluations").rglob("generation_report.json")
                   if "-candidate-" in str(p)]
        complete = len(reports) == 3 * len(cycles) and all(
            r["completed"] and len(r["samples"]) == 18 and r["content_hash"] == proof.EVAL_PACK_HASH for r in reports)
        if code == 0:
            disposition = "certified_promoted"
        elif len(cycles) == len(SPEC["candidates"]) and complete:
            disposition = "scientific_rejection" if all(
                c["comparison"]["deterministic_gate_status"] == "failed"
                or c["comparison"]["primary_outcome"] in {"regressed", "equivalent_within_evidence"}
                for c in cycles) else "certification_failure"
        else:
            disposition = "infrastructure_failure" if not reports else "incomplete_evidence"
        result.update({"wave_id": SPEC["wave_id"], "disposition": disposition,
                       "candidate_reports": len(reports), "complete_candidate_evidence": complete,
                       "admission_ref": str(ADMISSION_ROOT / "admission.json")})
        ORIGINAL_ATOMIC_JSON(root() / "wave_result.json", result)
        ORIGINAL_ATOMIC_JSON(execution_root / "driver_result.json", result)
        print("MODEL_WAVE_RESULT_JSON " + json.dumps(result, sort_keys=True))
        return code
    except Exception as exc:
        diagnostic("wave", exc)
        ORIGINAL_ATOMIC_JSON(execution_root / "driver_result.json", {
            "status": "failed", "disposition": "infrastructure_failure",
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "training_performed": False, "original_research_lineage_mutated": False})
        raise


proof.PinnedChatTemplateBackend = AdmittedV5Backend
proof.RealModelPromotionDriver = EvidencePreservingDriver
proof.materialize_model = materialize_model
proof.atomic_json = defer_terminal
proof.CANDIDATES = tuple({**c, "judge_model_id": SPEC["reviewer"]["model_id"],
                         "judge_revision": SPEC["reviewer"]["revision"],
                         "judge_license": SPEC["reviewer"]["license"],
                         "prior_rejection_evidence": [r["summary_key"] for r in SPEC["prior_evidence"]["cycles"]]}
                        for c in SPEC["candidates"])

if __name__ == "__main__":
    raise SystemExit(main())
