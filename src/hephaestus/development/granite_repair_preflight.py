"""Offline Granite Evaluator diagnosis/planning preflight; NO model or training call.

This imports already-exposed, hash-pinned historical material as observations,
not trusted diagnoses. It cannot certify a repair or grant training authority.
Usage: python -m hephaestus.development.granite_repair_preflight
    --registry configs/models/hephaestus_role_model_stack_v1.json
    --index /absolute/source/index.json
    --index-sha256 sha256:<independently retained digest>
    --evidence-root /absolute/source/evidence
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from hephaestus.diagnosis.pinned_historical import (
    PinnedEvidenceError, PinnedHistoricalEvidenceAdapter, _parse_json,
)
from hephaestus.diagnosis.service import EvidenceBasedDiagnosisService
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest


_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SCHEMA = "granite-evaluator-repair-preflight.v1"
_EXPECTED_MODEL = "ibm-granite/granite-3.3-8b-instruct"


def _pinned_json(path: Path, expected: str, *, max_bytes: int = 1024 * 1024) -> dict:
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise PinnedEvidenceError("index is not independently hash-pinned")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise PinnedEvidenceError("invalid or oversized index")
    raw = path.read_bytes()
    if len(raw) > max_bytes or "sha256:" + hashlib.sha256(raw).hexdigest() != expected:
        raise PinnedEvidenceError("index source identity mismatch")
    try:
        value = _parse_json(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PinnedEvidenceError("invalid UTF-8 index") from exc
    if not isinstance(value, dict):
        raise PinnedEvidenceError("index must be a JSON object")
    return value


def preflight(*, registry: Path, index: Path, index_sha256: str,
              evidence_root: Path, run_id: str, lineage_id: str) -> dict[str, object]:
    source = _pinned_json(index, index_sha256)
    if (source.get("schema") != "granite-repair-historical-index.v1" or
            source.get("historical_exposed") is not True):
        raise PinnedEvidenceError("only explicitly exposed historical evidence is admitted")
    registry_bytes = registry.read_bytes()
    catalog = _parse_json(registry_bytes.decode("utf-8"))
    if not isinstance(catalog, dict):
        raise PinnedEvidenceError("model registry is invalid")
    frozen = catalog.get("roles", {}).get("evaluator", {})
    adapter = frozen.get("adapter", {})
    if (frozen.get("model_id") != _EXPECTED_MODEL or
            source.get("model_id") != frozen.get("model_id") or
            source.get("model_revision") != frozen.get("revision") or
            source.get("adapter_sha256") != adapter.get("sha256")):
        raise PinnedEvidenceError("historical evidence differs from frozen Evaluator identity")
    hashes = source.get("files")
    if (not isinstance(hashes, dict) or not hashes or
            not all(isinstance(key, str) and isinstance(value, str)
                    for key, value in hashes.items())):
        raise PinnedEvidenceError("historical evidence index has no pinned source files")
    if not run_id or not lineage_id:
        raise PinnedEvidenceError("run and lineage identities are required")
    req = DiagnosisRequest(
        request_id="granite-repair-preflight:" + index_sha256.removeprefix("sha256:")[:16],
        run_id=run_id, lineage_id=lineage_id,
        stage_name="continuation_repair",
        evidence_refs=sorted(hashes),
        requested_by="offline_operator_preflight",
    )
    # Verify all files *before* diagnosis. The generic diagnosis service
    # intentionally catches unavailable adapters, but this preflight must
    # fail closed if a pinned archive is absent, altered, or malformed.
    admitted = PinnedHistoricalEvidenceAdapter(evidence_root, hashes).load(req)
    if not admitted:
        raise PinnedEvidenceError("pinned historical evidence contains no records")
    from dataclasses import replace
    diagnosis = EvidenceBasedDiagnosisService().diagnose(
        replace(req, evidence_refs=[], observed_failures=admitted)
    )
    interventions = ClosedLoopExperimentPlanner().propose_interventions(diagnosis)
    if not all(p.intervention_kind == "collect_more_evidence" for p in interventions):
        raise PinnedEvidenceError("historical-only intake unexpectedly admitted intervention")
    return {
        "schema": _SCHEMA,
        "execution_kind": "read_only_deterministic_preflight",
        "model_inference_performed": False,
        "training_updates": 0,
        "scientific_repair_demonstrated": False,
        "independent_certification": False,
        "historical_evidence_exposed": True,
        "index_sha256": index_sha256,
        "registry_sha256": "sha256:" + hashlib.sha256(registry_bytes).hexdigest(),
        "frozen_parent_model": frozen["model_id"],
        "frozen_parent_revision": frozen["revision"],
        "frozen_parent_adapter_sha256": adapter["sha256"],
        "diagnosis": diagnosis.to_dict(),
        "admissible_diagnostic_proposals": [p.to_dict() for p in interventions],
        "next_gate": "independently_verify_raw_evaluation_and_scoring_before_training",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-id", default="granite-historical-review")
    parser.add_argument("--lineage-id", default="granite-frozen-evaluator")
    args = parser.parse_args(argv)
    result = preflight(
        registry=args.registry, index=args.index,
        index_sha256=args.index_sha256, evidence_root=args.evidence_root,
        run_id=args.run_id, lineage_id=args.lineage_id,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
