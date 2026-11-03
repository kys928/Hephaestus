"""Read-only re-score of two SHA-pinned, previously exposed Granite-D shards.

This is historical eight-state research, NOT the certified six-state Evaluator.
No GPU, model inference, training, S3 mutation, holdout, or certification.
Never publish raw generations, prompts, or case text.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from hephaestus.diagnosis.pinned_historical import _parse_json, PinnedEvidenceError
from hephaestus.evaluation.granite_dev_scoring import (
    RESEARCH_ONTOLOGY, _parse_generated, score_development_decisions,
)

D_SOURCE_SHA = "ffa0458d5850f9d572a06d5ee6660fd63fc8acc6"
D_ADAPTER_SHA = "5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1"
DEVELOPMENT_SHA = "ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912"
SCREEN_ORDER_SHA = "98c3346fde66cc92d355bae019de811cb9c7614746073d1e4f93e6a069dbe11c"
ARCHIVE_SHA = "25ff27dea3153f601876d3ffb6bf5628cba734596f89364e7911286a1e4173ff"
BUCKET = "cviwpryzao"
BASE = ("hephaestus/scientific/v5/evaluator_semantic_repair_v2/"
        "evaluator-semantic-v2-D-38945578442/checkpoints/step-096/evaluation/samples/")
# Original byte hashes reported in workflow 38962439250, artifact 12072909151.
SHARDS = (
    (BASE + "shard-0001.json", 124152, "934363b71fd76d835b1d05bbc8892645605b3644e1415fa3a82d82e385f67dbc"),
    (BASE + "shard-0002.json", 124363, "471897bcfaa66a707599f234e5a7405b8d3783c67d876f34ff029f3c8fc2d1e1"),
)
ARCHIVED_STATE_CORRECT = {
    "certification_ready": 7, "equivalent": 0, "improved": 4,
    "incomplete_evidence": 5, "inconclusive": 8,
    "recheck_required": 0, "regressed": 4, "scientific_rejection": 5,
}
ARCHIVED_CORRECT = 33


def _read_pinned(client, key: str, size: int, sha: str):
    if not key.startswith(BASE) or not key.endswith(".json") or key not in {s[0] for s in SHARDS}:
        raise PermissionError("only frozen historical D shard keys are admissible")
    if type(size) is not int or not 0 < size <= 200_000:
        raise PinnedEvidenceError("invalid bounded shard size")
    result = client.get_object(Bucket=BUCKET, Key=key)
    body = result["Body"]
    try:
        blob = body.read(size + 1)
    finally:
        body.close()
    if len(blob) != size or hashlib.sha256(blob).hexdigest() != sha:
        raise PinnedEvidenceError("historical S3 shard byte identity mismatch")
    try:
        records = _parse_json(blob.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PinnedEvidenceError("non UTF-8 historical shard") from exc
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise PinnedEvidenceError("historical shard must be JSON object list")
    return records, {"key": key, "bytes": size, "sha256": sha}


def _references(manifest: object) -> list[dict]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "source_repo_sha", "development_sha256",
        "screen_case_order_sha256", "ontology", "references",
    }:
        raise PinnedEvidenceError("historical development reference manifest shape invalid")
    if (manifest["schema"] != "granite-d-exposed-screen-references.v1"
            or manifest["source_repo_sha"] != D_SOURCE_SHA
            or manifest["development_sha256"] != DEVELOPMENT_SHA
            or manifest["screen_case_order_sha256"] != SCREEN_ORDER_SHA
            or manifest["ontology"] != RESEARCH_ONTOLOGY):
        raise PinnedEvidenceError("frozen reference source identity drift")
    refs = manifest["references"]
    if not isinstance(refs, list) or len(refs) != 64:
        raise PinnedEvidenceError("D reference screen must contain 64 cases")
    if any(not isinstance(r, dict) or set(r) != {"case_id", "expected_decision"}
           or not isinstance(r["case_id"], str) or not isinstance(r["expected_decision"], str)
           for r in refs):
        raise PinnedEvidenceError("malformed historical D reference")
    ids = [r["case_id"] for r in refs]
    if len(set(ids)) != 64:
        raise PinnedEvidenceError("duplicate D reference ID")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if digest != SCREEN_ORDER_SHA:
        raise PinnedEvidenceError("historical screen case ordering drift")
    if Counter(r["expected_decision"] for r in refs) != Counter({k: 8 for k in ARCHIVED_STATE_CORRECT}):
        raise PinnedEvidenceError("historical screen state balance drift")
    return refs


def rescore_readonly(client, manifest: object, *, shards=SHARDS) -> dict:
    """Score original generated outputs, retaining only aggregate scientific evidence.

    The shards parameter exists for fake-client tests; the CLI always uses the
    hardcoded SHA-pinned original two-shard manifest, not arbitrary S3 keys.
    """
    refs = _references(manifest)
    entries, receipts = [], []
    for key, size, sha in shards:
        rows, receipt = _read_pinned(client, key, size, sha)
        entries.extend(rows)
        receipts.append(receipt)
    if len(entries) != 64 or len({r.get("case_id") for r in entries}) != 64:
        raise PinnedEvidenceError("saved D predictions incomplete or duplicated")
    reference = {r["case_id"]: r["expected_decision"] for r in refs}
    if {r["case_id"] for r in entries} != set(reference):
        raise PinnedEvidenceError("saved D case IDs differ from pinned reference IDs")
    observed, earlier_correct = [], Counter()
    strict_disagreements = 0
    for r in entries:
        raw = r.get("generation", {}).get("projected_output")
        recorded = r.get("score", {}).get("decision_exact")
        if not isinstance(raw, str) or type(recorded) is not bool:
            raise PinnedEvidenceError("missing original projected generation or Boolean saved score")
        state = reference[r["case_id"]]
        earlier_correct[state] += int(recorded)
        parsed, error = _parse_generated(raw)
        strict_correct = error is None and parsed is not None and parsed["decision"] == state
        strict_disagreements += int(strict_correct != recorded)
        observed.append({"case_id": r["case_id"], "ontology": RESEARCH_ONTOLOGY,
                         "partition": "exposed_development", "raw_generation": raw})
    refrows = [{"case_id": r["case_id"], "ontology": RESEARCH_ONTOLOGY,
                "partition": "exposed_development", "expected_decision": r["expected_decision"]}
               for r in refs]
    fresh = score_development_decisions(refrows, observed, ontology=RESEARCH_ONTOLOGY)
    archive_matches = (sum(earlier_correct.values()) == ARCHIVED_CORRECT
                       and all(earlier_correct[k] == n for k, n in ARCHIVED_STATE_CORRECT.items()))
    return {
        "schema": "prometheus-granite-d-readonly-raw-rescore.v1",
        "scope": "historical_exposed_eight_state_development_only",
        "source_repo_sha": D_SOURCE_SHA,
        "original_archive_zip_sha256": ARCHIVE_SHA,
        "historical_adapter_sha256_claim": D_ADAPTER_SHA,
        "model_weights_independently_verified": False,
        "s3_shards_sha256_verified": True,
        "s3_read_receipts": receipts,
        "recorded_score_comparison": {
            "saved_correct": sum(earlier_correct.values()),
            "archive_correct": ARCHIVED_CORRECT,
            "by_state_matches_archive": archive_matches,
            "strict_per_case_score_disagreements": strict_disagreements,
        },
        "independent_strict_scoring": fresh,
        "historical_exposed": True,
        "independent_final_assessment": False,
        "model_training_executed": False,
        "new_gpu_inference": False,
        "protected_final_holdout_accessed": False,
        "scientific_causality_established": False,
        "scientific_certification": False,
        "no_raw_generations_published": True,
        "next_gate": "source-verified, appropriately scoped observations into diagnosis and planning",
    }


def main() -> int:
    import boto3
    from botocore.config import Config

    if any(not os.environ.get(k) for k in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY")):
        raise PermissionError("read-only S3 access is not configured")
    manifest = _parse_json(Path(os.environ["HEPHAESTUS_D_REFERENCES"]).read_text(encoding="utf-8"))
    s3 = boto3.client(
        "s3", endpoint_url="https://s3api-eu-cz-1.runpod.io/",
        region_name="EU-CZ-1", aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(retries={"mode": "standard", "max_attempts": 3}),
    )
    outcome = rescore_readonly(s3, manifest)
    path = Path(os.environ.get("HEPHAESTUS_D_REPORT", "granite-d-readonly-rescore.json"))
    path.write_text(json.dumps(outcome, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = outcome["recorded_score_comparison"]
    print("GRANITE_D_RAW_RESCORE " + json.dumps({
        "count": outcome["independent_strict_scoring"]["count"],
        "strict_correct": outcome["independent_strict_scoring"]["decision_correct"],
        "saved_correct": summary["saved_correct"],
        "matches_archive": summary["by_state_matches_archive"],
        "strict_disagreements": summary["strict_per_case_score_disagreements"],
    }, sort_keys=True))
    if not summary["by_state_matches_archive"]:
        raise PinnedEvidenceError("saved per-case scores do not reproduce archival aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
