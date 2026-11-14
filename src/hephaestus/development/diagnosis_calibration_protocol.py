"""Precommitted Prometheus contrast admission and scoring (no model or provider IO).

This module does not create evidence, run inference, infer gold labels, or
authorize GPU use. CPU fixtures can validate the evaluator only; real-science
admission requires separately verified bytes, independent labels and a
reviewed external custodian. Never pass a self-declared JSON 'verified' flag
straight to a paid launcher.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from hephaestus.llm.role_contract import RoleMasteryContract
from hephaestus.policy.model_proposals import DIAGNOSIS_DOMAINS

SCHEMA = "prometheus-diagnosis-evidence-contrast-precommit.v1"
SEEDS = (271828, 314159, 161803)
ARMS = ("unaugmented_frozen_role", "deterministic_evidence_calibration")
SLOTS = ("M0", "E1", "D1", "C1", "E2", "D2", "N0", "N1")
MODEL = "mistralai/Ministral-3-3B-Reasoning-2512"
REVISION = "4a36357c811bf511a7b625d132e12f22408aac91"
ADAPTER_SHA = "534d1929d76fc64d07d8d36e85115d5660c113e63b022206bd500d288bdff32c"
ABSTENTIONS = frozenset({"inconclusive", "observability_limit"})
POSITIVE = frozenset(DIAGNOSIS_DOMAINS) - ABSTENTIONS


class CalibrationProtocolError(ValueError):
    """A source, trial, score or precommit boundary cannot be trusted."""


def canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def load_protocol(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32_768:
        raise CalibrationProtocolError("missing/unbounded protocol precommit")
    raw = path.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("status") != "FROZEN_PROTOCOL_NO_GPU_AUTHORIZATION"):
        raise CalibrationProtocolError("not a frozen non-executable protocol")
    cases = value.get("case_slots", [])
    pairs = value.get("paired_tests", [])
    samples = value.get("sampling", {})
    if (tuple(row["id"] for row in cases) != SLOTS
            or tuple(samples.get("seeds", [])) != SEEDS
            or tuple(samples.get("arms", [])) != ARMS
            or samples.get("invocation_ceiling") != len(SLOTS) * len(SEEDS) * len(ARMS)
            or samples.get("do_not_launch_until_source_coverage_and_separate_authorization") is not True
            or {p["kind"] for p in pairs} != {"causal_evidence_change", "irrelevant_metadata"}
            or value["action_policy"].get("training_and_adapter_changes_authorized") is not False
            or value["action_policy"].get("planner_allowed_only_after_actual_supported_accepted_diagnosis") is not True):
        raise CalibrationProtocolError("protocol identity or authority drift")
    labels = {c["id"]:c["label"] for c in cases}
    if any(labels[x] != "inconclusive" for x in ("M0", "C1")) or labels["E1"] != "evaluation_integrity" or labels["E2"] != "evaluation_integrity":
        raise CalibrationProtocolError("precommitted contrast labels changed")
    if not all(c["status"] != "authentic_source_admitted" for c in cases):
        raise CalibrationProtocolError("source precommit claims unverified admission")
    return value, sha256(raw).hexdigest()


def source_coverage(protocol: dict, evidence: dict) -> dict:
    """Fail closed on absent genuine positive source; never trust a fixture.

    `evidence` is only a *proposed inventory*, not proof of independent
    attestation. A separate custodian must rehash the original sources and
    lock labels before any model receives its prompts.
    """
    entries = evidence.get("cases", {}) if isinstance(evidence, dict) else {}
    missing = []
    inspected = []
    for slot in SLOTS:
        entry = entries.get(slot)
        if not isinstance(entry, dict):
            missing.append(f"{slot}:missing_source")
            continue
        refs = entry.get("source_refs", [])
        independent = entry.get("independent_verifier_receipts", [])
        adjudication = entry.get("independent_label_receipt")
        if (entry.get("source_kind") != "authentic" or not isinstance(refs, list)
                or not refs or not all(isinstance(x, str) and x.startswith("sha256:") and len(x) == 71 for x in refs)
                or len(set(refs)) != len(refs)
                or not isinstance(independent, list) or len(set(independent)) < 2
                or not isinstance(adjudication, str) or not adjudication.startswith("sha256:")):
            missing.append(f"{slot}:unverified_source_or_independent_label")
        else:
            inspected.append(slot)
    return {
        "schema":"prometheus-source-coverage-audit.v1",
        "possible_label_attestations":inspected,
        "missing_or_unverified":missing,
        "external_original_bytes_rehash_required":True,
        "independent_label_custody_required":True,
        "scientifically_admitted":False,
        "gpu_launch_authorized":False,
        "reason":"self-reported attestation references are not verified bytes or independent human review",
    }


def verify_raw_trial(data: bytes, *, content_sha256: str, expected_refs: set[str],
                     expected_seed: int, contract: RoleMasteryContract) -> dict:
    """Read one authentic-looking receipt, requiring real CUDA and frozen ID.

    This validates receipt *structure and content*, NOT the external source
    custody, GPU provider identity, or independent truth label. Those remain
    separately guarded by the experiment controller and source reviewer.
    """
    if len(data) > 3_000_000 or sha256(data).hexdigest() != content_sha256:
        raise CalibrationProtocolError("raw receipt has wrong content digest/size")
    row = json.loads(data)
    if not isinstance(row, dict):
        raise CalibrationProtocolError("receipt must be an object")
    execution = row.get("execution", {})
    if (row.get("schema") != "role-inference.v1"
            or row.get("role") != "diagnosis"
            or row.get("generator") != "PinnedTransformersRoleGenerator"
            or row.get("model_id") != MODEL or row.get("base_revision") != REVISION
            or row.get("adapter_sha256") != ADAPTER_SHA
            or row.get("seed") != expected_seed
            or not isinstance(execution, dict)
            or execution.get("seed") != expected_seed
            or execution.get("device") != "cuda:0"
            or not isinstance(execution.get("gpu_name"), str) or not execution["gpu_name"]
            or not isinstance(execution.get("generated_token_ids"), list)
            or not execution["generated_token_ids"]
            or row.get("evidence_refs") != sorted(expected_refs)
            and set(row.get("evidence_refs", [])) != expected_refs):
        raise CalibrationProtocolError("raw receipt lacks exact frozen CUDA provenance")
    for field, digest_field in (("prompt", "prompt_sha256"), ("raw_output", "output_sha256")):
        if not isinstance(row.get(field), str) or sha256(row[field].encode()).hexdigest() != row.get(digest_field):
            raise CalibrationProtocolError("raw inference byte identity differs")
    if row.get("status") == "accepted":
        parsed = contract.parse("diagnosis", row["raw_output"], expected_refs)
        if parsed.to_dict() != row.get("answer"):
            raise CalibrationProtocolError("accepted raw answer differs from frozen parser")
        return {"decision":parsed.decision, "evidence_refs":list(parsed.evidence_refs),
                "confidence":parsed.confidence, "schema_valid":True, "cuda_receipt_valid":True}
    if row.get("status") == "rejected":
        try:
            contract.parse("diagnosis", row["raw_output"], expected_refs)
        except ValueError:
            return {"decision":None,"evidence_refs":[],"confidence":None,
                    "schema_valid":False,"cuda_receipt_valid":True}
        raise CalibrationProtocolError("rejected receipt unexpectedly parses as valid")
    raise CalibrationProtocolError("unrecognized inference status")


def _rate(num: int, den: int) -> dict:
    return {"numerator":num,"denominator":den,"rate":num/den if den else None}


def score_contract_fixtures(protocol: dict, labels: dict[str, str],
                            trials: list[dict]) -> dict:
    """Score pre-labeled *fixtures*, never authorize science or a GPU launch.

    A future verified executor may reuse the arithmetic only AFTER separate
    raw-receipt and label/source custody admission.
    """
    if set(labels) != set(SLOTS):
        raise CalibrationProtocolError("not all precommitted case labels supplied")
    for slot in SLOTS:
        kind = next(row for row in protocol["case_slots"] if row["id"] == slot)["label"]
        if kind in {"inconclusive", "evaluation_integrity"} and labels[slot] != kind:
            raise CalibrationProtocolError("reference label conflicts with precommit")
        if kind == "same_as_M0" and labels[slot] != labels["M0"]:
            raise CalibrationProtocolError("metadata changed its reference label")
        if kind == "same_as_E1" and labels[slot] != labels["E1"]:
            raise CalibrationProtocolError("metadata changed its reference label")
        if kind == "one_explicit_supported_domain" and labels[slot] not in POSITIVE - {"evaluation_integrity"}:
            raise CalibrationProtocolError("other-domain evidence cannot be relabeled as evaluation")
    expected = {(slot, arm, seed) for slot in SLOTS for arm in ARMS for seed in SEEDS}
    seen = set()
    for t in trials:
        key = (t.get("case_id"), t.get("arm"), t.get("seed"))
        if key not in expected or key in seen:
            raise CalibrationProtocolError("unexpected/duplicate trial or unbounded budget")
        seen.add(key)
        if t.get("decision") is not None and t["decision"] not in DIAGNOSIS_DOMAINS:
            raise CalibrationProtocolError("decision outside frozen vocabulary")
        if t.get("schema_valid") is not True and t.get("decision") is not None:
            raise CalibrationProtocolError("invalid schema cannot yield a scientific decision")
    if seen != expected or len(trials) != 48:
        raise CalibrationProtocolError("incomplete 48-trial precommitted comparison")
    scored = {}
    for arm in ARMS:
        current = {(t["case_id"],t["seed"]):t for t in trials if t["arm"] == arm}
        all_rows = list(current.values())
        pp = [r for r in all_rows if r.get("decision") in POSITIVE]
        correct_positive = sum(r["decision"] == labels[r["case_id"]] for r in pp)
        abstain = [r for r in all_rows if labels[r["case_id"]] in ABSTENTIONS]
        appropriate = sum(r.get("decision") in ABSTENTIONS for r in abstain)
        unsupported = sum(r.get("decision") in POSITIVE and r["decision"] != labels[r["case_id"]] for r in all_rows)
        invalid = sum(not r.get("schema_valid") for r in all_rows)
        citation = sum(r.get("citation_exact") is True and r.get("citation_material") is True for r in all_rows)
        by_pair = {"causal_evidence_change":[0,0], "irrelevant_metadata":[0,0]}
        for pair in protocol["paired_tests"]:
            a,b,kind = pair["a"],pair["b"],pair["kind"]
            for seed in SEEDS:
                first,second=current[(a,seed)],current[(b,seed)]
                da,db=first.get("decision"),second.get("decision")
                success = (first.get("schema_valid") is True and second.get("schema_valid") is True
                    and da==labels[a] and db==labels[b]
                    and ((da!=db) if kind=="causal_evidence_change" else (da==db)))
                by_pair[kind][0] += int(success)
                by_pair[kind][1] += 1
        conf = [float(r["confidence"]) for r in all_rows if
                type(r.get("confidence")) in (int,float) and 0 <= r["confidence"] <= 1]
        scored[arm] = {
            "supported_domain_precision":_rate(correct_positive,len(pp)),
            "appropriate_abstention":_rate(appropriate,len(abstain)),
            "unsupported_domain_rate":_rate(unsupported,len(all_rows)),
            "invalid_output_rate":_rate(invalid,len(all_rows)),
            "exact_and_material_citations":_rate(citation,len(all_rows)),
            "evidence_sensitivity":_rate(*by_pair["causal_evidence_change"]),
            "nuisance_invariance":_rate(*by_pair["irrelevant_metadata"]),
            "confidence_descriptive_only":{"n":len(conf),"mean":sum(conf)/len(conf) if conf else None},
        }
    return {"schema":"prometheus-contrast-fixture-score.v1",
            "source":"synthetic_contract_fixture_only",
            "scientific_completion_claim":False,"gpu_launch_authorized":False,
            "independent_truth_validation":False,
            "trained_model_repair":False,
            "arms":scored}


def fixed_controls(labels: dict[str,str]) -> dict:
    """Non-learned reference predictions; no labels enter an actual control."""
    if set(labels) != set(SLOTS):
        raise CalibrationProtocolError("missing fixed control slots")
    return {
        "always_inconclusive":{"decisions":{slot:"inconclusive" for slot in SLOTS}},
        "fixed_repair_evaluation":{"decisions":{slot:"evaluation_integrity" for slot in SLOTS}},
        "precommitted_signal_rule":{"requires_source_signals":True,
                                     "order":["eval_integrity_failed", "evaluation_settings_mismatch",
                                              "tokenizer_mismatch", "data_coverage_gap"],
                                     "default":"inconclusive",
                                     "uses_reference_labels":False},
    }
