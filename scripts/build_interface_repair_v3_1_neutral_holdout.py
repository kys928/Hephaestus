#!/usr/bin/env python3
"""Build the paired V3.1 neutral-identifier diagnostic holdout.

This is not a new certification set and it is not training data. It is a causal
counterfactual diagnostic over the already-burned V3.1 regression cases: scientific
facts, evidence order, machine state, advisory handoff, expected state, case kind,
and semantic root are preserved exactly. Only model-visible identifiers that reveal
the evaluation namespace are replaced with deterministic opaque identifiers.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any

import build_interface_repair_v3_1_evaluator as source

ROLE = source.ROLE
SOURCE_PARTITION = "regression"
NEUTRAL_PARTITION = "neutral_holdout"
FORBIDDEN_IDENTIFIER_TOKENS = (
    "regression", "regressed", "certification", "certify", "improved",
    "equivalent", "inconclusive", "recheck", "rejection",
)


def _opaque(prefix: str, value: str, width: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:width].upper()
    return f"{prefix}{digest}"


def _neutralize_case(case: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(case)
    old_context = str(case["context"])
    new_context = _opaque("ctx_", old_context, 8)
    ref_map = {
        str(ref): _opaque("E-", str(ref), 12)
        for ref in case["allowed_evidence_refs"]
    }

    row["source_case_id"] = str(case["case_id"])
    row["case_id"] = _opaque("NH-", str(case["case_id"]), 14)
    row["context"] = new_context
    row["situation"] = str(case["situation"]).replace(old_context, new_context)
    row["evidence"] = [
        {"ref": ref_map[str(item["ref"])], "fact": str(item["fact"])}
        for item in case["evidence"]
    ]
    row["allowed_evidence_refs"] = [ref_map[str(ref)] for ref in case["allowed_evidence_refs"]]
    return row


def build_pack() -> dict[str, Any]:
    pack = source.build_pack()
    source_rows = list(pack["partitions"][ROLE][SOURCE_PARTITION])
    neutral = [_neutralize_case(row) for row in source_rows]
    pack["pack_id"] = "hephaestus-interface-repair-v3.1-neutral-holdout"
    pack["pack_version"] = 5
    pack["diagnostic_design"] = {
        "purpose": "paired_counterfactual_identifier_leakage_test",
        "source_run_id": "interface-repair-v3-evaluator-full-36738914957",
        "source_partition": "regression-v31",
        "source_partition_is_burned_diagnostic_only": True,
        "training_performed": False,
        "adapter_mutation_allowed": False,
        "paired_case_count": len(neutral),
        "only_model_visible_identifier_fields_changed": True,
        "certification_claim_allowed": False,
    }
    pack["partitions"][ROLE][NEUTRAL_PARTITION] = neutral
    return pack


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(pack: dict[str, Any]) -> None:
    source.validate(pack)
    part = pack["partitions"][ROLE]
    original = list(part[SOURCE_PARTITION])
    neutral = list(part[NEUTRAL_PARTITION])
    if len(original) != 128 or len(neutral) != 128:
        raise ValueError("neutral holdout must be a 128-case paired transform of V3.1 regression")

    roots = Counter(str(row["semantic_root"]) for row in neutral)
    if set(roots.values()) != {16} or set(roots) != set(source.base.VOCAB["decision"]):
        raise ValueError(f"neutral holdout must contain 16 cases per scientific state: {roots}")

    seen_ids: set[str] = set()
    for before, after in zip(original, neutral):
        if after["source_case_id"] != before["case_id"]:
            raise ValueError("paired source case identity drift")
        if after["case_id"] in seen_ids:
            raise ValueError("duplicate neutral case id")
        seen_ids.add(str(after["case_id"]))

        # Scientific content is frozen. Only identifiers may change.
        for field in (
            "role", "kind", "semantic_root", "focused_boundary", "upstream_role",
            "upstream_output", "verified_facts", "expected", "target_rationale",
            "forbidden_actions",
        ):
            if after.get(field) != before.get(field):
                raise ValueError(f"scientific field changed in neutral transform: {field}")
        if [x["fact"] for x in after["evidence"]] != [x["fact"] for x in before["evidence"]]:
            raise ValueError("evidence facts or ordering changed")
        expected_situation = str(before["situation"]).replace(str(before["context"]), str(after["context"]))
        if after["situation"] != expected_situation:
            raise ValueError("situation changed beyond context identifier replacement")

        identifiers = [str(after["case_id"]), str(after["context"]), *map(str, after["allowed_evidence_refs"])]
        lowered = " ".join(identifiers).lower()
        if any(token in lowered for token in FORBIDDEN_IDENTIFIER_TOKENS):
            raise ValueError(f"semantic leakage remains in neutral identifiers: {identifiers}")
        if str(before["context"]) in str(after["situation"]):
            raise ValueError("source regression context leaked into neutral situation")
        if any(str(ref) in str(after["evidence"]) for ref in before["allowed_evidence_refs"]):
            raise ValueError("source regression evidence ref leaked into neutral evidence")

    source_contexts = {
        name: {str(row["context"]) for row in part[name]}
        for name in ("train", "preflight", "certification", "regression")
    }
    neutral_contexts = {str(row["context"]) for row in neutral}
    if any(neutral_contexts & contexts for contexts in source_contexts.values()):
        raise ValueError("neutral context collides with an existing V3.1 context")


def main() -> int:
    pack = build_pack()
    validate(pack)
    neutral = pack["partitions"][ROLE][NEUTRAL_PARTITION]
    print(json.dumps({
        "pack_id": pack["pack_id"],
        "sha256": canonical_sha256(pack),
        "neutral_cases": len(neutral),
        "state_counts": dict(sorted(Counter(str(x["semantic_root"]) for x in neutral).items())),
        "training_performed": False,
        "certification_claim_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
