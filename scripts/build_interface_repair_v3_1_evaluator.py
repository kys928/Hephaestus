#!/usr/bin/env python3
"""Build V3.1 Evaluator repair data with a dedicated preflight split.

V3 preflight sampled the sealed certification partition. Because its aggregate
format failure informed this revision, those sampled V3 cases are now diagnostic
only. V3.1 keeps the 1,536-case training curriculum unchanged, creates a separate
8-case preflight partition, and rotates certification/regression onto fresh,
disjoint contexts and case IDs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import build_interface_repair_v3_evaluator as base

ROLE = base.ROLE

PREFLIGHT_CONTEXTS = [
    "format_probe_orbit", "format_probe_ember", "format_probe_delta", "format_probe_sable",
    "format_probe_lattice", "format_probe_nova", "format_probe_anchor", "format_probe_helios",
]

CERT_CONTEXTS_V31 = [
    "v31_cert_aurora", "v31_cert_borealis", "v31_cert_cinder", "v31_cert_drift",
    "v31_cert_equinox", "v31_cert_forge", "v31_cert_garnet", "v31_cert_horizon",
    "v31_cert_ion", "v31_cert_junction", "v31_cert_kepler", "v31_cert_lumen",
    "v31_cert_mosaic", "v31_cert_nimbus", "v31_cert_orchid", "v31_cert_pulse",
]

REGRESSION_CONTEXTS_V31 = [
    "v31_reg_quartz", "v31_reg_radian", "v31_reg_summit", "v31_reg_tangent",
    "v31_reg_umbra", "v31_reg_vector", "v31_reg_weld", "v31_reg_xenon",
    "v31_reg_yarrow", "v31_reg_zenith", "v31_reg_aperture", "v31_reg_bracket",
    "v31_reg_cascade", "v31_reg_domain", "v31_reg_echelon", "v31_reg_flux",
]


def _sealed(split: str, contexts: list[str], offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ci, context in enumerate(contexts):
        for ri, _ in enumerate(base.ROOTS):
            variant = (ci * 11 + ri * 13 + offset) % 37
            rows.append(base.make_case(split, context, ci, ri, variant))
    return rows


def _preflight() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ri, context in enumerate(PREFLIGHT_CONTEXTS):
        variant = (ri * 17 + 5) % 41
        rows.append(base.make_case("preflight-v31", context, ri, ri, variant))
    return rows


def build_pack() -> dict[str, Any]:
    pack = base.build_pack()
    train = pack["partitions"][ROLE]["train"]
    cert = _sealed("certification-v31", CERT_CONTEXTS_V31, 7)
    regression = _sealed("regression-v31", REGRESSION_CONTEXTS_V31, 19)
    preflight = _preflight()

    pack["pack_id"] = "hephaestus-interface-repair-v3.1-evaluator"
    pack["pack_version"] = 4
    pack["diagnostic_burned_sets"] = {
        **pack["diagnostic_burned_sets"],
        "v3_preflight_run_id": "interface-repair-v3-evaluator-preflight-36687021465",
        "v3_sampled_certification_cases_reused_in_v31": False,
        "v31_preflight_cases_reused_in_certification": False,
        "v31_preflight_cases_reused_in_regression": False,
    }
    pack["design"] = {
        **pack["design"],
        "dedicated_preflight_partition": True,
        "preflight_cases": len(preflight),
        "fresh_v31_certification_cases": len(cert),
        "fresh_v31_regression_cases": len(regression),
    }
    pack["partitions"] = {
        ROLE: {
            "train": train,
            "preflight": preflight,
            "certification": cert,
            "regression": regression,
        }
    }
    return pack


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(pack: dict[str, Any]) -> None:
    base.validate(pack)
    part = pack["partitions"][ROLE]
    preflight = part["preflight"]
    if len(preflight) != 8:
        raise ValueError("V3.1 preflight must contain exactly 8 cases")

    groups = {
        "train": part["train"],
        "preflight": preflight,
        "certification": part["certification"],
        "regression": part["regression"],
    }
    contexts = {name: {row["context"] for row in rows} for name, rows in groups.items()}
    names = list(contexts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if contexts[left] & contexts[right]:
                raise ValueError(f"V3.1 context leakage: {left}/{right}")

    all_rows = [row for rows in groups.values() for row in rows]
    ids = [row["case_id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate V3.1 case IDs")
    if {row["semantic_root"] for row in preflight} != set(base.VOCAB["decision"]):
        raise ValueError("V3.1 preflight must cover every semantic root exactly once")
    if pack["diagnostic_burned_sets"]["v3_sampled_certification_cases_reused_in_v31"]:
        raise ValueError("V3 sampled certification cases may not be reused")


def main() -> int:
    pack = build_pack()
    validate(pack)
    part = pack["partitions"][ROLE]
    print(json.dumps({
        "pack_id": pack["pack_id"],
        "sha256": canonical_sha256(pack),
        "train": len(part["train"]),
        "preflight": len(part["preflight"]),
        "certification": len(part["certification"]),
        "regression": len(part["regression"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
