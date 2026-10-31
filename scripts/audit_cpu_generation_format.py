#!/usr/bin/env python3
"""Read-only classification of original archived CPU generations, not certification.

Verifies original archived bytes. Reports aggregate format errors without leaking
any held-out prompts, expected targets or model-generated answer text.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

GENERATION = "cpu-prospective-real-generation.v1"
HEXDIGEST = re.compile(r"[0-9a-f]{64}\Z")


def inspect_archive(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or "archive_sha256.json" not in names:
            raise ValueError("duplicate names or missing evidence manifest")
        hashes = json.loads(archive.read("archive_sha256.json"))
        if not isinstance(hashes, dict) or set(hashes) != set(names)-{"archive_sha256.json"}:
            raise ValueError("original archive inventory mismatch")
        receipts = []
        for name, digest in hashes.items():
            raw = archive.read(name)
            if (name.startswith("/") or ".." in Path(name).parts or not
                    isinstance(digest, str) or not HEXDIGEST.fullmatch(digest) or
                    sha256(raw).hexdigest() != digest):
                raise ValueError("unsafe evidence member or SHA mismatch")
            if name.startswith("artifacts/objects/sha256/"):
                if name.split("/")[-1] != digest:
                    raise ValueError("content-addressed object pathname mismatch")
                try:
                    item = json.loads(raw)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(item, dict) and item.get("schema") == GENERATION:
                    receipts.append(item)
        report = json.loads(archive.read("cpu_campaign_report.json"))
        if report.get("status") != "completed" or report.get("completed_operations") != 45:
            raise ValueError("not a completed native CPU research campaign")
        return report, receipts


def category(value):
    if not isinstance(value, str):
        raise ValueError("expected original generated text")
    value = value.strip()
    if not value:
        return "empty"
    if value.startswith("```python"):
        return "python_code_fence"
    if value.startswith("```"):
        return "other_code_fence"
    if value.startswith("{"):
        return "json_opening"
    if value.startswith("["):
        return "array_opening"
    return "other_text"


def audit(archives):
    if not archives:
        raise ValueError("provide at least one original research archive")
    workflows, campaigns = set(), []
    for path in archives:
        report, receipts = inspect_archive(path)
        workflow = report.get("workflow_id")
        if not isinstance(workflow, str) or workflow in workflows:
            raise ValueError("absent or duplicate campaign workflow identity")
        workflows.add(workflow)
        baseline = [x for x in receipts if x.get("checkpoint_ref") is None and x.get("repeat_index") == 0]
        candidate = [x for x in receipts if x.get("checkpoint_ref") is not None and x.get("repeat_index") == 0]
        if len(baseline) != 1 or len(candidate) != 1:
            raise ValueError("need one original baseline and one original candidate receipt")
        left, right = baseline[0], candidate[0]
        if (left.get("protocol_ref") != right.get("protocol_ref") or
                left.get("generation_settings_id") != right.get("generation_settings_id") or
                [x["sample_id"] for x in left["samples"]] !=
                [x["sample_id"] for x in right["samples"]]):
            raise ValueError("frozen evaluation protocol/sample identity changed")
        summary = {}
        for label, receipt in (("baseline", left), ("candidate", right)):
            if (receipt.get("external_certification") is not False or
                    receipt.get("count") != len(receipt.get("samples", []))):
                raise ValueError("invalid or unscoped research generation receipt")
            tally = Counter(category(s["output"]) for s in receipt["samples"])
            summary[label] = {"format_counts": dict(sorted(tally.items())),
                              "json_valid": receipt["json_valid"], "correct": receipt["correct"],
                              "count": receipt["count"],
                              "mean_teacher_forced_target_nll":
                                  receipt.get("mean_teacher_forced_target_nll")}
        campaigns.append({"workflow_id": workflow, "generation_settings_id": left["generation_settings_id"],
                          **summary})
    totals = {phase: Counter() for phase in ("baseline", "candidate")}
    for campaign in campaigns:
        for phase in totals:
            totals[phase].update(campaign[phase]["format_counts"])
    return {"schema": "cpu-generation-format-audit.v1", "claim_scope": "posthoc_original_evidence",
            "archives": len(campaigns), "campaigns": campaigns,
            "format_totals": {phase: dict(sorted(totals[phase].items())) for phase in totals},
            "original_model_outputs_published": False, "external_certification": False,
            "GPU_replay_performed": False, "causal_diagnosis_established": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("archives", nargs="+", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite scientific diagnostic record")
    report = audit(args.archives)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"archives": report["archives"], "formats": report["format_totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
