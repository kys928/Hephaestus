#!/usr/bin/env python3
"""Read-only candidate-vs-original-CUDA GPU replay *preflight*, never a GPU launcher.

A passing preflight only says one development repair merits a separately
authorized exact-five-role scientific GPU proposal. Issue #102 requires an
actual GPU 45/45 trace plus independent historical reference certification.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

ROLE_SET={"controller","diagnosis","planner","evaluator","judge"}
SHA=re.compile(r"[0-9a-f]{64}\Z")
COMMIT=re.compile(r"[0-9a-f]{40}\Z")


def authenticated_research_archive(path):
    with ZipFile(path) as archive:
        names=archive.namelist()
        if len(set(names)) != len(names) or "archive_sha256.json" not in names:
            raise ValueError("missing or duplicate SHA-256 archive member manifest")
        digests=json.loads(archive.read("archive_sha256.json"))
        if not isinstance(digests, dict) or set(digests) != set(names)-{"archive_sha256.json"}:
            raise ValueError("archived member inventory mismatch")
        for name,expected in digests.items():
            if (name.startswith("/") or ".." in Path(name).parts or not isinstance(expected,str)
                    or not SHA.fullmatch(expected) or sha256(archive.read(name)).hexdigest()!=expected):
                raise PermissionError("unverified or unsafe CPU research archive member")
        if "report.json" not in digests or "adapter/adapter_model.safetensors" not in digests:
            raise ValueError("real CPU research report and adapter required")
        report=json.loads(archive.read("report.json"))
        adapter=digests["adapter/adapter_model.safetensors"]
        if report.get("adapter_sha256")!="sha256:"+adapter:
            raise PermissionError("trained adapter checksum mismatch")
        return report, {"archive_sha256":"sha256:"+sha256(Path(path).read_bytes()).hexdigest(),
                        "adapter_sha256":"sha256:"+adapter}


def original_frozen_roles(path):
    raw=Path(path).read_bytes()
    frozen=json.loads(raw)
    if (frozen.get("record_id")!="hephaestus-role-model-stack-v1-certification" or
            frozen.get("status")!="certified_and_artifact_verified" or
            frozen.get("decision",{}).get("production_certified") is not True or
            frozen.get("decision",{}).get("automatic_role_dispatch_enabled") is not False or
            set(frozen.get("roles",{}))!=ROLE_SET):
        raise PermissionError("original certified role registry changed")
    for role,receipt in frozen["roles"].items():
        if (not isinstance(receipt.get("model_id"),str) or
                not isinstance(receipt.get("revision"),str) or
                not COMMIT.fullmatch(receipt["revision"]) or
                not isinstance(receipt.get("adapter_sha256"),str) or
                not SHA.fullmatch(receipt["adapter_sha256"])):
            raise PermissionError("malformed original frozen CUDA role identity")
    return frozen, "sha256:"+sha256(raw).hexdigest()


def evaluate(report, provenance, original, original_digest):
    if (report.get("schema")!="cpu-chat-json-repair-result.v1" or
            report.get("claim_scope")!="original_generated_development_only" or
            report.get("optimizer_steps")!=96 or report.get("epoch_completed") is not True or
            report.get("weights_changed") is not True or
            report.get("evaluation_holdout_used") is not False or
            report.get("GPU_used") is not False or
            report.get("model_promoted") is not False or
            report.get("external_certification") is not False or
            report.get("frozen_five_cuda_roles_executed") is not False):
        raise PermissionError("CPU scientific provenance or nonpromotion boundary invalid")
    before, after=report.get("baseline"),report.get("candidate")
    if not isinstance(before,dict) or not isinstance(after,dict):
        raise ValueError("same-sample baseline and candidate evidence required")
    for s in (before,after):
        if s.get("n")!=24 or len(s.get("examples",[]))!=24:
            raise ValueError("exactly 24 authenticated original development tasks required")
    left,right=before["examples"],after["examples"]
    if [x.get("sample_id") for x in left]!=[x.get("sample_id") for x in right]:
        raise PermissionError("CPU evaluation sample identities changed")
    if len({x["sample_id"] for x in left})!=24:
        raise ValueError("repeated development task")
    for b,c in zip(left,right):
        if (b["exact_correct"] and not c["exact_correct"]) or (b["valid_json"] and not c["valid_json"]):
            raise PermissionError("CPU candidate introduces observed hard regression")
    improvements=sum(bool(r["exact_correct"]) for r in right)-sum(bool(r["exact_correct"]) for r in left)
    valid_gain=sum(bool(r["valid_json"]) for r in right)-sum(bool(r["valid_json"]) for r in left)
    if (improvements<=0 or valid_gain<=0 or
            improvements!=report.get("development_exact_correct_change") or
            valid_gain!=report.get("development_valid_json_change")):
        raise PermissionError("no demonstrated actual generated-answer improvement; cannot propose GPU")
    return {"schema":"frozen-gpu-replay-preflight.v1",
            "cpu_development_candidate_admissible_for_external_review":True,
            "cpu_archive_sha256":provenance["archive_sha256"],
            "cpu_adapter_sha256":provenance["adapter_sha256"],
            "cpu_exact_correct_gain":improvements,"cpu_json_valid_gain":valid_gain,
            "original_role_certification_sha256":original_digest,
            "frozen_roles":{k:{"model_id":v["model_id"],"revision":v["revision"],
                             "adapter_sha256":v["adapter_sha256"]}
                            for k,v in sorted(original["roles"].items())},
            "original_five_roles_must_run_unchanged":True,
            "cpu_candidate_does_not_replace_original_cuda_roles":True,
            "gpu_execution_authorized":False,"gpu_executed":False,
            "independent_scientific_certification":False,
            "historical_holdout_custody_attested":False,
            "issue_102_closable":False,
            "remaining_gates":["approved bounded GPU allocation and 45/45 original role execution",
                               "independent original train/dev/holdout leakage audit",
                               "Granite comparisons and signed human custody/review"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cpu-archive",type=Path,required=True)
    p.add_argument("--frozen-role-certification",type=Path,
                   default=Path("docs/evidence/role-model-stack-v1/certification.json"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite original GPU preflight evidence")
    report,provenance=authenticated_research_archive(args.cpu_archive)
    original,digest=original_frozen_roles(args.frozen_role_certification)
    result=evaluate(report,provenance,original,digest)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"candidate_ready":True,"gpu_executed":False,
                      "remaining_gates":result["remaining_gates"]},sort_keys=True))


if __name__=="__main__":
    main()
