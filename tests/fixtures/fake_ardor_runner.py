from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--max-steps", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--stage-name")
    parser.add_argument("--lineage-id")
    parser.add_argument("--contract-version", default="ardor_runtime_contract.v1")
    parser.add_argument("--fail-launch", action="store_true")
    parser.add_argument("--fail-runtime", action="store_true")
    parser.add_argument("--omit-metrics", action="store_true")
    parser.add_argument("--omit-checkpoint", action="store_true")
    parser.add_argument("--malformed-contract", action="store_true")
    parser.add_argument("--unsupported-state", action="store_true")
    parser.add_argument("--legacy-contract", action="store_true")
    parser.add_argument("--malformed-candidate", action="store_true")
    parser.add_argument("--omit-optional-refs", action="store_true")
    args = parser.parse_args()
    if args.fail_launch:
        return 2
    ar = Path(args.artifact_root); ar.mkdir(parents=True, exist_ok=True)
    files = {n: ar / n for n in ["ardor_metrics.json","ardor_probe.json","ardor_deterministic.json","ardor_runtime.log","dataset_manifest.json","training_recipe.json","tokenizer.json","architecture_config.json","eval_report.json","eval_pack.json"]}
    ck1, ck2 = ar / "ardor_checkpoint_step_100.ckpt", ar / "ardor_checkpoint_step_200.ckpt"
    files["ardor_runtime.log"].write_text("ardor runtime\n")
    if not args.omit_metrics: files["ardor_metrics.json"].write_text(json.dumps({"metrics": {"probe_score": 0.81}}))
    files["ardor_probe.json"].write_text("{}")
    files["ardor_deterministic.json"].write_text(json.dumps({"deterministic_passed": not args.fail_runtime}))
    if not args.omit_optional_refs:
        for k in ["dataset_manifest.json","training_recipe.json","tokenizer.json","architecture_config.json","eval_report.json","eval_pack.json"]:
            files[k].write_text("{}")
    checkpoint_candidates=[]
    checkpoint_scores={}
    if not args.omit_checkpoint:
        ck1.write_text("ckpt100"); ck2.write_text("ckpt200")
        checkpoint_scores={str(ck1):0.76,str(ck2):0.81}
        checkpoint_candidates=[{"checkpoint_ref":str(ck1),"step":100,"score":0.76,"probe_score":0.76,"content_hash":"abc123","hash_type":"sha256","metadata":{"source":"fake"}},{"checkpoint_ref":str(ck2),"step":200,"probe_score":0.81,"metadata":{}}]
        if args.malformed_candidate:
            checkpoint_candidates=[{"step":"bad"}]
    print(f"EVENT|status|100|ardor_running|{files['ardor_runtime.log']}")
    contract_path=Path(args.contract_path)
    if args.malformed_contract:
        contract_path.write_text("not-json"); return 0
    status = "unsupported" if args.unsupported_state else ("failed" if args.fail_runtime else "succeeded")
    artifacts={"metrics_ref":"" if args.omit_metrics else str(files["ardor_metrics.json"]),"probe_ref":str(files["ardor_probe.json"]),"deterministic_ref":str(files["ardor_deterministic.json"]),"runtime_log_ref":str(files["ardor_runtime.log"]),"dataset_manifest_ref":"" if args.omit_optional_refs else str(files["dataset_manifest.json"]),"training_recipe_ref":"" if args.omit_optional_refs else str(files["training_recipe.json"]),"tokenizer_ref":"" if args.omit_optional_refs else str(files["tokenizer.json"]),"architecture_config_ref":"" if args.omit_optional_refs else str(files["architecture_config.json"]),"eval_report_ref":"" if args.omit_optional_refs else str(files["eval_report.json"]),"eval_pack_ref":"" if args.omit_optional_refs else str(files["eval_pack.json"]),"checkpoint_refs":[c.get('checkpoint_ref') for c in checkpoint_candidates if c.get('checkpoint_ref')]}
    if args.legacy_contract:
        contract={"run_id":args.run_id,"status":status,"artifacts":{"metrics_ref":artifacts["metrics_ref"],"probe_ref":artifacts["probe_ref"],"deterministic_ref":artifacts["deterministic_ref"],"runtime_log_ref":artifacts["runtime_log_ref"],"checkpoint_refs":artifacts["checkpoint_refs"]},"checkpoint_scores":checkpoint_scores}
    else:
        contract={"contract_version":args.contract_version,"run_id":args.run_id,"status":status,"stage_name":args.stage_name,"lineage_id":args.lineage_id,"created_at":"2026-05-05T00:00:00Z","artifacts":artifacts,"checkpoint_candidates":checkpoint_candidates,"checkpoint_scores":checkpoint_scores}
    contract_path.write_text(json.dumps(contract, indent=2))
    return 1 if args.fail_runtime else 0

if __name__ == "__main__":
    raise SystemExit(main())
