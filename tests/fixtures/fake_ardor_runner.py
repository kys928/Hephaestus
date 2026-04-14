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
    parser.add_argument("--fail-launch", action="store_true")
    parser.add_argument("--fail-runtime", action="store_true")
    parser.add_argument("--omit-metrics", action="store_true")
    parser.add_argument("--omit-checkpoint", action="store_true")
    parser.add_argument("--malformed-contract", action="store_true")
    parser.add_argument("--unsupported-state", action="store_true")
    args = parser.parse_args()

    if args.fail_launch:
        return 2

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    metrics = artifact_root / "ardor_metrics.json"
    probe = artifact_root / "ardor_probe.json"
    deterministic = artifact_root / "ardor_deterministic.json"
    runtime_log = artifact_root / "ardor_runtime.log"
    checkpoint_100 = artifact_root / "ardor_checkpoint_step_100.ckpt"
    checkpoint_200 = artifact_root / "ardor_checkpoint_step_200.ckpt"

    runtime_log.write_text("ardor runtime\n")
    if not args.omit_metrics:
        metrics.write_text(json.dumps({"metrics": {"probe_score": 0.81, "toxicity": 0.02}}))
    probe.write_text(json.dumps({"samples": ["ok"]}))
    deterministic.write_text(json.dumps({"deterministic_passed": not args.fail_runtime}))

    checkpoint_refs: list[str] = []
    checkpoint_scores: dict[str, float] = {}
    if not args.omit_checkpoint:
        checkpoint_100.write_text("ckpt100")
        checkpoint_200.write_text("ckpt200")
        checkpoint_refs = [str(checkpoint_100), str(checkpoint_200)]
        checkpoint_scores = {str(checkpoint_100): 0.76, str(checkpoint_200): 0.81}

    print(f"EVENT|status|100|ardor_running|{runtime_log}")
    if not args.omit_metrics:
        print(f"EVENT|metric|200|ardor_metrics_ready|{metrics}")
    print(f"EVENT|probe|200|ardor_probe_ready|{probe}")
    print(f"EVENT|deterministic_check|200|deterministic checks {'fail' if args.fail_runtime else 'pass'}|{deterministic}")

    contract_path = Path(args.contract_path)
    if args.malformed_contract:
        contract_path.write_text("not-json")
        return 0

    status = "unsupported" if args.unsupported_state else ("failed" if args.fail_runtime else "succeeded")
    contract = {
        "run_id": args.run_id,
        "status": status,
        "artifacts": {
            "metrics_ref": "" if args.omit_metrics else str(metrics),
            "probe_ref": str(probe),
            "deterministic_ref": str(deterministic),
            "runtime_log_ref": str(runtime_log),
            "checkpoint_refs": checkpoint_refs,
        },
        "checkpoint_scores": checkpoint_scores,
    }
    contract_path.write_text(json.dumps(contract, indent=2))
    return 1 if args.fail_runtime else 0


if __name__ == "__main__":
    raise SystemExit(main())
