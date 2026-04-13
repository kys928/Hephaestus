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
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--omit-metrics", action="store_true")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    metrics = artifact_root / "metrics.json"
    probe = artifact_root / "probe.json"
    deterministic = artifact_root / "deterministic.json"
    checkpoint = artifact_root / "checkpoint_step_200.ckpt"

    if not args.omit_metrics:
        metrics.write_text(json.dumps({"probe_score": 0.72, "toxicity": 0.03}))
    probe.write_text(json.dumps({"sample": "ok"}))
    deterministic.write_text(json.dumps({"deterministic_passed": not args.fail}))
    if not args.fail:
        checkpoint.write_text("checkpoint")

    print(f"EVENT|status|100|running|")
    print(f"EVENT|metric|200|metrics_written|{metrics}")
    print(f"EVENT|probe|200|probe_written|{probe}")
    if args.fail:
        print(f"EVENT|deterministic_check|200|deterministic checks fail|{deterministic}")
        return 1
    print(f"EVENT|deterministic_check|200|deterministic checks pass|{deterministic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
