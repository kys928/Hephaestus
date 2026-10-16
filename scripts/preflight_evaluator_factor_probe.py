#!/usr/bin/env python3
"""CPU-only proof that the next 64 factor probes are immutable and label-free.

Prints ONLY metadata hashes. The generated development prompts and reference
factors stay in memory; no S3 writes, model inference or GPU allocation.
"""
import json
from evaluator_semantic_factor_probe_protocol import panel,plan_manifest

if __name__=="__main__":
    records=panel()
    print("EVALUATOR_FACTOR_PROBE_PREFLIGHT "+json.dumps(plan_manifest(records),sort_keys=True))
