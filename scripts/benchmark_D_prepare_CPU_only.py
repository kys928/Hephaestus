#!/usr/bin/env python3
"""Time the frozen D curriculum preparation without GPU, S3 or pod."""
import json,time
import prepare_evaluator_semantic_repair_v2 as mod

def main():
    start=time.monotonic()
    print("D_PREPARE_TIMING "+json.dumps({"stage":"start"}),flush=True)
    cfg,pack,selected,m=mod.prepare("D")
    elapsed=time.monotonic()-start
    print("D_PREPARE_TIMING "+json.dumps({
        "stage":"done","elapsed_seconds":round(elapsed,3),
        "selected_cases":len(selected),"arm":m["arm"],
        "scientific_config_sha256":m["scientific_config_sha256"],
        "schedule_sha256":m["selected_schedule_sha256"],
    },sort_keys=True),flush=True)
if __name__=="__main__":
    main()
