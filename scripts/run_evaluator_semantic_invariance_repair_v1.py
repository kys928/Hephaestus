#!/usr/bin/env python3
"""Entrypoint for Evaluator V3.1 semantic-invariance repair training/evaluation."""
from __future__ import annotations
import argparse,os
import evaluator_semantic_repair_common as common
import evaluator_semantic_repair_eval as evaluation
import evaluator_semantic_repair_train as training

def main()->int:
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--train',action='store_true');g.add_argument('--evaluate-checkpoint',type=int);a=ap.parse_args();cfg,parent=common.cfg_and_parent();run_id=os.environ.get('HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID','').strip()
    if not run_id:raise RuntimeError('HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID required')
    return training.run(cfg,parent,run_id) if a.train else evaluation.run(cfg,parent,run_id,int(a.evaluate_checkpoint))
if __name__=='__main__':raise SystemExit(main())
