#!/usr/bin/env python3
"""Reproducible non-paid preparation; never allocates GPU resources."""
import argparse
import json
from pathlib import Path
import build_evaluator_semantic_repair_v2 as b
from validate_evaluator_semantic_repair_v2 import validate_pack, validate_config, require
from evaluator_semantic_v2_supervision import target_answer

CFG_PATH = b.common.ROOT/'configs/experiments/hephaestus_evaluator_semantic_repair_v2.json'


def prepare(arm, matched_control=False):
    cfg = json.loads(CFG_PATH.read_text())
    parent = json.loads((b.common.ROOT/cfg['parent_candidate_manifest']).read_text())
    validate_config(cfg, parent)
    pack = b.build_pack(arm)
    report = validate_pack(pack)
    selected, meta = b.schedule(pack, cfg)
    require(report['pack_sha256'] == cfg['arms'][arm]['pack_sha256'], 'pack hash mismatch')
    require(b.digest(selected) == cfg['arms'][arm]['schedule_sha256'], 'selected schedule mismatch')
    if matched_control:
        require(arm == 'E', 'matched control uses E scientific evidence')
        selected = b.matched_handoff_control(selected)
    manifest = dict(version=2, arm='D_matched' if matched_control else arm,
        base_model=parent['model']['model_id'], base_revision=parent['model']['revision'],
        parent_adapter_sha256=parent['model']['adapter_sha256'], parent_adapter_s3_key=parent['model']['adapter_s3_key'],
        scientific_config_sha256=b.digest(cfg), curriculum_sha256=b.digest(pack),
        selected_schedule_sha256=b.digest(selected), selected_case_order_sha256=b.digest([r['case_id'] for r in selected]),
        target_serialization_sha256=b.digest([target_answer(r) for r in selected]),
        training=cfg['training'], decoding=parent['model']['decoding'],
        development_diagnostic_sha256=report['development_sha256'],
        evidence_inference_performed=False, paid_execution_authorized=False,
        scientific_improvement_demonstrated=False, certification_claim=False,
        comparison=cfg['arms'][arm]['comparison'], validation=report, selection=meta)
    return cfg, pack, selected, manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list('BCDE'), default='E')
    ap.add_argument('--matched-handoff-control', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    _,_,_,manifest=prepare(args.arm,args.matched_handoff_control)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # No overwrite of an immutable prepared experiment.
    with args.output.open('x') as f:
        json.dump(manifest,f,sort_keys=True,indent=2)
        f.write('\n')
    print(json.dumps(manifest,sort_keys=True))


if __name__=='__main__':
    main()
