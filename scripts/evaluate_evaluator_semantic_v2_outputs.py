#!/usr/bin/env python3
"""Re-score persisted development outputs without inference or trusted input scores."""
import argparse
import json
from pathlib import Path
import build_evaluator_semantic_repair_v2 as b
import evaluator_semantic_v2_metrics as metrics


def evaluate_outputs(pack, partition, outputs, frozen_reference=None):
    if partition not in ('development_diagnostic', 'controlled_diagnostic'):
        raise ValueError('training rows cannot serve as development evidence')
    cases=pack['partitions']['evaluator'][partition]
    by_id={r['case_id']:r for r in cases}
    rows=[]
    for row in outputs:
        if row['case_id'] not in by_id:
            raise ValueError('undeclared case output')
        raw=row.get('raw_output')
        if raw is None:
            raw=row['generation']['projected_output']
        rows.append(dict(case_id=row['case_id'],score=metrics.score(pack,by_id[row['case_id']],raw)))
    return metrics.summarize(cases,rows,frozen_reference)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();outputs=[json.loads(line) for path in a.inputs for line in path.read_text().splitlines() if line.strip()]
    report=evaluate_outputs(b.build_pack('E'),'development_diagnostic',outputs)
    with a.output.open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(report,sort_keys=True))
