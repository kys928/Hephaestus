#!/usr/bin/env python3
"""GET/HEAD-only audit of frozen V1 evidence; emits no prompts or credentials."""
import hashlib
import json
from pathlib import Path
import evaluator_semantic_repair_common as c
import build_evaluator_semantic_invariance_repair_v1 as old
import run_interface_repair_v1 as runtime
from evaluate_evaluator_semantic_v2_outputs import evaluate_outputs


def audit(client, bucket, cfg):
    reads = []
    def read(key, lines=False):
        response = client.get_object(Bucket=bucket, Key=key)
        try:
            raw = response['Body'].read()
        finally:
            response['Body'].close()
        reads.append(dict(key=key, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        return [json.loads(x) for x in raw.decode().splitlines() if x.strip()] if lines else json.loads(raw)
    prefix = cfg['execution']['s3_prefix'] + '/evaluator-semantic-repair-37005231044'
    baseline = read(cfg['baseline']['s3_key'])
    training = read(prefix + '/training/result.json')
    pack, dsha, order = c.pack_and_hashes(cfg)
    if (baseline['candidate_adapter_sha256'] != cfg['required_parent_adapter_sha256']
            or baseline['development_diagnostic_sha256'] != dsha
            or baseline['development_case_order_sha256'] != order
            or training['parent_adapter_sha256'] != cfg['required_parent_adapter_sha256']
            or training['status'] != 'training_complete'):
        raise ValueError('frozen lineage/development drift')
    cases = {r['case_id']: r for r in pack['partitions']['evaluator']['development_diagnostic']}
    results = {}
    for step in cfg['training']['checkpoint_steps']:
        ep = f'{prefix}/checkpoints/step-{step:03d}/evaluation'
        result = read(ep + '/result.json')
        if result['status'] != 'evaluation_complete' or result['development_diagnostic_sha256'] != dsha:
            raise ValueError('incomplete checkpoint evaluation or diagnostic drift')
        rows = [row for shard in range(1, 9) for row in read(f'{ep}/samples/development_diagnostic/shard-{shard:04d}.jsonl', True)]
        if len(rows) != len(cases) or {r['case_id'] for r in rows} != set(cases):
            raise ValueError('checkpoint sample completeness mismatch')
        confusion = {s: {} for s in old.STATE_ORDER}
        for row in rows:
            want = cases[row['case_id']]['expected']['decision']
            parsed = row['score'].get('parsed') or {}
            got = parsed.get('decision', '<invalid>')
            confusion[want][got] = confusion[want].get(got, 0) + 1
        art = result['checkpoint_adapter']
        expected = next(x['artifact'] for x in training['checkpoint_artifacts'] if x['optimizer_step'] == step)
        if art['sha256'] != expected['sha256']:
            raise ValueError('checkpoint lineage mismatch')
        head = client.head_object(Bucket=bucket, Key=art['s3_key'])
        if head['ContentLength'] != art['bytes']:
            raise ValueError('checkpoint size mismatch')
        results[str(step)] = dict(summary=result['development_summary'], confusion=confusion,
            gates=result['checkpoint_selection_gates'], adapter=art,
            v2_rescored=evaluate_outputs(pack, 'development_diagnostic', rows, baseline['development_analysis']),
            sample_decision_accuracy=sum(row['score']['decision_exact'] for row in rows) / len(rows))
    return dict(version=2, baseline_summary=baseline['development_summary'],
        baseline_analysis=baseline['development_analysis'], checkpoints=results, object_reads=reads,
        original_training_repo_sha=training['repo_sha'], original_training_summary=training.get('training'), development_sha256=dsha,
        s3_writes=0, gpu_allocations=0, verdict='no_checkpoint_passed' if not any(r['gates']['passed'] for r in results.values()) else 'inspect_passed_checkpoint')


if __name__ == '__main__':
    cfg = json.loads(c.CFG_PATH.read_text())
    report = audit(runtime.s3_client(), runtime.bucket(), cfg)
    Path('semantic-v2-evidence-audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('SEMANTIC_V2_FROZEN_EVIDENCE ' + json.dumps(report, sort_keys=True))
