"""Independent CPU-only paired assessment of persisted C/reference outputs."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import build_evaluator_semantic_invariance_repair_v1 as old
from launch_evaluator_semantic_repair_v2 import s3_client
import run_interface_repair_v1 as runtime
RUN='evaluator-semantic-v2-C-38935682810'
ROOT='hephaestus/scientific/v5/evaluator_semantic_repair_v2/'+RUN
BASE='hephaestus/scientific/v4/evaluator_development_baseline/evaluator-v3-1-dev-baseline-36998587411'
V1='hephaestus/scientific/v4/evaluator_semantic_invariance_repair/evaluator-semantic-repair-37005231044/checkpoints/step-096/evaluation'
def exact_p(b,c):
    n=b+c
    return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/(2**n)) if n else 1.
assert exact_p(0,0)==1 and exact_p(5,5)==1 and exact_p(10,0)==2/1024
def assess():
    client=s3_client();bucket=runtime.bucket();reads=[]
    def get(key,lines=False):
        obj=client.get_object(Bucket=bucket,Key=key)
        try:raw=obj['Body'].read()
        finally:obj['Body'].close()
        reads.append(dict(key=key,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        return [json.loads(s) for s in raw.decode().splitlines() if s.strip()] if lines else json.loads(raw)
    cases={r['case_id']:r for r in old.build_pack()['partitions']['evaluator']['development_diagnostic']}
    training=get(ROOT+'/training/result.json');manifest=get(ROOT+'/training/manifest.json')
    result=get(ROOT+'/checkpoints/step-096/evaluation/result.json')
    assert training['status']=='training_complete' and result['status']=='evaluation_complete'
    assert training['repo_sha']==result['repo_sha']=='80366f51dac74141134ffdeb3f19c4f8a942c943'
    checkpoint=next(r['artifact'] for r in training['checkpoint_artifacts'] if r['optimizer_step']==96)
    assert checkpoint==result['checkpoint_adapter']
    assert result['development_diagnostic_sha256']==manifest['development_diagnostic_sha256']
    crows=[r for i in range(1,9) for r in get(ROOT+f'/checkpoints/step-096/evaluation/samples/shard-{i:04d}.json')]
    brows=[r for i in range(1,9) for r in get(BASE+f'/samples/development_diagnostic/shard-{i:04d}.jsonl',True)]
    vrows=[r for i in range(1,9) for r in get(V1+f'/samples/development_diagnostic/shard-{i:04d}.jsonl',True)]
    def flags(rows):
        assert len(rows)==256 and len({r['case_id'] for r in rows})==256 and {r['case_id'] for r in rows}==set(cases)
        out={}
        for r in rows:
            parsed=r['score'].get('parsed') or {}
            correct=parsed.get('decision')==cases[r['case_id']]['expected']['decision']
            assert correct==bool(r['score']['decision_exact'])
            out[r['case_id']]=correct
        return out
    candidate=flags(crows)
    comparisons={}
    for name,rows in [('frozen_parent',brows),('V1_step96',vrows)]:
        reference=flags(rows);b=sum(reference[k] and not candidate[k] for k in cases);c=sum(candidate[k] and not reference[k] for k in cases)
        comparisons[name]=dict(reference_accuracy=sum(reference.values())/256,candidate_accuracy=sum(candidate.values())/256,
            accuracy_difference=(sum(candidate.values())-sum(reference.values()))/256,
            reference_only_correct=b,candidate_only_correct=c,both_correct=sum(reference[k] and candidate[k] for k in cases),
            both_wrong=sum(not reference[k] and not candidate[k] for k in cases),
            exploratory_exact_paired_binomial_p=exact_p(b,c))
    head=client.head_object(Bucket=bucket,Key=checkpoint['s3_key']);assert head['ContentLength']==checkpoint['bytes']
    report=dict(version=2,run_id=RUN,source_repo_sha=training['repo_sha'],checkpoint=checkpoint,
        development_sha256=manifest['development_diagnostic_sha256'],comparisons=comparisons,
        evaluation=result,training_runtime=manifest['runtime'],training_manifest=manifest,
        object_reads=reads,checkpoint_head_size_verified=True,gpu_allocations=0,s3_writes=0,
        statistical_note='Paired tests are exploratory, unadjusted and cannot replace fixed acceptance gates or independent certification.')
    Path('semantic-v2-assessment.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('V2_INDEPENDENT_ASSESSMENT '+json.dumps(dict(comparisons=comparisons,report=result['report'],checkpoint=checkpoint,source_repo_sha=training['repo_sha']),sort_keys=True))
    return report
if __name__=='__main__':assess()
