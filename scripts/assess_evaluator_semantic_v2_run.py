"""Independent CPU-only paired assessment of persisted C/reference outputs."""
import hashlib,json,math
from botocore.exceptions import ClientError
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
    try:
        result=get(ROOT+'/checkpoints/step-096/evaluation/result.json')
    except ClientError as exc:
        if exc.response['Error']['Code'] not in ('NoSuchKey','404'):raise
        result=None
    complete=bool(result and result.get('status')=='evaluation_complete')
    assert training['status']=='training_complete'
    assert training['repo_sha']=='80366f51dac74141134ffdeb3f19c4f8a942c943'
    if complete:assert result['repo_sha']==training['repo_sha']
    checkpoint=next(r['artifact'] for r in training['checkpoint_artifacts'] if r['optimizer_step']==96)
    if complete:
        assert checkpoint==result['checkpoint_adapter']
        assert result['development_diagnostic_sha256']==manifest['development_diagnostic_sha256']
    listing=client.list_objects_v2(Bucket=bucket,Prefix=ROOT+'/checkpoints/step-096/evaluation/samples/')
    keys=sorted(x['Key'] for x in listing.get('Contents',[]) if x['Key'].endswith('.json'))
    crows=[r for key in keys for r in get(key)]
    assert crows
    candidate_ids={r['case_id'] for r in crows}
    assert len(candidate_ids)==len(crows) and candidate_ids<=set(cases)
    if complete:assert len(crows)==256
    from evaluator_semantic_v2_metrics import score
    pack=old.build_pack()
    for row in crows:
        rescored=score(pack,cases[row['case_id']],row['generation']['projected_output'])
        assert rescored==row['score']
    n=len(crows)
    brows=[r for i in range(1,9) for r in get(BASE+f'/samples/development_diagnostic/shard-{i:04d}.jsonl',True)]
    vrows=[r for i in range(1,9) for r in get(V1+f'/samples/development_diagnostic/shard-{i:04d}.jsonl',True)]
    def flags(rows):
        assert len(rows)==len({r['case_id'] for r in rows}) and {r['case_id'] for r in rows}<=set(cases)
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
        assert len(rows)==256 and {r['case_id'] for r in rows}==set(cases)
        reference=flags(rows);b=sum(reference[k] and not candidate[k] for k in candidate_ids);c=sum(candidate[k] and not reference[k] for k in candidate_ids)
        comparisons[name]=dict(reference_accuracy=sum(reference[k] for k in candidate_ids)/n,candidate_accuracy=sum(candidate.values())/n,
            accuracy_difference=(sum(candidate.values())-sum(reference[k] for k in candidate_ids))/n,
            reference_only_correct=b,candidate_only_correct=c,both_correct=sum(reference[k] and candidate[k] for k in candidate_ids),
            both_wrong=sum(not reference[k] and not candidate[k] for k in candidate_ids),
            exploratory_exact_paired_binomial_p=exact_p(b,c))
    head=client.head_object(Bucket=bucket,Key=checkpoint['s3_key']);assert head['ContentLength']==checkpoint['bytes']
    import run_interface_repair_v3_evaluator as v3
    counts=Counter(cases[k]['expected']['decision'] for k in candidate_ids)
    correct=Counter(cases[k]['expected']['decision'] for k,v in candidate.items() if v)
    from evaluator_semantic_v2_metrics import summarize
    observed_report=summarize([cases[r['case_id']] for r in crows],crows)
    observed_report['frozen_development_gates']=result['report']['frozen_development_gates'] if complete else None
    for state in observed_report['per_state_decision_accuracy']:
        if not counts[state]:observed_report['per_state_decision_accuracy'][state]=None
    partial=dict(enhanced_report=observed_report,completed_cases=n,total_cases=256,full_diagnostic_complete=complete,
        gates_assessable=complete,summary=v3.summarize_v3(crows),
        per_state={state:dict(total=counts[state],correct=correct[state],accuracy=correct[state]/counts[state] if counts[state] else None) for state in ('scientific_rejection','incomplete_evidence','inconclusive','recheck_required','regressed','equivalent','improved','certification_ready')},
        first_case_latency_seconds=crows[0]['generation']['latency_seconds'],
        max_case_latency_seconds=max(r['generation']['latency_seconds'] for r in crows),
        saved_generation_seconds=sum(r['generation']['latency_seconds'] for r in crows),
        saved_generated_tokens=sum(r['generation']['generated_tokens'] for r in crows),
        first_saved_case=crows[0]['case_id'],last_saved_case=crows[-1]['case_id'],
        limitation='Ordered incomplete prefix; comparisons apply only to these identical cases and cannot establish full-diagnostic improvement or acceptance.')
    report=dict(version=3,partial_evidence=partial,run_id=RUN,source_repo_sha=training['repo_sha'],checkpoint=checkpoint,
        development_sha256=manifest['development_diagnostic_sha256'],comparisons=comparisons,
        evaluation=result,training_runtime=manifest['runtime'],training_manifest=manifest,
        object_reads=reads,checkpoint_head_size_verified=True,gpu_allocations=0,s3_writes=0,
        statistical_note='Paired tests are exploratory, unadjusted and cannot replace fixed acceptance gates or independent certification.')
    Path('semantic-v2-assessment.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('V2_INDEPENDENT_ASSESSMENT '+json.dumps(dict(comparisons=comparisons,report=result['report'] if complete else partial,checkpoint=checkpoint,source_repo_sha=training['repo_sha']),sort_keys=True))
    return report
if __name__=='__main__':assess()
