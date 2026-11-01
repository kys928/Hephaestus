"""Real bounded CPU campaigns, concurrent inspection, full restore and cold replay.

Run each cohort in its own process with an immutable source installation. Failed
attempts and quarantined roots are retained, never overwritten as accepted work.
"""
import hashlib
import json
import os
from pathlib import Path
from importlib.metadata import version
import platform
import sqlite3
import subprocess
import sys
import time

from hephaestus.schemas.persistence_endurance import PersistenceEndurancePlan
from hephaestus.storage.durability import atomic_write_bytes
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes
from hephaestus.state.campaign_snapshot import snapshot_campaign, restore_campaign
from hephaestus.development.cpu_autonomy_benchmark import prepare


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_inventory():
    root = Path(__file__).resolve().parents[3]
    return {str(p): _hash(p) for directory in (root/'src',root/'configs')
            for p in sorted(directory.rglob('*')) if p.is_file()
            and (directory.name != 'src' or p.suffix == '.py')}


def _state_commitment(repository):
    with repository._db.connection() as connection:
        rows = [dict(row) for row in connection.execute('SELECT * FROM state_records ORDER BY sequence')]
    operations = [r['operation_id'] for r in rows]
    sequences = [r['sequence'] for r in rows]
    if len(set(operations)) != len(rows) or sequences != sorted(set(sequences)):
        raise ValueError('committed operation identities or event order are inconsistent')
    finals = repository.all('cpu_autonomy_final')
    return {'records':len(rows), 'ordered_records_sha256':hashlib.sha256(canonical_bytes(rows)).hexdigest(),
            'completed_programs':sum(r['status']=='completed' for r in finals),
            'consumed_programs':sum(r['status']=='consumed' for r in finals)}


_INSPECTOR = '''
import json,sys,time
from pathlib import Path
from hephaestus.storage.sqlite import SQLiteStateRepository
r=SQLiteStateRepository(Path(sys.argv[1]),read_only=True)
stop=Path(sys.argv[2]); reads=0; maximum=0
while not stop.exists():
    with r._db.connection() as connection:
        connection.execute('BEGIN')
        count=connection.execute('SELECT count(*) FROM state_records').fetchone()[0]
        assert count>=maximum
        maximum=count
        connection.execute('SELECT sequence,operation_id FROM state_records ORDER BY sequence DESC LIMIT 1').fetchone()
    reads+=1
    if reads%50==0:r.assert_integrity()
    time.sleep(.01)
r.assert_integrity()
print(json.dumps({'coherent_read_transactions':reads,'maximum_records':maximum,'integrity':'ok'}))
'''


def _child(action, root, policy, seed, destination):
    command = [sys.executable,'-m','hephaestus.development.cpu_autonomy_benchmark',action,
               str(root),'--policy',policy,'--seed',str(seed)]
    with (destination.with_suffix('.stderr.log')).open('wb') as error:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=error, timeout=300, check=False)
    atomic_write_bytes(destination, result.stdout, overwrite=False)
    if result.returncode != 0:
        raise RuntimeError(f'{action} failed with {result.returncode}; retain {destination}')
    return json.loads(result.stdout)


def run_cohort(root, evidence_root, *, cohort, accepted_source_commit):
    root, evidence_root = Path(root).absolute(), Path(evidence_root).absolute()
    plan = PersistenceEndurancePlan()
    if cohort not in range(plan.cohorts) or len(accepted_source_commit) != 40:
        raise ValueError('cohort and separately verified accepted source commit required')
    evidence_root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    inventory = _source_inventory()
    contract = {'plan':plan.to_dict(),'cohort':cohort,'accepted_source_commit':accepted_source_commit,
                'source_files':inventory, 'absolute_root':str(root),
                'environment':{'python':sys.version,'sqlite':sqlite3.sqlite_version,
                    'platform':platform.platform(),
                    'frameworks':{name:version(name) for name in ('torch','transformers','peft','tokenizers','safetensors')}},
                'scope':'public tiny random-model engineering endurance; no protected data or superiority claim'}
    atomic_write_bytes(evidence_root/'precommitted-contract.json', canonical_bytes(contract), overwrite=False)
    manifest, ref = prepare(root)
    atomic_write_bytes(root/'endurance-contract.json', canonical_bytes(contract), overwrite=False)
    atomic_write_bytes(evidence_root/'source-manifest.json',canonical_bytes({'manifest_ref':ref}),overwrite=False)
    results = []
    for seed in manifest['plan']['seeds']:
      for policy in manifest['plan']['policies']:
        program = f'cpu-{policy}-{seed}'
        receipt = evidence_root/program
        receipt.mkdir()
        database = root/'state'/'production.sqlite3'
        stop = receipt/'stop-inspector'
        inspector = subprocess.Popen([sys.executable,'-c',_INSPECTOR,str(database),str(stop)],
                                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            result = _child('program',root,policy,seed,receipt/'program.json')
        finally:
            stop.touch()
            stdout, stderr = inspector.communicate(timeout=30)
            atomic_write_bytes(receipt/'inspector.stdout.json',stdout,overwrite=False)
            atomic_write_bytes(receipt/'inspector.stderr.log',stderr,overwrite=False)
            if inspector.returncode != 0:
                raise RuntimeError('concurrent inspector failed; retain this attempt')
        repository = SQLiteStateRepository(database,initialize=False)
        before = _state_commitment(repository)
        snapshot = snapshot_campaign(root,repository,receipt/'snapshot')
        restore_campaign(receipt/'snapshot',root,expected_manifest_sha256=snapshot['manifest_sha256'],
                         quarantine_existing_to=receipt/'quarantined-campaign')
        restored = SQLiteStateRepository(database,initialize=False)
        after = _state_commitment(restored)
        if before != after:
            raise ValueError('restore changed record identities, order or completed publications')
        repeated = _child('program',root,policy,seed,receipt/'restart.json')
        if result != repeated or _state_commitment(restored) != after:
            raise ValueError('restart repeated a committed scientific action')
        cold = _child('cold-program',root,policy,seed,receipt/'cold.json')
        if not cold['all_optimizer_tensors_reproduced'] or not cold['all_fixed_models_reexecuted']:
            raise ValueError('cold scientific reexecution was incomplete')
        if _source_inventory() != inventory:
            raise ValueError('endurance source changed during execution')
        entry = {'program_id':program,'state':after,'snapshot':snapshot,
                 'inspector':json.loads(stdout),'cold':cold,'same_result_after_restart':True,
                 'actual_optimizer_updates':result['actual_optimizer_updates']}
        atomic_write_bytes(receipt/'verified.json',canonical_bytes(entry),overwrite=False)
        results.append(entry)
        print(json.dumps({'cohort':cohort,'program':program,'verified':True,
                          'records':after['records']}),flush=True)
    summary = {'schema':'cpu-persistence-endurance-result.v1','cohort':cohort,
               'accepted_source_commit':accepted_source_commit,'programs':results,
               'program_count':len(results),'cycles':3*len(results),
               'actual_optimizer_updates':sum(r['actual_optimizer_updates'] for r in results),
               'cold_optimizer_updates':12*len(results),'final_state':results[-1]['state'],
               'elapsed_seconds':time.monotonic()-start,'external_certification':False,
               'historical_corruption_cause_established':False,'protected_holdout_exposed':False}
    if (len(results)!=9 or summary['final_state']['completed_programs']!=9
            or summary['final_state']['consumed_programs']!=9):
        raise ValueError('endurance completed publication counts differ from frozen dose')
    atomic_write_bytes(evidence_root/'result.json',canonical_bytes(summary),overwrite=False)
    return summary


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('evidence_root',type=Path)
    parser.add_argument('--cohort',type=int,required=True)
    parser.add_argument('--accepted-source-commit',required=True)
    args=parser.parse_args()
    run_cohort(args.root,args.evidence_root,cohort=args.cohort,
               accepted_source_commit=args.accepted_source_commit)
