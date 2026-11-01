"""Complete public CPU launch fixture with retained external service and restore.

All authority keys, examples and models are fixtures. This does not authorize or
generate a final protected batch, or demonstrate institutional independence.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from hephaestus.development.prospective_rehearsal import run_rehearsal
from hephaestus.evaluation.prospective_pipeline import ProspectiveCustodyRunner, generation_profile
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.state.campaign_snapshot import snapshot_campaign,restore_campaign
from hephaestus.state.campaign_witness import CampaignRollbackError
from hephaestus.state.prospective_session_store import ProspectiveSessionStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.durability import atomic_write_bytes
from hephaestus.storage.verified import canonical_bytes,read_verified_json


def run_launch_rehearsal(root,evidence_root,external_fixture_root,*,source_commit):
    root,evidence_root,external_fixture_root=map(lambda p:Path(p).absolute(),(root,evidence_root,external_fixture_root))
    locations=(root,evidence_root,external_fixture_root)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(locations) for b in locations[i+1:]):
        raise ValueError('campaign, evidence and fixture service require disjoint roots')
    evidence_root.mkdir(parents=True,exist_ok=False)
    captured={}
    def before(repository,artifacts,ledger_options):
        captured['ledger']=ledger_options['exposure_ledger']
        captured['before']=snapshot_campaign(root,repository,evidence_root/'pre-exposure-snapshot')
        atomic_write_bytes(evidence_root/'pre-exposure-receipt.json',canonical_bytes(captured['before']),overwrite=False)
    result=run_rehearsal(root,campaign_witness=True,external_fixture_root=external_fixture_root,before_assessment=before)
    repository=SQLiteStateRepository(root/'custody.sqlite3',initialize=False)
    artifacts=FileSystemArtifactStore(root/'custody-artifacts')
    session=ProspectiveSessionStore(repository).get('cpu-fixture-rehearsal')
    bundle=read_verified_json(artifacts,session['bundle_ref'])
    if bundle['schema']!='prospective-custody-bundle.v2' or session['status']!='completed':
        raise ValueError('launch fixture did not complete its external-consumption custody bundle')
    identity=repository.storage_identity()
    try:
        restore_campaign(evidence_root/'pre-exposure-snapshot',root,
            expected_manifest_sha256=captured['before']['manifest_sha256'],
            quarantine_existing_to=evidence_root/'forbidden-stale-quarantine')
    except CampaignRollbackError:
        pass
    else:
        raise ValueError('pre-exposure backup restored a consumed permission')
    if repository.storage_identity()!=identity or not root.exists():
        raise ValueError('denied stale restore changed the live campaign')
    # The independently retained SERVICE FIXTURE additionally denies reuse even
    # with a fresh nonce; campaign snapshots never contain or roll it backward.
    request=deepcopy(read_verified_json(artifacts,bundle['exposure_authority']['request_ref']))
    request['attempt_nonce']=str(uuid4())
    try:
        captured['ledger'].consume_once(request,bundle['grant'],submission=bundle['submission'])
    except PermissionError:
        pass
    else:
        raise ValueError('external fixture service reissued a spent assessment')
    exposures=read_verified_json(artifacts,bundle['holdout_ref'])['rows']
    atomic_write_bytes(evidence_root/'permanent-public-fixture-exclusions.json',
        canonical_bytes({'sample_ids':sorted(row['sample_id'] for row in exposures),
                         'purpose':'permanently public fixture; never final protected material'}),overwrite=False)
    saved=snapshot_campaign(root,repository,evidence_root/'completed-campaign-snapshot')
    restore_campaign(evidence_root/'completed-campaign-snapshot',root,
        expected_manifest_sha256=saved['manifest_sha256'],quarantine_existing_to=evidence_root/'completed-campaign-quarantine')
    trust=external_fixture_root/'fixture-exposure-trust.json'
    cold=subprocess.run([sys.executable,'-m','hephaestus.development.prospective_rehearsal','cold',str(root),
        '--expected-bundle-ref',session['bundle_ref'],'--external-trust-file',str(trust)],
        capture_output=True,text=True,timeout=180,check=False)
    atomic_write_bytes(evidence_root/'cold.stdout.json',cold.stdout.encode(),overwrite=False)
    atomic_write_bytes(evidence_root/'cold.stderr.log',cold.stderr.encode(),overwrite=False)
    if cold.returncode!=0:raise ValueError('new-process launch fixture cold verification failed; retain logs')
    checked=json.loads(cold.stdout)
    if not checked['token_decoding_verified'] or not checked['model_reexecution_verified']:
        raise ValueError('launch fixture cold generation verification was incomplete')
    # Only design METADATA is passed; no final rows, seed, seal or grant exist.
    try:
        ProspectiveCustodyRunner(SQLiteStateRepository(root/'custody.sqlite3',initialize=False),artifacts,{}).run(
            study_id='blocked-final-metadata-only',design=ProspectiveDesign(),rows=None,
            exposure_sample_ids=None,bindings=None,selection=None,grant=None)
    except PermissionError as exc:
        if 'final launch blocked' not in str(exc):raise
    else:
        raise ValueError('missing external final authorization was not denied')
    if len(captured['ledger'].repository.all('fixture_exposure_consumptions'))!=1:
        raise ValueError('launch fixture exposure consumption count changed')
    final={'schema':'prospective-launch-dry-run.v1','purpose':'public-nonconfirmatory-fixtures-only',
        'source_commit':source_commit,'execution_profile':generation_profile(),'actual_optimizer_updates':12,
        'bundle_ref':session['bundle_ref'],'complete_snapshot':saved,'cold':checked,
        'stale_pre_exposure_restore_denied':True,'retained_external_fixture_reuse_denied':True,
        'missing_external_final_authority_denied_before_material_access':True,
        'permanently_consumed_external_fixture_records':1,'rehearsal_result':result,
        'protected_final_holdout_exposed':False,'independent_authority_established':False,
        'final_scientific_authorization_conferred':False}
    atomic_write_bytes(evidence_root/'result.json',canonical_bytes(final),overwrite=False)
    return final


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('evidence_root',type=Path)
    parser.add_argument('external_fixture_root',type=Path)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    print(json.dumps(run_launch_rehearsal(args.root,args.evidence_root,args.external_fixture_root,
        source_commit=args.source_commit),indent=2))
