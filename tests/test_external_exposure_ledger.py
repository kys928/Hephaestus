"""Adversarial public software fixtures, never independent scientific evidence."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import subprocess
import sys
import threading
from uuid import uuid4

import pytest

from test_prospective_custody_pipeline import custody
from hephaestus.development.fixture_exposure_ledger import FixtureExposureLedger
from hephaestus.evaluation.prospective_pipeline import ProspectiveCustodyRunner,verify_cold_bundle,digest
from hephaestus.schemas.exposure_ledger import ExposureReservation
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.state.exposure_authority import verify_exposure_receipt
from hephaestus.state.single_writer import WriterBusyError
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes,read_verified_json


@pytest.fixture
def external_fixture(custody,tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
    old,args,calls=custody
    key=Ed25519PrivateKey.generate()
    keys={'fixture-ledger-key':key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex()}
    ledger=FixtureExposureLedger(SQLiteStateRepository(tmp_path/'retained-external-fixture'/'ledger.sqlite3',campaign_witness=True),
        private_key=key,authority_id='fixture-ledger-key',trusted_custodian_keys=old.trusted_public_keys,
        namespace='fixture/public-unit-only')
    runner=ProspectiveCustodyRunner(old.repository,old.artifacts,old.trusted_public_keys,
        exposure_ledger=ledger,trusted_exposure_keys=keys,trusted_ledger_namespace=ledger.namespace)
    return runner,args,calls,ledger,keys


def test_missing_or_fixture_authority_blocks_final_before_reading_any_protected_material(external_fixture):
    runner,_,calls,_,_=external_fixture
    class Unavailable:
        def __iter__(self):raise AssertionError('protected material was touched')
        def __getitem__(self,key):raise AssertionError('protected material was touched')
    args=dict(study_id='not-a-final-session',design=ProspectiveDesign(),rows=Unavailable(),
              exposure_sample_ids=Unavailable(),bindings=Unavailable(),selection=Unavailable(),grant=Unavailable())
    with pytest.raises(PermissionError,match='final launch blocked'):runner.run(**args)
    runner.exposure_ledger=None
    with pytest.raises(PermissionError,match='final launch blocked'):runner.run(**args)
    assert calls==[] and runner.sessions.get('not-a-final-session') is None


def test_fixture_receipt_is_persisted_and_requires_separately_pinned_cold_trust(external_fixture):
    runner,args,calls,ledger,keys=external_fixture
    bundle=runner.run(**args)
    assert len(calls)==4 and bundle['schema']=='prospective-custody-bundle.v2'
    assert bundle['exposure_authority']['authority_scope']=='nonconfirmatory-fixture'
    state=runner.sessions.get(args['study_id'])
    assert state['exposure_receipt_ref']==bundle['exposure_authority']['receipt_ref']
    assert len(ledger.repository.all('fixture_exposure_consumptions'))==1
    assert runner.run(**args)==bundle and len(calls)==4
    with pytest.raises(PermissionError,match='separately entrusted'):
        verify_cold_bundle(bundle,runner.artifacts,expected_bundle_ref=state['bundle_ref'],trusted_public_keys=runner.trusted_public_keys)
    checked=verify_cold_bundle(bundle,runner.artifacts,expected_bundle_ref=state['bundle_ref'],
        trusted_public_keys=runner.trusted_public_keys,trusted_exposure_keys=keys,trusted_ledger_namespace=ledger.namespace)
    assert checked['verified'] and not checked['external_independent_certification']


def test_local_backup_restore_cannot_reset_external_consumption_even_without_local_witness(external_fixture,tmp_path):
    runner,args,calls,ledger,keys=external_fixture
    # Deliberately simulate a stronger local rollback than the supported whole-
    # campaign restore allows. This legacy disposable unit store has no witness;
    # the external service lives elsewhere and is never restored with it.
    backup=tmp_path/'pre-exposure.sqlite3'
    runner.repository.snapshot_to(backup)
    runner.run(**args); calls.clear()
    restored=ProspectiveCustodyRunner(SQLiteStateRepository(backup,initialize=False),runner.artifacts,
        runner.trusted_public_keys,exposure_ledger=ledger,trusted_exposure_keys=keys,
        trusted_ledger_namespace=ledger.namespace)
    with pytest.raises(PermissionError,match='permanently consumed'):restored.run(**args)
    assert calls==[] and restored.sessions.get(args['study_id'])['status']=='failed'
    assert len(ledger.repository.all('fixture_exposure_consumptions'))==1


def test_unknown_remote_ack_blocks_generation_and_permanently_spends_fixture_material(external_fixture):
    runner,args,calls,ledger,_=external_fixture
    class AckLost:
        namespace=ledger.namespace;authority_scope=ledger.authority_scope
        def consume_once(self,*a,**kw):
            ledger.consume_once(*a,**kw)
            raise TimeoutError('service committed but response was lost')
    runner.exposure_ledger=AckLost()
    with pytest.raises(TimeoutError):runner.run(**args)
    assert calls==[] and runner.sessions.get(args['study_id'])['status']=='failed'
    assert len(ledger.repository.all('fixture_exposure_consumptions'))==1
    runner.exposure_ledger=ledger
    with pytest.raises(PermissionError,match='consumed'):runner.run(**args)


def test_wrong_external_key_consumes_no_generation_permission(external_fixture):
    runner,args,calls,ledger,_=external_fixture
    runner.trusted_exposure_keys={'fixture-ledger-key':'00'*32}
    with pytest.raises(PermissionError,match='exactly scoped'):runner.run(**args)
    assert calls==[] and len(ledger.repository.all('fixture_exposure_consumptions'))==1
    assert runner.sessions.get(args['study_id'])['status']=='failed'


def test_old_signed_receipt_cannot_be_replayed_for_fresh_nonce_after_local_rollback(external_fixture,tmp_path):
    runner,args,calls,ledger,keys=external_fixture
    backup=tmp_path/'before.sqlite3';runner.repository.snapshot_to(backup)
    bundle=runner.run(**args);calls.clear()
    old_receipt=read_verified_json(runner.artifacts,bundle['exposure_authority']['receipt_ref'])
    class Replay:
        namespace=ledger.namespace;authority_scope=ledger.authority_scope
        def consume_once(self,*a,**kw):return old_receipt
    restarted=ProspectiveCustodyRunner(SQLiteStateRepository(backup,initialize=False),runner.artifacts,
        runner.trusted_public_keys,exposure_ledger=Replay(),trusted_exposure_keys=keys,
        trusted_ledger_namespace=ledger.namespace)
    with pytest.raises(PermissionError,match='exactly scoped'):restarted.run(**args)
    assert calls==[]


def test_fixture_service_denies_concurrent_changed_study_ids_for_same_material(external_fixture):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
    runner,args,_,ledger,_=external_fixture
    key=Ed25519PrivateKey.generate()
    ledger.trusted_custodian_keys['second-fixture']=key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex()
    material='sha256:'+'7'*64
    now=datetime.now(timezone.utc)
    requests=[]
    for index in range(2):
        study=f'public-concurrency-{index}'
        submission={'study_id':study,'holdout_commitment':material}
        payload={'study_id':study,'purpose':'rehearsal','allowed_sessions':1,'submission_commitment':digest(submission),
            'authority_id':'second-fixture','issued_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        grant={'payload':payload,'signature_hex':key.sign(canonical_bytes(payload)).hex()}
        reservation=ExposureReservation(study,'rehearsal',digest(submission),material,digest(grant),str(uuid4())).to_dict()
        requests.append((reservation,grant,submission))
    barrier=threading.Barrier(2)
    def consume(request):
        barrier.wait()
        try:return ledger.consume_once(request[0],request[1],submission=request[2])
        except (PermissionError,WriterBusyError):return None
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(consume,requests))
    assert sum(result is not None for result in results)==1
    assert len(ledger.repository.all('fixture_exposure_consumptions'))==1
    # Changing material while keeping the old signed submission must not let a
    # caller turn one signed grant into permissions for arbitrary other data.
    changed=deepcopy(requests[0][0]);changed['material_commitment']='sha256:'+'8'*64
    with pytest.raises(PermissionError,match='not bound'):
        ledger.consume_once(changed,requests[0][1],submission=requests[0][2])


def test_corrupt_receipt_and_namespace_do_not_pass_independent_readback(external_fixture):
    runner,args,_,ledger,keys=external_fixture
    bundle=runner.run(**args)
    reservation=read_verified_json(runner.artifacts,bundle['exposure_authority']['request_ref'])
    receipt=read_verified_json(runner.artifacts,bundle['exposure_authority']['receipt_ref'])
    receipt['payload']['monotonic_sequence']=True
    with pytest.raises(PermissionError):verify_exposure_receipt(receipt,reservation=reservation,
        trusted_keys=keys,namespace=ledger.namespace,authority_scope=ledger.authority_scope)


@pytest.mark.parametrize('cut',['receipt_publication','generation'])
def test_real_process_death_after_external_consumption_never_reissues_permission(external_fixture,tmp_path,cut):
    from cryptography.hazmat.primitives.serialization import Encoding,PrivateFormat,NoEncryption
    runner,args,calls,ledger,keys=external_fixture
    specification={'repository':str(runner.repository.path),'artifacts':str(runner.artifacts.root),
        'ledger':str(ledger.repository.path),'custodian_keys':runner.trusted_public_keys,
        'ledger_private_fixture_hex':ledger.key.private_bytes(Encoding.Raw,PrivateFormat.Raw,NoEncryption()).hex(),
        'ledger_public_fixture_keys':keys,'cut':cut,'args':{**args,'design':args['design'].to_dict(),
            'bindings':[b.to_dict() for b in args['bindings']],'exposure_sample_ids':sorted(args['exposure_sample_ids'])}}
    path=tmp_path/'private-unit-fixture-only.json';path.write_text(json.dumps(specification))
    child=subprocess.run([sys.executable,'-c','''
import os,json,sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from hephaestus.development.fixture_exposure_ledger import FixtureExposureLedger
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.schemas.prospective_model_binding import ProspectiveModelBinding
from hephaestus.state.prospective_session_store import ProspectiveSessionStore
from hephaestus.evaluation import prospective_pipeline as p
x=json.loads(Path(sys.argv[1]).read_text());a=x['args']
a['design']=ProspectiveDesign.from_dict(a['design'])
a['bindings']=[ProspectiveModelBinding.from_dict(b) for b in a['bindings']]
a['exposure_sample_ids']=set(a['exposure_sample_ids'])
ledger=FixtureExposureLedger(SQLiteStateRepository(Path(x['ledger']),initialize=False),
 private_key=Ed25519PrivateKey.from_private_bytes(bytes.fromhex(x['ledger_private_fixture_hex'])),
 authority_id='fixture-ledger-key',trusted_custodian_keys=x['custodian_keys'],namespace='fixture/public-unit-only')
p.generation_profile=lambda:{'stub':True}
p.verify_model_binding=lambda *args:None
p.generate_cpu=lambda *args,**kwargs:os._exit(78)
if x['cut']=='receipt_publication':ProspectiveSessionStore.retain_authority_receipt=lambda *args:os._exit(78)
p.ProspectiveCustodyRunner(SQLiteStateRepository(Path(x['repository']),initialize=False),
 FileSystemArtifactStore(Path(x['artifacts'])),x['custodian_keys'],exposure_ledger=ledger,
 trusted_exposure_keys=x['ledger_public_fixture_keys'],trusted_ledger_namespace=ledger.namespace).run(**a)
''',str(path)],capture_output=True,text=True,timeout=30)
    assert child.returncode==78,child.stderr
    assert len(ledger.repository.all('fixture_exposure_consumptions'))==1
    assert runner.sessions.get(args['study_id'])['status']=='consumed'
    with pytest.raises(PermissionError,match='consumed'):runner.run(**args)
    assert calls==[]


def test_generation_failure_preserves_every_already_published_raw_sample(external_fixture,monkeypatch):
    from hephaestus.evaluation import prospective_pipeline as pipeline
    runner,args,_,_,_=external_fixture
    original=pipeline.generate_cpu
    def fail_candidate(binding,rows,design,**kwargs):
        if binding.seed is not None:raise RuntimeError('fixture failure after baseline raw samples')
        return original(binding,rows,design,**kwargs)
    monkeypatch.setattr(pipeline,'generate_cpu',fail_candidate)
    with pytest.raises(RuntimeError,match='fixture failure'):runner.run(**args)
    session=runner.sessions.get(args['study_id'])
    links=read_verified_json(runner.artifacts,session['partial_raw_index_ref'])['links']
    assert session['status']=='failed' and len(links)==len(args['rows'])
    assert all(link['model_label']=='baseline' for link in links)
    assert len({link['sample_id'] for link in links})==len(links)
    for link in links:assert read_verified_json(runner.artifacts,link['raw_sample_ref'])['sample_id']==link['sample_id']
