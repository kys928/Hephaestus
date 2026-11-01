"""Storage health regressions discovered during the nine-program CPU campaign."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import gc
import os
from pathlib import Path
import sqlite3

import pytest

from hephaestus.storage.sqlite import SQLiteStateRepository,StateIntegrityError
from hephaestus.state.evidence_store import ScientificEvidenceStore,EVIDENCE_BINDINGS
from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.verified import canonical_bytes


def test_repeated_state_reads_close_connections_without_relying_on_gc(tmp_path):
    if not Path('/proc/self/fd').is_dir(): pytest.skip('Linux descriptor regression')
    repo=SQLiteStateRepository(tmp_path/'state.sqlite3');repo.append('probe',{'id':'x','value':1})
    gc.collect();was_enabled=gc.isenabled();gc.disable()
    before=len(os.listdir('/proc/self/fd'))
    try:
        for _ in range(200):
            assert repo.all('probe')==[{'id':'x','value':1}]
            assert repo.get_latest('probe','id','x')['value']==1
            assert repo.schema_version()==1
        assert len(os.listdir('/proc/self/fd'))<=before+3
    finally:
        if was_enabled:gc.enable()
        gc.collect()


def test_concurrent_writes_reads_and_reconstruction_leave_integrity_and_all_rows(tmp_path):
    path=tmp_path/'state.sqlite3';repo=SQLiteStateRepository(path)
    def work(worker):
        local=SQLiteStateRepository(path)
        for i in range(100):
            local.append('load',{'worker':worker,'i':i,'evidence':'x'*2000},operation_id=f'op-{worker}-{i}')
            assert local.get_latest('load','worker',worker)['i']==i
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work,range(4)))
    repo.assert_integrity();assert len(repo.all('load'))==400
    cold=SQLiteStateRepository(path,read_only=True)
    assert cold.ready() and len(cold.all('load'))==400
    with pytest.raises(PermissionError):cold.append('load',{'forbidden':'write'})
    snapshot=tmp_path/'consistent-snapshot.sqlite3'
    cold.snapshot_to(snapshot)
    assert SQLiteStateRepository(snapshot,read_only=True).all('load')==cold.all('load')
    with pytest.raises(FileExistsError):cold.snapshot_to(snapshot)


def test_damaged_secondary_index_blocks_health_even_if_primary_table_is_readable(tmp_path):
    path=tmp_path/'corrupt.sqlite3';repo=SQLiteStateRepository(path)
    for i in range(30):repo.append('probe',{'i':i})
    with closing(sqlite3.connect(path)) as connection:
        root=connection.execute("SELECT rootpage FROM sqlite_master WHERE name='state_latest_key'").fetchone()[0]
        page_size=connection.execute('PRAGMA page_size').fetchone()[0]
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    gc.collect()
    # Deliberate offline corruption, after every SQLite connection is closed.
    with path.open('r+b') as stream:
        stream.seek((root-1)*page_size);stream.write(b'\x7f')
    with closing(sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)) as connection:
        assert connection.execute('SELECT count(*) FROM state_records NOT INDEXED').fetchone()[0]==30
    assert not repo.ready()
    with pytest.raises(StateIntegrityError):repo.assert_integrity()
    from hashlib import sha256
    from hephaestus.production.composition import ProductionCompositionRoot,ProductionCompositionSettings
    before=sha256(path.read_bytes()).hexdigest()
    settings=ProductionCompositionSettings(state_root=tmp_path/'state',artifact_root=tmp_path/'artifacts',database_path=path)
    with pytest.raises(StateIntegrityError):ProductionCompositionRoot(settings).build()
    assert sha256(path.read_bytes()).hexdigest()==before


def test_repeated_evidence_carry_binding_is_idempotent_and_cannot_reassign_identity(tmp_path):
    repo=SQLiteStateRepository(tmp_path/'state.sqlite3');artifacts=FileSystemArtifactStore(tmp_path/'artifacts')
    ref=artifacts.put_bytes(canonical_bytes({'run_id':'source','lineage_id':'L','stage_name':'smoke_test',
        'kind':'eval_report','payload':{'real':'immutable evidence'}})).artifact_ref
    binding=ScientificEvidenceBinding('binding-1','target','source','L','smoke_test','eval_report',ref)
    store=ScientificEvidenceStore(repo,artifacts)
    for _ in range(200):store.register(binding)
    assert len(repo.all(EVIDENCE_BINDINGS))==1
    from dataclasses import replace
    with pytest.raises(PermissionError):store.register(replace(binding,target_run_id='other'))
