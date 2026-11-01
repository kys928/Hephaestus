"""Retained single-host commit witness outside campaign snapshots.

This detects campaign rollback while this separately retained journal survives.
It does not resist administrator rollback of the whole host and is not an
independent protected-data exposure service. Never infer scientific authority.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

from hephaestus.policy.campaign_storage import campaign_control_path
from hephaestus.policy.storage_admission import reject_symlinks
from hephaestus.schemas.campaign_storage import CampaignWitnessRecord, StateDirectoryBinding
from hephaestus.schemas.storage_admission import LocalStorageProfile
from hephaestus.storage.durability import append_bytes, atomic_write_bytes
from hephaestus.storage.verified import canonical_bytes
from .single_writer import single_writer


class CampaignRollbackError(RuntimeError):
    """Preserve both sides; never manufacture a new campaign to resume."""


def bind_state_directory(root, repository):
    root = reject_symlinks(root)
    path = root / '.campaign-storage.json'
    binding = StateDirectoryBinding(str(repository.path), repository._db.profile.to_dict()).to_dict()
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise CampaignRollbackError('state directory cannot be rebound to another database or storage profile')
    else:
        atomic_write_bytes(path, canonical_bytes(binding), overwrite=False)


@contextmanager
def state_directory_owner(root):
    """Fence legacy JSON mirrors together with the bound production database."""
    path = reject_symlinks(root) / '.campaign-storage.json'
    if not path.exists():
        yield
        return
    binding = json.loads(path.read_text())
    if binding.get('schema') != 'campaign-state-directory.v1':
        raise CampaignRollbackError('unsupported state directory binding')
    from hephaestus.storage.sqlite import SQLiteStateRepository
    repository = SQLiteStateRepository(Path(binding['database']), initialize=False,
        storage_profile=LocalStorageProfile(**binding['storage_profile']))
    witness = repository._db.witness()
    if witness is None:
        raise CampaignRollbackError('bound state directory lost its retained campaign witness')
    with witness.owner():
        with repository._db.connection() as connection:
            repository._db.assert_witness(connection, witness)
        yield


class CampaignWitness:
    def __init__(self, database, profile):
        self.database = reject_symlinks(database)
        self.root = campaign_control_path(self.database, profile)
        self.journal = self.root / 'commits.jsonl'
        self._pid = os.getpid()

    @contextmanager
    def owner(self, *, blocking=True):
        if os.getpid() != self._pid:
            raise CampaignRollbackError('reconstruct campaign witness after fork')
        reject_symlinks(self.root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.root.stat().st_uid != os.geteuid() or self.root.stat().st_mode & 0o077:
            raise CampaignRollbackError('campaign control directory must be private and owned')
        with single_writer(self.root, name='campaign', blocking=blocking):
            yield

    @staticmethod
    def _decode(raw):
        try:
            entry = json.loads(raw)
            record = entry['record']
            if (hashlib.sha256(canonical_bytes(record)).hexdigest() != entry['hash']
                    or record['schema'] != 'campaign-commit-witness.v1'
                    or record['kind'] not in {'prepared', 'committed', 'aborted'}
                    or type(record['sequence']) is not int):
                raise ValueError('invalid witness entry')
            return entry
        except (ValueError, KeyError, TypeError) as exc:
            raise CampaignRollbackError('campaign witness is corrupt or incomplete; retain it for investigation') from exc

    def tail(self):
        reject_symlinks(self.journal)
        if not self.journal.exists():
            return None
        if self.journal.stat().st_nlink != 1:
            raise CampaignRollbackError('campaign witness must not have hard-link aliases')
        with self.journal.open('rb') as stream:
            size = stream.seek(0, 2)
            if not size:
                raise CampaignRollbackError('empty retained witness cannot authorize existing state')
            stream.seek(max(0, size - 8192))
            raw = stream.read()
        if not raw.endswith(b'\n'):
            raise CampaignRollbackError('torn witness tail; automatic truncation is forbidden')
        return self._decode(raw.splitlines()[-1])

    def verify_chain(self):
        previous, sequence = '0' * 64, 0
        if not self.journal.exists():
            return None
        with self.journal.open('rb') as stream:
            for raw in stream:
                if not raw.endswith(b'\n'):
                    raise CampaignRollbackError('torn witness tail')
                entry = self._decode(raw)
                record = entry['record']
                if record['sequence'] != sequence + 1 or record['previous'] != previous:
                    raise CampaignRollbackError('campaign witness history is missing or reordered')
                previous, sequence = entry['hash'], record['sequence']
        return self.tail()

    def _append(self, kind, before, after):
        tail = self.tail()
        record = CampaignWitnessRecord(
            1 if tail is None else tail['record']['sequence'] + 1,
            '0' * 64 if tail is None else tail['hash'], kind, before, after).to_dict()
        entry = {'record': record, 'hash': hashlib.sha256(canonical_bytes(record)).hexdigest()}
        append_bytes(self.journal, canonical_bytes(entry) + b'\n')
        return entry

    def assert_current(self, identity, *, has_records=False, required=False, reconcile=True):
        tail = self.tail()
        if tail is None:
            if required or has_records:
                raise CampaignRollbackError('retained witness missing; existing scientific state cannot be automatically enrolled')
            return
        record = tail['record']
        if record['kind'] == 'prepared':
            if not reconcile:
                raise CampaignRollbackError('unresolved commit intent blocks snapshot restore')
            # Only the original live SQLite store may resolve an interrupted SQL
            # commit. A restored backup is never allowed to prove rollback.
            if identity == record['after']:
                self._append('committed', record['before'], identity)
            elif identity == record['before']:
                self._append('aborted', identity, identity)
            else:
                raise CampaignRollbackError('SQLite identity matches neither side of interrupted commit')
            return
        if identity != record['after']:
            raise CampaignRollbackError('campaign rollback or foreign state detected; retained commit witness differs')

    def prepare(self, before, after):
        return self._append('prepared', before, after)

    def committed(self, before, after):
        return self._append('committed', before, after)

    def receipt(self):
        tail = self.tail()
        if tail is None or tail['record']['kind'] == 'prepared':
            raise CampaignRollbackError('campaign witness has no settled commit')
        return {'sequence': tail['record']['sequence'], 'hash': tail['hash'],
                'control_root': str(self.root), 'identity': tail['record']['after']}
