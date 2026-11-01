# Durable local publication primitives

SQLite `synchronous=FULL` protects its own transaction; it does not flush a
separately written artifact, checkpoint or lineage filename. A successful
research publication must complete its file and directory barriers before a
state record can refer to that evidence.

The local primitives now flush file bytes before an atomic rename/link, then
flush the containing directory and ancestor entries. Immutable content paths
use a no-overwrite link; concurrent same-content publication is verified and
flushed again, while different content is rejected. Failed directory sync is
an error, so the integration sink cannot commit an artifact reference after
that failure. Legacy JSON history appends are serialized and flushed, but
still provide no multi-record transaction or automatic torn-tail repair.

The actual HF worker flushes every checkpoint component, including weights,
optimizer/scheduler/RNG state and the integrity manifest, before publishing
the checkpoint directory. The parent directory is then flushed before the
durable checkpoint record is written. Worker JSON and the pre-optimizer
progress fence share the same file/name durability contract. A failed barrier
can leave unacknowledged bytes; those bytes alone never authorize a lineage
transition or an automatic post-update retry.

`SQLiteStateRepository.snapshot_to(NEW_PATH)` uses SQLite's backup API into
an explicitly partial sibling. It checks integrity, closes the private backup,
flushes it and publishes its final name without overwriting another file.
An interrupted backup never has that completed name. A failure after publication
but before acknowledgement remains an unknown caller outcome; inspect the
existing identity instead of blindly overwriting or replaying an action.
Never hash or copy a live main SQLite file through ordinary file I/O while
another thread owns an SQLite transaction. Use a closed private backup for
hashing and transport.

These are Linux/POSIX primitives with local atomic link/rename and real fsync
semantics. They do not prove that an arbitrary mount, network share, workspace
snapshot service, container lifecycle or disk honors those requirements. A
process kill is not a physical power-loss experiment. The enforceable campaign
storage profile and full restore/endurance validation are separate work under
[#179](https://github.com/kys928/Hephaestus/issues/179); historical causality
remains unassigned by these repairs.

`test_persistence_publication_durability.py` exercises actual artifact/state
integration, lineage and worker publication, injected directory-I/O failure,
a process killed during a real SQLite backup and a genuine optimized tiny
checkpoint whose components must all be flushed before directory publication.
Six new regressions failed on the unchanged `68119ca...` source before repair.
Existing checkpoint, failed-run recovery, storage and actual training-dose
tests also cover the integrated callers. No protected holdout or paid resource
is involved.

Primary references: [SQLite atomic commit](https://www.sqlite.org/atomiccommit.html),
[safe backups and corruption boundaries](https://www.sqlite.org/howtocorrupt.html),
[WAL assumptions](https://www.sqlite.org/wal.html).
