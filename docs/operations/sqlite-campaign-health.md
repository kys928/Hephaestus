# SQLite campaign health and evidence snapshots

The first nine-program CPU comparison exposed a failed persistence validation:
its computation returned nine results, but the database later failed
`PRAGMA integrity_check`. A secondary-index tree referenced a page beyond the
file, several indexes disagreed with the primary table, and the ninth final
publication was absent from a direct primary-table scan. The original database
and every model/output artifact were retained. The cohort is not accepted as a
complete verified comparison.

An offline forensic derivative preserved all 31,739 surviving primary records
and rebuilt valid indexes, but it could not recover the missing final
publication. No missing historical completion was invented from the summary.
The original corrupt SHA-256 is
`e0eeb7cfe4fb6a884b0dfd43b07085eaffe54b983480601d853f58c702c2d05c`.
The full-row derivative and unsuccessful cold attempt remain separate evidence.
The ultimate cause of the corruption has not been established. SQLite 3.53.1
was used; the older documented WAL-reset defect is already patched in that
version, so it is not assigned as the cause.

Investigation independently reproduced a concrete repository defect: read
methods used `with sqlite3.Connection`, which commits/rolls back but does not
close the connection. With cyclic GC disabled, 200 reads retained **401 extra
file descriptors**. All read paths now explicitly close, including error
paths; the same probe retains zero extra descriptors. This removes resource
leaks and connections surviving until later worker creation. It is a confirmed
repair, not proof that it alone caused or cures every corruption scenario.

Additional guards and resource fixes:

- Read-only repository mode opens SQLite with `mode=ro`, skips initialization,
  and rejects write transactions.
- Readiness uses full integrity checking; production construction checks an
  existing database read-only before any migration/reconciliation writes.
- Latest-record reads stream newest-first, preserving Python value comparison
  semantics while avoiding full collection materialization.
- Repeated identical scientific evidence carry-forward no longer appends the
  same binding on every pending poll. Conflicting binding reassignment fails.
- `snapshot_to(NEW_PATH)` uses SQLite's backup API and checks the resulting
  database. It never overwrites an existing destination. Copying a live main
  database without its consistent WAL state is not an evidence snapshot.

The CPU benchmark checks integrity before final assessment and after final
publication. Cold verification opens a read-only state view with no production
action reconciliation or training service. It may explicitly reexecute fixed
optimizer jobs only in new temporary verification directories; original
models, raw outputs and publication records remain unchanged.

`test_sqlite_connection_lifetime.py` covers GC-independent descriptor closure,
concurrent append/read/reconstruction, read-only mutation denial, consistent
snapshots, deliberately corrupted secondary indexes with readable primary
rows, and duplicate/conflicting evidence registration. Existing storage,
model-evidence and multi-cycle suites also run. A new complete matched cohort
uses the unchanged scientific policy/data/dose after these engineering fixes;
the unsuccessful cohort is preserved and reported, not silently resumed or
removed.

Use host-local storage with reliable locks. Unexplained corruption remains a
material reliability risk to investigate if it recurs; passing a short stress
test is not proof that all hardware/filesystem faults are solved. No automatic
database repair, deletion, completion reconstruction or loss of raw history is
authorized by a failed readiness check.

## Complete isolated cohort and remaining investigation

A second workspace-resident cohort failed after its first completed program,
even with the reader-closure repair. Therefore that repair is not presented as
the root-cause resolution. [Issue #179](https://github.com/kys928/Hephaestus/issues/179)
tracks both failures, preserved bytes and the unassigned cause.

The third cohort used a fresh isolated host-local `/tmp` root, the same frozen
policy/data/dose and the final integrity/read-only changes. All nine programs
published with integrity checks passing. A fresh process verified 27 cycles,
1,215 completed native operations, every raw inference and all adapter,
optimizer, scheduler and RNG tensors by actual reexecution. Original database
SHA-256 stayed `d3c309414600006267524a3e58ed7db02a83fdfdcc244ad3cfc3586a884bdb89`
before and after cold verification. Its consistent SQLite backup passed
integrity as well. The [campaign report](../scientific-reports/remediation-campaign-20261101.md)
records the exact publication identities and negative behavioral results.

This is evidence for the bounded isolated run, not proof of a general cure or
of synchronized storage as the cause. Keep active databases outside synchronized
workspace trees for this profile; retain checked backups and quiescent raw
evidence. Long-running production readiness, host/filesystem diagnostics and
portable state migration remain unresolved. Do not close #179 from this run.

## Subsequent restricted-boundary acceptance

PRs #181–#184 and #187 added enforced storage admission, durable publication,
transaction identities, retained witnesses, worker fences and complete latest-only
recovery. Two fresh cohorts then verified 18 programs, 54 cycles, 46,025 state
records, 216 optimizer updates plus 216 cold updates and 18 complete restores.
Their full raw evidence and latest separate control checkpoints were retained
outside the temporary runtime. The [restricted acceptance report](../scientific-reports/179-restricted-storage-resolution-20261101.md)
supports #179's restricted-boundary alternative. It does not retroactively accept
the failed cohorts, explain their cause, validate a persistent operator deployment
or claim path portability. Historical causal uncertainty remains [#189](https://github.com/kys928/Hephaestus/issues/189).

Primary references: [Python connection context semantics](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-the-connection-context-manager),
[SQLite corruption and safe snapshots](https://www.sqlite.org/howtocorrupt.html),
[SQLite WAL-reset scope/fix](https://www.sqlite.org/wal.html#the_wal_reset_bug).
