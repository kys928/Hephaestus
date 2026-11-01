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

Primary references: [Python connection context semantics](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-the-connection-context-manager),
[SQLite corruption and safe snapshots](https://www.sqlite.org/howtocorrupt.html),
[SQLite WAL-reset scope/fix](https://www.sqlite.org/wal.html#the_wal_reset_bug).
