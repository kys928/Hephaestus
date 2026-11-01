# Retained witnesses and whole-campaign snapshots

The production composition root enrolls fresh SQLite state in a separately
retained, host-local commit witness. This is a technical rollback fence, not an
independent scientific authority. A nonempty legacy database cannot be enrolled
automatically; retain it for read-only inspection or perform a separately reviewed
evidence migration. No missing publication is reconstructed.

`state.campaign_witness` owns the journal. Its default location is
`/tmp/hephaestus-campaign-control-<uid>/<SHA256(absolute database path)>` for
bounded rehearsals, and `/var/lib/hephaestus/control/<...>` for retained research.
The campaign and backup must be disjoint from this location. Retain this control
root separately. Removing it does not create a new authorization to resume.
The filesystem and ownership requirements in [local-storage-admission.md](local-storage-admission.md)
apply. Overlay remains admitted only for container/host-lifetime rehearsals;
retained research requires the operator-managed admitted local filesystem.

## Commit and restart

A kernel fence outside the campaign root serializes controllers, SQLite writers,
and the production directory's legacy JSON mutations. It has no timeout takeover.
Each transaction durably appends an intent binding its previous database/commit
UUIDs to a new commit UUID, commits SQL with WAL/FULL durability, then durably
acknowledges the witness before returning success. State and job operations retain
their operation identities when acknowledgement fails. Retrying a scientific
action still requires its own idempotent identity and workflow reconciliation.

A restart checks SQLite integrity and the witness chain before native execution.
An interrupted intent can be settled only against the original live database:
the old marker means SQL rollback; the new marker means committed. Any other
identity, lost journal, torn record or database replacement blocks execution.
Restore rejects every unresolved intent. It never uses an old backup to infer
that a newer transaction did not commit. Reopening a repository cannot reset a
consumed session or invent an absent completion.

## Complete snapshot and restore

Use `state.campaign_snapshot.snapshot_campaign(root, repository, destination)`
at a settled native-controller boundary. The root must contain the database,
artifacts, all retained checkpoints and the local inputs needed by the campaign.
All asynchronous training workers must be settled and reaped; prepared cross-store
actions and in-flight assessments block snapshot creation. Direct background
mutation outside these adapters is unsupported.

The snapshot holds the external campaign fence, uses SQLite's backup API to
include live WAL state, and copies other regular files. It rejects symlinks,
hard-link aliases, unknown extra databases, partial publications, missing typed
artifact references and mismatched producer checkpoint commitments. It seals the
complete file set, byte lengths, hashes, SQL identity, relationship counts and
retained witness receipt. File and directory barriers precede atomic publication.
A killed copy has only a visibly partial directory, never a completed snapshot.
No live SQLite database is hashed or copied using raw file handles.

Worker quiescence uses the lifecycle's persisted reaped exit acknowledgement,
not the continued presence/absence of a bare numeric PID. PIDs can be reused or
name different processes across PID namespaces. A terminal handle with an unknown
worker exit blocks snapshots even if its numeric PID is absent locally.

Retain the returned manifest SHA separately from the bundle. Use
`verify_snapshot(..., expected_manifest_sha256=...)` for cold readback.
`restore_campaign(..., expected_manifest_sha256=...)` validates the bundle and
requires its SQL identity and witness receipt to match the separately retained
latest settled history. A stale snapshot is rejected even when SQLite integrity
passes. The target must be absent. For a deliberate restore of the latest snapshot,
`quarantine_existing_to=<new disjoint path>` verifies quiescence and atomically
retains the old root under the same external fence before installation. Restart
constructs new repository objects; old objects reject the replaced inode.
If installation is interrupted, preserve partials and recover from the sealed
latest bundle. Never delete the control journal to make restoration succeed.

Absolute artifact paths are preserved: restore at the original logical root.
This is not path relocation or full portability. Source code and any dependencies
outside that root must independently match the campaign's retained commitments.
An entire-host loss requires a separately retained latest control journal and its
independently retained checkpoint. An older bundle alone cannot resume safely.

## Trust and limits

The witness detects campaign-only rollback while separately retained control
history survives. Administrator rollback of both the campaign and control history,
storage that lies about fsync, bad hardware, kernel failure and cross-host sharing
are outside this guarantee. Process-kill tests exercise application crash ordering;
they do not simulate a physical power failure. An entrusted external service with
independently durable, one-use exposure history is additionally required for final
protected assessments. A local witness or fixture signature never supplies that
independence. Historical SQLite corruption remains causally unresolved.
