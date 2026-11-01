# Local storage admission and transaction outcomes

The default research runtime is a **Linux, single-host, bounded-rehearsal**
profile rooted under `/tmp`. State, artifacts, cache and custom database paths
are checked before production composition creates directories. Database and
profile-bound artifact writes recheck their paths. Paths are bound absolutely;
parent traversal, symlinks, foreign-owned entries and nonsticky directories
writable by other users are rejected. SQLite hard-link aliases are rejected.

| Deployment | Admission and guarantee |
| --- | --- |
| Isolated `/tmp` or `/var/tmp`, local ext4/XFS/Btrfs/overlay | Bounded rehearsal; process-crash recovery within the current container/host lifetime, assuming locks and fsync are honored |
| Operator-managed `/var/lib/hephaestus/...`, local ext4/XFS/Btrfs | Eligible for explicitly configured `persistent_research`; actual deployment, backup retention and hardware/VFS sync semantics still require verification |
| Overlay for persistent research; temporary root for persistent research | Rejected; container/session retention is not persistent scientific storage |
| NFS, CIFS, FUSE, 9p, virtiofs, tmpfs, unknown mounts | Rejected; no inferred local durability or shared-host locking guarantee |
| Synchronized workspace, arbitrary mount/path outside the declared isolated root | Rejected for execution; unchanged forensic files can still be inspected read-only |
| Multiple hosts | Unsupported; a local SQLite lease is not distributed fencing |

Configure a `LocalStorageProfile` with an absolute root and purpose, then pass
it in `ProductionCompositionSettings.storage_profile`. The profile cannot
authorize compute, protected evaluation, scientific certification or promotion.
The default is deliberately suitable for bounded CPU preparation, not a claim
that the current ephemeral environment is a persistent production deployment.

State, lease and job-queue adapters share the same checked SQLite connection
and transaction implementation. Every connection is short-lived and closed;
read methods use SQLite read-only mode without changing journal settings.
Writers verify WAL and `synchronous=FULL`. A per-database initialization lock
serializes fresh constructors; DDL and its migration record share one actual
transaction, without `executescript()`'s implicit commit. Unknown schema
versions are rejected rather than silently downgraded.

Existing healthy admitted legacy databases receive a clearly technical storage
identity during explicit initialization. This is not retroactive scientific
custody or trust in an old checkpoint. Read-only legacy inspection adds no
identity. Writable `initialize=False` requires an already initialized identity.
Repository objects bind the process, file device/inode and logical database
identity. Replaced/disappeared files, in-place foreign databases and inherited
objects after fork are blocked. Reconstruct an object in a fresh process;
never carry active SQLite connections across fork. These checks cannot identify
an in-place rollback of the entire environment and are not an independent
anti-rollback ledger.

A failure during/after SQL commit, directory flush, acknowledgement validation
or connection closure raises `StateCommitUncertain`, with retained operation
identities (or job identities). It does not mean the transaction rolled back.
Inspect the exact committed receipt after reopening; reuse its original
idempotency identity only when reconciliation establishes the outcome. Never
blindly retry with a new operation ID. Prepared scientific side effects retain
their separate execution/reconciliation fences. Backup/restore must also
retain those fences; copying a database is not sufficient authorization to resume.

First acquisition of execution ownership checks full SQLite integrity, so a
runtime constructed before offline corruption cannot resume domain operations.
Job readiness also checks full integrity. Invalid originals are preserved;
no automatic index rebuild, database deletion or missing-publication creation
is permitted. Immutable backup copies use the [durable publication API](durable-publication.md).
Whole-campaign consistency, restore fencing and representative endurance are
documented in the [restricted #179 acceptance record](../scientific-reports/179-restricted-storage-resolution-20261101.md).
The original corruption mechanism remains separately unresolved in #189.

`test_storage_admission_and_transactions.py` covers actual pre-mutation denial,
aliases/replacement, real migration rollback, unknown-commit reconciliation,
mount-policy cases, fresh competing processes, descriptor lifetime, read-only
inspection and corruption after construction. Policy mount fixtures verify
denial logic; they are not live NFS or physical power-loss experiments.
