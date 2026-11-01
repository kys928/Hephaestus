# #179 restricted storage resolution — 2026-11-01

**Historical root cause: unresolved. Restricted engineering boundary: passed.**
The alternative acceptance in #179 is supported for the declared single-host
Linux local bounded profile, with complete latest checkpoints and separately
retained control history. This is not approval of arbitrary workspace storage,
an unverified persistent deployment, physical power-loss behavior or cross-host
operation. The remaining historical causal investigation is [#189](https://github.com/kys928/Hephaestus/issues/189).

## Original evidence and confidence

Both failed original databases remain unchanged. They were rehashed before and
after immutable read-only inspection with SQLite 3.53.1. Confidence in their
structural damage is high; confidence in a particular causal mechanism is low.

| Original | Direct observation |
| --- | --- |
| First database, SHA-256 `e0eeb7cfe4fb6a884b0dfd43b07085eaffe54b983480601d853f58c702c2d05c` | 68,296,704 bytes; 4,096-byte pages; physical/header page count 16,674. Index tree 8/page 2945/cell 68 points beyond EOF to page 16,706. |
| First surviving primary/index state | 31,739 primary rows. Operation index: 31,753 entries, 31,660 distinct row IDs, 93 duplicate-row-ID entries; 96 primary row IDs absent from index; 17 index row IDs absent from primary. Surviving final statuses: eight consumed and eight completed. |
| Second database, SHA-256 `445e27dbe5ce1d06be94483dbf45d07c2b4b1fd45e2cb5fa956bf2b1a4167a4a` | 20,635,648 bytes; 5,038 physical pages versus 5,120 declared header pages. Even schema inspection reports malformed. Its original append failure occurred after the reader-leak repair. |

The absent ninth publication was not reconstructed. Index repair derivatives
remain forensic derivatives, not accepted scientific completions. Structural
inconsistency does not identify whether bytes were lost during original writes,
copy/capture, synchronization or another mechanism. Neither the confirmed reader
leak, the later shared-core descriptor leak, implicit transaction behavior nor a
subsequent temporary-runtime reset is assigned as the historical cause.

Original bytes and failed campaign/checkpoint/replay records are in
[the original campaign archive](https://chatgpt.com/library?file_id=libfile_00b412f5ea708191b0482c66c0a01efe&account_id=personal),
SHA-256 `455923defc0e4cb518817a08badfd9b88c8dc6ef7ad916b6b5183a85aa54b46e`.
Its 13,354 manifested source members were reverified after the temporary-runtime
reset. Owner permissions govern archive access.

## Corrected implementation and enforceable scope

The [storage policy](../operations/local-storage-admission.md),
[durable publication contract](../operations/durable-publication.md) and
[complete snapshot runbook](../operations/campaign-snapshots.md) are the normative
operating instructions. Decision-critical workflow state still passes through
the existing state adapters; the eight-phase control spine is unchanged.

| Environment | Disposition |
| --- | --- |
| Isolated `/tmp` or `/var/tmp`, admitted local ext4/XFS/Btrfs/overlay | Tested bounded single-host process-crash profile; retention is limited to host/container lifetime unless complete checkpoints are exported. |
| Operator-managed `/var/lib/hephaestus`, admitted local ext4/XFS/Btrfs | Explicit persistent profile exists; actual mount, hardware/VFS sync semantics and backup operation require operator deployment verification. This environment did not test that deployment. |
| Overlay for persistent use, temporary roots for persistent use | Rejected. |
| Synchronized workspace, arbitrary roots, network/FUSE/9p/virtiofs/tmpfs/unknown mounts | Rejected for campaign execution. Read-only forensic inspection remains possible. |
| Multiple hosts or administrator rollback of all data and control history | Unsupported. |

Writes use checked WAL/FULL SQLite transactions and short-lived connections.
Initialization is atomic, without `executescript` implicit commits. PID,
device/inode and logical database UUID pins deny inherited or replaced handles.
Artifact/checkpoint publications flush files and ancestor directories before
atomic installation; immutable collisions do not overwrite accepted bytes.

The separate hash-chained witness records prepared intent, SQL identity and
durable acknowledgement under a kernel fence. Unknown commits preserve operation
identities and reconcile against the original live SQL marker; a new operation
ID is not a retry strategy. A missing required journal, inconsistent identity,
unresolved restore intent or integrity failure blocks native execution. The
worker fence covers execution and exception normalization; a recovered terminal
worker without an OS return code requires its matching bound result under that
fence. No exit status is invented from a PID.

Snapshots use SQLite's backup API to include WAL state, retain all regular
campaign inputs/artifacts/checkpoints, check references and seal the exact file
set. Restore requires the latest matching separately retained witness and rejects
stale snapshots. Existing roots are quarantined. Restoration preserves original
absolute paths; relocation and full portability are not claimed. Local technical
witnesses are not independent one-use scientific exposure authorities.

## Failure and endurance evidence

The maintained tests in `test_persistence_publication_durability.py`,
`test_storage_admission_and_transactions.py` and
`test_campaign_witness_and_snapshot.py` cover competing processes, bounded file
descriptors, real read-only access, migration rollback, aliases/replacement,
unknown acknowledgements, prepared/committed witness reconciliation, corruption
denial, worker ownership, stale restore and complete artifact/checkpoint checks.
Real child-process deaths exercise before/body/prepared/after-commit boundaries,
reconciliation, snapshot copy/seal and restore copy/install. These are application
process-kill tests, not a simulation of physical power failure or lying hardware.

Fresh endurance used the exact merged #187 tree
`ab2ec2a4143091fc418cf65a5c0d922c9fd4047b`, commit
`fdaef02445136fa1576339a5e85ae23bbe41ee55`, in two separately rooted cohorts.
Each ran the frozen [endurance plan](179-endurance-plan-v1.md), with authentic
native CPU training, concurrent state inspectors and cold optimizer/model replay.

The observed current mount is Linux overlay at `/`, with
`rw,nosuid,nodev,relatime`, running as UID 0. The admitted `/tmp` roots isolate
campaign paths; they do not demonstrate a separate physical filesystem from the
workspace. Other filesystem admission cases are policy fixtures, not live
persistent-deployment tests. The original host's mount, write and copy traces
were not retained, so this current observation supplies no historical cause.

| Measurement | Cohort 0 | Cohort 1 |
| --- | --- | --- |
| Programs / native cycles | 9 / 27 | 9 / 27 |
| Real optimizer updates / separate cold updates | 108 / 108 | 108 / 108 |
| Ordered state records | 22,971 | 23,054 |
| Consumed / completed final publications | 9 / 9 | 9 / 9 |
| Retained checkpoint relationships | 27 | 27 |
| Complete restores / coherent inspector reads | 9 / 69,331 | 9 / 69,455 |
| Runtime | 964.05 seconds | 967.19 seconds |
| Result SHA-256 | `34902560a65894a70a61df8726da875659abd109c3d6b96ac5c44765ec858613` | `d82e7b4bdfd1b4ff2eece93fadb4d8ebd6ed021df11b91943cf60d0edf09dc5b` |

Totals: 18 programs, 54 cycles, 2,430 native operations, 216 updates plus 216
cold updates, 46,025 records, 54 checkpoint relationships, 18 complete restores,
18 preserved zero-update bootstrap failures/repairs and 138,786 coherent reads.
Restore/cached restart preserved ordered record identities; cold verification
checked actual inference, NLL, policy decisions, lineage/memory and optimizer,
scheduler and RNG state. This is bounded engineering replication, not new
independent scientific training-seed evidence or a days-long endurance claim.

Two earlier fresh attempts were safely blocked because a recovered terminal
worker had no original OS return code. Their bytes remain retained; #187 repaired
that overrestriction using the worker fence. A still earlier observed success
and unfinished pilot lost their unpublished raw bytes in a temporary-runtime
reset. They are not counted as accepted retained evidence. The fresh completed
cohorts and their latest sealed recovery bundles were exported before continuing.

## Latest retained recovery authorities

Per-program snapshots become historical after a subsequent technical commit,
even when scientific rows do not change. Use these final latest seals only:

| Anchor | Cohort 0 | Cohort 1 |
| --- | --- | --- |
| Complete manifest SHA-256 | `d874aabb877d902bab263a2130108441ed4b2ba77488a9811baf131dec7b4233` | `67705804b849ca87ed73749e751d63d2b732d33b2f366fa276f6d5146af8789c` |
| Separate control journal SHA-256 | `a408a9ea7cab714ac2dff89db06199af571abbafb790906446602bcf7dd83568` | `baee5088cef89fd066f48334361bf75f7765a519ac2121677a7a9d801ff26772` |
| CAS objects / retained checkpoints | 4,473 / 27 | 4,480 / 27 |

Complete accepted receipts, snapshots, quarantines, inspector/cold logs, separately
retained controls, blocked attempts and forensic reverification are in
[Hephaestus_Persistence_Recovery_Evidence_20261101.zip](https://chatgpt.com/library?file_id=libfile_56606634f7448191a1a4d35048fcb152&account_id=personal),
version 2, 248,052,221 bytes, SHA-256
`8d9c8ee1fe3cf0f4d4aeda9c72dc77f6bd480fdffd8b18185df7939f63d4a64a`.
Source code remains in Git. Restoring a complete exported bundle still requires
the exact admitted logical root, separately retained latest control checkpoint,
matching source/dependencies and operator verification of the destination.

## Merged repairs and acceptance

| PR | Ordinary merge | Full PR-head CPU CI |
| --- | --- | --- |
| [#181](https://github.com/kys928/Hephaestus/pull/181), durable publication | `eb7d5166721a60d90735bffd668f44b442296a69` | 1,184 passed / 21 skipped; run 41321198610 |
| [#182](https://github.com/kys928/Hephaestus/pull/182), admission and transaction identities | `1b501547980f4bd538cb0f469fcfc2aeec419241` | 1,207 / 21; 41324340035 |
| [#183](https://github.com/kys928/Hephaestus/pull/183), witness and complete recovery | `b8960b7f1d0f2fe9a0f645d0ef4c0e2bbf85bbc7` | 1,228 / 21; 41326716285 |
| [#184](https://github.com/kys928/Hephaestus/pull/184), endurance harness | `dadd332afadf2be6629c9510cd92ad0d4521f92c` | 1,228 / 21; 41327207773 |
| [#187](https://github.com/kys928/Hephaestus/pull/187), recovered-worker fence | `fdaef02445136fa1576339a5e85ae23bbe41ee55` | 1,249 / 21; 41357127233 |

Each merge was verified to retain two parents and the tested PR tree. These are
PR-head CI results, not separately executed post-merge CI. #188 additionally
denies generic writable admission when required witness history is absent; its
full PR-head CPU suite passed 1,255 / 21, run 41358459233.

#179 is eligible for closure under its restricted-boundary alternative. Unknown
historical causality stays open in #189. Mission B can use this bounded verified
CPU foundation, with externally saved checkpoints; independent scientific design
acceptance, production exposure authority and authorization remain separate gates.
Original R10 evidence and the prior negative CPU scientific findings are unchanged.
