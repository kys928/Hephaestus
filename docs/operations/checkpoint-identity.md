# Accepted checkpoint identity and recovery

A checkpoint's own manifest proves internal consistency. It does not prove that
the bytes are the checkpoint accepted earlier. Future HF lifecycle receipts
retain `checkpoint_manifest_hashes` outside the checkpoint directory. The staged
checkpoint resolver verifies that producer commitment and writes a lineage-bound
`CheckpointIdentity` through `state/checkpoint_identity_store.py` before exposing
the checkpoint to evaluation and continuation.

`continue_from_checkpoint` requires both lineage membership and the independently
retained manifest digest. Replacing weights and rebuilding the manifest at the
same path fails. Conflicting retained commitments fail closed. The same
preconditions run during prepared-action recovery while the original before-state
is still current. If the exact after-state is already committed, recovery only
finishes the journal and mirror; it does not repeat or revoke a past mutation.

## Legacy migration

An old path without a retained digest cannot continue automatically. A trusted
operator/custodian must recover its original digest from a previously retained
producer receipt, immutable archive or evaluation record, verify that provenance,
then retain an explicit `CheckpointIdentity(lineage_id, checkpoint_ref,
manifest_hash, evidence_ref)`. This is a deterministic state import, not a model
proposal or console action. Computing a new digest from the current path alone
does **not** establish historical identity. Missing original evidence remains a
blocker. Original R10 records are preserved and are not migrated by this change.

These are integrity controls within the trusted host/state boundary. They do not
protect against an attacker who can rewrite both the accepted state repository
and all producer evidence, nor do they authorize checkpoint promotion.
