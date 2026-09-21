# Durable Model Storage and Repository Evidence Architecture

Status: Accepted design direction  
Date: 2026-09-21

## Decision

Hephaestus must separate temporary execution storage from durable user-owned model storage.

Before any non-trivial training run that is expected to produce a persistent model, the project must have an admitted durable storage target. Supported targets may include:

- Hugging Face model repositories
- Kaggle Models
- S3-compatible object storage
- local or custom filesystem paths
- future providers such as GCS or Azure Blob Storage

A project may use transient working storage for checkpoints, caches, temporary evaluation outputs, and runtime state, but selected, rollback-worthy, or certified model artifacts must be persisted to the configured durable target according to project policy.

## Credential boundary

Language-model roles must never receive raw provider credentials.

Models may emit typed storage or repository intents. A deterministic provider layer resolves those intents against an approved storage target and a secret reference.

Example:

```text
Planner/Controller intent
    -> persist_model(lineage=L14, checkpoint=C93, target=user_primary)
    -> StorageProvider
    -> secret reference resolution
    -> provider API
```

The role context must not contain Hugging Face tokens, Kaggle credentials, GitHub tokens, cloud keys, or equivalent secrets.

## Storage admission

Before expensive training begins, Hephaestus should verify the configured durable target.

A storage admission should, where supported:

1. validate the provider and target identity;
2. resolve the credential reference;
3. authenticate;
4. verify requested write permissions;
5. verify target visibility and policy;
6. verify capacity or quota when observable;
7. write a small probe object;
8. read the probe back and verify its bytes/hash;
9. delete the probe when policy permits;
10. persist a StorageAdmissionRecord.

A failed storage admission blocks persistent training.

## Provider abstraction

The rest of Hephaestus should depend on a narrow provider contract rather than vendor-specific APIs.

```text
ArtifactStorageProvider
    LocalStorageProvider
    S3StorageProvider
    HuggingFaceModelProvider
    KaggleModelProvider
    GCSStorageProvider
    AzureBlobStorageProvider
```

Storage targets should be project-scoped rather than independently selected by each agent.

## Persistence policy

Do not upload every optimizer checkpoint to durable model registries.

Recommended policy:

```text
ephemeral optimizer checkpoint
    -> working storage only

milestone / rollback checkpoint
    -> working storage
    -> durable storage when project policy requires

selected candidate
    -> durable storage

certified model
    -> durable storage plus immutable Hephaestus provenance/evaluation metadata
```

A durable model reference should include at least:

- provider
- repository or target path
- immutable revision/version where supported
- weight hash
- lineage ID
- checkpoint ID
- model/config/tokenizer identity
- certification state
- provenance/evaluation evidence references

## Hugging Face / Kaggle policy

Hugging Face is the preferred first-class remote model-registry backend for language-model artifacts.

Kaggle Models should be supported as an additional model-registry backend.

Neither provider should be hard-coded into role logic.

Default write scopes should be narrow. Dangerous capabilities should be denied unless explicitly approved, especially:

- deletion
- changing private artifacts to public
- rewriting unrelated repositories
- destructive history operations

## Git repository evidence provider

Hephaestus may support GitHub or another Git provider for durable evidence and metadata, but generated scientific evidence must not be mixed casually into the Hephaestus source tree.

The preferred design is a separate user-selected evidence repository, or a tightly bounded metadata branch/path in a designated repository.

Model roles do not call the GitHub API directly and do not receive GitHub credentials.

They produce typed repository intents such as:

```text
persist_evidence_manifest
persist_run_summary
persist_lineage_metadata
open_evidence_pull_request
update_model_card_metadata
```

A deterministic GitRepositoryEvidenceProvider executes the action using an approved credential reference.

## GitHub default permission boundary

A GitHub evidence integration should default to the minimum useful capability.

Recommended default allowlist:

- read repository metadata;
- create a bounded evidence branch;
- create/update files only under an approved prefix;
- create commits containing approved metadata/evidence;
- optionally open a pull request for review;
- read back committed bytes and record commit SHA.

Default deny:

- force push;
- delete branches;
- rewrite arbitrary source files;
- mutate repository settings;
- read or write Actions secrets;
- change collaborators or permissions;
- create releases unless explicitly enabled;
- delete repositories;
- make private repositories public;
- bypass protected branches;
- merge its own evidence PR unless project policy explicitly permits it.

Prefer GitHub App installation credentials or similarly narrow, revocable credentials over broad long-lived personal tokens when possible.

## Evidence repository contents

Git repositories are appropriate for compact, inspectable, immutable metadata such as:

- run manifests
- dataset manifests
- model lineage records
- hashes
- certification summaries
- deterministic scorecards
- experiment plans and outcomes
- pointers to large S3/Hugging Face/Kaggle artifacts
- compact sampled outputs where policy allows
- model cards and provenance records

Large model weights, datasets, dense logs, and high-volume generated state should remain in artifact storage rather than Git.

## Integrity model

Every external write should be verifiable.

For stored models or evidence, Hephaestus should record:

- logical artifact identity
- provider
- destination
- immutable provider revision/commit where available
- SHA-256 or equivalent content digest
- byte size
- writer action ID
- credential reference identifier, never the secret value
- timestamp
- readback verification result

A successful API response is not sufficient proof of persistence.

## Governance principle

External APIs are capabilities of Hephaestus infrastructure, not capabilities handed directly to language models.

The intended flow is:

```text
Model role
    -> typed intent
    -> deterministic policy validation
    -> approval check when required
    -> provider adapter
    -> external API
    -> readback verification
    -> persisted audit record
```

This principle applies equally to model storage, GitHub, cloud storage, compute providers, and future integrations.

## Rationale

Hephaestus is expected to become increasingly autonomous and capable. Durable external storage and repository access materially increase that capability. The correct control mechanism is not to prevent useful external actions, but to make them typed, least-privilege, auditable, reversible where possible, and inaccessible to prompt-level secret exfiltration.

User ownership remains the default: compute may be temporary, but final models and their evidence should be persistable into infrastructure controlled by the user.
