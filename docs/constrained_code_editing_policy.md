# Constrained Code Editing Policy (Proposal-Only)

## Scope

Hephaestus supports **proposal-first** constrained code editing. This protocol classifies and stores edit proposals, but does **not** execute patches or mutate files.

## Protocol guarantees

- Code editing is proposal-first and approval-gated.
- The system may classify and persist proposals, but does not execute them.
- All code-edit proposals require `operator_approval`.
- Forbidden paths are blocked and marked `not_approvable_forbidden_path`.
- Frozen eval packs, state/run history, artifacts, model weights, secrets, and external data are protected.

## Required proposal content

Every `CodeEditProposal` must provide:

- purpose
- target files
- test plan
- rollback plan
- evidence references

A proposal may be created before a concrete diff exists, but target files are still mandatory.

## Path restrictions

Default allowed path prefixes:

- `src/hephaestus/`
- `tests/`
- `docs/`
- `configs/`

Default forbidden path prefixes include:

- `.git/`, `secrets/`, `private/`, `data/`, `artifacts/`, `state/`, `runs/`, `checkpoints/`, `model_weights/`, `eval_packs/frozen/`, `frozen_eval_packs/`, `external_data/`

Also blocked:

- secret filenames such as `.env`, `.env.local`, `id_rsa`, `id_ed25519`, `credentials.json`, `token.json`
- model weight file extensions (`.pt`, `.safetensors`, `.bin`, `.ckpt`, `.pth`)
- files exceeding configured size limits (when size metadata is available)

## Risk classification

- `low`: docs/tests only
- `medium`: `src/hephaestus/` non-critical files
- `high`: policy/control/schemas/state areas
- `forbidden`: any forbidden target path

## Future integration constraint

Any future coder agent must use this protocol and remain approval-gated with path restrictions. This policy intentionally does **not** add autonomous code mutation, self-healing patching, or planner-driven rewrites.

## Stage 11 governed proposal flow

Stage 11 adds a small, auditable workflow helper for code-edit governance. It remains **proposal + governance only**:

1. A caller supplies `run_id`, `lineage_id`, `requested_by`, `purpose`, `target_files`, `rollback_plan`, `test_plan`, and optional metadata.
2. The workflow constructs a `CodeEditProposal` and normalizes it through `evaluate_code_edit_proposal` before persistence.
3. `CodeEditProposalStore` persists the evaluated record in JSONL state storage.
4. Operators may query:
   - pending proposals (`approval_required`)
   - blocked proposals
   - proposals for a run
   - proposals for a lineage
5. Approval resolution is explicit:
   - approval-required proposals may become `approved`
   - approval-required proposals may become `rejected`
   - blocked proposals cannot become approved
   - forbidden-path proposals remain `blocked`
6. Execution is not implemented. The Stage 11 execution helper only returns an auditable dry-run record:
   - unapproved proposals return `refused`
   - approved proposals return `dry_run_ready`
   - no target files are mutated

This flow does not grant autonomous file editing authority and does not bypass operator approval policy.
