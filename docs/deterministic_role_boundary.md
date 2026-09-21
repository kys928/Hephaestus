# Deterministic Role Boundary

Status: implemented v1  
Date: 2026-09-21

## Why

The post-training adversarial audit showed that the certified role models still fail
on several machine-verifiable cases: hard-gate precedence, provenance aliases,
checkpoint integrity, stale/duplicate actions, evidence-reference hallucination,
model admission, protocol identity, and related governance edges.

Those responsibilities do not require language-model intelligence and must not be
left to probabilistic reasoning.

## Boundary

The role model may reason about:

- causal hypotheses;
- ambiguous scientific interpretation;
- trade-offs between bounded experiments;
- expected information gain;
- uncertainty;
- whether evidence supports one scientific explanation more than another.

The deterministic control plane owns:

- evidence-reference existence;
- run and lineage ownership;
- evidence state and supersession;
- content-hash/checksum equality;
- approval validity;
- action-registry legality;
- stage action legality;
- checkpoint integrity and lineage identity;
- provenance validity;
- eval-pack/config/tokenizer identity;
- required recheck completion;
- model admission;
- budget availability;
- terminal-action/idempotency state.

## Runtime flow

1. Resolve evidence references through an EvidenceRegistry.
2. Reject unknown, foreign-lineage, foreign-run, superseded, unverified, or
   hash-mismatched evidence.
3. Merge machine-verifiable facts from trusted infrastructure.
4. Give the role model those facts as facts, not questions.
5. Let the model reason only over the ambiguous/scientific remainder.
6. Pass the model's proposed executable action through the DeterministicRoleBoundary.
7. Block any action contradicted by verified facts or governance.
8. Only an allowed effective action may reach mutation-capable infrastructure.

A model saying that a hard gate passed does not make it true. A model citing a
nonexistent approval does not create an approval. A model repeating an action whose
idempotency key is already terminal cannot trigger a duplicate mutation.

## Training consequence

Future role hardening must not train the models to become checksum comparators,
approval databases, or stage-policy interpreters.

Training examples should expose verified facts explicitly and ask the role model to
reason about what follows from them.

For example:

- Deterministic layer: `deterministic_gate_passed=false`
- Judge reasoning task: explain the correct governed next action in the presence of
  otherwise promising semantic evidence.

The frozen 600-case post-training red-team pack remains an untouched holdout. New
hardening data must use neighboring scenarios and must not copy its exact cases or
outputs.

## CI

Tests must cover the deterministic boundary separately from model quality. CI should
fail if unknown evidence, wrong-lineage evidence, invalid hashes, missing approvals,
stage-forbidden actions, or terminal duplicate actions can cross the boundary.
