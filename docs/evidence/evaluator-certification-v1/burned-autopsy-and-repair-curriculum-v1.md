# Evaluator sealed-v1 burned autopsy and semantic-invariance repair curriculum

## Status

The fresh sealed Evaluator certification run `evaluator-sealed-cert-36929796355` is scientifically rejected and permanently burned for certification use. Its 256 certification/regression rows are used below only as diagnostic evidence. No new inference or training was performed during this autopsy. The rejected candidate remains frozen, and no production promotion or automatic dispatch is authorized.

## Diagnostic source

- Candidate: `evaluator-v3.1-boundary-v1`
- Granite revision: `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`
- Adapter SHA-256: `913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed`
- Burned sealed pack SHA-256: `1314d28c978538ca8bb5527c24e4200d5463f7ad90b084aaae1e656d0b530d25`
- Source run: `evaluator-sealed-cert-36929796355`
- Full autopsy: `hephaestus/scientific/v4/evaluator_certification/evaluator-sealed-cert-36929796355/burned-autopsy-v1.json`
- Conditional autopsy: `hephaestus/scientific/v4/evaluator_certification/evaluator-sealed-cert-36929796355/burned-autopsy-conditionals-v1.json`
- Autopsy workflow run: `36994649504`

## Combined 256-case state accuracy

| Scientific state | Exact decisions | Accuracy |
| --- | ---: | ---: |
| `certification_ready` | 32/32 | 100.000% |
| `incomplete_evidence` | 32/32 | 100.000% |
| `regressed` | 32/32 | 100.000% |
| `scientific_rejection` | 31/32 | 96.875% |
| `inconclusive` | 23/32 | 71.875% |
| `equivalent` | 19/32 | 59.375% |
| `recheck_required` | 17/32 | 53.125% |
| `improved` | 8/32 | 25.000% |

The failure is therefore strongly state-conditional rather than a uniform collapse. Four states remain robust at or above 95%, while four neighboring states account for almost all scientific-state instability.

## Exact confusion structure

The burned evidence contains these scientific-state misses:

- `improved -> continue_lineage_best`: 13. These are invalid action tokens emitted in the `decision` field.
- `equivalent -> certification_ready`: 10.
- `recheck_required -> inconclusive`: 10.
- `improved -> certification_ready`: 9.
- `inconclusive -> incomplete_evidence`: 6.
- `recheck_required -> scientific_rejection`: 5.
- `inconclusive -> scientific_rejection`: 3.
- `improved -> recheck_required`: 2.
- `equivalent -> recheck_required`: 2.
- `equivalent -> scientific_rejection`: 1.
- `scientific_rejection -> regressed`: 1.

There are 13 invalid decisions in total; every one is an action token, every one occurs on an `improved` case, and every one is `continue_lineage_best`. This is a distinct field/authority-contamination failure, not merely a neighboring-state classification error.

## Planner-handoff effect

Across all states, direct/rehearsal cases score 79.6875% decision exact while Planner-handoff cases score 74.4792%. Handoff is therefore a material stressor but does not explain the entire failure.

The important exception is `improved`:

- overall: 8/32 = 25.0%
- direct: 6/8 = 75.0%
- Planner handoff: 2/24 = 8.33%
- all 24 `improved` handoff cases contain incorrect upstream action advice
- 24/32 `improved` cases trigger upstream-copy violation telemetry

Thus the dominant `improved` failure is upstream/action-field contamination. Granite often follows the Planner's policy vocabulary strongly enough to place `continue_lineage_best` in the scientific-state field.

By contrast, `equivalent`, `inconclusive`, and `recheck_required` remain weak in direct cases as well. Their failures therefore require semantic/lexical repair in addition to handoff isolation.

## Wording/template sensitivity

The fresh sealed evidence exposed severe phrase-family sensitivity:

- `equivalent`: template families 0 and 2 score 90% and 100%, while template family 3 scores 0/11. The failed family includes the formulation that effect magnitude remains inside a predefined neutral interval. Direct and handoff cases both fail, so this is genuine wording sensitivity rather than a Planner-only effect.
- `inconclusive`: template family 2 scores 4/12 = 33.3%, with six cases mapped to `incomplete_evidence` and two to `scientific_rejection`; the other principal families are strong. This exposes a provenance-versus-evidence-completeness boundary weakness.
- `recheck_required`: template family 0 scores 0/5 and is always called `scientific_rejection`; template family 2 scores 4/12 and is usually called `inconclusive`; template family 3 scores 11/11. This is strong evidence of variance-language brittleness.
- `improved`: direct performance is much stronger than handoff performance, but individual wording families also vary. The dominant failure mechanism remains action/upstream contamination.

## Training-shortcut diagnosis

The V3/V3.1 training curriculum contained only three evidence fact templates per scientific state. In addition, every `improved` training case included a model-visible note explicitly saying that improvement authorizes continuation from the candidate checkpoint and is neither lineage fallback nor certification; every `equivalent` training case included a model-visible note explicitly saying equivalence never upgrades authority and to retain the lineage best absent certification-ready evidence. V3.1 reused that 1,536-case training curriculum unchanged.

The fresh sealed pack removed these state-specific hints and asked the model to infer the state from evidence alone. Together with the sealed failure pattern, this strongly suggests that V3.1 learned useful state structure but also learned shortcut features tied to the narrow training wording and state-specific situation notes. This is a supported causal hypothesis, not proof from a controlled ablation.

## Repair objective

The next repair should preserve the V3.1 ontology and deterministic boundary while teaching semantic invariance. Exact burned prompts are not copied into training. The model-visible problem is rebuilt from independent scientific variables:

1. hard-gate pass/fail,
2. evidence completeness,
3. provenance validity,
4. effect class: positive / neutral / negative,
5. variance stability,
6. mandatory recheck satisfaction.

The deterministic conceptual precedence represented in the curriculum is:

`hard gate failure -> scientific_rejection`

`else missing evidence -> incomplete_evidence`

`else invalid provenance -> inconclusive`

`else excessive variance -> recheck_required`

`else negative material effect -> regressed`

`else neutral effect -> equivalent`

`else positive material effect + satisfied mandatory recheck -> certification_ready`

`else positive material effect -> improved`

This factorization makes the distinction between neighboring states a change in scientific evidence rather than a change in label-associated prose.

## New curriculum

- Builder: `scripts/build_evaluator_semantic_invariance_repair_v1.py`
- Curriculum ID: `hephaestus-evaluator-semantic-invariance-repair-v1`
- Canonical SHA-256: `0d01fe1853742d1259a333fa9a92dec46e8789b69f2019ed34e17d815fc01a3c`
- Validation workflow run: `36994488837` — passed

Partitions:

- 1,536 training cases
- 8 format-preflight cases
- 256 development-diagnostic cases, explicitly non-certification

Every state has exactly 192 training cases and 32 development cases. Training is exactly balanced at 768 direct/rehearsal cases and 768 Planner-handoff cases, including a 96/96 split for each scientific state.

Training contains three equal-state components:

- 768 factorial-generalization examples, 96 per state
- 512 minimal-contrast examples, 64 per state, forming 256 controlled pairs
- 256 precedence/field-isolation examples, 32 per state

### Controlled conceptual contrasts

The contrast set directly stresses the boundaries implicated by the autopsy, particularly:

- `improved <-> equivalent`
- `improved <-> certification_ready`
- `equivalent <-> certification_ready`
- `inconclusive <-> recheck_required`
- `inconclusive <-> incomplete_evidence`
- `recheck_required <-> scientific_rejection`
- `scientific_rejection <-> regressed`

Contrast-pair members share the same opaque context. When a Planner handoff is present, both members receive the same advisory action, so upstream advice cannot reveal which scientific state is correct.

### Anti-shortcut controls

Validation enforces all of the following:

- no exact evidence sentence from the old V3/V3.1 curriculum is reused;
- no exact evidence sentence from the burned sealed certification is reused;
- training wording and development-diagnostic wording are disjoint;
- training and development contexts are disjoint;
- model-visible case/context/evidence identifiers are opaque and contain no state or split names;
- every case carries six unique evidence references;
- every state has equal training count;
- every state has equal direct/handoff count;
- Planner advice is balanced or held constant across contrast pairs;
- the role contract explicitly requires the `decision` field to contain a scientific-state token and never an action token.

## Proposed conservative training protocol — not authorized or launched

The recorded proposal keeps the V3.1 adapter as immutable parent and uses the existing LoRA geometry `r=16`, `alpha=32`, `dropout=0`, BF16, AdamW at `2e-6`, microbatch 1, gradient accumulation 8, max sequence 2048, and at most 96 optimizer steps. Proposed checkpoints are 16, 32, 48, 64, 80, and 96.

Before any weight update, the frozen V3.1 parent must be evaluated on the new 256-case development diagnostic to establish an uncontaminated baseline. This development diagnostic cannot certify anything. Checkpoint selection must use this development set, not the burned sealed certification evidence.

A checkpoint is selectable only if it clears all recorded development gates, including >=95% overall decision/semantic exactness, 100% schema, >=99% grounding, <=1% hallucination, 100% deterministic primary/action projection, 100% system-action exactness, zero system escalation, <=1% upstream-copy violations, >=90% on each previously weak state, and >=95% on each previously robust state. If no checkpoint clears those gates by step 96, this repair protocol is scientifically rejected rather than blindly continued.

## Certification remains separate

No replacement certification set has been generated. If and only if a repair checkpoint is selected, Hephaestus must generate another fresh 128-case sealed certification partition and a fresh 128-case sealed regression partition whose wording/template families are disjoint from training and development. The previous sealed v1 cases remain permanently burned. Existing certification gates are unchanged. Live Planner -> Evaluator -> Judge handoff certification and production canary/shadow operation remain required before authoritative dispatch.
