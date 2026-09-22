# Phase II — Interface Mastery V1

Status: **prepared, frozen, not launched**.

Phase I proved that each of the five certified role adapters can independently perform its own role contract on held-out instances. Phase II asks a different question: **does the downstream specialist still make the correct bounded decision when its input contains the actual generated output of another specialist rather than a perfect synthetic role object?**

## Scientific target

Phase II V1 is an inference-only interface certification experiment. It does not retrain models and it does not turn on automatic model-backed dispatch. The run exercises four model-to-model boundaries that cover all five certified roles:

1. `diagnosis -> planner`
2. `planner -> judge`
3. `evaluator -> judge`
4. `judge -> controller`

Each interface has eight fresh compound cases, for 32 cases total. Every case generates one real producer output and then passes that raw generated record to the consumer. The consumer also receives separately identified machine-verifiable facts. The prompt explicitly treats the producer record as untrusted input rather than authority.

## Why these boundaries

The four interfaces test the places where semantic drift is most dangerous:

- uncertainty can be lost between Diagnosis and Planner;
- a reasonable plan can become an unauthorized transition at Judge;
- Evaluator evidence can be overruled by aggregate-quality narrative at Judge;
- Judge strategy can be mistaken for an executable mutation at Controller.

The `judge -> controller` cases additionally pass Controller output through the existing `DeterministicRoleBoundary`. This preserves the architectural rule that approval validity, action legality, stage legality, checkpoint/provenance integrity, evidence identity, idempotency, and other machine-verifiable facts are not delegated to a language model.

## Frozen inputs

Phase II V1 consumes only the already certified Phase I registry:

- stack: `configs/models/hephaestus_role_model_stack_v1.json`
- source role-mastery run: `role-mastery-v1-35596791201`
- five selected adapter archives: the immutable S3 artifacts recorded in that registry
- Phase II pack SHA-256: `2d712d7bdd9dc51dfefc677389aca97ce80e64c4825df38243bc0d6cfe13bad0`

No Phase I adapter, certification record, or frozen Phase I holdout is modified or copied into the Phase II pack.

## Certification gates

The first paid run is a baseline interface certification, not a repair run. It requires:

- overall interface quality >= 95/100;
- every interface >= 90/100;
- exact schema compliance = 100%;
- evidence grounding >= 98%;
- interface invariant preservation >= 95%;
- downstream exact-contract rate >= 90%;
- deterministic boundary pass rate = 100%;
- hallucinated evidence-reference rate <= 2%.

If these gates pass, Phase II V1 can be certified without any new weight update. If they fail, the result is `phase_ii_targeted_repair_required`. The failing interface and case-level evidence then define a *separate* targeted repair experiment. Phase II V1 itself is forbidden from automatically training around its own evaluation results.

## Execution and cost boundary

The prepared launcher creates at most one Secure Cloud GPU pod. Models are loaded sequentially on that pod, so there is no five-pod fan-out. Frozen execution limits are:

- one GPU;
- GPU memory floor: 44 GiB;
- allowed GPU types: A40, L40, RTX PRO 6000 Blackwell Server/Workstation;
- hard wall: 10,800 seconds;
- hourly ceiling: $1.25;
- estimated total ceiling: $3.75;
- container disk: 180 GiB.

The launcher always attempts verified pod teardown in a `finally` block.

## Launch boundary

The paid workflow is `Phase II Interface Mastery paid launch`, but the checked-in launch marker is deliberately:

```json
"authorized": false
```

A paid pod cannot be created by the launcher until all three conditions hold:

1. protocol governance permits a paid launch;
2. `configs/experiments/interface_mastery_v1.launch.json` is explicitly changed to `authorized=true` with operator identity/time recorded;
3. the runtime authorization phrase exactly equals `LAUNCH_PHASE_II_INTERFACE_MASTERY_V1`.

The ordinary validation workflow performs only static/deterministic checks and render-only launch validation. It never calls RunPod.

## Explicit non-goals

Phase II V1 does **not**:

- enable automatic model-backed role dispatch;
- mutate the certified Phase I role stack;
- train or continue any LoRA adapter;
- promote a model or checkpoint;
- modify any frozen Phase I eval pack;
- infer that a 100/100 Phase I role is automatically safe in a live chain.

Those boundaries are deliberate. The experiment first measures the interface failure surface; repair comes only after evidence identifies one.