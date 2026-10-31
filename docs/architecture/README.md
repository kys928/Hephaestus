# Architecture

Hephaestus is a governed research-engineering platform, **not** an unconstrained multi-agent swarm. Specialist model roles make bounded proposals; deterministic schemas, control, policies and evidence determine allowed actions. The current mainline includes genuine research/training modules, but its original CUDA45 verification is for one scoped campaign.

## Component map

| Responsibility | Maintained source |
| --- | --- |
| Eight-phase coordination, stage transitions | [`src/hephaestus/control/`](../../src/hephaestus/control/) |
| Specialist role adapters and structured model inference | [`roles/`](../../src/hephaestus/roles/), [`llm/`](../../src/hephaestus/llm/) |
| Real governed staged research and CUDA composition | [`production/`](../../src/hephaestus/production/) |
| Diagnosis and intervention proposals | [`diagnosis/`](../../src/hephaestus/diagnosis/), [`planning/`](../../src/hephaestus/planning/) |
| Data discovery, admission, preprocessing | [`data/`](../../src/hephaestus/data/), [`providers/`](../../src/hephaestus/providers/) |
| PEFT, checkpointing and target-model training | [`training/`](../../src/hephaestus/training/) |
| Frozen comparisons, scoring and assessment | [`evaluation/`](../../src/hephaestus/evaluation/) |
| Approval, action, stage and promotion gates | [`policy/`](../../src/hephaestus/policy/) |
| Cross-boundary typed records and persistence | [`schemas/`](../../src/hephaestus/schemas/), [`state/`](../../src/hephaestus/state/) |
| Runtime, storage/evidence custody and recovery | [`runtime/`](../../src/hephaestus/runtime/), [`storage/`](../../src/hephaestus/storage/) |

## Follow the scientific execution path

```text
Judge entry -> Planner -> Data acquisition & audit -> Preprocessor
  -> Training engineer -> Runtime monitor -> Evaluator -> Judge exit
             |                   |               |
     typed manifests       approved execution   frozen scorecards
             +------------- evidence / state / lineage --------+
                  deterministic gates own effective action
```

The **eight control phases** are the governing process; the **45 native operations** are a separate R10 experiment-specific execution count. Do not conflate them.

- [Phases, roles and authority](control-and-governance.md)
- [Data, state and immutable evidence](data-and-state.md)
- [Operator and deployment boundaries](../operations/run-safety.md)
- [Architecture mission](../../PROJECT_MISSION.md) and [code ownership history](../component_ownership.md)

**Not yet proven:** broadly autonomous from-scratch pretraining, reliable target-model improvement across seeds, universal model-role accuracy, or production authorization of arbitrary paid experiments.
