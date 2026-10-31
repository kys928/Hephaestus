# Hephaestus

**A governed, evidence-first research-engineering system for training, diagnosing, evaluating, and recovering language models.**

Hephaestus connects a deterministic eight-phase scientific workflow with specialist language-model roles, immutable research evidence, real target-model PEFT training, evaluation and safety-gated decisions.

## Verified capability and limitations

| Capability | Current evidence |
| --- | --- |
| Original 45-stage native governed research campaign | **Completed 45/45 in original R10 CUDA experiment** |
| Five original pinned GPU specialist-role inferences | **5/5 independently accepted in R10** |
| Real target-model AdamW/LoRA updates | **96/96 in R10**; not a claim of behavioral improvement by itself |
| Native archive, checkpoint and role inference hashes | Independent cold verification passed |
| RunPod cost/Pod custody | Real scoped prepaid Pod, verified deletion |
| External historical scientific certification | **Pending** (original private data custody, contamination audit and Granite comparisons) |
| General fully autonomous model-development factory | **Not yet demonstrated** |
| Automatic production role dispatch or checkpoint promotion | **Disabled unless separately authorized** |

**Primary evidence:** [R10 original CUDA45 scientific report](docs/evidence/original-cuda45-r10/README.md) · [GPU execution](https://github.com/kys928/Hephaestus/actions/runs/41059067130) · [Independent cold verifier](https://github.com/kys928/Hephaestus/actions/runs/41059339245).

R10's target training generated a verifiable checkpoint, but demonstrating consistent **held-out improvement** is a separate experiment. The five specialist adapters were trained in an earlier model-role mastery campaign; R10's 96 updates were *not* further tuning of the Evaluator.

## Development quick start

The project uses Python 3.11+ and an installable `src/hephaestus` package.

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests/
hephaestus-infrastructure --help
```

Pinned real frozen-model CPU regression dependencies used to validate the original R10 research integration:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,peft-training,role-inference,s3,independent-certification]"
python -m pytest -q tests/
```

The second install includes heavier model dependencies. Neither test recipe should rent a GPU, download original model weights for live production inference, or mutate the protected R10 scientific archive.

## Source layout

- `src/hephaestus/control/`: orchestrates the mandatory eight-phase control spine.
- `src/hephaestus/production/`: governed native research services and the authentic staged campaign.
- `src/hephaestus/roles/`, `llm/`: expert roles, frozen strict contracts and opt-in pinned inference.
- `src/hephaestus/data/`, `training/`, `evaluation/`: reviewed source custody, actual training and quantitative evaluation.
- `src/hephaestus/schemas/`, `policy/`, `state/`, `storage/`: explicit contracts, permission decisions, durable record and immutable evidence verification.
- `src/hephaestus/providers/`, `runtime/`, `backends/`: scoped resource allocation, recovery and worker abstractions.
- `configs/`: versioned configurations, frozen evaluation packs and research recipes.
- `tests/`: deterministic contracts, system integration, negative-path and scientific evidence tests.
- `docs/`: architecture, operating boundaries, research reports and integration decisions.
- `scripts/`: reference operator/research utilities; historically pinned one-off runners are **not** automatically production-authorized.
- `.github/workflows/`: maintained CI and manual tools only. Historical campaign-specific workflows remain in version history, not the active tree.

Full project design and boundaries: [PROJECT_MISSION.md](PROJECT_MISSION.md), [AGENTS.md](AGENTS.md), [REPO_STRUCTURE_PLAN.md](REPO_STRUCTURE_PLAN.md).

## Eight-phase mandatory control spine

`Judge entry → Planner → Data acquisition & audit → Preprocessor → Training engineer → Runtime monitor → Evaluator → Judge exit`.

The role models provide evidence-bound, non-authoritative proposals. The deterministic control and explicit approval boundaries decide what may actually happen.

## Scientific evidence and roadmap

1. **Original R10 internal CUDA45 execution:** [published](docs/evidence/original-cuda45-r10/README.md), cold verified, no re-execution needed for that milestone.
2. **Consolidated source integration:** [issue #151](https://github.com/kys928/Hephaestus/issues/151); keep historical Git ancestry, avoid unreviewed or expensive workflow triggers, validate a clean checkout.
3. **Prospective actual behavior improvement:** [issue #152](https://github.com/kys928/Hephaestus/issues/152); pre-registered, leakage-screened frozen baseline/candidate and multiple independent seeds.
4. **Outstanding historical external custody/certification:** [issue #102](https://github.com/kys928/Hephaestus/issues/102); internal CUDA completion **does not** self-issue that certificate or authorize checkpoint promotion.

Do **not** commit private training data, Hugging Face weights, S3 credentials, generated checkpoints, runtime SQLite or content-addressed protected artifacts into Git. Do not manually dispatch a historical paid script with an old authorization.
