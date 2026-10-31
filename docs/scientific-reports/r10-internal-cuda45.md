# R10 — verified internal CUDA45 execution

**Status:** completed, independently cold-verified *internal execution milestone* (2026-10-30). **Not:** independent historical external certification, behavior-improvement proof, approved checkpoint promotion, or unrestricted autonomous operation.

## Registered observations

| Criterion | Evidence-backed outcome |
| --- | --- |
| True native workflow operations | **45/45** |
| Original five CUDA model roles | **5/5** Controller, Diagnosis, Planner, Evaluator, Judge |
| Real CUDA target-model optimizer updates | **96/96** |
| Final native staged state | Completed |
| Independent post-execution cold readback | Passed |
| Pod cleanup | Verified deletion; no matching Pod left by this campaign |

The original executed source commit is [`d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf`](https://github.com/kys928/Hephaestus/commit/d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf). Exact private original output/checkpoint/archive hashes, five distinct role inference receipt hashes and detailed run identity are retained in the [immutable R10 evidence index](../evidence/original-cuda45-r10/README.md). We do **not** duplicate raw private evidence in this report.

The [actual GPU execution](https://github.com/kys928/Hephaestus/actions/runs/41059067130) and [separate cold verification](https://github.com/kys928/Hephaestus/actions/runs/41059339245) succeeded. The reported roughly USD 0.12 usage remains a **cost estimate**, not a provider invoice.

## Evaluator's precise role

The selected **Granite 3.3 8B Instruct** Evaluator LoRA adapter was trained and chosen in the earlier role-mastery process; the R10 target model's 96 updates are **not** Evaluator training. R10 verified a conforming frozen Evaluator inference on actual CUDA. Earlier role scores and zero-hallucination measurements were tied to their original restricted evaluation conditions and must not be advertised as real-world perfection.

Later experimental semantic Evaluator variants had rejected or inconclusive results. R10 did not authorize swapping the registered production role model or eliminating deterministic evaluation gates.

## What R10 does not answer

1. Did the trained target checkpoint improve over its frozen baseline on a prospective independent holdout, across multiple seeds and with bounded uncertainty? **Not established by R10 operation counts.**
2. Are private historical source/holdout leakage, custody, original Granite comparisons and external independent certification complete? **No; tracked in issue #102.**
3. Is it safe to run an arbitrary expensive pipeline or auto-promote weights? **No; each scope needs fresh admission and governance.**

## Source-integration context

[PRs #153–#157](https://github.com/kys928/Hephaestus/issues/151) subsequently brought the R10 source lineage and guarded runtime into `main` via non-squashed merges and retired historical one-off paid workflows. This **does not change the original 2026-10-30 outcome**. Read [mainline integration ledger](../integration_notes/151-r10-full-history-and-runtime.md) for provenance.

Next scientific experiment: [#152 — frozen baseline versus independently evaluated candidate improvement](https://github.com/kys928/Hephaestus/issues/152).
