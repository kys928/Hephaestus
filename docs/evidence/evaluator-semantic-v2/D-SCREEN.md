# D experiment: balanced-screen scientific rejection

**Verdict: D step 96 is a no-go.** This is a mathematically conclusive rejection under the frozen development acceptance gates, not a complete 256-case metric estimate or a certification result.

- Paid GitHub Actions run: [38945578442](https://github.com/kys928/Hephaestus/actions/runs/38945578442)
- Immutable source SHA: `ffa0458d5850f9d572a06d5ee6660fd63fc8acc6`
- Run ID: `evaluator-semantic-v2-D-38945578442`
- Training completed 96 optimizer steps and 768 examples from frozen parent; all six checkpoint archives persisted.
- Frozen diagnostic SHA: `ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912`.
- Screen selection SHA: `98c3346fde66cc92d355bae019de811cb9c7614746073d1e4f93e6a069dbe11c`.
- Evaluated exactly 64 fixed development cases, 8 per scientific state.
- D step 96 got **33/64 = 51.5625%** correct.
- Maximum possible across all 256 cases, even with 192 perfect remaining predictions: **225/256 = 87.890625%**, below the fixed **95%** decision gate.
- No 256-case full diagnostic was executed. The unseen 192 outcomes are unknown.
- No checkpoint selection, certification, model promotion or authoritative dispatch occurred.

| State | Correct on 8-case screen | Full per-state gate still reachable? |
|---|---:|---|
| scientific_rejection | 5/8 | No |
| incomplete_evidence | 5/8 | No |
| improved | 4/8 | No |
| regressed | 4/8 | No |
| equivalent | 0/8 | No |
| inconclusive | 8/8 | Yes (not proven passed) |
| recheck_required | 0/8 | No |
| certification_ready | 7/8 | Yes (not proven passed) |

The observed pattern still contains severe failures in `equivalent` and `recheck_required`. D changes true factorial examples, precedence examples and difficulty ordering together; the negative result cannot isolate one change's causal contribution. The 64-case screen is a balanced, predeclared subset of the old frozen development diagnostic, not an independent test or a new certification partition.

## Execution and costs

Training pod `2sb00owcma1ydn` completed successfully, saved six checkpoints and was deleted with verified absence. Its estimated GPU cost was **$0.255209**.

Evaluation pod `0wcpud3152qc6m` persisted the balanced screen and terminal `screen_rejected` evidence before teardown. It was deleted with verified absence. Its estimated GPU cost was **$0.234942**.

**Combined estimated GPU spend: $0.490151.** These are $2.49/hour elapsed-time estimates, not authoritative provider billing charges. The experiment respected its $1.50 training and $1.50 evaluation ceilings. The $2.00 evaluation contingency was never used.

## Independent assessment

Independent [GPU-free assessment #38947323158](https://github.com/kys928/Hephaestus/actions/runs/38947323158) **passed** and separately rescored the 64 saved generations against the original case identities and expected scores. The audit recorded D **33/64 (51.5625%)** versus V1 step 96 **34/64 (53.125%)** on *exactly the same* 64 cases.

Paired outcomes: 31 both correct, 28 both incorrect, three V1-only correct, two D-only correct. This is not evidence of a general accuracy difference or improvement. Both `equivalent` and `recheck_required` are 0/8 for D and for V1 on this screen. D is 8/8 on `inconclusive` versus V1 7/8. The observed screen is a subset of the frozen development diagnostic, **not an independent certification partition**.

The independent workflow publishes `evaluator-semantic-v2-D-independent-screen` with its JSON report. Further paid D/E experiments remain disabled pending a scientifically distinct intervention.

No additional GPU experiment is authorized by this outcome. Draft PRs #70 and #71 remain unmerged.
