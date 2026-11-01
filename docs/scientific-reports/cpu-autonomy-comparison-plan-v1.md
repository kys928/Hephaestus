# Public tiny CPU research comparison — plan v1

This engineering experiment is separate from #152. It uses already exposed
examples from the original seed-20261028 authored corpus: four interleaved
training examples (one per family), eight development examples and sixteen
test examples. It creates no independent or protected holdout. One actual
37,472-parameter randomly initialized GPT-2 base, initialization seed 152 and
its byte tokenizer are shared by all conditions. Only training RNG seeds vary:
1117, 2227 and 3337. This is not full-parameter language-model pretraining.

Each of nine programs runs three native 45-operation cycles through the
standard production driver and loop. Every cycle performs four actual LoRA
updates, batch size one and accumulation one over the complete four-example
training set. Rank=2, alpha=4, dropout=.1, targets=`c_attn`, CPU float32,
AdamW and a constant within-cycle scheduler are fixed. Accepted parents use
verified adapter warm starts with explicitly reset optimization state at each
cycle. Each program has twelve optimizer updates and a 240-second wall cap.
All conditions receive the same first-cycle pre-optimizer bootstrap fault;
the existing recovery service must diagnose it and the staged driver may
create only one new attempt. Failed predecessors cannot count as cycles.

Policies are frozen before the comparison:

| Policy | Learning-rate rule |
| --- | --- |
| Fixed | .01 for every cycle |
| Predetermined heuristic | .02, .01, .005; outcomes do not change it |
| Hephaestus adaptive | Start .02. After accepted development improvement, move one position up [.005,.01,.02,.03,.04]; otherwise move down one. Clamp at the endpoints. Read only immutable preceding experiment memory. |

The adaptive rule is a finite deterministic experimental Hephaestus policy.
It does not invoke or evaluate the five frozen specialist models. The human
heuristic is a conventional decreasing schedule authored before these runs.
No policy, hyperparameter grid, sample or threshold may be tuned from test
outputs. Programs are interleaved by training seed, then fixed/heuristic/
adaptive. They share model/data, update counts, failure injection and limits;
actual wall time is measured and is not assumed equal or causally comparable.

After every cycle, evaluate actual greedy development generations and
teacher-forced target NLL. Accept a checkpoint only if mean per-example target
NLL decreases by more than 1e-6 and exact-answer count does not regress.
Otherwise reject it and retain the verified parent. No production checkpoint
promotion, external scientific certification, branch, stage change, data
change or code-edit action is available. Stop after exactly three completed
cycles. Reconstruct production composition after the first completed cycle.

Final test assessment is consumed once per frozen selected program, after its
three cycles. Its primary diagnostic outcome is mean per-example target-token
NLL. Exact strict-JSON accuracy, EOS/truncation, repair count, injected-failure
diagnosis, repeated failures, candidate regression frequency, optimizer dose,
wall time and steps to .1 development-NLL gain are also reported. A rejected
candidate count is a descriptive proxy, not proof of causally unnecessary
intervention. Lower teacher-forced NLL alone does not establish useful or
exact-correct generated behavior.

Compare adaptive with fixed and heuristic using all three matched training
seed differences. Report every result, the paired mean and the empirical
27-resample seed-bootstrap range. The latter is a weak uncertainty estimate
with only three seeds and fixed tasks. Use exact two-sided seed sign tests
with Holm correction across the two comparisons. Even three wins out of
three yield unadjusted p=.25: this design cannot establish superiority at .05.
It can expose large regressions and provide an authentic controlled pilot.
No generalization to new templates, bases, data seeds or large models follows.

Freeze an immutable manifest with all source/data/model/recipe identities
before execution. Cold verification checks every completed 45-operation
trace, failed predecessor, actual dose, historical memory decision, parent
checkpoint and final publication root. In a new process, reexecute all fixed
development/test inference and all three optimizer jobs, comparing every
adapter tensor, optimizer/scheduler/RNG state and sample counters exactly.
Verification work is extra cost, reported separately from matched training
budgets. It is performed by the same engineering agent and supplies no
independent external certification.

Earlier operational preflights are retained separately. They are excluded
from the matched comparison and cannot be represented as fresh heldout science.
Changes to verification code or operational correctness must retain those
attempts and freeze a new complete source identity before the comparison.
