# #152 prospective protocol v2 — distribution, uncertainty and custody

Status: **pre-registration candidate; confirmatory execution remains blocked**.
This amends the unexecuted [v1 candidate](152-prospective-protocol-v1.md), which
remains a historical record. No new confirmatory items or model outcomes were
inspected to choose this design. The machine-readable contract is
[`prospective_model_improvement_v2.json`](../../configs/research/prospective_model_improvement_v2.json).

## Population and capacity

Retain four original task families and their fixed authored templates. Change the
arithmetic/integer proposal range prospectively from `0..19` to **`0..99`**;
symbolic length/alphabet and five-node graph proposals remain those of original
generator v1. This is a declared distribution change, including arithmetic
generalization beyond the original training range, not a relabeling of v1 data.
Compare baseline and every candidate on exactly the same new distribution.

The old arithmetic domain has 820 canonical cases, of which **79** hash into
evaluation bucket 9. The new domain has 20,100 canonical cases and **2,021** in
that bucket. Request **512 canonical cases per family, 2,048 total**, after
excluding every known historical, training, development, code-test and rehearsal
identity. The independent parser collapses commutative operands, list order and
graph edge-order aliases. Rights/provenance and exposure completeness remain
independent custody requirements.

Sampling is explicitly **unique draws from the original proposal sampler,
conditional on canonical SHA bucket 9**, with range100. It is not uniform over
canonical equivalence classes. Freeze the actual generator, validator, Python
environment, difficulty, family counts and exposure-manifest hash before the
custodian's secret seed is used. Exhaustion fails; it never expands the range or
changes templates automatically. The training agent must not materialize the
real final holdout. Public fixture rehearsals have a separate purpose/commitment,
at most 16 cases per family and 16 generation tokens.

## Two different seed claims

Train the same original-source 96-example recipe with seeds **1117, 2227, 3337**,
using verified original base/tokenizer/chat identities, LoRA rank8/alpha16,
`q_proj`/`v_proj`, AdamW 5e-4 and 96 updates per successful attempt. Record every
failure and cost. A new versioned recipe must admit these three seeds; never
weaken the frozen original R10 admission guard to reuse its run identity.

Before holdout access, select one primary seed by development exact typed
correctness, then valid JSON, then lowest seed. Commit all development evidence,
attempt statuses, selected seed and checkpoint hashes to the custodian.

- **Selected-procedure improvement:** only that predetermined seed is the primary
  comparison, at two-sided alpha .05. The other final results cannot change it.
- **Observed three-seed replication:** assess all three already committed seeds
  in the same single custody session, report each, and require all three to meet
  the behavioral gates with Bonferroni p < .05/3 and simultaneous 98.333% paired
  intervals if claiming three successful replications. Missing/failed seeds defeat
  that claim. Three observed successes do not estimate universal seed reliability.

One signed unlock authorizes this complete fixed batch, not repeated selection
on the holdout. No candidate may be replaced after any final response is seen.

## Measurement and uncertainty

Use `strict-json-object-finite-typed.v1` for every response. Capture original token
IDs, raw decoded text, prompt/model/tokenizer/template hashes and actual EOS/
truncation evidence. No fences, duplicate keys, nonfinite decoded values, wrong
keys or type coercion. Preserve immutable raw receipts outside public Git.

Retain v1's primary practical gain >=5pp, exact McNemar p < .05, positive lower
paired 95% interval, aggregate nonregressing format, nonnegative family gains,
correct-to-incorrect losses <=1%, all candidate EOS and zero truncation gates.
Use a **stratified paired bootstrap**, resampling canonical cases within each
fixed family together for baseline/candidate, 2,000 draws, seed152. Report the
pooled exact-McNemar result and its independent-pair/exchangeability assumptions.
These are conditional fixed-family/template claims. Four templates cannot supply
a valid population uncertainty estimate for unseen templates or general language.
If random template/family-cluster inference is desired, a new prospective design
with enough independently sampled clusters is required before final exposure.

## Pre-test power calculation

`evaluation/prospective_design.py::planning_power` simulates matched
wins/losses/ties, exact McNemar, the actual practical/loss/family gates and the
stratified bootstrap. Each row below uses 1,000 planning trials, 2,000 bootstrap
draws, seeds152000 through152005, and no learned model. Intervals measure Monte
Carlo error, not uncertainty about a real model's effect.

| Assumed true gain | Assumed regression rate | Assumed family ICC | Statistical gate power | MC 95% interval |
| --- | --- | --- | --- | --- |
| 5pp | 0.5% | 0 | 50.6% | 47.5–53.7% |
| 5pp | 1.0% | 0 | 33.8% | 30.9–36.8% |
| 8pp | 0.5% | 0 | 99.8% | 99.3–99.9% |
| 8pp | 1.0% | 0 | 51.9% | 48.8–55.0% |
| 8pp | 0.5% | .01 | 87.7% | 85.5–89.6% |
| 8pp | 0.5% | .05 | 69.6% | 66.7–72.4% |

The ICC rows use a declared Dirichlet-mixture sensitivity model; they do not
estimate actual template correlation. Power conditions on nonregressing format
and nontruncated outputs. At hypothetical independent per-case EOS success .999,
the probability that **all 2,048** terminate is only **12.886%**. This can dominate
full-gate power. A 5pp effect exactly on the observed >=5pp boundary is also not
an 80%-power design merely because N is large.

Therefore do **not** call this final design adequately powered yet. A separate
blinded, nonconfirmatory pilot must establish plausible discordance, regressions,
EOS/truncation and dependence assumptions; the independent custodian must accept
the sample/gate plan or authorize another amendment **before** real holdout
materialization. Publish this limitation rather than relaxing gates after results.

## Source and custody prerequisites

The [source-recovery record](152-source-recovery-20261031.md) verifies the historical
CPU archive, adapter and reconstructed train/development bytes. Original base/
tokenizer byte recovery, separate R10 target archive and external source custody
are distinct checks. No new optimizer run or held-out improvement is claimed.

The operator must establish the independent custodian/public key out of band.
The software verifies signed commitments and one-use consumption; it cannot make
its own operator an independent scientific authority. A crash after exposure
cannot permit blind re-evaluation. Final cold verification must inspect original
raw bytes and model identities; promotion, external historical #102 certification
and paid execution retain their separate requirements.

Primary methodological references: [exact McNemar API](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
[paired resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
[binomial model](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html).

Independent custody, pre-test pilot acceptance and any fresh paid authorization
remain external blockers. Negative, null, regressed and inconclusive outcomes
must be published with the same provenance as a positive result.
