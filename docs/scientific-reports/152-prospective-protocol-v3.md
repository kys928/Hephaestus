# #152 candidate protocol v3 — no-launch decision and redesign handoff

**Current decision: NO LAUNCH with the original 96-update recipe.** This is a
versioned review record, not an accepted or executable final contract. It
supersedes v2 as the current preparation recommendation; historical v1/v2 remain
unchanged. Authorized new protected cases in this assignment: **zero**.

The [completed public pilot](152-public-pilot-results-20261101.md) selected seed
1117 on development and observed 3/64 canonical exact answers (4.6875pp over
baseline0/64), with complete JSON/EOS. Seeds2227/3337 scored4/64 and3/64. All
adapters answered every graph with the training-modal distance2 and failed every
JSON sum/count and reversal task. This does not support the optimistic effect
assumptions needed to spend a one-use confirmatory batch. It is not proof that
all future recipes fail, nor permission to select2227 on the pilot.

No larger N can create a missing effect. Under an optimistic IID plug-in of
selected wins3/64/losses0, the old gain/test/loss subset has only36.62% power at
N512 and24.57% at N2048. These calculations omit other gates and are not actual
stratified population-power estimates. The selected pilot interval is wide;
the decision is insufficient feasibility evidence, not a conclusive null result.

## Conditional candidate for independent review

There is **no accepted confirmatory N for the current recipe**. The concrete
smaller design below is a conditional review candidate for a revised recipe,
not an instruction to run it. Before any final material exists, independently
review a new training recipe and its separate public pilot, establish plausible
discordance/termination/dependence, validate the full decision procedure, then
either accept this candidate or register another version. No extra training or
final selection is authorized by this document.

| Design item | Frozen review candidate and required interpretation |
| --- | --- |
| Primary hypothesis | The development-selected procedure has positive paired terminated-exact gain, conditional on the declared four-family population and sampling assumptions. Null: no positive gain. This is not a test of general autonomous-research superiority. |
| Primary endpoint | Real EOS followed by strict finite typed JSON correctness, on identical baseline/candidate tasks. Both models' nontermination is incorrect even if the text parses. No repair, extraction or coercion. |
| Practical threshold and planning alternative | Require observed gain >=5pp and a positive lower95% paired interval. This supports a positive effect with a sufficiently large point estimate, not a confidence claim that the population gain is >=5pp. A **10pp true gain /0.5% losses** is a conditional planning alternative requiring new public support. |
| Sample size | Candidate **128 canonical cases per family, N512**. Current launch authorization remains N0. N512 is the smallest evaluated grid size above90% for the test/gain/loss subset at the stated10pp alternative (95.42%); N256 gives86.06%. Neither is certified full-gate power. Do not retain N2048 merely because v2 used it. |
| Independent unit | One canonical task; model outputs remain paired. Wrappers/reexecutions do not add units. Training seeds are separate replications sharing tasks, not512 new independent training experiments. |
| Population and weights | Original four fixed task grammars, arithmetic/list range0..99, reversal length3..8, five-node graphs with the original backbone. Weights .25 each; no unseen-family/template or general-language inference. |
| Sampling and dependence | Unique original proposal draws conditional on bucket9 and the complete exclusion catalog. This is nonuniform over canonical classes and without replacement. Working independence/exchangeability and bootstrap calibration require explicit independent acceptance; exact McNemar is not automatically exact for every heterogeneous finite-population null. |
| Paired test | Positive two-sided exact McNemar p<.05 for the single development-selected primary, conditional on its accepted paired-null assumptions. Report wins/losses/ties and family effects regardless of verdict. |
| Confidence intervals | Candidate10,000-draw stratified paired percentile bootstrap, resampling paired canonical cases within family, seed152; primary95%, secondary98.333% intervals. Report approximation/degeneracy and shared-template limits. Do not claim population equivalence from a degenerate interval. |
| Multiplicity | One primary selected before exposure. Only after its complete success may the all-three-observed-seeds claim be considered, requiring every precommitted seed to pass at .05/3 with simultaneous98.333% intervals. Report other outcomes descriptively if the primary fails; no final-data selection. |
| Generation and termination | Candidate128 new tokens, original pinned chat/tokenizer/system instruction, greedy CPUfloat32 batch1/one beam, explicit attention mask, fresh GenerationConfig, min0/repetition1/EOSpad2/no forced EOS/BOS. Retain **all candidate EOS and zero truncation** hard gates. They are not relaxed to manufacture feasibility. |
| Regression limits | Nonnegative correctness gain in every family, nonregressing aggregate valid JSON and correct-to-incorrect losses <=1% (at N512, at most5). Wrong-key/type, refusal, malformed and nonterminated answers remain endpoint failures. |
| Outcome rules | Positive only when all endpoint/statistical/regression gates pass. A hard regression-gate failure is ineligible, with the failed gate reported. Statistically negative gain is regressed. A no-material-change claim additionally needs accepted nondegenerate uncertainty contained within the practical equivalence region; otherwise report inconclusive. No-support does not imply equivalence. |
| Stopping and exclusions | No interim efficacy/futility inspection, adaptive N, post-exposure exclusions, candidate replacement, cap change or answer repair. Invalid pre-seal generation exhausts/aborts by rule; partial/failed exposure remains permanently spent and inconclusive, never permission to retry. |
| Selected-seed estimand | Best-of-three selected **only** from all original development attempt receipts by terminated exact, valid JSON, then lowest seed. Freeze actual job/checkpoint/selection identities before the independent final seal. A revised recipe must be fixed before its three fresh jobs. |
| Three-seed replication estimand | Three specific predeclared successful jobs, seeds1117/2227/3337, same recipe except seed. Report all failures. All three successful observed comparisons do not estimate universal seed reliability; even3/3 IID random successes has a95% lower reliability bound of only .2924, and fixed seed values alone do not justify that model. |

## Why this is not accepted yet

The analytic finite sum is a model-based upper bound for a subset of gates, not
a proof of full power under the actual sampler. Shared grammar, finite support,
proposal weights, family heterogeneity and termination can change the result.
Four authored templates cannot identify a random-template population variance.
The reviewer must accept the intended narrow claim or require a new public
development distribution and enough independent template/semantic clusters.

At independent per-case EOS .999, all512 terminate with probability59.91%; at
N2048 it is12.89%. Pilot64/64 EOS is not evidence that failure probability is
below.1%: its two-sided95% binomial failure upper limit is5.601% under IID.
Keeping the hard all-EOS gate therefore remains a material feasibility risk.
If an independent reviewer recommends another termination policy, register and
validate a new version **before** final exposure. Do not alter it after a result.

The N512 candidate assumes an improved recipe and a10pp alternative that the
current selected pilot does not support. A new recipe should address arithmetic
range transfer, list computation, actual reversal and graph-label collapse;
format-only success is insufficient. A broader claim would require a new task
and template design, not relabeling these four grammars. Do not retrain repeatedly
on this pilot or turn it into a final test.

The runtime `ProspectiveDesign` intentionally continues to recognize its existing
v2 contract only; changed versions/parameters are rejected. **No executable v3
final-launch support is claimed.** After an independent decision fixes a revised
recipe and protocol, its matching versioned implementation and complete fixture
rehearsal must precede any final seal or grant. The existing public128-token
measurement and custody/recovery mechanics have been verified separately.

## Custody, source and external handoff

The [custodian handoff](../operations/152-independent-custodian-handoff.md) is
normative. Retain the exact public source, three optimization-complete bindings,
development selection and all failures. The historical public CPU adapter and
private R10 target are separate; no public substitute resolves #102.

The current known exposure catalog has2,612 identities, commitment
`sha256:7a2cd6aceead43f177f34a82a6d2e625163a28df2c60eebf80ac39f8889703e1`,
including the explicitly public synthetic large scorer fixture. Remaining known
range100 arithmetic bucket9 capacity is1,992, before further external exposures.
The custodian must reconcile dynamic/external history, near-duplicates and source
rights; canonical hashing cannot prove absence from pretrained general knowledge.

Only a genuinely independent custodian may later generate/seal its final batch
and retain its private seed. An independently operated irreversible exposure
service must bind global material/grant/study uniqueness, a fresh nonce and signed
receipt outside operator rollback. Local consumed intent precedes that service;
verified durable acknowledgement precedes generation. Unknown acknowledgements,
partial output and crashes remain spent. Final launch is blocked before material
access while separately entrusted service/namespace/keys are absent or fixture-scoped.

The independent verifier must reproduce source/checkpoint bindings, development
selection, raw token decoding, strict scores, statistics and claim flags from
retained immutable commitments, with the reviewed exact version. Resource
authorization and any model promotion remain separate. This assignment grants
no paid GPU spending or protected-material exposure.

## Readiness decision

Bounded engineering preparation, authentic public optimization/generation,
recovery and the complete CPU custody fixture are verified. Scientific design
acceptance is **not passed**: the current recipe receives a no-launch decision,
and this conditional redesign is not approved or executable. External scientific
authorization is **not passed**: no independent custodian/service/keys/final
materials/grant have been provided. Keep #152 open.

The next external action is independent review of this negative pilot and the
proposed redesign, followed by an explicit recipe/protocol decision. It is not
generation of a final batch. Complete the resulting versioned implementation,
new public feasibility work and deployment rehearsal before any future approval
to seal or expose protected material. No protected final material was consumed.
