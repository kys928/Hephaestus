# Public Smol pilot results — 2026-11-01

**Nonconfirmatory result: format learning is reproducible; the current recipe
does not justify a confirmatory launch.** Development selected seed 1117. It
answered 3/64 canonical pilot tasks exactly, versus baseline 0/64. Seeds 2227
and 3337 answered 4/64 and 3/64. Do not select seed 2227 using these pilot results.
No protected final seed, batch, answer, grant or assessment was obtained.

## Authentic source, separation and dose

The [predeclared pilot plan](152-public-pilot-plan-v1.md) and contract
`sha256:5e49230adf7d011104bfcd3f140c15584c0e2e6bad297cc35e1bd1b05d6e74a2`
were fixed before generation. Execution used merged #188, commit
`823fda9d5cbacc544ba1dec30cb5f093e8921b7a`, tree
`a6ddf1f5b0150d04adb22719bfe9dea722a75f7e`, not a later analysis installation.

| Foundation | Verified identity |
| --- | --- |
| Public base | `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; 134,515,008 parameters |
| Base safetensors SHA-256 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| Original tokenizer JSON SHA-256 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| Original chat-template SHA-256 | `872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788` |
| Historical public CPU adapter SHA-256 | `a782d0ca344559659893a64d8bc1e8a90f597cf9cf0bbef06b433c555c1da195` |

All seven pinned snapshot files, architecture and special token IDs were checked.
All 120 adapter tensor keys, shapes, finite values and loaded values were checked.
The historical public adapter lacks complete optimization provenance and is not
a final candidate or the separate private R10 CUDA target.

Three actual CPU float32 LoRA jobs used the original 96 training examples,
rank8/alpha16/dropout0, q_proj/v_proj, AdamW 5e-4, constant scheduler, batch1 and
96 updates each. A separate seed-1117 job repeated 96 updates. Every adapter
tensor and optimizer/scheduler/RNG/dose field reproduced exactly: **288 training
updates plus 96 repeat updates**. No development or pilot gradients were used.
The original train and development source commitments were reverified.

Pilot sampling used public seed 202611010152, range100 proposals and canonical
development bucket 8: 16 tasks per family, 64 unique canonical units, excluding
all original 144 public identities. Thirty-two predeclared wrapper surfaces repeat
eight of those tasks per family. They add no independent observations. This is
a public feasibility distribution, not a hidden test. It is permanently excluded
from confirmation, including canonical and prompt aliases.

There were **696 unique fixed measurement requests**: baseline and historical
adapter each 72 development-cap measurements plus 96 pilot surfaces; three fresh
seeds each 24 development measurements plus 96 pilot surfaces. Cap diagnostics
used the original 24 public development cases, never a future protected batch.
The complete generation invocation took 1,841.16 seconds, excluding prior
training and external checkpoint pauses.

## Canonical results at the frozen 128-token public contract

| Model | Exact with EOS /64 | Valid JSON /64 | EOS /64 | Truncated /64 | Median / max tokens | Mean target NLL including EOS |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 12 | 52 | 128 / 128 | 2.573925 |
| Historical public adapter | 3 | 64 | 64 | 0 | 8.5 / 12 | .845246 |
| Seed 1117, development-selected | 3 | 64 | 64 | 0 | 8.5 / 12 | .845157 |
| Seed 2227 | 4 | 64 | 64 | 0 | 8 / 12 | .837939 |
| Seed 3337 | 3 | 64 | 64 | 0 | 8.5 / 12 | .847477 |

The selected seed has three wins, zero losses and 61 ties: observed gain 4.6875pp.
The other fresh seeds have four and three wins, with zero observed losses.
These small paired counts are planning observations, not confirmatory evidence.
Even zero losses in 64 pairs has an IID-binomial two-sided 95% upper limit of
5.601%, not proof of a <=1% population loss rate. The analogous selected-seed
accuracy interval is approximately .98%–13.09% under that working model; fixed
strata, proposal weights and shared grammar limit its interpretation.

Development selection was fixed by terminated exact count, valid JSON and lowest
seed: 1117 scored 6/24, 2227 scored 6/24, 3337 scored 5/24; all had 24 valid JSON.
Seed 1117 therefore remains selected despite the larger pilot count for 2227.

| Family, 16 canonical cases each | Baseline | Historical | 1117 | 2227 | 3337 |
| --- | --- | --- | --- | --- | --- |
| Arithmetic | 0 | 1 | 1 | 2 | 1 |
| JSON count/sum | 0 | 0 | 0 | 0 | 0 |
| String reversal | 0 | 0 | 0 | 0 | 0 |
| Five-node shortest path | 0 | 2 | 2 | 2 | 2 |

All four adapters predicted `{"distance":2}` on every canonical graph task.
The training mode was distance2 on 12/24 graph examples; only 2/16 pilot graphs
had that target. Thus the observed graph successes equal a training-modal label
baseline, not evidence of learned path solving. The selected seed's only exact
arithmetic answer was among two cases whose largest operand was below20; it
scored 0/14 when the largest operand was20..99. Seed2227 added one success among
12 cases in50..99. All 16 JSON tasks were in the50..99 band, and reversal scored
0/6 for lengths3..5 and0/10 for lengths6..8. Counts in each difficulty cell are
small; no difficulty trend is estimated reliably.

Inference: the large NLL reduction and complete JSON/EOS behavior are consistent
with format learning, while task correctness and out-of-range generalization
remain weak. This does not isolate every training limitation or prove that a
different recipe cannot work. It does not demonstrate autonomous-policy superiority.

## Generation diagnosis and paired wrapper sensitivity

The exact system message is `Return only one valid JSON object with exactly the
requested answer, with no markdown, code, or prose.` The original pinned chat
template supplies system/user roles and the assistant generation prefix. A fresh
GenerationConfig freezes greedy batch1 CPU float32, one beam, min0, max128,
repetition penalty1, EOS/pad2, no forced EOS/BOS and an explicit attention mask.
Real terminal EOS is removed only for scoring. Nonterminated or malformed output
is incorrect; finite JSON types and exact keys/values are required. No repair,
fence extraction, answer coercion or favorable stopping was used.

| Public development cap, 24 cases | Baseline exact / valid / EOS / truncation | Historical adapter exact / valid / EOS / truncation |
| --- | --- | --- |
| 16 | 0 / 0 / 0 / 24 | 6 / 24 / 24 / 0 |
| 64 | 0 / 0 / 0 / 24 | 6 / 24 / 24 / 0 |
| 128 | 0 / 0 / 5 / 19 | 6 / 24 / 24 / 0 |

A larger cap alone did not make the baseline follow the required JSON format.
Adapter outputs demonstrate that the actual EOS token and strict parser work.
Verified original loading, prompt/token receipts and exact training replay supply
evidence against a missing adapter, wrong EOS ID or silent model substitution.
They do not turn incorrect arithmetic/reversal answers into correct ones. The
old randomly initialized 37,472-parameter /16-token experiment remains separate.

Wrapper diagnostics compare the **same 32 tasks** on both surfaces:

| Model | Canonical exact /32 | Wrapped exact /32 | Wrapper wins / losses | Canonical EOS / wrapped EOS |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 0 / 0 | 6 / 22 |
| Historical | 0 | 2 | 2 / 0 | 32 / 32 |
| 1117 | 0 | 2 | 2 / 0 | 32 / 32 |
| 2227 | 1 | 2 | 2 / 1 | 32 / 32 |
| 3337 | 0 | 2 | 2 / 0 | 32 / 32 |

Baseline wrapped valid JSON was2/32 versus0/32 on matching canonical cases,
without any exact answers. This sensitivity matters, but one additional fixed
wrapper cannot establish generalization to unseen templates. Wrapper scores did
not select a seed, change a cap or modify a final endpoint.

## Public exposures, capacity and scientific decision

The old arithmetic domain has820 canonical tasks /79 bucket9 cases. Range100 has
20,100 /2,021. The expanded known public catalog contains **2,612 identities**,
including all splits of the explicitly synthetic large scorer fixture, generator
test seeds31/93/413, original material, public rehearsals, the pilot and literal
valid grammar prompts. It excludes29 range100 arithmetic bucket9 identities,
leaving **1,992**. The earlier limited 237-identity preflight left2,007; that count
is superseded, not a complete contamination clearance.

Catalog commitment:
`sha256:7a2cd6aceead43f177f34a82a6d2e625163a28df2c60eebf80ac39f8889703e1`.
The public large mock fixture is explicitly labeled synthetic, not the real final
evaluation. No final seed is used to reconstruct it. External/dynamic exposure
history and near-duplicate limits still need independent reconciliation.

[Analytic power checks](152-analytic-power-checks.md) show why simply enlarging
the study does not rescue this recipe. Under a deliberately optimistic IID
plug-in model of wins3/64 and losses0, only the old gain/test/loss subset has
36.62% power at N512 and24.57% at N2048; additional gates can reduce power.
These are sensitivities, not estimates of actual population power. A true5pp
effect on the observed5pp boundary has about half power, while all-EOS can add
another severe veto. At per-case EOS .999, all2,048 terminate with probability
12.89% under independence. The pilot cannot establish such a rare failure rate.

**Do not launch the current recipe under protocol v2.** The selected pilot gain
is below its observed5pp requirement, generalization is weak, and the small pilot
does not validate the optimistic power assumptions. Preserve a no-launch result
and review a prospective recipe/distribution/power revision before spending a
one-use final batch. The [v3 review record](152-prospective-protocol-v3.md) specifies
the decision and conditional candidate; it is not scientific acceptance.

## Recovery and retained evidence

Accepted analysis ref:
`sha256:3198b888e5b270ab32c5d67b365935595e0dcc91982a51f92f7d1c2a30eefe62`.
The complete final snapshot has715 state records,666 CAS objects and4 retained
optimization checkpoints (three seeds plus the independent repeat). Manifest:
`e6968036afb435fac1bd3c0103af92a7e5576df6a1b03272aaeb9b26eca23cc6`;
separate control journal SHA-256:
`3c615030fd0c98164706494a87631a9d11ec71288dbbca502d02c8c8c3e09bf0`.
CAS deduplication does not reduce the696 distinct measurement identities.

All raw generations, jobs, optimizer/checkpoint state, public foundation bytes,
phase seal, separate control, public catalog, matched sensitivity and full CPU
launch-fixture evidence are retained in
[Hephaestus_Public_Pilot_Evidence_20261101.zip](https://chatgpt.com/library?file_id=libfile_fca2530d8e548191b1d04ec3c66cd616&account_id=personal).
Version4 was saved before restore:244,896,560 bytes, SHA-256
`fcd3196a6220b32f9e31b8a5825031d046a98068ec1c857ca0b7c87c658a8eda`.
Earlier phase versions remain historical checkpoints. The final report records
the later cold-verification evidence version separately.

A fresh extraction of that packaged snapshot restored at the original absolute
logical root, retaining the old root in quarantine. All715 ordered records
matched before/after, SHA-256
`1d41a08baf0ee7380df3bc66a59da8fcfbf087f0ddee9065935c4b603d8ff077`.
No relocation or independent scientific certificate is claimed.

After that restore, a new process using the exact frozen #188 source decoded
and rescored all retained raw samples, rechecked all optimization states and
exactly reexecuted the40 predeclared generation/NLL comparisons across all five
models. The cold receipt reports23,613 token checks (including repeated checking
of shared development/calibration receipts), not23,613 distinct generated tasks.
It passed without changing the selected seed or original publication.

## Complete launch fixture and readiness gates

A separate explicit `development.prospective_launch_rehearsal` used the accepted
#188 source and a declared37,472-parameter random GPT-2 fixture: three seeds,
four updates each,12 real updates total and three retained checkpoint commitments.
It verified source/partition bindings, checkpoint loading, fixture authorization,
raw generation/scoring/reporting, immutable publication, complete snapshot/restore
and new-process token/model reexecution. A stale pre-exposure restore was denied.
The separately retained fixture service kept one spent exposure record and denied
reuse. Missing production authority was denied using metadata only, before any
final material access. Wrong-key/namespace, changed-model, partial-publication,
nonce replay, hard-crash and local-rollback boundaries have separate adversarial
regressions in the maintained suite.

Fixture bundle:
`sha256:dd8fa57a1420d64261d211d3d82f187b2f1ccbd9c25349d4acfb1fe5ed957035`.
Completed snapshot manifest:
`cfa211933df7ff0a64afa85f6fe02afa0f634467cf312312c4a5291be30f573f`.
Its negative result remains0/8 exact, with eight truncated/no-EOS candidate
responses at the fixture16-token cap. Fixture keys are explicitly nonindependent;
this test is pipeline evidence, not public Smol quality or production custody.

Engineering preparation is complete for the bounded verified mechanisms.
**SCIENTIFIC DESIGN ACCEPTED: not passed. EXTERNAL AUTHORIZATION READY: not
passed.** An actual future revised final study also requires its accepted
version's matching implementation and full rehearsal; no executable approved v3
launch is claimed. The next external action is independent review/redesign,
not production of a protected final batch.

Attempts are explicit: an initial memory-guard stop had zero updates; a later
observed288+96-update attempt lost unpublished raw data in a temporary-runtime
reset and supplies no accepted scientific outcome. The fresh complete attempt
is separately identified and was externally checkpointed before continuation.
No guard was weakened, no missing historical result was invented, no model was
promoted and no paid GPU was launched. #152 and #102 remain open.
