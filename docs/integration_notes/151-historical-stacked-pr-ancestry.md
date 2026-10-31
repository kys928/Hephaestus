# Issue #151 — reconciliation of the historical stacked development pull requests

## Immutable source and method

- Original genuine R10 CUDA45 source Git commit: `d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf`.
- Original R10 second-parent history merge: `8d2af83ce792e46d7ecafe6407ee7f7b7191cb10`.
- Main integration [PR #154](https://github.com/kys928/Hephaestus/pull/154), merge SHA `1942b6e8ca1b7056661811012d04c0da89d8a25d`, preserved **both** main's and original R10's commit ancestry. No history squashing.
- Compared each head commit of the 58 **then-open** stacked PRs (#81–#141 range) against exact R10 using GitHub commit ancestry comparisons. Exactly **55** had `behind_by=0` (original tip is an ancestor of R10); three branches **diverged** with unique commits not reachable from original R10.
- Read-only provider/catalog operations and experimental paid YAML workflows were intentionally **excluded** from the maintained active tree; their original blobs still exist in the R10 ancestry. Ancestry inclusion does not assert that every individual experimental result is accepted, certified, currently active or promoted.

## Superseded historical stack — branch tips already in main's Git history

Every listed branch-tip commit is reachable via the original R10 source parent and therefore now from `main`. These historically stacked PRs can be **closed as superseded by PR #154** after documenting their exact original head SHA. **Do not label them as newly independently scientifically certified**. Do not delete branches or rewrite Git history.

| Original PR | Original head SHA (full) | Historical PR topic |
| --- | --- | --- |
| [#82](https://github.com/kys928/Hephaestus/pull/82) | `f1a8470e5b0f6e5b296eca1c2b7d705878d98bfd` | Certified five-role PEFT runtime, verified S3 adapters, and guarded staged model proposals |
| [#83](https://github.com/kys928/Hephaestus/pull/83) | `c1f4c81bff321397d16426847357e92b5d8e7bf6` | Verify action approvals and fence lineage recovery; pin frozen role contracts |
| [#84](https://github.com/kys928/Hephaestus/pull/84) | `0173ada8cc7407de2622e7e206e5eb3e3c891990` | Verify scientific artifact restoration and persist replayable role inference receipts |
| [#85](https://github.com/kys928/Hephaestus/pull/85) | `4b69d6090b89337982d446db01c52090efea8ee6` | Admit model scientific selections before execution and restore experiment history |
| [#86](https://github.com/kys928/Hephaestus/pull/86) | `f3e4ddb0dbe069316294c574912726abd879f40a` | Reconcile accepted RunPod allocations, publish immutable adapters and close verification gaps |
| [#87](https://github.com/kys928/Hephaestus/pull/87) | `9ca4e2fb54ee405b89a7199727b1a082ff5a1732` | RunPod REST v2 lifecycle with allocation and termination recovery |
| [#88](https://github.com/kys928/Hephaestus/pull/88) | `12d46a4e0d95a48a9479ba040ce6a5eeeeb98aaa` | Production LoRA training, verified adapter warm starts and exact interrupted resume |
| [#89](https://github.com/kys928/Hephaestus/pull/89) | `1cef5b234da161d6cfa99caaab08fc7e0ef0e616` | Scientific-state holdout contrast scoring with immutable evidence (CPU-only) |
| [#90](https://github.com/kys928/Hephaestus/pull/90) | `0d0b632a130b450ba00fd53e00a35b8bc1e3f25d` | Separately authorized Evaluator role activation and rollback with runtime routing |
| [#91](https://github.com/kys928/Hephaestus/pull/91) | `50151d890b5bd527c3f796a692c231d18eb2815a` | Durable three-cycle autonomous-repair control flow and model-backed production handoff |
| [#92](https://github.com/kys928/Hephaestus/pull/92) | `77988696ddbe46d0789831f97e74b0fa32a87562` | Wire verified Planner discovery to real providers and guarded PEFT training handoff |
| [#93](https://github.com/kys928/Hephaestus/pull/93) | `9a9e495e2c63d3f50071566b4d243de09d7d1d06` | Resolve verified baseline and completed PEFT candidate into frozen generation and evaluation |
| [#94](https://github.com/kys928/Hephaestus/pull/94) | `51f3b28e6c57f10b26224771418e1d3c8dda4506` | Bind approved local dataset bytes to real preprocessing and trainable-data evidence |
| [#95](https://github.com/kys928/Hephaestus/pull/95) | `63babaf4b44d2bb2250d17bcbfe5512b90ede085` | Finalize verified data/model selections into authorized PEFT proposals and prove real tiny CPU LoRA |
| [#96](https://github.com/kys928/Hephaestus/pull/96) | `ade1ba4290ede32c22171fd125b5590c3a4414ae` | Guard 45-step staged composition and require independent semantic custody review |
| [#97](https://github.com/kys928/Hephaestus/pull/97) | `09269b237a29f4e64648f6e691efcc7792717dc4` | Bind actual staged scientific entry, diagnosis, runtime, recovery and Judge policies |
| [#98](https://github.com/kys928/Hephaestus/pull/98) | `bc62873a0ee0b38c32a65d62375fd081b53287cb` | CPU research: actual five-role-conditioned Torch inference and real PEFT smoke, nonproduction |
| [#101](https://github.com/kys928/Hephaestus/pull/101) | `175f6e55583d011bd2a667e6f53a53b4e32b44e7` | Research CPU: pinned pretrained causal-LM five-role inference and governed Planner admission |
| [#103](https://github.com/kys928/Hephaestus/pull/103) | `db41b72baf01e263a0831ea3fe1b093a49723776` | Integrate real dataset discovery and independently pending acquisition approvals into CPU science spine |
| [#104](https://github.com/kys928/Hephaestus/pull/104) | `f5c3a2b8e9af31bb6922b75732a9b3bec9de7284` | Read-only original-diagnosis dataset discovery and independent real-source review |
| [#105](https://github.com/kys928/Hephaestus/pull/105) | `d66cb3eb9bdf62cbffe555f6070f53f22ea7cb3a` | Four independent capability-targeted Hugging Face searches with frozen-tokenizer review |
| [#106](https://github.com/kys928/Hephaestus/pull/106) | `f42fb8be2b90ce2a9bb216854763037033a34781` | Review real dataset schemas, pinned rights, English-only filtering and tokenizer continuation windows (read-only) |
| [#107](https://github.com/kys928/Hephaestus/pull/107) | `19f75beae9e4fb048a627756d02680bb0207ba4f` | Bind reviewed HF datasets to staged acquisition and freeze verified research findings |
| [#108](https://github.com/kys928/Hephaestus/pull/108) | `29be72d94a199bd03b8f42d045785c39e4fac7d0` | Run exact-source quarantine screening and enforce external reference custody |
| [#109](https://github.com/kys928/Hephaestus/pull/109) | `90c3ced7f415c722cb05b759fa5652d25c9db07c` | First real CPU development loop: auto evaluation plan → autograd → frozen regression → Judge → replay |
| [#110](https://github.com/kys928/Hephaestus/pull/110) | `300a87e7e66be34f41a0891bb57f80c9026db48a` | Integrate real PEFT lifecycle and Transformers generation into CPU development cycle |
| [#111](https://github.com/kys928/Hephaestus/pull/111) | `5a6300022f62d9933ceedbad67b36d1a4a79bb36` | Resume only the authentic governed 45-stage campaign: SHA-verified prefix and scoped acquisition admission |
| [#113](https://github.com/kys928/Hephaestus/pull/113) | `37a9eb9c69090e41401b6573b96416a903d0d829` | Start NEW governed 45-stage campaign with authentic zero-history durable custody |
| [#114](https://github.com/kys928/Hephaestus/pull/114) | `acde3aaa00b5097431ace7b4bf254aa542939388` | Measure one exact WikiText continuation source without fabricating clearance or acquisition |
| [#115](https://github.com/kys928/Hephaestus/pull/115) | `f7ec89f166615319107b13edbc4633dc69a9f120` | Fail closed: prevent approved WikiText from training via unreviewed generic preprocessor |
| [#116](https://github.com/kys928/Hephaestus/pull/116) | `77467cc0c4073a9a5996a9a45eb52b896f013f09` | Preserve exact reviewed WikiText token windows through governed native training |
| [#117](https://github.com/kys928/Hephaestus/pull/117) | `dcfa6e5260bcebc895d81e42f4d97ecf618a2c26` | Separate bounded internal data admission from independent certification |
| [#118](https://github.com/kys928/Hephaestus/pull/118) | `e92a397c8f9a652e4d3606889407f41693248def` | Add bounded source discovery, rejection and full-corpus internal screening |
| [#119](https://github.com/kys928/Hephaestus/pull/119) | `759f787a69514166cb259c5cf80235ada4b8f04a` | Generate original validated tasks and select an eligible internal source |
| [#120](https://github.com/kys928/Hephaestus/pull/120) | `7e3231cda018fadcb91f18388f9ec9be00a4df77` | Execute genuine internal CPU research through the native 45-operation spine |
| [#121](https://github.com/kys928/Hephaestus/pull/121) | `3eb9e9f295dcefb9340f1e9aa4074f8669d14c1c` | Generalize governed CPU campaign identity, seeds, budgets and immutable resume scope |
| [#122](https://github.com/kys928/Hephaestus/pull/122) | `c95e7ae462839b5bcf984133f9055e41f6dcdd0a` | Run three genuine CPU training seeds and measure frozen held-out learning |
| [#123](https://github.com/kys928/Hephaestus/pull/123) | `b57cb7f4fa7763cbf44a76c3fdbdb4ac4561518a` | Preserve verified CPU campaign archives on durable mounts and enforce merge gates |
| [#124](https://github.com/kys928/Hephaestus/pull/124) | `12c2bf40e63417437600eddcfed78343a7f51dc6` | Diagnose genuine CPU generation-format failures and define governed GitHub evidence releases |
| [#125](https://github.com/kys928/Hephaestus/pull/125) | `8f75e63b22ef7fe37e52727d26a465151ccba7cf` | Test frozen development JSON failures with real CPU chat-template and token-limit ablations |
| [#126](https://github.com/kys928/Hephaestus/pull/126) | `bec0a33db45277f35cfe47c3328ae78b0a58e6a3` | Test real chat-aligned supervised JSON repair with 96 genuine CPU LoRA updates |
| [#127](https://github.com/kys928/Hephaestus/pull/127) | `720c81d3c84c07219fd28973c6060d1e9b62dacf` | Govern immutable GitHub, Hugging Face and Kaggle evidence-provider references |
| [#128](https://github.com/kys928/Hephaestus/pull/128) | `699cdcbcebc2bea4d10c8a498eb8b28316ff6d40` | Gate original five frozen CUDA roles on verified CPU answer improvements before GPU allocation |
| [#129](https://github.com/kys928/Hephaestus/pull/129) | `a2cee007bd1b0f706ed8f85da940d05aec78a418` | Replicate real CPU JSON-answer improvement across two further frozen-seed experiments |
| [#130](https://github.com/kys928/Hephaestus/pull/130) | `30086e4319ed9113503993129ab5a46d5b6b88a3` | Pin Kaggle Model variation versions as first-class durable scientific weight references |
| [#131](https://github.com/kys928/Hephaestus/pull/131) | `12768aa0ba4497fb1024c598e03f3768f93a3346` | Confirm real JSON-answer improvement once on untouched original generated evaluation |
| [#132](https://github.com/kys928/Hephaestus/pull/132) | `238009a328580dca5a2b72e856608b9d7ee6364c` | Execute one real governed 45-stage CPU campaign with hash-bound 96-step JSON repair |
| [#133](https://github.com/kys928/Hephaestus/pull/133) | `66000cfbfb2798299fd6f636a1e3ff598b31333f` | Verify original complete CPU 45/45 archive before any original five-role CUDA GPU replay |
| [#134](https://github.com/kys928/Hephaestus/pull/134) | `6b791a5c6c6bf547b6ab1d74caae75ff613f2cbb` | Compose authentic frozen five-CUDA roles with existing governed 45-stage production services |
| [#135](https://github.com/kys928/Hephaestus/pull/135) | `f550d22591d1fec73407dba66b99b6143fd8dfd5` | Rebind verified Tier A original JSON/EOS training data to five frozen CUDA role dependencies |
| [#136](https://github.com/kys928/Hephaestus/pull/136) | `e41afa04bd5e1be90d47a724cb6316e20b3c2dcb` | Attach all five real frozen CUDA inference roles to native Tier A research stage observations |
| [#137](https://github.com/kys928/Hephaestus/pull/137) | `0d94cfcfe6c311de63f76f56838fec22688a9662` | Implement genuine CUDA-only Tier A target LoRA training, checkpoint recovery and native evaluation |
| [#138](https://github.com/kys928/Hephaestus/pull/138) | `d04e452d2c04dfdc5765ba086a855af0aa104712` | Gate original Tier A CUDA worker on independently signed compute authorization and durable native custody |
| [#139](https://github.com/kys928/Hephaestus/pull/139) | `f6ffa1e804133cbab674e1ae28d4bfda18052b1e` | Add RunPod live-price/balance budgeting, 15-minute bounded pause, and detached scientific worker |
| [#141](https://github.com/kys928/Hephaestus/pull/141) | `905625a9057fb8df7159cec887bf49df675f7a5e` | Prepare full original CUDA 45 campaign on private RunPod volume and audit pinned model/storage feasibility |

## Divergent / unmerged — preserve and review separately

**Do NOT close, force-merge or discard these three branches** until their unique tip commits have been reviewed against the production code already integrated:

| PR | Unique history relative to original R10 | Exact original head | Scope |
| --- | --- | --- | --- |
| [#99](https://github.com/kys928/Hephaestus/pull/99) | GitHub compare R10 behind by 22 commits | `b4a7df83f44ccddda2c3256607f48386c0b72281` | CUDA production role preflight |
| [#112](https://github.com/kys928/Hephaestus/pull/112) | GitHub compare R10 behind by 1 commit | `bafe3f153dfbc3708b959d6e9b224cce815d3f09` | CPU evidence recovery / HF source |
| [#140](https://github.com/kys928/Hephaestus/pull/140) | GitHub compare R10 behind by 13 commits | `e749ae7cb9fe3a688107fe7178e49e6f6e3dc5af` | RunPod live prepaid and quote checks |

## Review and cleanup rules

1. Do not delete or mutate historical S3 receipts, source tags, original closed-failure evidence, R10 original checkpoint or scientific archives.
2. A GitHub PR marked `closed` as **superseded** is not the same thing as a GitHub `merged` PR, nor evidence the full feature was active; original commits are independently traversable from the preserved R10 parent, while integrated code remains subject to full CPU tests and explicit activation boundaries.
3. Maintain the historical PR links and source SHAs in this ledger. Avoid 55 redundant merge commits that would replay historical 45-stage workflows or reintroduce obsolete YAML.
4. For #99, #112 and #140, inspect only their unique diffs, identify whether they contain needed production code not already in main, then safely port or close as rejected/obsolete under their own records.
5. Original role-model automatic dispatch stays `false`; no old paid workflow or signed grant should be reactivated by this maintenance cleanup.
6. Separate externally signed scientific certification and model promotion from R10's verified internal CUDA45 completion, and do not close [issue #102](https://github.com/kys928/Hephaestus/issues/102) on the basis of commit ancestry.

### Repository status after PR #154

- Single maintainable `main` includes real production code, frozen original five-role inference, CPU/GPU target training, scientific and immutable archive verification, and working CI for original external approval boundaries.
- Exactly 109 obsolete R10-only one-off workflow files omitted from the **active** tree; all retained in immutable old ancestry.
- [R10 report](../evidence/original-cuda45-r10/README.md) remains independent, and all original optimizer evidence remains off-repository.
- Full pinned CPU regression: **1,058 passed, 21 skipped**. Python 3.11/3.12 existing main CI succeeded. No new paid Pod created for integration.

**This ledger is metadata provenance only.** It does not alone close issue #151: final divergent PR review, ongoing curated workflow maintenance, mainline confidence tests and clean documentation are still required.
