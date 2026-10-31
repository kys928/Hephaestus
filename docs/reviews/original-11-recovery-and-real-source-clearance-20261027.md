# Historical 11-operation state recovery + training-source clearance — evidence and decision

**Date:** 2026-10-27  
**Disposition:** HISTORICAL TEST RECOVERED AS TWO REPORTS ONLY; **NO ORIGINAL RESUMABLE STATE**; NO NEW SOURCE CLEARED FOR ACQUISITION. Original 45-step production campaign is **not** advanced by this work.

## 1. Actual original 11-step execution and state

The exact original [GitHub CPU research smoke run #40216554433](https://github.com/kys928/Hephaestus/actions/runs/40216554433), artifact **#12607209791**, was re-downloaded and cryptographically verified via authenticated read-only REST without altering source bytes.

Original artifact ZIP SHA-256: `db2963b6bbceb607260448f3a5eadf7e9f02e61b6e8267b499ca0e98cbaf61f4`. Two and only two included files:
- `cpu-staged-causal-planner.json`: SHA-256 `2eef80e12c31424e8f556a0a51532ed4c1b45d32134083cbfb443d1d78af1eb1`. Its **status vector** records 11 completed and the native operation `data_acquisition_audit.acquisition_approval_gate` as pending; it is **not the original `StagedWorkflowState` or `WORKFLOW_RECORD_COLLECTION`**.
- `cpu-real-pretrained-causal-role-output.json`: SHA-256 `b139b0c91d86e89998db461f42466eb6989b75828c24a74e2ae66e03006ecc0b`. This is actual SmolLM2 135M research-model inference evidence, **not the five frozen distinct CUDA production roles**.

Inspection of the exact original executable `tests/test_cpu_causal_staged_planner.py` at commit `db41b72baf01e263a0831ea3fe1b093a49723776` proves the workflow was created from `pytest.tmp_path`, and the workflow uploaded only `evidence/*.json`, not its SQLite DB, content-addressed artifact store, tokenizer or original scientific input files. We cannot legitimately synthesize an absent workflow state from the summary.

## 2. Complete S3 storage discovery and direct database opening

[Actual source commit `6f43ac84...` live workflow #40572417275](https://github.com/kys928/Hephaestus/actions/runs/40572417275) performed paginated read-only `ListObjectsV2` over **the whole assigned network-volume bucket**: **45,990 objects in 46 pages**, listing complete, no truncation. It found 16 SQLite candidates, all paths under `hephaestus/scientific/v1/positive_promotion/`, not the test's temporary GitHub runner location. Evidence artifact [#12767001238](https://github.com/kys928/Hephaestus/actions/runs/40572417275/artifacts/12767001238). ZIP SHA256 `186cbcd71060b8cb5ed02987e62d671ddf478140d59a469a82b94d26dad330ea`. Internal custody file SHA independently checked all three result members.

[Direct SQLite verification workflow #40574321928](https://github.com/kys928/Hephaestus/actions/runs/40574321928) opened **all 16** candidate databases read-only from temporary storage (max 512 KiB each; ~32–106 KiB observed), performed bounded `sqlite_master` / collection-count queries, then discarded the SQLite bytes. **0/16 contain `staged_workflow_states` or `staged_workflow_records` collections**. [Read-only DB metadata artifact #12766776389](https://github.com/kys928/Hephaestus/actions/runs/40574321928/artifacts/12766776389), downloaded ZIP SHA256 `759748dde4bb5614c7ad9ca815485a41d24c04f8d48f87f48933b739b1b2decf`. No private contents or downloaded DBs were published.

**Conclusion:** Original 11/45 is a historically genuine **temporary smoke test** with verified output, but **not** a recoverable stored production campaign. Repeating it can start a NEW, explicitly named durable run; such re-execution does not retroactively recover old state or count past operations. The `GovernedStagedOrchestrator.resume()` must not be invoked against a guessed/generated history.

## 3. One-source clearance research, and second candidate recheck

All public previews used the *exact original* S3 tokenizer, SHA-256 `3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25`, `sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce`. No vocab substitution or 1% threshold relaxation.

[Real screened HF dataset workflow #40574321746](https://github.com/kys928/Hephaestus/actions/runs/40574321746), actual live Hub card and 100 preview rows **at matched pinned Hub and viewer revisions**, successfully executed. [Evidence artifact #12767006934](https://github.com/kys928/Hephaestus/actions/runs/40574321746/artifacts/12767006934). Downloaded ZIP SHA256 `712adce672561632b812a7cf52bfa140283cf7b201d5031aebb407c5e0840580`.

| Source | Exact revision | Rights as declared by Hub | Raw original-tokenizer preview | Decision |
| --- | --- | --- | --- | --- |
| `x0root/math-hq-10k` | `d4ab3182f35395c6feb8f81819de75e886d4e436` | MIT, publisher describes SymPy-based synthetic generation | **2,408 / 32,178 unknown = 7.4834%**, 100 paired rows | **HOLD**: fails original ≤1%, not the structured-JSON/termination proof the failed model specifically needs |
| `roneneldan/TinyStories` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | CDLA-Sharing-1.0, GPT-generated stories | **675 / 22,444 unknown = 3.0075%**, 100 raw story rows | **HOLD**: raw source fails ≤1%. A conservative same-story 128-prefix+32-target, no-PII-regex, no-duplicate, per-row ≤1% profile yields only **2/100 rows, 348 tokens and 2 windows**, insufficient |

No corpus downloaded beyond public preview. Uncorrected raw newline results are preserved; newline/whitespace flattening never used to misstate eligibility.

### Strongest existing alternative: WikiText-2 raw, still **not cleared**

Prior frozen [PR #108 source-screening](https://github.com/kys928/Hephaestus/pull/108) inspected **36,718 WikiText-2 train rows**, reported **0% original unknown rate within its separately accepted English subset**, **2,060 repeated nonempty rows** and **54 privacy-regex flags pending review**. Its previous preview yielded 40 valid 128+32 same-row next-token windows, not JSON or short-answer capability evidence.

Rights still need careful treatment. The Hub metadata advertises `cc-by-sa-3.0` plus `gfdl`; the README text calls it CC BY-SA 4.0. Original Wikipedia data dates to before the Wikipedia June 2023 CC BY-SA 4.0 change; Creative Commons says earlier Wikipedia edits remain under 3.0 ([CC explanation](https://creativecommons.org/2023/06/29/wikipedia-moves-to-cc-4-0-licenses/)). This explains the apparent vintage split but **does not certify legal provenance, eliminate GFDL obligations or automatically authorize training**. The defensible research route is pinning exact older license terms and attribution/share-alike obligations, excluding identified privacy/duplicate risks, and screening the precisely selected train subset against predeclared development/frozen checks.

**Next experiment with highest expected utility:** use a specifically authorized bounded **WikiText-2 raw English continuation subset**, keep source revision/license obligations and original tokenizer fixed, isolate evaluation windows by article/group, deduplicate/PII-filter, and evaluate continuation on a frozen general pack as *noncertified CPU development*. It is NOT a justified replacement for JSON/short-answer capability datasets and cannot yield formal promotion without independent evidence.

## 4. True external boundary

A real candidate must be explicitly bound to exact corpus revision, files, rights profile, source-owner approval, original-tokenizer compatibility and frozen train/development/evaluation split policies. The current *production* operation 12 also requires accepted independent source screening and an authentic matching acquisition-approval decision, not only dataset selection approval. That status remains **pending**.

No external dataset acquired for training, no training launched, no GPU spend, no original campaign DB fabricated, no stage count increased, no independent certification or source rights clearance falsely asserted. The only next legitimate continuation is **NEW durable scientific campaign creation** after source readiness/authorization, with persisted state and artifact custody from the first stage, never importing old pytest status into authoritative state.
