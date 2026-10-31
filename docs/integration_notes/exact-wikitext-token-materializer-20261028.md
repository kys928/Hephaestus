# Exact WikiText token materializer and native training handoff

Base: PR #115, `f7ec89f166615319107b13edbc4633dc69a9f120`. This change implements an explicit exact-token alternative while retaining the generic WikiText refusal. It does not admit the currently uncleared dataset or claim a completed campaign.

## Source and processing identity

`PINNED_WIKITEXT_SUBSET` fixes the original source commit, one Parquet file and exact byte SHA, original tokenizer artifact/file SHA, public evaluation-pack SHA, 36,718 source rows, 7,232 selected source rows, 1,755,120 eligible tokens, zero unknown tokens, 13,380 windows and the existing source-position/normalized-content commitment. The immutable typed contract is part of the independent review's preprocessing-profile hash. A different source, row ordering, tokenizer, public pack, selection commitment, count or processing policy invalidates that review.

The original screen and materializer share row acceptance code and the exact existing English heuristic. Within each accepted normalized row, the materializer slices token IDs into 128-prefix + 32-target windows at stride 64. Every record binds source index, normalized source-content SHA and token offset. Labels mask exactly the prefix with `-100`. It never joins rows, inserts EOS/wrappers, flattens newlines, decodes/re-encodes IDs, truncates or samples the reviewed windows.

The current contract reproduces the technical screen. If an actual independent privacy/leakage review requires more exclusions, the source processing policy, selection commitment and counts must be revised and reviewed explicitly. The current 13,380-window count is not a scientific obligation to retain unsafe examples.

## Native boundaries

`AutonomousDataPreprocessor.process_reviewed_continuation_acquisition` is the opt-in entrypoint. It requires an exact continuation contract, the actual frozen tokenizer checker and raw-text mapper, an independently accepted full-corpus review with all existing rights/reference/audit requirements, and a matching selection/acquisition approval and completed remote receipt. Generic extra filters/remapping are rejected. No trainable output is written before all source, tokenizer, public-pack and inventory checks complete. Output publication is atomic and refuses conflicting immutable bytes.

`StagedApprovedHuggingFaceDataFactory` takes this route only when the preprocessor has an explicit continuation contract and the composer receives `public_eval_pack_path`. Its ordinary route retains PR #115's fail-closed guard. Staged finalization retains model/recipe/baseline approval boundaries and independently verifies the selected model's actual tokenizer backend against the original frozen tokenizer.

## Explicit loader contract

Token records use `trainable-token-ids.v1`, distinct from the existing text schema. Preparation and the worker reject schema/evidence disagreement. The reader checks all IDs and exact masks, ordered source indices, hashes, offset strides, overlapping token consistency, accepted-row commitment and complete window count. Recipe limits must fit every window; for this subset the total input budget is at least **2,140,800 tokens**. A smaller budget blocks rather than silently shrinking the training set.

The frozen tokenizer's artifact identity and its HF loading directory are distinct fields. Preparation verifies exact original bytes; the worker checks the loaded backend against those original bytes. Original tokenizer bytes must remain accessible for initial loading and resume. At consumption the worker rehashes data, trainable contract and processing evidence, including resumed jobs. Existing text-loading behavior remains on its explicit schema.

## Verification and limits

The dedicated `exact-wikitext-materializer-cpu.yml` checks fixture-native materialization, approval refusal, the unchanged generic guard, explicit loader validation, staged handoff, real tiny CPU optimizer/checkpoint behavior, and existing CPU HF/PEFT regression tests.

Its separate live job reuses the already accepted source-only quarantine scope. The exact real source and original S3 tokenizer are used to reproduce and validate every window in memory. Only aggregate counts and a token-window JSONL digest are uploaded. No source text, token records, training acquisition, protected-reference access or real-source training is authorized by that job.

Independent source rights, attribution, privacy and original-text evaluation custody remain unresolved. The fresh durable production count remains 0/45. Existing production admission also requires certified CUDA role inference; a CPU-only production role contract is a separate engineering prerequisite, as detailed in the campaign-readiness review. No GPU replay may be relabeled as completion of the CPU prerequisite.
