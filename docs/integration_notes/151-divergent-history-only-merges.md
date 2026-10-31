# Issue #151 — preserve the three historically diverged PR branches without activating old code

## Context

After original R10 source integration into `main` via ordinary non-squash [PR #154](https://github.com/kys928/Hephaestus/pull/154), GitHub compared all 58 originally open stacked PR heads (#81–#141) against the verified original R10 source `d6bcec7b61bcd3b285a4bcf4cdd50036440b8edf`.

Fifty-five were exact ancestors. Three had independent historical commits not reachable through original R10. **None required old scripts/workflows to be restored to active main**. Rather than erasing those commits, this integration branch creates a deliberate series of three standard **two-parent history-only (ours-tree) Git merges**. Each merge's resulting tree SHA is exactly the pre-existing main tree SHA `265da697c59e667f7381ba9fc1894a703a8cb789`, and each new merge's second parent is its original intact historical PR HEAD.

This is **preservation of original source history only**. It does not assert that the old PR was directly merged, that all its proposals are deployed, that its results passed current scientific standards, or that any old experimental paid workflow is authorized.

| Historical PR | Historical head preserved as real second parent | Additional original commits versus its source merge-base | Audit of unique changed files | Explicit history merge commit |
| --- | --- | ---: | --- | --- |
| [#112](https://github.com/kys928/Hephaestus/pull/112) | `bafe3f153dfbc3708b959d6e9b224cce815d3f09` | 1 | Three `pytest.importorskip` lines. Current main file `tests/test_historical_recovery_and_one_source.py` verified **byte-identical** to historical #112 HEAD (Git blob `9d135449ea53d67236dce1b68d95d6afee7690ba`). | `2bf523f1b089c7214f599816fecb3e08b3029055` |
| [#99](https://github.com/kys928/Hephaestus/pull/99) | `b4a7df83f44ccddda2c3256607f48386c0b72281` | 22 | Three obsolete one-off CUDA launch/diagnostic workflows and five early manually bounded five-role preflight/probe scripts. The current R10 model runtime and scoped prepaid execution supersede these scripts; do not reintroduce the old execution YAML. | `cc86422b1547e543c3aec23e5b1d1d402089ecac` |
| [#140](https://github.com/kys928/Hephaestus/pull/140) | `e749ae7cb9fe3a688107fe7178e49e6f6e3dc5af` | 13 | Seven old RunPod-specific read-only repair, diagnosis and transient Pod inspection workflows. Retain them in ancestry, not in current automatic CI. | `fb5b919805feb8cefaeda4006ebeeff75dacec18` |

## Chain verification

Starting exact existing `main`: `eb072487618085631d28c21c6091a088f498b206`.

- Merge #112 history-only: first parent existing main, second parent #112 head, original **same** main tree.
- Merge #99 history-only: first parent previous history-only merge, second parent #99 head, original **same** main tree.
- Merge #140 history-only: first parent previous history-only merge, second parent #140 head, original **same** main tree.
- Add this public-safe explanatory documentation in a separate commit, then merge this documentation branch into main via a **regular merge commit**. Never squash, force-push, amend or delete old science.
- Original historical PR branches and all raw source commit bytes remain reachable from main after that merge; the three **PRs themselves can be closed as superseded, never represented as individually merged scientific successes**.

## Safety boundary

No new auto-triggered GPU workflow, CUDA Pod, S3 mutation, secret injection, role-routing activation, old signed approval, scientific result rewrite, external certification or checkpoint promotion is created by these metadata-only merges.

The current real R10 scientific archive and the production source remain at their previously cold-verified immutable hashes; [R10 report](../evidence/original-cuda45-r10/README.md).

Older separate investigation PRs outside the R10 stacked chain (e.g. previously rejected Evaluator experiments) are not automatically reclassified by this PR and require independent retention/closure decisions.

**Scientific integrity standard:** a preserved Git branch history is not proof that its experiments are reproducible, independently valid, or suitable for live production.
