# Runtime monitoring — stable reference path

A runtime monitor records real events, failures, bounded run progress and evidence references; it **cannot** invent completed stages, re-authorize a spent compute budget, or silently change a frozen GPU/run configuration.

For safe operational decisions use [run safety](operations/run-safety.md), [bounded recovery](autonomous_recovery.md), [compute execution policy](compute_execution_policy.md) and [replay verification](replay_verification_policy.md). The authentic R10 cold-readback and teardown outcome is linked from [the original scientific report](evidence/original-cuda45-r10/README.md).

An interrupted job is not automatically resumable. Confirm original checkpoint, once-claim and durable native state evidence first.
