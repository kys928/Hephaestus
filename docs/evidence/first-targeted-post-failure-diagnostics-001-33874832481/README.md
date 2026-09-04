# First targeted post-failure diagnostic proof

This directory freezes the exact deterministic outputs from GitHub Actions run `33874832481`.

Upstream chain:
- training: `first-bounded-scientific-training-001-33866198758`
- semantic evaluation: `first-semantic-evaluation-001-33869352751`
- first post-failure diagnosis: `diag-49904c2b7fa6cd1a` (`inconclusive`)

Targeted result:
- diagnosis status: `completed`
- leading domain: `data_coverage`
- confidence: `0.825`
- undertraining signal: not supported by the measured loss tail
- optimizer/scheduler pathology: not supported; recorded schedule conformed
- processed rows scanned: `54,878`
- prompt/target rows: `0`
- structured target rows: `0`
- Planner next intervention: `replace_or_mix_dataset`
- one primary variable: `dataset_mixture`
- next contract boundary: `dataset_discovery_and_selection`

No training, model mutation, promotion, Judge action, dataset mutation, or Planner action was executed by this diagnostic run.
