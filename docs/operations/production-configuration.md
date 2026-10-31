# Effective production configuration

`ProductionCompositionSettings.config_dir` is resolved once at construction.
Supply an absolute path when starting outside the checkout. Planner, action
executor and scientific recovery share the same `ApprovalPolicy` instance;
recovery stage profiles and behavioral evaluation use that same configuration
root. Generation and model-evidence adapters already receive the explicit root.

Frozen evaluation-pack identities and approval policy fingerprints remain their
own contracts. A configuration directory is not approval to run training or edit
an evaluation pack. The unrelated-CWD regression copies a configuration tree,
changes a policy value, builds from another directory and verifies actual policy
and evaluator bindings. No service may quietly fall back to a different checkout
merely because its default was relative `configs/`.
