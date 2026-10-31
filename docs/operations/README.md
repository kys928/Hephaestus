# Operations

These runbooks distinguish safe inspection and CPU validation from actions that require a separately approved compute, storage or production authorization.

1. [Developer quickstart](developer-quickstart.md): clean checkout, install, CPU checks, demo and replay inspection.
2. [Run safety](run-safety.md): experiment admission, resource/credential boundaries, incident response and paid execution.
3. [Release and rollback](release-checklist.md): protected main, CI, integration review, history preservation and recovery.

Current terminal/console reference documentation remains available at [local app quickstart](../local_app_quickstart.md) and [operator console](../operator_console.md). These are **local bounded inspection tools**, not an unrestricted experiment-launch UI.

The maintained GitHub Actions inventory is documented in [issue #151 workflow retirement](../integration_notes/151-active-workflow-retirement.md). Historical one-off paid launchers should be read from source history only, not restored or dispatched with previously spent approval.
