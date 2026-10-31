# Controller ownership and interrupted operations

The supported SQLite and JSONL staged runtimes use one host-local kernel lock per
workflow; the production loop similarly owns a program before reading or changing
its state. Concurrent and recursive contenders receive `WriterBusyError` before
invoking services. Locks cover the entire invocation, have no time-based takeover,
and are released by process death. Durable lock files are coordination names, not
expiring leases. Never delete or replace them while a controller is alive.

Before calling a staged service, the controller persists an invocation intent.
After a return it persists the exact result, content-addressed when an artifact
store is configured. A crash after that receipt but before the workflow snapshot
reuses the returned result. A crash after intent but without a returned receipt
blocks with `ambiguous_operation_outcome`; it does not blindly repeat an unknown
side effect. Independent provider/worker reconciliation is required before that
intent can be resolved. A known returned pending result may be polled again with
the same stable operation identity and the next observation attempt.

These controls preserve the existing RunPod allocation-intent reconciliation;
they do not replace it or authorize a second allocation. Arbitrary custom cycle
drivers must still provide their own external-operation reconciliation. There is
no exactly-once guarantee for arbitrary external systems lacking idempotency.

This is **one-host** coordination on trustworthy local filesystem locks, not
cross-host fencing. Multi-host/shared-network-filesystem deployments are not a
supported execution topology. Other state repository types fail closed until
their own fenced ownership adapter is implemented and tested.
