# Hephaestus — user-owned provider-aware cloud spending and resilient experiments

Status: draft policy + RunPod read-only quote + detached worker; no live account balance query, GPU allocation, source promotions or historical certification. The original CPU 45/45 remains verified and original GPU 45/45 remains unexecuted.

## Owner delegation

One-time authorized funding mode may be `fixed_prepaid_limit` or explicitly opted-in `available_prepaid_balance_opt_in`. Absence of a numeric limit must **not** be interpreted as consent for a user who has not opted into account-balance delegation. The owner can revoke/change the scope, or select a fixed cap at any point. The current complete original-five-role CUDA campaign has a user-issued authorization to spend its **currently available** RunPod account balance, not later top-ups, debt, external certification fees, or other provider accounts.

At launch, obtain the authenticated live RunPod prepaid balance and current competing spend. Freeze an immutable ceiling no greater than available credit minus already committed active usage, expected storage/evidence preservation reserve and user-specific reserves. Future account top-ups must never increase the original ceiling without fresh owner approval. Read-only input uses RunPod GraphQL `myself { clientBalance currentSpendPerHr isAutoPayEnabled }` while RunPod REST API v2 offers `catalog/gpus` priced GPU listings. The RunPod v2 provider must be refreshed before allocation and during expenditure; if the API becomes unavailable, no further spending may be autonomously authorized. Auto-pay must be disabled for prepaid-balance exhaustion mode, otherwise new credit could be charged invisibly.

## Provider-neutral estimator

Projected total USD = selected live on-demand GPU $/GPU-hour × benchmark-derived GPU hours + existing concurrent resource spend over this window + storage/network-volume/evidence retention and one-off setup/transfer overhead. Quote selected actual GPU ID, region/datacenter, cloud tier, VRAM, hourly price, expected duration with uncertainty, current balance, competing charges, storage reserve, approved hard ceiling and reference timestamp. Check true available stock through the RunPod adapter's pinned datacenter placement API; a price listing alone does not prove capacity. Show the user estimated cost and predicted remaining balance before starting. Use measured step timing and current actual provider charges to revise ETA and cost during the run.

## 15-minute pause and fallback

When estimated cost worsens or the current experiment is projected to exceed its initially estimated spend, save a verified checkpoint and append a durable `PAUSE_NOTIFY` decision with timestamp and revised quote. Notify the owner. During pause, GPU compute needs to be stopped/released by a real off-pod controller: an idling paid Pod can still exhaust the entire budget in the 15-minute wait.

After 900 seconds of no response, the controller can enter `RESUME_WITHIN_EXISTING_APPROVAL` **only** when the latest authenticated current remaining prepaid balance, current GPU price, storage reserve, original owner-authorized envelope and exact experiment scope still cover the next bounded operation. A missing reply cannot expand any cap or override a refusal. If budget exhausted or insufficient, state becomes `STOP_NO_AVAILABLE_PREPAID_FUNDS` and any active GPU Pod must be terminated by an independent controller. Fresh deposits, a higher user cap or a different cloud provider require renewed approval.

**Important:** the current PR implements deterministic decisions, not a fully deployed distributed pause scheduler or live cost meter. It does not override the existing external Ed25519-signed GPU compute lease; an expiring signed GPU grant still needs a separately authorized renewal. Connecting the signed lease to an always-on controller is remaining work.

## Connection/disaster recovery

Use `nohup setsid flock` on an already authorized Pod to survive an SSH/Jupyter/browser disconnection without a second concurrent worker. Place checkpoint/optimizer/RNG state, work manifests, science receipts and logs on a real independently retained network volume. The worker should be PID 1 or under a restart-capable service supervisor for container crashes, and should periodically checkpoint and validate each actual step; a logged exit, lost SSH or stopped Pod is not automatically a failed scientific run.

A terminated Pod cannot keep executing. Recovery requires an off-pod controller to reconcile the old Pod, obtain new verified capacity and budget/grant, provision a compatible replacement Pod with the same network volume, and resume the exact original checkpoint/RNG/sample lineage. Failure to verify the checkpoint must block resumption, not silently start a new science lineage. RunPod prepaid exhaustion may stop Pod and eventually endanger even network-volume retention; retain important evidence outside RunPod and reserve storage credit.

## Remaining real deployment gates

Live RunPod balance/price observations have **not** been performed by these tests, because this conversation has no authenticated RunPod management connection. Actual externally signed compute grant, a funded/guarded cloud controller, a provider-level cost monitor, a separate always-on termination/redeployment supervisor, and measured GPU resource availability are still required. The original five-role CUDA 45 operation campaign has not executed; historical original-reference contamination/Granite comparisons, signed independent certification and issue #102 remain open.

Source references: https://www.runpod.io/pricing ; https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx ; https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod-migrate/reference/graphql-to-v2.md .