# Runtime Monitoring

Runtime monitoring converts backend execution output into typed events, incidents, health classification, and stop recommendations. It does not decide promotion or certification.

## Runtime monitor role

The runtime-monitor phase receives the experiment plan, training plan, launch config, trainable data contract, and resolved stage profile. It prepares and launches the backend job, collects backend events and outputs, classifies the outcome with `RuntimePolicy`, derives incidents, and returns a short monitor payload.

The coordinator persists incident reports and indexes event payload references as artifacts.

## Runtime events

Runtime events use `RuntimeEvent` with:

- `event_id`
- `run_id`
- `step`
- `category`
- `message`
- optional `payload_ref`

Backend and subprocess events must prefer `payload_ref` over inline logs or payloads. Subprocess event parsing recognizes `EVENT|category|step|message|payload_ref` lines and treats stderr as a runtime incident event.

## Incident derivation

`incident_from_event()` converts `INCIDENT` runtime events into `IncidentRecord` values. Incidents are medium severity by default and high severity when the event message contains hard-failure language. Launch failures are represented by a dedicated high-severity `runtime_launch` incident.

## Health classification

`RuntimePolicy.classify()` maps observed issues to monitor outcomes:

- Any deterministic failure count greater than zero returns `hard_abort`.
- Incident count at or above the waste threshold returns `waste_stop`; the threshold is 3 normally and 2 when stop sensitivity is high.
- Any nonzero incident count below the waste threshold returns `soft_suspicion`.
- No incidents and no deterministic failures returns `healthy`.

`stop_recommendation()` maps outcomes to operator-facing recommendations: `hard_abort` to `stop_now`, `waste_stop` to `stop_for_waste`, `soft_suspicion` to `pause_and_review`, and otherwise `continue`.

## Runtime artifacts

Runtime monitoring may observe metric files, probe outputs, deterministic-check outputs, logs, and other backend artifacts. It records only references. The evaluator later consumes backend `training_outputs`, especially `intermediate_eval` and `checkpoint_candidates`.

## Boundary invariants

- Runtime monitoring may stop, pause, or recommend review according to runtime policy.
- Runtime monitoring must not promote checkpoints.
- Runtime monitoring must not mutate lineage trust directly.
- Runtime monitoring must preserve raw heavy logs/artifacts outside control memory and expose path references.
