# Legacy execution is a demo/test route

The legacy `build_orchestrator` path is supported for `DryRunBackend` demos.
It does not authenticate real production training approvals. A caller-provided
`approval_request_id` is never authority. A failed launch guard now raises before
the runtime phase, and the runtime monitor independently enforces the boundary.

Repository integration tests may explicitly pass `legacy_fixture_mode=True`.
This permits only the fixed local `tests/fixtures/fake_trainer.py` and
`tests/fixtures/fake_ardor_runner.py` subprocesses after checking the prepared
execution specification. It cannot authorize an arbitrary configured trainer,
remote backend or real Ardor model. The fixture subprocess tests still run actual
processes and inspect artifacts; they are not evidence of real model training.

Real research execution belongs to the governed staged/lifecycle compositions
and their genuine approval, source, data, checkpoint and resource contracts.
This restriction does not change those production or frozen CUDA boundaries.
