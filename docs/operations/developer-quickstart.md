# Developer quickstart (safe, no GPU)

Run these commands from a clean checkout on Python 3.11+.

## Install and inspect

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests/
hephaestus-infrastructure --help
python -m hephaestus.cli.doctor --format json
```

The base install is deliberately lightweight. Optional model/CUDA dependencies are **not** necessary for the local demo.

## Inspect a deterministic demo

```bash
python -m hephaestus.cli.create_demo_state --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.cli.inspect_run --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.cli.verify_replay --state-root /tmp/hephaestus-demo-state --run-id demo-run --format json
python -m hephaestus.app.console --state-root /tmp/hephaestus-demo-state --host 127.0.0.1 --port 8765
```

The console is a local HTTP inspection interface. The demo uses `DryRunBackend`: recorded checkpoint names are references, **not** trained model weights. Do not commit the generated demo state. On Windows, substitute a writable temporary directory for `/tmp/hephaestus-demo-state`.

See the [existing local demo specification](../local_app_quickstart.md) and [read-only/restricted console routes](../operator_console.md).

## Pinned R10 CPU regression environment

To reproduce the integrated test dependency profile on a suitable machine, install the CPU Torch build and optional test extras (may download substantial packages):

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,peft-training,role-inference,s3,independent-certification]"
python -m pytest -q tests/
```

As recorded in [issue #151](https://github.com/kys928/Hephaestus/issues/151), the 2026-10-31 integration PR #157 full pinned CPU run yielded **1,060 passed / 21 skipped** and Python 3.11/3.12 lightweight suites each yielded **924 passed / 113 skipped**. These are dated CI outcomes, not a guarantee for another dependency platform, and they do not prove a new live GPU result.

## Where to start debugging

Confirm eval integrity → run/config identity → data/rights/manifest → tokenizer/wrapper → checkpoint/architecture loader → optimizer dynamics → model-family limitations. Never prescribe retraining before isolating the failure domain.

There is **no general, authorized "launch any GPU experiment" command** here. Old experiment scripts and historical Actions remain archival; see [run safety](run-safety.md).
