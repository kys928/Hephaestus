#!/usr/bin/env bash
# SSH/Jupyter disconnection-safe process handoff on an ALREADY APPROVED RunPod Pod.
# This is NOT crash-proof/host-shutdown-proof; restart is owned by an external supervisor.
set -euo pipefail
umask 077
if [[ "${1:-}" != /workspace/* ]]; then
  echo "First argument must be the already-approved persistent /workspace campaign directory" >&2
  exit 3
fi
WORKSPACE="$1"
shift
if [[ ! -d "$WORKSPACE" || ! -f "$WORKSPACE/context_ref.txt" ]]; then
  echo "No approved native CUDA workspace present" >&2
  exit 3
fi
if [[ ! -f "$WORKSPACE/external-original-cuda-grant.json" ]]; then
  echo "The independently signed original GPU grant is missing" >&2
  exit 3
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$WORKSPACE/original-cuda-detached-worker.log"
LOCK="$WORKSPACE/.original-cuda-worker.lock"
PID="$WORKSPACE/original-cuda-detached.pid"
if ! command -v nohup >/dev/null || ! command -v setsid >/dev/null || ! command -v flock >/dev/null; then
  echo "Missing nohup/setsid/flock; unattended launch refused" >&2
  exit 4
fi
# flock -n rejects a second real worker; do not kill or overwrite an existing one.
# Important: the parent connection can vanish without sending SIGHUP to the worker.
nohup setsid flock -n "$LOCK" \
  python -u "$ROOT/scripts/run_original_tier_a_cuda_worker.py" \
  --workspace "$WORKSPACE" \
  --signed-approval-file "$WORKSPACE/external-original-cuda-grant.json" \
  "$@" >> "$LOG" 2>&1 < /dev/null &
CHILD_PID=$!
printf '%s\n' "$CHILD_PID" > "$PID"
echo "Detached process launched with pid=$CHILD_PID; this is NOT proof of CUDA execution."
echo "Monitor original evidence in $LOG; independent pod/lease supervisor remains required."
