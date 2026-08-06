#!/usr/bin/env bash
set -euo pipefail

V7_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$V7_DIR/../.." && pwd)"
LOG_DIR="$V7_DIR/logs"
RUNTIME_DIR="$V7_DIR/runtime"
SOCKET_PATH="${GRASPNET_V7_SOCKET:-/tmp/graspnet-v7.sock}"
JOBS_ROOT="${GRASPNET_V7_JOBS_ROOT:-$V7_DIR/jobs}"
PID_FILE="${GRASPNET_SERVICE_PID_FILE:-$RUNTIME_DIR/graspnet-service.pid}"
LOG_FILE="${GRASPNET_SERVICE_LOG:-$RUNTIME_DIR/graspnet-service.log}"

mkdir -p "$LOG_DIR" "$RUNTIME_DIR" "$JOBS_ROOT"
if [ -e "$SOCKET_PATH" ] || [ -L "$SOCKET_PATH" ]; then
  echo "GraspNet V7 socket already exists: $SOCKET_PATH" >&2
  echo "Stop the old service first, or set GRASPNET_V7_SOCKET to another path." >&2
  exit 1
fi

GRASPNET_COMMAND=(
  "$ROOT_DIR/.venv/bin/python" "$V7_DIR/graspnet_service.py"
  --checkpoint-path "$ROOT_DIR/checkpoint/checkpoint-rs.tar"
  --num-view "${GRASPNET_V7_NUM_VIEW:-300}"
  --socket-path "$SOCKET_PATH"
  --allow-root "$JOBS_ROOT"
  --pid-file "$PID_FILE"
)

if [ "${GRASPNET_FOREGROUND:-0}" = "1" ]; then
  exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${GRASPNET_COMMAND[@]}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  nohup "${GRASPNET_COMMAND[@]}" >"$LOG_FILE" 2>&1 &

echo "Started GraspNet V7 service: pid=$!"
echo "PID file: $PID_FILE"
echo "Log: $LOG_FILE"
