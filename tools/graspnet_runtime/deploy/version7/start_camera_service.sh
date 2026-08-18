#!/usr/bin/env bash
set -euo pipefail
V7_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$V7_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$V7_DIR/../../../.." && pwd)"
LOG_DIR="$V7_DIR/logs"
SOCKET_PATH="${GRASPNET_V7_CAMERA_SOCKET:-/tmp/graspnet-v7-camera.sock}"
JOBS_ROOT="${GRASPNET_V7_JOBS_ROOT:-$V7_DIR/jobs}"
CAMERA_ARGS=(
  --socket-path "$SOCKET_PATH" --allow-root "$JOBS_ROOT"
  --d405-serial "${GRASPNET_V7_D405_SERIAL:-260322270600}"
  --d405-fps "${GRASPNET_V7_D405_FPS:-30}"
  --femto-serial "${GRASPNET_V7_FEMTO_SERIAL:-CL8L56301LK}"
  --femto-warmup-frames "${GRASPNET_V7_FEMTO_WARMUP_FRAMES:-3}"
  --femto-request-flush-frames "${CAMERA_BROKER_FEMTO_REQUEST_FLUSH_FRAMES:-30}"
  --femto-frame-read-retries "${CAMERA_BROKER_FEMTO_FRAME_READ_RETRIES:-10}"
  --min-valid-depth-ratio "${CAMERA_BROKER_MIN_VALID_DEPTH_RATIO:-0.01}"
  --max-capture-bundles "${CAMERA_BROKER_MAX_CAPTURE_BUNDLES:-200}"
  --arm-host "${PARM_ARM_HOST:-${GRASPNET_V7_ARM_HOST:-192.168.3.11}}"
  --arm-id "${GRASPNET_V7_ARM_ID:-1}"
)
if [ "${GRASPNET_V7_D405_POSTPROCESS:-1}" = "0" ]; then CAMERA_ARGS+=(--no-d405-postprocess); fi
# Version7 and Version9 both require this in their preflight configuration.
# Keep the persistent service default aligned with their launch scripts; set
# GRASPNET_V7_D405_HOLE_FILLING=0 only for an explicit no-hole-fill experiment.
if [ "${GRASPNET_V7_D405_HOLE_FILLING:-1}" = "1" ]; then CAMERA_ARGS+=(--d405-hole-filling); fi
if [ "${GRASPNET_V7_DISABLE_FEMTO:-0}" = "1" ]; then CAMERA_ARGS+=(--disable-femto); fi
if [ "${GRASPNET_V7_DIRECT_PLANNER:-0}" = "1" ]; then CAMERA_ARGS+=(--no-arm-state); fi
mkdir -p "$LOG_DIR" "$JOBS_ROOT"
if [ -e "$SOCKET_PATH" ] || [ -L "$SOCKET_PATH" ]; then
  echo "Camera socket already exists: $SOCKET_PATH" >&2; exit 1
fi
nohup "$PROJECT_ROOT/.venv/bin/python" "$V7_DIR/camera_service.py" \
  "${CAMERA_ARGS[@]}" \
  >"$LOG_DIR/camera_service.log" 2>&1 &
echo "$!" >"$LOG_DIR/camera_service.pid"
echo "Started Version7 camera service: pid=$(cat "$LOG_DIR/camera_service.pid")"
echo "Log: $LOG_DIR/camera_service.log"
