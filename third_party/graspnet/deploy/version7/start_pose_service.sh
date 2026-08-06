#!/usr/bin/env bash
set -euo pipefail
V7_DIR="$(cd "$(dirname "$0")" && pwd)"; ROOT_DIR="$(cd "$V7_DIR/../.." && pwd)"; LOG_DIR="$V7_DIR/logs"; SOCKET="${GRASPNET_V7_POSE_SOCKET:-/tmp/graspnet-v7-pose.sock}"
POSE_CONFIG="${GRASPNET_V7_POSE_CONFIG:-$ROOT_DIR/deploy/version6/config_template.json}"
POSE_ROOT="${GRASPNET_V7_POSE_OUTPUT_ROOT:-$ROOT_DIR/deploy/version6/output}"
mkdir -p "$LOG_DIR"
if [ -e "$SOCKET" ] || [ -L "$SOCKET" ]; then echo "Pose socket already exists: $SOCKET" >&2; exit 1; fi
mkdir -p "$POSE_ROOT"
nohup "$ROOT_DIR/.venv/bin/python" "$V7_DIR/pose_service.py" --socket-path "$SOCKET" --config "$POSE_CONFIG" --allow-root "$POSE_ROOT" >"$LOG_DIR/pose_service.log" 2>&1 &
echo "$!" >"$LOG_DIR/pose_service.pid"; echo "Started pose service: pid=$(cat "$LOG_DIR/pose_service.pid")"
