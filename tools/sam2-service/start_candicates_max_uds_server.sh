#!/usr/bin/env bash
# Independent resident service for /v1/first-frame/candicates-max experiments.
# It deliberately has its own socket, PID file and log target, so it never
# replaces or stops the production sam2-service instance.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SAM2_SERVICE_SOCKET="${SCRIPT_DIR}/runtime/candicates-max-service.sock"
export SAM2_SERVICE_PID_FILE="${SCRIPT_DIR}/runtime/candicates-max-service.pid"
exec "${SCRIPT_DIR}/start_uds_server.sh"
