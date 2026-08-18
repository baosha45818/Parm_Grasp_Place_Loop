#!/usr/bin/env bash
# Independent max-candidate service.  It never replaces sam2-service.sock.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SAM2_SERVICE_SOCKET="${SCRIPT_DIR}/runtime/sam2-service-max.sock"
export SAM2_SERVICE_PID_FILE="${SCRIPT_DIR}/runtime/sam2-service-max.pid"
exec "${SCRIPT_DIR}/start_uds_server.sh"
