#!/usr/bin/env bash
# One-command pure-traditional pick launcher.  No GraspNet service is started.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOOP_ROOT="${PARM_LOOP_ROOT:-$(cd -- "${HERE}/../../.." && pwd)}"
SAM_DIR="${LOOP_ROOT}/tool/sam2-service"
SAM_SOCKET="${PARM_SAM2_SOCKET:-${SAM_DIR}/runtime/sam2-service.sock}"
PYTHON_BIN="${LOOP_ROOT}/.venv/bin/python"
LOG_DIR="${HERE}/logs"

mkdir -p "${LOG_DIR}"

sam_ready() {
  [[ -S "${SAM_SOCKET}" ]] \
    && curl --silent --fail --max-time 2 --unix-socket "${SAM_SOCKET}" \
      http://localhost/healthz >/dev/null
}

if ! sam_ready; then
  # A Unix socket pathname can survive a service crash.  It is not evidence
  # that a process is listening, so remove only this known stale socket before
  # starting the resident service again.
  if [[ -S "${SAM_SOCKET}" ]]; then
    echo "-> stale SAM socket detected; restarting resident SAM service"
    rm -f -- "${SAM_SOCKET}"
  else
    echo "-> starting resident SAM service"
  fi
  "${SAM_DIR}/start_uds_server.sh" >"${LOG_DIR}/sam2-service.log" 2>&1 &
  SAM_PID="$!"
  for _ in {1..180}; do
    sam_ready && break
    if ! kill -0 "${SAM_PID}" 2>/dev/null; then
      tail -n 40 "${LOG_DIR}/sam2-service.log" >&2 || true
      exit 1
    fi
    sleep 1
  done
  sam_ready || {
    echo "SAM service did not become ready within 180 seconds; see ${LOG_DIR}/sam2-service.log" >&2
    exit 1
  }
else
  echo "-> reusing resident SAM service: ${SAM_SOCKET}"
fi

echo "-> verifying Camera Broker for traditional D405 capture"
(
  cd "${LOOP_ROOT}"
  "${PYTHON_BIN}" -c \
    'from modules.module_0_prepare.ensure_camera_broker import ensure_camera_broker; ensure_camera_broker()'
)

exec "${PYTHON_BIN}" "${HERE}/run_live_traditional_pick.py" \
  --sam-socket "${SAM_SOCKET}" "$@"
