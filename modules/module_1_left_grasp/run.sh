#!/usr/bin/env bash
# One-command pure-traditional pick launcher.  No GraspNet service is started.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOOP_ROOT="${PARM_LOOP_ROOT:-$(cd -- "${HERE}/../.." && pwd)}"
SAM_DIR="${LOOP_ROOT}/tools/sam2-service"
DEFAULT_SAM_SOCKET="${SAM_DIR}/runtime/sam2-service.sock"
LEGACY_SAM_SOCKET="${LOOP_ROOT}/tool/sam2-service/runtime/sam2-service.sock"
if [[ -n "${PARM_SAM2_SOCKET:-}" ]]; then
  SAM_SOCKET="${PARM_SAM2_SOCKET}"
elif [[ -S "${LEGACY_SAM_SOCKET}" ]] \
  && curl --silent --fail --max-time 2 --unix-socket "${LEGACY_SAM_SOCKET}" \
    http://localhost/healthz >/dev/null; then
  # Reuse a resident pre-migration process without copying runtime state.
  # The next cold start uses tools/sam2-service and DEFAULT_SAM_SOCKET.
  SAM_SOCKET="${LEGACY_SAM_SOCKET}"
else
  SAM_SOCKET="${DEFAULT_SAM_SOCKET}"
fi
SAM_LOCK_FILE="${SAM_SOCKET}.lock"
SAM_PID_FILE="${SAM2_SERVICE_PID_FILE:-$(dirname "${SAM_SOCKET}")/sam2-service.pid}"
PYTHON_BIN="${LOOP_ROOT}/.venv/bin/python"
LOG_DIR="${LOOP_ROOT}/output/module_1_left_grasp/logs"

MODULE_ARGS=("$@")
MODULE_MODE=""
EXPLICIT_D405_SERIAL=""
for ((ARG_INDEX = 0; ARG_INDEX < ${#MODULE_ARGS[@]}; ARG_INDEX++)); do
  case "${MODULE_ARGS[ARG_INDEX]}" in
    --mode)
      if ((ARG_INDEX + 1 >= ${#MODULE_ARGS[@]})); then
        echo "--mode requires a value" >&2
        exit 2
      fi
      MODULE_MODE="${MODULE_ARGS[ARG_INDEX + 1]}"
      ((ARG_INDEX += 1))
      ;;
    --mode=*)
      MODULE_MODE="${MODULE_ARGS[ARG_INDEX]#*=}"
      ;;
    --d405-serial)
      if ((ARG_INDEX + 1 >= ${#MODULE_ARGS[@]})); then
        echo "--d405-serial requires a value" >&2
        exit 2
      fi
      EXPLICIT_D405_SERIAL="${MODULE_ARGS[ARG_INDEX + 1]}"
      ((ARG_INDEX += 1))
      ;;
    --d405-serial=*)
      EXPLICIT_D405_SERIAL="${MODULE_ARGS[ARG_INDEX]#*=}"
      ;;
  esac
done
case "${MODULE_MODE}" in
  gzhu)
    D405_SERIAL="260322270242"
    D405_FPS=15
    ROBOT_HOST="192.168.88.1"
    GRIPPER_HOST="127.0.0.1"
    FORCE_SENSOR_HOST="127.0.0.1"
    ;;
  xiangxin)
    D405_SERIAL="260322270600"
    D405_FPS=30
    ROBOT_HOST="192.168.3.11"
    GRIPPER_HOST="192.168.3.11"
    FORCE_SENSOR_HOST="192.168.3.22"
    ;;
  "")
    echo "--mode is required; choose gzhu or xiangxin" >&2
    exit 2
    ;;
  *)
    echo "unsupported --mode ${MODULE_MODE}; choose gzhu or xiangxin" >&2
    exit 2
    ;;
esac
if [[ -n "${EXPLICIT_D405_SERIAL}" && "${EXPLICIT_D405_SERIAL}" != "${D405_SERIAL}" ]]; then
  echo "--mode ${MODULE_MODE} requires --d405-serial ${D405_SERIAL}" >&2
  exit 2
fi
export PARM_D405_SERIAL="${D405_SERIAL}"
if [[ -n "${PARM_D405_FPS:-}" ]]; then
  D405_FPS="${PARM_D405_FPS}"
fi
if [[ "${D405_FPS}" != "5" && "${D405_FPS}" != "15" && "${D405_FPS}" != "30" ]]; then
  echo "PARM_D405_FPS must be one of: 5, 15, 30" >&2
  exit 2
fi
export PARM_D405_FPS="${D405_FPS}"
export PARM_ARM_HOST="${ROBOT_HOST}"
export PARM_GRIPPER_HOST="${GRIPPER_HOST}"
export PARM_FORCE_SENSOR_HOST="${FORCE_SENSOR_HOST}"

mkdir -p "${LOG_DIR}"
# A pristine tools/ checkout has no runtime directory yet.  flock cannot
# create its lock file through a missing parent and that failure used to be
# mistaken for a lock held by another starter, causing a 180-second wait.
mkdir -p "$(dirname "${SAM_SOCKET}")"

sam_ready() {
  [[ -S "${SAM_SOCKET}" ]] \
    && curl --silent --fail --max-time 2 --unix-socket "${SAM_SOCKET}" \
      http://localhost/healthz >/dev/null
}

wait_for_sam_ready() {
  for _ in {1..180}; do
    sam_ready && return 0
    sleep 1
  done
  return 1
}

lock_is_held() {
  ! flock -n "${SAM_LOCK_FILE}" -c true 2>/dev/null
}

if ! sam_ready; then
  # ``flock`` locks are released by the kernel when a service exits, but the
  # lock pathname remains.  Never remove a lock held by a live service.  A
  # lock which can be acquired is stale state and can be removed together
  # with a stale socket/PID file before a cold start.
  if lock_is_held; then
    echo "-> SAM service is already starting; waiting for its health check"
    wait_for_sam_ready || {
      echo "SAM service holds ${SAM_LOCK_FILE} but did not become ready within 180 seconds" >&2
      exit 1
    }
  else
    echo "-> recovering stale SAM runtime state before cold start"
    rm -f -- "${SAM_SOCKET}" "${SAM_LOCK_FILE}" "${SAM_PID_FILE}"
    echo "-> starting resident SAM service"
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
  fi
else
  echo "-> reusing resident SAM service: ${SAM_SOCKET}"
fi

echo "-> mode ${MODULE_MODE}: D405 ${D405_SERIAL}, arm ${ROBOT_HOST}, gripper ${GRIPPER_HOST}, force sensor ${FORCE_SENSOR_HOST}"
echo "-> D405 stream: 640x480 color+depth @ ${D405_FPS} FPS"
echo "-> Module 1 will open D405 directly through pyrealsense2 for this capture"

exec "${PYTHON_BIN}" "${HERE}/main.py" \
  --sam-socket "${SAM_SOCKET}" "$@"
