#!/usr/bin/env bash
# Version7 grasp -> hold at graspup -> Module 2 standalone inspection flow.
#
# The Camera Broker owns D405/Femto for the whole Loop.  Module 1 and Module 2
# are clients; this script must never stop the Broker between them.
set -eEuo pipefail

V7_DIR="$(cd "$(dirname "$0")" && pwd)"
GRASPNET_SOURCE_ROOT="$(cd "$V7_DIR/../.." && pwd)"
V7_DIR="$GRASPNET_SOURCE_ROOT/deploy/version7"
LOOP_ROOT="${PARM_LOOP_ROOT:-$(cd "$GRASPNET_SOURCE_ROOT/../.." && pwd)}"
MODULE2_SCRIPT="${MODULE2_SCRIPT:-$LOOP_ROOT/modules/module_2_left_inspect_apart/run_inspect_then_place.py}"

V7_GRASPNET_SOCKET="${GRASPNET_V7_SOCKET:-/tmp/graspnet-v7.sock}"
V7_CAMERA_SOCKET="${CAMERA_BROKER_SOCKET:-${GRASPNET_V7_CAMERA_SOCKET:-/tmp/graspnet-v7-camera.sock}}"
V7_HANDOFF_FILE="$V7_DIR/logs/module2_handoff_ready.json"
GRASPNET_GRASP_SCRIPT="${GRASPNET_GRASP_SCRIPT:-$V7_DIR/run_grasp_v7.sh}"
MODULE2_MAX_ATTEMPTS="${MODULE2_MAX_ATTEMPTS:-3}"
V7_CAMERA_START_ATTEMPTS="${V7_CAMERA_START_ATTEMPTS:-3}"
V7_GRASPNET_STARTED_BY_THIS_SCRIPT=0
V7_CAMERA_STARTED_BY_THIS_SCRIPT=0
MODULE1_TIMER_ACTIVE=0
MODULE1_TIMER_SCOPE="graspnet-shell:${BASHPID}"

export GRASPNET_V7_JOBS_ROOT="${GRASPNET_V7_JOBS_ROOT:-$V7_DIR/jobs}"
export GRASPNET_V7_NUM_VIEW="${GRASPNET_V7_NUM_VIEW:-300}"
export GRASPNET_V7_SOCKET="$V7_GRASPNET_SOCKET"
export GRASPNET_V7_CAMERA_SOCKET="$V7_CAMERA_SOCKET"
# Module 2 reads the broker path from this canonical client-side setting.
export CAMERA_BROKER_SOCKET="$V7_CAMERA_SOCKET"
export GRASPNET_V7_ARM_HOST="${PARM_ARM_HOST:-${GRASPNET_V7_ARM_HOST:-192.168.3.11}}"
export GRASPNET_V7_ARM_ID="${GRASPNET_V7_ARM_ID:-1}"
export GRASPNET_V7_D405_POSTPROCESS="${GRASPNET_V7_D405_POSTPROCESS:-1}"
export GRASPNET_V7_D405_HOLE_FILLING="${GRASPNET_V7_D405_HOLE_FILLING:-1}"
# The singleton Camera Broker serves both Version7 and Module 2.
export GRASPNET_V7_DISABLE_FEMTO="${GRASPNET_V7_DISABLE_FEMTO:-0}"

report_module_metric() {
  if [[ -z "${PARM_MODULE_METRICS_DIR:-}" ]]; then
    return 0
  fi
  PARM_MODULE_REPORT_SCOPE="graspnet-shell:${BASHPID}" \
    "$LOOP_ROOT/.venv/bin/python" \
    "$LOOP_ROOT/modules/module_status_cli.py" "$@" || {
      echo "Warning: failed to persist module metric: $*" >&2
      return 0
    }
}

report_workpiece_timer() {
  if [[ -z "${PARM_WORKPIECE_ID:-}" ]]; then
    return 0
  fi
  "$LOOP_ROOT/.venv/bin/python" "$LOOP_ROOT/tool/workpiece_timer.py" "$@" || {
    echo "Warning: failed to persist workpiece timing: $*" >&2
    return 0
  }
}

wait_for_log() {
  local log_file="$1"
  local expected="$2"
  local timeout_s="$3"
  local pid_file="$4"
  local label="$5"
  local started_s=$SECONDS
  while ! grep -Fq "$expected" "$log_file" 2>/dev/null; do
    # A daemon that exits during initialization can never emit its ready
    # message.  Fail immediately instead of looking like a frozen script.
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(<"$pid_file")"
      if [[ "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
        echo "Version7 $label service exited before becoming ready." >&2
        echo "Last log lines ($log_file):" >&2
        tail -n 20 "$log_file" >&2 || true
        return 1
      fi
    fi
    if (( SECONDS - started_s >= timeout_s )); then
      echo "Timed out waiting for: $expected" >&2
      echo "Check log: $log_file" >&2
      return 1
    fi
    sleep 1
  done
}

expected_program_for_service() {
  case "$1" in
    camera) printf '%s\n' "$V7_DIR/camera_service.py" ;;
    GraspNet) printf '%s\n' "$V7_DIR/graspnet_service.py" ;;
    *) return 2 ;;
  esac
}

pid_is_expected_v7_service() {
  local pid="$1"
  local label="$2"
  local expected
  expected="$(expected_program_for_service "$label")" || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command_line" == *"$expected"* ]]
}

socket_owner_pids() {
  local socket_path="$1"
  command -v ss >/dev/null 2>&1 || return 0
  # ss reports the owning process PID for Unix-domain listeners owned by this
  # user.  Keep only numeric PIDs and tolerate systems where this information
  # is unavailable.
  ss -xlpnH 2>/dev/null | grep -F -- "$socket_path" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true
}

socket_listener_exists() {
  local socket_path="$1"
  command -v ss >/dev/null 2>&1 || return 1
  ss -xlH 2>/dev/null | grep -Fq -- "$socket_path"
}

stop_v7_service() {
  local label="$1"
  local pid_file="$2"
  local socket_path="$3"
  local pid=""
  local expected
  expected="$(expected_program_for_service "$label")" || return 2

  if [[ -f "$pid_file" ]]; then
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      if ! pid_is_expected_v7_service "$pid" "$label"; then
        echo "Refusing to stop pid=$pid from $pid_file: it is not $expected" >&2
        return 1
      fi
      echo "-> stopping Version7 $label service (pid=$pid)"
      kill -TERM "$pid"
      local started_s=$SECONDS
      while kill -0 "$pid" 2>/dev/null; do
        if (( SECONDS - started_s >= 30 )); then
          echo "Version7 $label service did not stop within 30 seconds; refusing to remove its socket." >&2
          return 1
        fi
        sleep 1
      done
    fi
  fi

  # A previous interrupted launcher can overwrite a pid file while its daemon
  # survives.  Discover Unix-socket owners as well, but stop only a process
  # whose exact command line is the expected Version7 service.
  local owner_pid
  while IFS= read -r owner_pid; do
    [[ -n "$owner_pid" ]] || continue
    if ! pid_is_expected_v7_service "$owner_pid" "$label"; then
      echo "Refusing to remove $socket_path: live owner pid=$owner_pid is not $expected" >&2
      return 1
    fi
    echo "-> stopping previous Version7 $label service discovered from socket (pid=$owner_pid)"
    kill -TERM "$owner_pid"
    local owner_started_s=$SECONDS
    while kill -0 "$owner_pid" 2>/dev/null; do
      if (( SECONDS - owner_started_s >= 30 )); then
        echo "Version7 $label socket owner pid=$owner_pid did not stop within 30 seconds." >&2
        return 1
      fi
      sleep 1
    done
  done < <(socket_owner_pids "$socket_path")

  # Do not unlink an active socket if this user cannot inspect its owner.
  if socket_listener_exists "$socket_path"; then
    echo "Refusing to remove active socket $socket_path whose owner could not be verified." >&2
    return 1
  fi

  # At this point there is no listener; remove only the stale pathname left
  # after an interrupted service.
  if [[ -S "$socket_path" ]]; then
    rm -f -- "$socket_path"
  fi
  rm -f -- "$pid_file"
}

start_v7_camera_with_retry() {
  if [[ ! "$V7_CAMERA_START_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "V7_CAMERA_START_ATTEMPTS must be a positive integer" >&2
    return 2
  fi

  local attempt=1
  local status=1
  while (( attempt <= V7_CAMERA_START_ATTEMPTS )); do
    echo "-> starting Version7 camera service (attempt $attempt/$V7_CAMERA_START_ATTEMPTS)"
    bash "$V7_DIR/start_camera_service.sh"
    V7_CAMERA_STARTED_BY_THIS_SCRIPT=1
    if wait_for_log \
      "$V7_DIR/logs/camera_service.log" \
      "camera service listening on $V7_CAMERA_SOCKET" \
      60 \
      "$V7_DIR/logs/camera_service.pid" \
      "camera"; then
      return 0
    else
      status=$?
    fi

    stop_v7_service \
      "camera" \
      "$V7_DIR/logs/camera_service.pid" \
      "$V7_CAMERA_SOCKET" || return 1
    V7_CAMERA_STARTED_BY_THIS_SCRIPT=0
    if (( attempt == V7_CAMERA_START_ATTEMPTS )); then
      echo "Version7 camera service failed after $V7_CAMERA_START_ATTEMPTS attempts." >&2
      return "$status"
    fi
    echo "-> camera initialization failed; waiting 3s before a clean-process retry"
    sleep 3
    ((attempt += 1))
  done
}

cleanup_v7_on_error() {
  local status=$?
  if (( status == 0 )); then
    return
  fi
  if (( MODULE1_TIMER_ACTIVE )); then
    report_workpiece_timer finish --module module_1_1_graspnet \
      --scope "$MODULE1_TIMER_SCOPE" --status FAILURE \
      --detail "Version7 shell异常退出，exit=$status"
    MODULE1_TIMER_ACTIVE=0
  fi
  if (( status == 20 )); then
    echo "-> no valid grasp candidate this round; keeping resident Version7 services for the scheduled retry"
  else
    echo "-> flow failed; keeping resident Version7 services for diagnosis and later retry" >&2
  fi
}

run_module_2() {
  if [[ ! -f "$MODULE2_SCRIPT" ]]; then
    echo "Module 2 standalone entry is unavailable: $MODULE2_SCRIPT" >&2
    return 1
  fi
  if [[ "${PARM_LOOP_WORKPIECE:-}" != "1" && "${PARM_LOOP_WORKPIECE:-}" != "2" ]]; then
    echo "PARM_LOOP_WORKPIECE must be explicitly set to 1 or 2 before Module 2 starts" >&2
    return 2
  fi
  local placement_gate_args=()
  if [[ -n "${MODULE2_PLACEMENT_READY_FILE:-}" || -n "${MODULE2_PLACEMENT_PERMIT_FILE:-}" ]]; then
    if [[ -z "${MODULE2_PLACEMENT_READY_FILE:-}" || -z "${MODULE2_PLACEMENT_PERMIT_FILE:-}" ]]; then
      echo "MODULE2_PLACEMENT_READY_FILE and MODULE2_PLACEMENT_PERMIT_FILE must be set together" >&2
      return 2
    fi
    placement_gate_args=(
      --placement-ready-file "$MODULE2_PLACEMENT_READY_FILE"
      --placement-permit-file "$MODULE2_PLACEMENT_PERMIT_FILE"
    )
  fi
  if [[ "${LOOP_AUTO_CONFIRM_MOTION:-0}" == "1" ]]; then
    placement_gate_args+=(--auto-confirm-motion)
  fi
  if [[ "${PARM_LOOP_SKIP_INSPECT_DIRECT:-0}" == "1" ]]; then
    placement_gate_args+=(--skip-inspect-direct)
  fi
  cd "$LOOP_ROOT"
  "$LOOP_ROOT/.venv/bin/python" "$MODULE2_SCRIPT" \
    --planner-host "${PARM_PLANNER_HOST:-127.0.0.1}" \
    --arm-host "${PARM_ARM_HOST:-${GRASPNET_V7_ARM_HOST:-192.168.3.11}}" \
    --gripper-host "${PARM_GRIPPER_HOST:-127.0.0.1}" \
    --workpiece "$PARM_LOOP_WORKPIECE" \
    --weight-handoff-file "$V7_HANDOFF_FILE" \
    "${placement_gate_args[@]}"
}

run_module_2_with_camera_retry() {
  local attempt=1
  local status=1
  if [[ ! "$MODULE2_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MODULE2_MAX_ATTEMPTS must be a positive integer" >&2
    return 2
  fi
  while (( attempt <= MODULE2_MAX_ATTEMPTS )); do
    echo "-> Module 2 inspection attempt $attempt/$MODULE2_MAX_ATTEMPTS"
    if run_module_2; then
      return 0
    else
      status=$?
    fi
    if (( attempt == MODULE2_MAX_ATTEMPTS )); then
      echo "-> Module 2 failed after $MODULE2_MAX_ATTEMPTS attempts" >&2
      return "$status"
    fi
    echo "-> Module 2 returned failure; waiting for Femto frames to stabilize before retrying"
    sleep 3
    ((attempt += 1))
  done
}

main() {
  report_workpiece_timer start --module module_1_1_graspnet \
    --scope "$MODULE1_TIMER_SCOPE" --detail "Version7服务启动、预测、抓取与graspup交接"
  MODULE1_TIMER_ACTIVE=1
  cd "$GRASPNET_SOURCE_ROOT"
  # Module 0 uses a real request/response health check.  Reuse it here instead
  # of treating ``ss`` output as readiness: an existing Unix socket can be
  # healthy even when a restricted process cannot enumerate its listener.
  if [[ "${PARM_LOOP_REQUIRE_PRESTARTED_SERVICES:-0}" == "1" ]]; then
    echo "-> requiring prestarted Module 0 GraspNet service"
    PYTHONPATH="$LOOP_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$LOOP_ROOT/.venv/bin/python" -c '
from modules.module_0_prepare.ensure_graspnet_service import require_graspnet_service
require_graspnet_service()
'
    if [[ "${PARM_LOOP_MANAGE_CAMERA_BROKER:-0}" == "1" ]]; then
      echo "-> ensuring loop-owned Camera Broker service"
      PYTHONPATH="$LOOP_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$LOOP_ROOT/.venv/bin/python" -c '
from modules.module_0_prepare.ensure_camera_broker import ensure_camera_broker
ensure_camera_broker()
'
    else
      echo "-> requiring prestarted Camera Broker service"
      PYTHONPATH="$LOOP_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$LOOP_ROOT/.venv/bin/python" -c '
from modules.module_0_prepare.ensure_camera_broker import require_camera_broker
require_camera_broker()
'
    fi
  else
    echo "-> verifying Module 0 GraspNet and Camera Broker services"
    PYTHONPATH="$LOOP_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$LOOP_ROOT/.venv/bin/python" -c '
from modules.module_0_prepare.ensure_camera_broker import ensure_camera_broker
from modules.module_0_prepare.ensure_graspnet_service import ensure_graspnet_service
ensure_graspnet_service()
ensure_camera_broker()
'
  fi

  echo "-> Version7 services ready; starting real grasp flow"
  echo "-> production grasp script: $GRASPNET_GRASP_SCRIPT"
  echo "-> configured post-grasp flow: $MODULE2_SCRIPT"
  rm -f -- "$V7_HANDOFF_FILE"
  local grasp_args=("$@")
  # The modular loop removes its own CLI flag before delegating to this shell
  # script, then supplies the same policy through this environment variable.
  if [[ "${GRASPNET_AUTO_CONFIRM_REAL_GRASP:-}" == "1" ]]; then
    grasp_args+=(--auto-confirm-real-grasp)
  elif [[ "${GRASPNET_AUTO_CONFIRM_REAL_GRASP:-}" == "0" ]]; then
    grasp_args+=(--no-auto-confirm-real-grasp)
  fi
  report_module_metric \
    "module_1_1_graspnet" "WAIT" \
    "开始 GraspNet 预测、真实抓取并移动到 graspup"
  if bash "$GRASPNET_GRASP_SCRIPT" \
    --handoff-success-file "$V7_HANDOFF_FILE" "${grasp_args[@]}"; then
    :
  else
    local grasp_status=$?
    if (( grasp_status == 20 )); then
      report_workpiece_timer finish --module module_1_1_graspnet --scope "$MODULE1_TIMER_SCOPE" \
        --status RETRY --detail "未得到有效抓取候选"
      MODULE1_TIMER_ACTIVE=0
      report_module_metric \
        "module_1_1_graspnet" "RETRY" \
        "未得到有效抓取候选；保持当前安全位并回退到 Module 1 重试" \
        --reason-code "NO_GRASP_PREDICTED"
    else
      report_workpiece_timer finish --module module_1_1_graspnet --scope "$MODULE1_TIMER_SCOPE" \
        --status FAILURE --detail "GraspNet抓取子流程失败，exit=$grasp_status"
      MODULE1_TIMER_ACTIVE=0
      report_module_metric \
        "module_1_1_graspnet" "FAULT" \
        "GraspNet 抓取子流程失败，exit=$grasp_status；交由总控安全停机" \
        --reason-code "EXIT_$grasp_status"
    fi
    return "$grasp_status"
  fi
  if [[ ! -s "$V7_HANDOFF_FILE" ]]; then
    report_workpiece_timer finish --module module_1_1_graspnet --scope "$MODULE1_TIMER_SCOPE" \
      --status FAILURE --detail "缺少有效graspup handoff"
    MODULE1_TIMER_ACTIVE=0
    report_module_metric \
      "module_1_1_graspnet" "FAULT" \
      "抓取子流程退出但缺少有效 graspup handoff，拒绝进入 Module 2" \
      --reason-code "MISSING_GRASP_HANDOFF"
    echo "-> Version7 ended without a verified graspup handoff; Module 2 is prohibited." >&2
    return 1
  fi
  report_module_metric \
    "module_1_1_graspnet" "SUCCESS" \
    "GraspNet 已确认单工件抓取并到达 graspup"
  report_workpiece_timer finish --module module_1_1_graspnet --scope "$MODULE1_TIMER_SCOPE" \
    --status SUCCESS --detail "已确认单件并到达graspup"
  MODULE1_TIMER_ACTIVE=0
  echo "-> Version7 verified one workpiece and reached graspup; handoff is permitted."
  echo "-> Version7 returned. The object must remain clamped at graspup before Module 2 can start."
  if [[ "${LOOP_AUTO_CONFIRM_MOTION:-0}" == "1" ]]; then
    echo "-> 已验证 graspup handoff，自动确认进入 Module 2。"
  else
    read -r -p "确认工件仍被左夹爪稳定夹持且PC/PC2路径安全，按回车继续模块2："
  fi

  echo "-> GraspNet and Camera Broker remain resident; starting configured post-GraspNet flow: $MODULE2_SCRIPT"
  run_module_2_with_camera_retry
}

on_interrupt() {
  # Make Ctrl-C deterministic: exit through the EXIT trap so only the
  # Version7 services created by this invocation are stopped.
  echo "-> interrupted; cleaning up Version7 services started by this script" >&2
  trap - INT TERM
  exit 130
}

trap on_interrupt INT TERM
trap cleanup_v7_on_error EXIT
main "$@"
