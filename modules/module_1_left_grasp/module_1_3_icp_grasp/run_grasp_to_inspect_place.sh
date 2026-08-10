#!/usr/bin/env bash
# Module 1 production entry: traditional SAM/STL/ICP pick, then Module 2.
#
# This deliberately does not start or invoke GraspNet.  The legacy picker
# performs the real pick and returns with the object clamped at graspup.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOOP_ROOT="${PARM_LOOP_ROOT:-$(cd "$MODULE_DIR/../../.." && pwd)}"
ICP_PICK_DIR="$MODULE_DIR"
ICP_PICK="$ICP_PICK_DIR/run_live_traditional_pick.sh"
MODULE2_SCRIPT="${MODULE2_SCRIPT:-$LOOP_ROOT/modules/module_2_left_inspect_apart/run_inspect_then_place.py}"

if [[ ! -x "$ICP_PICK" ]]; then
  echo "traditional Module 1 entry is unavailable or not executable: $ICP_PICK" >&2
  exit 1
fi
if [[ ! -f "$MODULE2_SCRIPT" ]]; then
  echo "Module 2 standalone entry is unavailable: $MODULE2_SCRIPT" >&2
  exit 1
fi
if [[ "${PARM_LOOP_WORKPIECE:-}" != "1" && "${PARM_LOOP_WORKPIECE:-}" != "2" ]]; then
  echo "PARM_LOOP_WORKPIECE must be explicitly set to 1 or 2 before Module 1 starts" >&2
  exit 2
fi
if [[ -z "${LOOP_GRASPNET_EXECUTION_PERMIT_FILE:-}" ]]; then
  echo "the loop must provide LOOP_GRASPNET_EXECUTION_PERMIT_FILE for the ICP weight handoff" >&2
  exit 2
fi
WEIGHT_HANDOFF_FILE="$(dirname "$LOOP_GRASPNET_EXECUTION_PERMIT_FILE")/icp_weight_handoff.json"

placement_gate_args=()
if [[ -n "${MODULE2_PLACEMENT_READY_FILE:-}" || -n "${MODULE2_PLACEMENT_PERMIT_FILE:-}" ]]; then
  if [[ -z "${MODULE2_PLACEMENT_READY_FILE:-}" || -z "${MODULE2_PLACEMENT_PERMIT_FILE:-}" ]]; then
    echo "MODULE2_PLACEMENT_READY_FILE and MODULE2_PLACEMENT_PERMIT_FILE must be set together" >&2
    exit 2
  fi
  placement_gate_args=(
    --placement-ready-file "$MODULE2_PLACEMENT_READY_FILE"
    --placement-permit-file "$MODULE2_PLACEMENT_PERMIT_FILE"
  )
fi
if [[ "${LOOP_AUTO_CONFIRM_MOTION:-0}" == "1" ]]; then
  placement_gate_args+=(--auto-confirm-motion)
fi

echo "-> Module 1: traditional pick; GraspNet is not invoked"
echo "-> running: cd $ICP_PICK_DIR && ./run_live_traditional_pick.sh --execute --confirm-real-grasp"
(
  cd "$ICP_PICK_DIR"
  ./run_live_traditional_pick.sh \
    --arm-host "${PARM_ARM_HOST:-${GRASPNET_V7_ARM_HOST:-192.168.88.1}}" \
    --weight-handoff-file "$WEIGHT_HANDOFF_FILE" \
    --execute --confirm-real-grasp
)

# The traditional picker has reached graspup with the gripper closed and its
# quality gate has produced a verified single-workpiece handoff.  Refuse to
# enter Module 2 if it is absent or empty.
if [[ ! -s "$WEIGHT_HANDOFF_FILE" ]]; then
  echo "ICP grasp ended without a verified weight handoff: $WEIGHT_HANDOFF_FILE" >&2
  exit 1
fi
# Module 1 and Module 2 share the persistent Broker.  It remains the sole
# owner of D405 and Femto throughout the complete run.
echo "-> traditional D405 capture completed through Camera Broker; starting Module 2"
echo "-> traditional pick completed at graspup; starting Module 2: $MODULE2_SCRIPT"
"$LOOP_ROOT/.venv/bin/python" "$MODULE2_SCRIPT" \
  --planner-host "${PARM_PLANNER_HOST:-127.0.0.1}" \
  --arm-host "${PARM_ARM_HOST:-${GRASPNET_V7_ARM_HOST:-192.168.88.1}}" \
  --gripper-host "${PARM_GRIPPER_HOST:-127.0.0.1}" \
  --workpiece "$PARM_LOOP_WORKPIECE" \
  --weight-handoff-file "$WEIGHT_HANDOFF_FILE" \
  "${placement_gate_args[@]}"
