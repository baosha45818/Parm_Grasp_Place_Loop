#!/usr/bin/env bash
# Experimental Version9: Version7 persistent flow with the URDF nominal TCP.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOOP_ROOT="$(cd "$MODULE_DIR/../../../.." && pwd)"
V9_GRASPNET_SOURCE_ROOT="${V9_GRASPNET_SOURCE_ROOT:-$LOOP_ROOT/third_party/graspnet}"
test -x "$V9_GRASPNET_SOURCE_ROOT/.venv/bin/python"
cd "$V9_GRASPNET_SOURCE_ROOT"
echo "-> EXPERIMENTAL Module 1-1 Version9; modular Loop does not call this entry"
echo "-> Version9 GraspNet source: $V9_GRASPNET_SOURCE_ROOT"

# Version9 defaults to plan-only.  The full Version7 physical workflow needs
# this shell-level opt-in; terminal confirmation remains enabled downstream.
PLAN_ONLY_ARGS=(--planner-plan-only)
FULL_V7_EXECUTION_ARGS=()
if [ "${V9_ALLOW_EXECUTE:-0}" = "1" ]; then
  PLAN_ONLY_ARGS=()
  FULL_V7_EXECUTION_ARGS=(
    --execute-grasp --confirm-real-grasp --allow-urdf-tcp-execution
    --max-no-grasp-retries 99
    --planner-candidate-top-k 3 --max-planner-unreachable-retries 3
    --grasp-weight-check
    --weight-empty-max-g 30
    --weight-single-min-g 30.000000001
    --weight-single-max-g 129.999999999
    --weight-multiple-min-g 130
    --weight-contiguous-single-band
    --max-weight-retries 20
    --max-operator-declines 20
    --force-sensor-host 127.0.0.1 --force-sensor-hand 1
    --postgrasp-lift-joints-deg -4.081 -52.722 3.637 100.478 -10.051 53.969 8.843
    --place-joints-deg 49.671 -56.037 -115.398 72.201 9.075 53.369 11.563
    --stop-after-graspup
  )
fi

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" V9_GRASPNET_SOURCE_ROOT="$V9_GRASPNET_SOURCE_ROOT" "$V9_GRASPNET_SOURCE_ROOT/.venv/bin/python" "$MODULE_DIR/run_urdf_tcp_graspnet.py" \
  --graspnet-socket /tmp/graspnet-v7.sock \
  --camera-socket /tmp/graspnet-v7-camera.sock \
  --graspnet-service-jobs "$V9_GRASPNET_SOURCE_ROOT/deploy/version7/jobs" \
  --num-point 50000 --num-view 300 \
  --wrist-calibration "$V9_GRASPNET_SOURCE_ROOT/deploy/calibration/output/wrist1/calibration_wrist1.json" \
  --laser-calibration "$V9_GRASPNET_SOURCE_ROOT/deploy/calibration/output/laser/calibration_laser.json" \
  --tcp-translation-calibration "$V9_GRASPNET_SOURCE_ROOT/deploy/calibration/output/tcp_grasp_tcp/calibration_translation2.json" \
  --output-dir "$MODULE_DIR/output" \
  --planner-bridge-calibration "$V9_GRASPNET_SOURCE_ROOT/deploy/calibration/output/planner_tf_bridge.json" \
  --planner-host 127.0.0.1 --planner-trajectory-port 5605 --planner-tf-port 5614 --planner-waypoint-port 5615 \
  --planner-execute-url http://127.0.0.1:5667/middle/waypoint/execute \
  --planner-base-frame body_base --planner-tool-frame left_hand_transfer_end --planner-group left_arm \
  --planner-max-vel 0.05 --planner-max-acc 0.05 --planner-pregrasp-mode rrtstar --planner-descend-mode lin \
  --arm-host 192.168.88.1 --arm-id 1 --move-observation \
  --d405-serial 260322270242 --d405-frame-count 5 --d405-postprocess --d405-hole-filling \
  --d405-z-min 0.10 --d405-z-max 1.20 --d405-x-abs-max 0.80 --d405-y-abs-max 0.80 \
  --femto-serial CL8L56301LK --femto-z-min 0.20 --femto-z-max 2.00 --femto-x-abs-max 1.20 --femto-y-abs-max 1.20 \
  --roi 0 0 512 336 --femto-roi 175 281 488 228 \
  --collision-thresh 0.5 --collision-voxel-size 0.005 --collision-approach-dist 0.01 \
  --min-grasp-approach-angle-deg 70 --min-grasp-edge-clearance-m 0.004 \
  --pregrasp-height-m 0.10 --grasp-downward-probe-m 0.01 \
  --gripper-host localhost --gripper-hand 1 --gripper-open-width-m 0.078 --gripper-width-margin-m 0.005 \
  --gripper-force 60 --gripper-close-force 100 \
  "${PLAN_ONLY_ARGS[@]}" \
  "${FULL_V7_EXECUTION_ARGS[@]}" \
  "$@"
