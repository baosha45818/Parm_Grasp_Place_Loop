#!/usr/bin/env bash
# Fixed arm1 Version7 grasp invocation.  The matching camera service must be
# started with GRASPNET_V7_D405_POSTPROCESS=1 and GRASPNET_V7_D405_HOLE_FILLING=1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# The canonical D405 eye-in-hand calibration is maintained with the Loop
# configuration rather than under the vendored GraspNet source tree.
LOOP_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
# Module 1-1 的所有重量/力稳定性阈值只从此共享配置读取。
# Module 2/3 会继承本次 handoff 中记录的实际值，而不各自维护一份阈值。
source "$LOOP_ROOT/config/grasp_weight_thresholds.conf"
cd "$ROOT_DIR"
# GraspNet needs a slightly faster approach than the rest of the Loop, but
# this must not change the shared left-arm speed used by Module 2/3.  This is
# passed directly to both RRTStar and LIN planning below (valid range (0, 1]).
GRASPNET_PLANNER_SPEED="${GRASPNET_PLANNER_SPEED:-0.3}"
if ! awk "BEGIN { exit !($GRASPNET_PLANNER_SPEED > 0 && $GRASPNET_PLANNER_SPEED <= 1) }"; then
  echo "Invalid GRASPNET_PLANNER_SPEED=$GRASPNET_PLANNER_SPEED; use a value in (0, 1]." >&2
  exit 2
fi
VISUALIZATION_ARGS=()
case "${GRASPNET_VISUALIZATION:-1}" in
  0|false|FALSE|no|NO)
    # The modular controller's --no-vis must reach the Python argparse layer;
    # otherwise its default is still vis=True.
    VISUALIZATION_ARGS+=(--no-vis)
    ;;
  1|true|TRUE|yes|YES|"")
    ;;
  *)
    echo "Invalid GRASPNET_VISUALIZATION=${GRASPNET_VISUALIZATION}; use 0 or 1." >&2
    exit 2
    ;;
esac

# Module 1-1 selects this production-native inference option for workpiece 2.
# Keep it explicit rather than importing the experimental Version11 wrapper:
# the V7 runner owns the raw cloud, derived cloud, and service request in one
# process, while all physical execution remains unchanged.
WORKPIECE_ARGS=()
case "${GRASPNET_WORKPIECE_ID:-}" in
  "")
    ;;
  1|2)
    WORKPIECE_ARGS+=(--workpiece-id "$GRASPNET_WORKPIECE_ID")
    # Dense workpiece-2 stacks frequently leave only a millimetre of visible
    # support near a usable contact edge.  Keep workpiece 1 at the global
    # 4 mm safety default; only workpiece 2 receives this ranking/filtering
    # relaxation.  Collision and closing-volume support checks remain active.
    if [[ "${GRASPNET_WORKPIECE_ID}" == "2" ]]; then
      WORKPIECE_ARGS+=(--min-grasp-edge-clearance-m 0.001)
    fi
    ;;
  *)
    echo "Invalid GRASPNET_WORKPIECE_ID=${GRASPNET_WORKPIECE_ID}; use 1 or 2." >&2
    exit 2
    ;;
esac
case "${GRASPNET_WORKPIECE2_CLOUD_THICKENING:-0}" in
  0|false|FALSE|no|NO|"")
    ;;
  1|true|TRUE|yes|YES)
    if [[ "${GRASPNET_WORKPIECE_ID:-}" != "2" ]]; then
      echo "GRASPNET_WORKPIECE2_CLOUD_THICKENING requires GRASPNET_WORKPIECE_ID=2." >&2
      exit 2
    fi
    WORKPIECE_ARGS+=(--enable-workpiece2-cloud-thickening)
    ;;
  *)
    echo "Invalid GRASPNET_WORKPIECE2_CLOUD_THICKENING=${GRASPNET_WORKPIECE2_CLOUD_THICKENING}; use 0 or 1." >&2
    exit 2
    ;;
esac

# Use the Loop's ordinary Planner as one coherent service group:
# trajectory/TF/waypoint = 5604/5609/5610 (ROS Domain 59).  The bridge below
# was captured against this live Planner on 2026-08-03 and independently
# verified at <=0.042 mm / <=0.007 deg.  Do not mix it with the GraspNet-only
# 5605/5614/5615 calibrated-overlay bridge.
# Dense multi-workpiece stacks keep finger/bottom collision filtering but
# disable its swept local approach volume.
# Temporary integration mode: do not capture, infer, or select Femto grasps.
# Remove this flag after Femto is recalibrated and physically revalidated.
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$ROOT_DIR/.venv/bin/python" "${GRASPNET_FLOW_RUNNER:-deploy/version7/run_persistent_graspnet.py}" \
  --graspnet-socket "${GRASPNET_V7_SOCKET:-/tmp/graspnet-v7.sock}" \
  --camera-socket "${CAMERA_BROKER_SOCKET:-${GRASPNET_V7_CAMERA_SOCKET:-/tmp/graspnet-v7-camera.sock}}" \
  --graspnet-service-jobs "${GRASPNET_V7_JOBS_ROOT:-$ROOT_DIR/deploy/version7/jobs}" \
  --graspnet-timeout-s 120 \
  --camera-timeout-s 30 \
  --camera-max-start-skew-s 0.2 \
  --femto-request-flush-frames "${CAMERA_BROKER_FEMTO_REQUEST_FLUSH_FRAMES:-30}" \
  --femto-frame-read-retries "${CAMERA_BROKER_FEMTO_FRAME_READ_RETRIES:-10}" \
  --max-no-grasp-retries 99 \
  --planner-candidate-top-k 3 \
  --max-planner-unreachable-retries 30 \
  --empty-failure-ban-count 2 \
  --empty-failure-position-radius-m 0.020 \
  --empty-failure-orientation-deg 15 \
  --num-point 80000 \
  --num-view 300 \
  --wrist-calibration "$LOOP_ROOT/tool/calibration/left_wrist_d405/calibration_wrist1.json" \
  --laser-calibration "$LOOP_ROOT/tool/calibration/graspnet_laser/calibration_laser.json" \
  --planner-bridge-calibration "$LOOP_ROOT/tool/calibration/graspnet_planner_bridge_5604_20260803/planner_tf_bridge_5604_candidate.json" \
  --planner-host 127.0.0.1 \
  --planner-trajectory-port 5604 \
  --planner-tf-port 5609 \
  --planner-waypoint-port 5610 \
  --planner-connect-timeout-s 30 \
  --planner-connect-retry-s 0.5 \
  --planner-execute-url http://127.0.0.1:5667/middle/waypoint/execute \
  --planner-base-frame body_base \
  --planner-tool-frame left_eef \
  --planner-direct-grasp-center \
  --planner-group left_arm \
  --planner-max-vel "$GRASPNET_PLANNER_SPEED" \
  --planner-max-acc "$GRASPNET_PLANNER_SPEED" \
  --planner-pregrasp-mode rrtstar \
  --planner-descend-mode lin \
  --arm-host 192.168.88.1 \
  --arm-id 1 \
  --move-observation \
  --d405-serial 260322270242 \
  --d405-frame-count 5 \
  --d405-postprocess \
  --d405-hole-filling \
  --disable-femto \
  --d405-z-min 0.10 --d405-z-max 1.20 \
  --d405-x-abs-max 0.80 --d405-y-abs-max 0.80 \
  --femto-serial CL8L56301LK \
  --femto-z-min 0.20 --femto-z-max 2.00 \
  --femto-x-abs-max 1.20 --femto-y-abs-max 1.20 \
  --roi 0 0 512 372 \
  --femto-roi 175 281 488 228 \
  --collision-thresh 0.5 \
  --collision-voxel-size 0.005 \
  --collision-approach-dist 0.01 \
  --min-grasp-approach-angle-deg 60 \
  --min-grasp-edge-clearance-m 0.004 \
  --pregrasp-height-m 0.10 \
  --grasp-downward-probe-m -0.02 \
  --gripper-host "${PARM_GRIPPER_HOST:-127.0.0.1}" \
  --gripper-hand 1 \
  --gripper-open-width-m 0.075 \
  --gripper-width-margin-m 0.005 \
  --gripper-force 100 \
  --gripper-close-force 100 \
  --grasp-weight-check \
  --weight-empty-max-g "$WEIGHT_EMPTY_MAX_G" \
  --weight-single-min-g "$WEIGHT_SINGLE_MIN_G" \
  --weight-single-max-g "$WEIGHT_SINGLE_MAX_G" \
  --weight-multiple-min-g "$WEIGHT_MULTIPLE_MIN_G" \
  --weight-max-axis-mad-n "$WEIGHT_MAX_AXIS_MAD_N" \
  --weight-check-samples "$WEIGHT_CHECK_SAMPLES" \
  --weight-check-timeout-s "$WEIGHT_CHECK_TIMEOUT_S" \
  --weight-check-settle-s "$WEIGHT_CHECK_SETTLE_S" \
  --weight-contiguous-single-band \
  --max-weight-retries 20 \
  --max-operator-declines 20 \
  --force-sensor-host 127.0.0.1 \
  --force-sensor-hand 1 \
  --postgrasp-lift-joints-deg -4.081 -52.722 3.637 100.478 -10.051 53.969 8.843 \
  --place-joints-deg 49.671 -56.037 -115.398 72.201 9.075 53.369 11.563 \
  --execute-grasp \
  --confirm-real-grasp \
  "${VISUALIZATION_ARGS[@]}" \
  "${WORKPIECE_ARGS[@]}" \
  "$@"
