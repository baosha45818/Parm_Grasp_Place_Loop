#!/usr/bin/env bash
# Module 1-1 production entry: run Version7 grasp, then the Module 2 entry
# selected by the modular controller.  Workpiece 2 alone enables V7's native
# inference-cloud thickening; all robot/camera/planner behaviour remains V7.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOOP_ROOT="$(cd "$MODULE_DIR/../../.." && pwd)"
PRODUCTION_V7_FLOW="$LOOP_ROOT/third_party/graspnet/deploy/version7/run_grasp_to_module2.sh"
case "${PARM_LOOP_WORKPIECE:-}" in
  2)
    # Never inherit a parent-shell experimental runner into production.  The
    # vendored V7 process creates a separate D405 inference cloud itself.
    unset GRASPNET_FLOW_RUNNER GRASPNET_V11_V7_DIR GRASPNET_GRASP_SCRIPT
    export GRASPNET_WORKPIECE_ID=2
    export GRASPNET_WORKPIECE2_CLOUD_THICKENING=1
    echo "-> Module 1-1 workpiece 2: native Version7 D405 thickening enabled (only plane-to-table angle >=65deg)"
    ;;
  1)
    # A parent-shell experimental override must not accidentally change the
    # production workpiece-1 pipeline.
    unset GRASPNET_FLOW_RUNNER GRASPNET_V11_V7_DIR GRASPNET_GRASP_SCRIPT
    unset GRASPNET_WORKPIECE_ID GRASPNET_WORKPIECE2_CLOUD_THICKENING
    echo "-> Module 1-1 workpiece 1: production Version7; no point-cloud thickening"
    ;;
  *)
    echo "PARM_LOOP_WORKPIECE must be explicitly set to 1 or 2 before Module 1-1 starts" >&2
    exit 2
    ;;
esac

echo "-> Module 1-1 production entry: Version7 (third_party/graspnet)"
echo "-> Module 1-1 resolved flow: $PRODUCTION_V7_FLOW"
exec bash \
  "$PRODUCTION_V7_FLOW" \
  "$@"
