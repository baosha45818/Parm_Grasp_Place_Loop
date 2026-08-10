#!/usr/bin/env bash
# Legacy compatibility entry.
#
# The production traditional ICP entry moved to module_1_3_icp_grasp so that
# the name of the active path matches the actual grasp algorithm.  Keep this
# small forwarding script for any old command that still references it.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="$MODULE_DIR/../module_1_3_icp_grasp/run_grasp_to_inspect_place.sh"

if [[ ! -x "$TARGET" ]]; then
  echo "traditional ICP production entry is unavailable: $TARGET" >&2
  exit 1
fi

echo "-> module_1_1_graspnet is a legacy alias; forwarding to module_1_3_icp_grasp"
exec "$TARGET" "$@"
