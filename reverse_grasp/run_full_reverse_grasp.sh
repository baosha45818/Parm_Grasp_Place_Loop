#!/usr/bin/env bash
# Fresh head RGB-D -> SAM -> fixed-reverse STL/ICP -> convex-tab point -> preflight -> optional grasp.
# All generated artifacts stay under reverse_grasp.  Robot/gripper motion requires --execute.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
run_name=""
execute=0

usage() {
  cat <<'EOF'
Usage:
  bash reverse_grasp/run_full_reverse_grasp.sh [--run-name NAME] [--execute]

The convex tab is selected automatically from the SAM workpiece point cloud's
ICP-frame +X extreme after each fresh RGB-D/SAM/ICP run; no pixel input is required.
By default it stops after Planner preflight.  --execute performs the actual
left-arm grasp and lift only after a successful preflight.

EOF
}

while (($#)); do
  case "$1" in
    --run-name) run_name="${2:?--run-name requires NAME}"; shift 2 ;;
    --execute) execute=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$run_name" ]]; then
  run_name="full_reverse_$(date +%Y%m%d_%H%M%S)"
fi
run_dir="reverse_grasp/runs/$run_name"
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists: $run_dir" >&2
  exit 2
fi

"$python_bin" reverse_grasp/capture_sam_fixed_icp.py --run-dir "$run_dir"

pose_json="$run_dir/capture/table_reverse_fixed_icp/fixed_reverse_pose.json"
capture_dir="$run_dir/capture"
overlay="$run_dir/capture/table_reverse_fixed_icp/fixed_reverse_axes_overlay.png"
echo "Fresh pose overlay: $overlay"

point_dir="$run_dir/capture/convex_tab_grasp"
"$python_bin" reverse_grasp/predict_convex_tab_grasp.py \
  --pose "$pose_json" --capture-dir "$capture_dir" --output-dir "$point_dir" \
  --grasp-offset-config reverse_grasp/grasp/manual_grasp_offset.json

prediction_json="$point_dir/convex_tab_grasp_prediction.json"
visual_dir="reverse_grasp/grasp/visualizations/$run_name"
"$python_bin" reverse_grasp/grasp/visualize_grasp_prediction.py \
  --prediction-json "$prediction_json" --output-dir "$visual_dir"
echo "Grasp prediction overlay: $visual_dir/grasp_prediction_overlay.png"
preflight_dir="reverse_grasp/grasp/runs/${run_name}_preflight"
"$python_bin" reverse_grasp/grasp/execute_convex_tab_grasp.py \
  --prediction-json "$prediction_json" --output-dir "$preflight_dir" --preflight-only

echo "Preflight succeeded: $preflight_dir/grasp_plan.json"
if (( ! execute )); then
  echo "No arm or gripper command was sent. Re-run with --execute to grasp."
  exit 0
fi

execute_dir="reverse_grasp/grasp/runs/${run_name}_execute"
"$python_bin" reverse_grasp/grasp/execute_convex_tab_grasp.py \
  --prediction-json "$prediction_json" --output-dir "$execute_dir" --execute
echo "Grasp execution finished: $execute_dir/grasp_plan.json"
