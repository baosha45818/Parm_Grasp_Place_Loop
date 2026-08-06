#!/usr/bin/env bash
# 新工件 2 反面自动抓取：Module 4-1 视觉预测 + Module 5-1 Planner 执行。
set -euo pipefail

pipeline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
loop_root="$(cd "$pipeline_root/../../../.." && pwd)"
module4="$loop_root/modules/module_4_head_judge/module_4_1_reverse_grasp_prediction/workpiece_2_pipeline"
python_bin="${PYTHON_BIN:-$loop_root/.venv/bin/python}"
run_name=""
execute=0

usage() {
  cat <<'EOF'
Usage:
  bash modules/module_5_left_pick/module_5_1_reverse_grasp/workpiece_2_pipeline/run_full_reverse_grasp.sh [--run-name NAME] [--execute]

Without --execute, only performs fresh head RGB-D, SAM2, fixed-reverse ICP,
convex-tab prediction, and Planner preflight.  --execute additionally moves
the left arm, commands the gripper, and lifts after a successful preflight.
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
  run_name="workpiece2_reverse_$(date +%Y%m%d_%H%M%S)"
fi
run_dir="$module4/runs/$run_name"
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists: $run_dir" >&2
  exit 2
fi

"$python_bin" "$module4/capture_sam_fixed_icp.py" --run-dir "$run_dir"
point_dir="$run_dir/03_grasp"
prediction_json="$point_dir/convex_tab_grasp_prediction.json"
if [[ ! -f "$prediction_json" ]]; then
  echo "视觉阶段未生成凸耳抓取预测: $prediction_json" >&2
  exit 1
fi
# capture_sam_fixed_icp.py 已完成 SAM2、固定反面 ICP 和凸耳抓取预测；
# 此处只复用结果，不能再次写入同一个 03_grasp 目录。
visual_dir="$pipeline_root/grasp/visualizations/$run_name"
"$python_bin" "$pipeline_root/grasp/visualize_grasp_prediction.py" \
  --prediction-json "$prediction_json" --output-dir "$visual_dir"

preflight_dir="$pipeline_root/grasp/runs/${run_name}_preflight"
"$python_bin" "$pipeline_root/grasp/execute_convex_tab_grasp.py" \
  --prediction-json "$prediction_json" --output-dir "$preflight_dir" --preflight-only

echo "Preflight succeeded: $preflight_dir/grasp_plan.json"
if (( ! execute )); then
  echo "No arm or gripper command was sent. Re-run with --execute to grasp."
  exit 0
fi

execute_dir="$pipeline_root/grasp/runs/${run_name}_execute"
"$python_bin" "$pipeline_root/grasp/execute_convex_tab_grasp.py" \
  --prediction-json "$prediction_json" --output-dir "$execute_dir" --execute
echo "Grasp execution finished: $execute_dir/grasp_plan.json"
