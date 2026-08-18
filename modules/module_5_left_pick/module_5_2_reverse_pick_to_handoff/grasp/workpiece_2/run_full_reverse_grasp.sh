#!/usr/bin/env bash
# 新工件 2 反面自动抓取：Module 4 方向/物体位姿 + Module 5-1 抓取预测与执行。
set -euo pipefail

pipeline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
loop_root="$(cd "$pipeline_root/../../../../.." && pwd)"
python_bin="${PYTHON_BIN:-$loop_root/.venv/bin/python}"
run_name=""
module4_result=""
execute=0
pregrasp_m="${PARM_REVERSE_PREGRASP_M:-0.06}"
grasp_down_offset_m="${PARM_REVERSE_GRASP_DOWN_OFFSET_M:--0.005}"
approach_angles="${PARM_REVERSE_APPROACH_ANGLES_DEG:-90 15 -15}"
planning_inflight="${PARM_REVERSE_PLANNING_INFLIGHT:-3}"
close_force="${PARM_REVERSE_CLOSE_FORCE:-100}"

usage() {
  cat <<'EOF'
Usage（反面分支工件2）：
  bash modules/module_5_left_pick/reverse_pick_to_handoff/grasp/workpiece_2/run_full_reverse_grasp.sh --module4-result FILE [--run-name NAME] [--execute]

FILE 必须是 Module 4-1 发布的唯一 BACK 工件2正反+位姿报告。本脚本不再采集或
运行第二次 ICP；它只生成凸耳抓取 TCP，并执行 Planner 预检/抓取。
EOF
}

while (($#)); do
  case "$1" in
    --run-name) run_name="${2:?--run-name requires NAME}"; shift 2 ;;
    --module4-result) module4_result="${2:?--module4-result requires FILE}"; shift 2 ;;
    --execute) execute=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$module4_result" || ! -f "$module4_result" ]]; then
  echo "--module4-result 必须指向有效的 Module 4-1 报告: $module4_result" >&2
  exit 2
fi

if [[ -z "$run_name" ]]; then
  run_name="workpiece2_reverse_$(date +%Y%m%d_%H%M%S)"
fi
point_dir="$pipeline_root/grasp/runs/${run_name}_prediction"
"$python_bin" "$pipeline_root/grasp/predict_convex_tab_grasp.py" \
  --module4-result "$module4_result" \
  --grasp-down-offset-m "$grasp_down_offset_m" \
  --output-dir "$point_dir"
prediction_json="$point_dir/convex_tab_grasp_prediction.json"
if [[ ! -f "$prediction_json" ]]; then
  echo "视觉阶段未生成凸耳抓取预测: $prediction_json" >&2
  exit 1
fi
# Module 4 已完成 SAM2 与固定反面 ICP；凸耳抓取预测由 Module 5-1 生成。
visual_dir="$pipeline_root/grasp/visualizations/$run_name"
"$python_bin" "$pipeline_root/grasp/visualize_grasp_prediction.py" \
  --prediction-json "$prediction_json" --output-dir "$visual_dir"

preflight_dir="$pipeline_root/grasp/runs/${run_name}_preflight"
# Profile values are whitespace-separated numeric arguments produced by the
# validated Python loader. Deliberate word splitting turns them into argv.
read -r -a approach_args <<< "$approach_angles"
"$python_bin" "$pipeline_root/grasp/execute_convex_tab_grasp.py" \
  --prediction-json "$prediction_json" --output-dir "$preflight_dir" --preflight-only \
  --pregrasp-m "$pregrasp_m" --approach-angles-deg "${approach_args[@]}" \
  --planning-inflight "$planning_inflight" --close-force "$close_force"

echo "Preflight succeeded: $preflight_dir/grasp_plan.json"
if (( ! execute )); then
  echo "No arm or gripper command was sent. Re-run with --execute to grasp."
  exit 0
fi

execute_dir="$pipeline_root/grasp/runs/${run_name}_execute"
"$python_bin" "$pipeline_root/grasp/execute_convex_tab_grasp.py" \
  --prediction-json "$prediction_json" --output-dir "$execute_dir" --execute \
  --pregrasp-m "$pregrasp_m" --approach-angles-deg "${approach_args[@]}" \
  --planning-inflight "$planning_inflight" --close-force "$close_force"
echo "Grasp execution finished: $execute_dir/grasp_plan.json"
