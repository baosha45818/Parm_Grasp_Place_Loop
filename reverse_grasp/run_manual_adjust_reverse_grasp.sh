#!/usr/bin/env bash
# Fresh RGB-D -> SAM -> fixed-reverse STL/ICP -> point-cloud tab -> manual 5 mm jog grasp.
# This is separate from run_full_reverse_grasp.sh and does not change its automatic behavior.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
execute=0
run_name=""

usage() {
  cat <<'EOF'
Usage: bash reverse_grasp/run_manual_adjust_reverse_grasp.sh [--run-name NAME] --execute

Performs a fresh head RGB-D/SAM/ICP run. With --execute, it moves the open
left gripper to the grasp point, accepts terminal jog keys in the ICP object
frame (5 mm each), then closes only after you press c.
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
if (( ! execute )); then
  echo "Manual jog sends robot motion; explicitly pass --execute." >&2
  exit 2
fi
if [[ -z "$run_name" ]]; then run_name="manual_reverse_$(date +%Y%m%d_%H%M%S)"; fi
run_dir="reverse_grasp/runs/$run_name"
if [[ -e "$run_dir" ]]; then echo "Run directory already exists: $run_dir" >&2; exit 2; fi

"$python_bin" reverse_grasp/capture_sam_fixed_icp.py --run-dir "$run_dir"
capture_dir="$run_dir/capture"
candidate_dir="$capture_dir/manual_grasp_candidates"
"$python_bin" reverse_grasp/prepare_manual_grasp_candidates.py --capture-dir "$capture_dir" --output-dir "$candidate_dir"
selection_json="$candidate_dir/manual_selected_candidate.json"
prediction_json="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["prediction"])' "$selection_json")"
overlay="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["overlay"])' "$selection_json")"
echo "Selected grasp prediction overlay: $overlay"
"$python_bin" reverse_grasp/show_grasp_prediction.py --image "$overlay" --title "selected reverse grasp prediction"
"$python_bin" reverse_grasp/grasp/manual_adjust_convex_tab_grasp.py \
  --prediction-json "$prediction_json" \
  --output-dir "reverse_grasp/grasp/runs/${run_name}_manual" \
  --execute
