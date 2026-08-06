#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dainanjun/Public/Parm_Grasp_Place_Loop"
PYTHON="$ROOT/.venv/bin/python"
PACKAGE="tests.reversed_head_observation_2"

usage() {
  printf '%s\n' \
    "用法: $0 <命令> [参数...]" \
    "" \
    "  test                         离线检查模型坐标、碰撞包围盒和代码" \
    "  check-observation            只规划左手到头部观察位" \
    "  move-observation             执行左手到头部观察位（原流程逐段确认）" \
    "  estimate [参数]              D435i+交互式SAM2+ICP；交接须加 --output-frame body_base" \
    "  reprocess --result FILE      用工件2坐标系重新处理已保存采集" \
    "  prepare-config --icp-result FILE --center-m X Y Z" \
    "                               生成附着碰撞 handoff_config.ready.json" \
    "  left-plan --result FILE      只规划左手送到胸前交接区" \
    "  left-execute --result FILE   执行左手送到胸前交接区" \
    "  right-plan --result FILE     只规划右手抓工件+X端" \
    "  right-execute --result FILE  执行右手交接；还需显式几何确认参数" \
    "  place-execute                从右手放置准备位继续Module 9/10放置" \
    "  full-execute [参数]           左手反抓→观察→交接→右手放置" \
    "                               已持件续跑需加 --already-left-held"
}

if [[ ! -x "$PYTHON" ]]; then
  printf '运行环境不存在: %s\n' "$PYTHON" >&2
  exit 2
fi

command_name="${1:-}"
if [[ -z "$command_name" ]]; then
  usage
  exit 2
fi
shift
cd "$ROOT"

case "$command_name" in
  test)
    exec "$PYTHON" -m unittest tests.reversed_head_observation_2.test_workpiece_2 -v "$@"
    ;;
  check-observation)
    exec "$PYTHON" -m "$PACKAGE.check_plan" "$@"
    ;;
  move-observation)
    exec "$PYTHON" -m "$PACKAGE.execute_plan" "$@"
    ;;
  estimate)
    exec "$PYTHON" -m "$PACKAGE.estimate_pose_head_d435" "$@"
    ;;
  reprocess)
    exec "$PYTHON" -m "$PACKAGE.reprocess_head_d435_result" "$@"
    ;;
  prepare-config)
    exec "$PYTHON" -m "$PACKAGE.configure_handoff" set-target "$@"
    ;;
  left-plan)
    exec "$PYTHON" -m "$PACKAGE.move_object_center" "$@"
    ;;
  left-execute)
    exec "$PYTHON" -m "$PACKAGE.move_object_center" --execute "$@"
    ;;
  right-plan)
    exec "$PYTHON" -m "$PACKAGE.execute_handoff" "$@"
    ;;
  right-execute)
    exec "$PYTHON" -m "$PACKAGE.execute_handoff" --execute "$@"
    ;;
  place-execute)
    exec "$PYTHON" -m "$PACKAGE.execute_first_workpiece_place" \
      --execute --confirm-reuse-first-workpiece-placement "$@"
    ;;
  full-execute)
    exec "$PYTHON" -m "$PACKAGE.execute_full_process" --execute "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf '未知命令: %s\n' "$command_name" >&2
    usage >&2
    exit 2
    ;;
esac
