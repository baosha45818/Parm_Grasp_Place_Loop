"""Argument parsing, control snapshots, and workpiece-profile helpers."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import time
import os
from pathlib import Path
import time
from toml_config import load as load_toml
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PLANNER_CONFIG = ROOT / "config.hole_a_first_live_fast_return.json"
INITIAL_RIGHT_JOINTS_DEG = [
    22.43,
    23.33,
    0.52,
    66.99,
    -62.20,
    -64.44,
    25.51
]


def make_event_logger(
    output_file: object, *, started_at: float | None = None,
) -> object:
    """Return a flushing JSONL event logger for an already-open log file.

    When ``started_at`` is supplied, each record also includes the elapsed
    monotonic time since that point.
    """
    def log_event(event: str, **fields: object) -> None:
        record: dict[str, object] = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        }
        if started_at is not None:
            record["elapsed_s"] = round(time.monotonic() - started_at, 3)
        record["event"] = event
        record.update(fields)
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[attr-defined]
        output_file.flush()  # type: ignore[attr-defined]

    return log_event



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="右臂回固定关节起点后，沿配置的世界坐标轴逐毫米搜索，并以力特征确认孔已进入。"
    )
    parser.add_argument(
        "--initial-motion", choices=("planner-movej", "joint-servo"), default="planner-movej",
        help="回固定关节起点的方式；默认规划器避碰 move_j，joint-servo 仅用于已验证的近距离动作",
    )
    parser.add_argument(
        "--initial-right-joints-deg", nargs=7, type=float,
        default=INITIAL_RIGHT_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="示教得到的右臂起始关节角（度，7 个值）；默认使用本文件内置的已验证姿态",
    )
    parser.add_argument(
        "--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG,
        help=f"规划器配置；默认 {DEFAULT_PLANNER_CONFIG}",
    )
    parser.add_argument(
        "--initial-joint-planner", choices=("rrtstar", "ompl"), default="rrtstar",
        help="回固定关节起点的规划器；默认 rrtstar，ompl 使用 RRTConnect 绕障",
    )
    parser.add_argument(
        "--trajectory-workpiece", type=str, default="1",
        help="固定 place-ready→起点轨迹缓存的工件编号（由外层配置传入）",
    )
    parser.add_argument("--live", action="store_true", help="连接真实硬件")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--force-host", default=os.environ.get("PARM_FORCE_SENSOR_HOST", "192.168.3.22"))
    parser.add_argument(
        "--reset-right-task-before-servo",
        action="store_true",
        help="规划回起点前调用 cancel_task(2) 清理右臂残留任务；总控串联时使用",
    )
    parser.add_argument("--pre-search-leg", action="append", help="搜索前运动段：name|axis|direction|distance_mm|speed_mm_s")
    parser.add_argument(
        "--pre-search-press-mm", type=float, default=1.0,
        help="第 1 轮搜索前沿世界 Z− 的 LIN 下压距离，默认 1 mm",
    )
    parser.add_argument(
        "--retry-search-press-increment-mm", type=float, default=1.5,
        help="每次重试相对前一轮额外下压距离，默认 1.5 mm",
    )
    parser.add_argument(
        "--pre-search-press-speed-mm-s", type=float, default=1.0,
        help="每轮搜索前 LIN 下压速度，默认 1 mm/s",
    )
    parser.add_argument("--pre-search-press-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--pre-search-press-direction", choices=("negative", "positive"), default="negative")
    parser.add_argument("--baseline-settle-s", type=float, default=0.5)
    parser.add_argument("--baseline-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--search-attempt", type=int, default=1,
        help="当前搜索轮次；第 N 轮从初始点累计下压 N×pre-search-press-mm，默认第 1 轮",
    )
    parser.add_argument(
        "--return-search-exhausted", action="store_true",
        help="达到最大搜索距离时返回内部重试状态码，而非直接报错；仅供工件配置入口使用",
    )
    parser.add_argument("--search-axis", choices=("x", "y"), default="x", help="直线搜索使用的世界坐标轴，默认 X")
    parser.add_argument("--search-direction", choices=("negative", "positive"), default="negative", help="沿搜索轴的方向，默认负方向")
    parser.add_argument(
        "--continuous-force-monitor-hz", type=float, default=50.0,
        help="单段连续直线搜索中的力监测频率；默认 50 Hz",
    )
    parser.add_argument(
        "--continuous-force-output-hz", type=float, default=10.0,
        help="连续力监测的输出频率；默认 10 Hz（每 100 ms），0 表示不额外输出",
    )
    parser.add_argument(
        "--continuous-release-frames", type=int, default=6,
        help="连续力监测判进洞所需的连续下降帧数；默认 6 帧（约 100 ms）",
    )
    parser.add_argument(
        "--continuous-release-fx-rise-tolerance-n", type=float, default=0.4,
        help="连续下降期间相邻帧 dFx 允许的最大回升量，默认 0.4 N",
    )
    parser.add_argument(
        "--continuous-release-ty-rise-tolerance-nm", type=float, default=0.1,
        help="连续下降期间相邻帧 dTy 允许的最大回升量，默认 0.1 N*m",
    )
    right_scale = float(os.environ.get("PARM_RIGHT_SPEED_SCALE", "1.0"))
    parser.add_argument("--fast-linear-speed-mm-s", type=float, default=7.0 * right_scale, help="快速段线速度，默认 7 mm/s × 右臂速度倍率")
    parser.add_argument("--fast-linear-accel-mm-s2", type=float, default=20.0 * right_scale, help="快速段线加速度，默认 20 mm/s² × 右臂速度倍率")
    parser.add_argument("--pre-search-leg-accel-mm-s2", type=float, default=20.0, help="搜索前接近段线加速度，单位 mm/s²")
    parser.add_argument("--pre-search-press-accel-mm-s2", type=float, default=10.0, help="搜索前预压段线加速度，单位 mm/s²")
    parser.add_argument("--slow-after-back-mm", type=float, default=15.0, help="累计后拉达到该距离后降速，默认 15 mm")
    parser.add_argument("--slow-linear-speed-mm-s", type=float, default=7.0 * right_scale, help="降速后短段线速度，默认 7 mm/s × 右臂速度倍率")
    parser.add_argument("--slow-linear-accel-mm-s2", type=float, default=20.0 * right_scale, help="降速后短段线加速度，默认 20 mm/s² × 右臂速度倍率")
    parser.add_argument(
        "--quick-endpoint-tolerance-mm", type=float, default=0.5,
        help="快速摆动段开始力采样前允许的最大末端位置误差，默认 0.5 mm",
    )
    parser.add_argument(
        "--quick-endpoint-wait-s", type=float, default=0.45,
        help="快速摆动段到达后最多额外等待多久；未确认则自动退回完整到位等待，默认 0.45 秒",
    )
    parser.add_argument("--max-back-mm", type=float, default=40.0, help="最大累计后拉距离，默认 40 mm")
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument("--sample-timeout", type=float, default=2.0)
    parser.add_argument(
        "--max-force-delta-n", type=float, default=20.0,
        help="相对力基线的合力硬限；默认 20 N，超过立即停止",
    )
    parser.add_argument("--max-torque-delta-nm", type=float, default=5.0, help="相对力基线的合扭矩硬限；默认 5 N*m")
    # Entry is a two-frame release: Fx must drop sharply from contact, while
    # Ty must return close to zero.  The Ty absolute-value guard rejects a
    # transition into an opposite-direction collision.
    parser.add_argument("--entry-pre-fx-n", type=float, default=10.0, help="进洞前一步所需最小 dFx")
    parser.add_argument("--entry-pre-ty-nm", type=float, default=2.5, help="进洞前一步所需最小 dTy")
    parser.add_argument("--entry-fx-drop-n", type=float, default=10.0, help="进洞单步所需最小 dFx 下降量")
    parser.add_argument("--entry-ty-drop-nm", type=float, default=2.5, help="进洞单步所需最小 dTy 下降量")
    parser.add_argument("--entry-ty-abs-nm", type=float, default=1.0, help="进洞当前步允许的最大 |dTy|")
    # A confirmed insertion can also unload from a moderate contact load,
    # rather than from the older >=10 N / >=2.5 N*m peak.  Keep this as a
    # separate condition so the original strong-jump rule is unchanged.
    parser.add_argument("--entry-relaxed-pre-fx-n", type=float, default=6.0, help="中等进洞条件的前一步最小正向 dFx")
    parser.add_argument("--entry-relaxed-pre-ty-nm", type=float, default=1.5, help="中等进洞条件的前一步最小正向 dTy")
    parser.add_argument("--entry-relaxed-fx-drop-n", type=float, default=4.0, help="中等进洞条件的最小 dFx 下降量")
    parser.add_argument("--entry-relaxed-ty-drop-nm", type=float, default=1.0, help="中等进洞条件的最小 dTy 下降量")
    # A smaller, simultaneous force/torque unload can also mean that the part
    # has entered the hole.  Unlike the large-jump rules above, require the
    # load to remain low while holding the current pose.  The previous load
    # may be the cycle centre and the unload may occur during a lateral wiggle.
    parser.add_argument("--entry-release-pre-force-n", type=float, default=2.0, help="卸载进洞前一步所需最小 |dF|")
    parser.add_argument("--entry-release-pre-torque-nm", type=float, default=0.4, help="卸载进洞前一步所需最小 |dT|")
    parser.add_argument("--entry-release-force-n", type=float, default=1.5, help="卸载进洞候选当前点及确认采样中位数允许的最大 |dF|")
    parser.add_argument("--entry-release-torque-nm", type=float, default=0.4, help="卸载进洞候选当前点及确认采样中位数允许的最大 |dT|")
    parser.add_argument("--entry-release-force-drop-n", type=float, default=2.0, help="卸载进洞所需最小 |dF| 下降量")
    parser.add_argument("--entry-release-torque-drop-nm", type=float, default=0.4, help="卸载进洞所需最小 |dT| 下降量")
    parser.add_argument("--entry-release-confirm-samples", type=int, default=3, help="卸载进洞原地确认采样次数，默认 3")
    parser.add_argument("--entry-release-confirm-interval-s", type=float, default=0.1, help="卸载进洞确认采样间隔，默认 0.1 秒")
    parser.add_argument(
        "--entry-resultant-pre-force-n", type=float, default=0.0,
        help="合力连续卸载判据的峰值最小 |dF|；0 表示关闭此判据",
    )
    parser.add_argument(
        "--entry-resultant-pre-torque-nm", type=float, default=0.0,
        help="合力连续卸载判据的峰值最小 |dT|；0 表示关闭此判据",
    )
    parser.add_argument("--entry-resultant-force-n", type=float, default=0.0, help="合力连续卸载后的最大 |dF|")
    parser.add_argument("--entry-resultant-torque-nm", type=float, default=0.0, help="合力连续卸载后的最大 |dT|")
    parser.add_argument("--entry-resultant-force-drop-n", type=float, default=0.0, help="合力连续卸载所需最小 |dF| 降幅")
    parser.add_argument("--entry-resultant-torque-drop-nm", type=float, default=0.0, help="合力连续卸载所需最小 |dT| 降幅")
    parser.add_argument("--entry-resultant-release-frames", type=int, default=0, help="合力连续卸载所需连续下降帧数；0 表示关闭")
    parser.add_argument(
        "--release-events-required", type=int, default=1,
        help="连续搜索中需要出现几次合格卸载事件才按进洞停止；默认 1",
    )
    parser.add_argument("--entry-fx-reversal-pre-n", type=float, default=0.0, help="Fx 翻转进洞判据的前一帧最小正 dFx；0 表示关闭")
    parser.add_argument("--entry-fx-reversal-current-n", type=float, default=0.0, help="Fx 翻转进洞判据的当前帧最大负 dFx")
    parser.add_argument("--entry-fx-reversal-drop-n", type=float, default=0.0, help="Fx 翻转进洞判据所需相对局部正峰值的最小下降量")
    parser.add_argument("--entry-fx-reversal-confirm-frames", type=int, default=0, help="Fx 翻转后保持负值的确认帧数；0 表示关闭")
    parser.add_argument("--entry-fx-reversal-peak-window-mm", type=float, default=0.0, help="Fx 翻转判据保存正向局部峰值的最大搜索距离；0 表示关闭")
    # Some insertions happen during the lateral search rather than the next
    # back-centre step.  Confirm this with a Fx/Ty sign reversal plus a large
    # drop, rather than requiring one fixed final magnitude.  Confirmed runs:
    # +7.73 -> -12.50 N / +1.90 -> -3.88 N*m, and
    # +9.63 -> -7.52 N / +2.28 -> -2.32 N*m.
    parser.add_argument("--wiggle-entry-pre-fx-n", type=float, default=5.0, help="摆动进洞前中心点所需最小 dFx")
    parser.add_argument("--wiggle-entry-pre-ty-nm", type=float, default=1.5, help="摆动进洞前中心点所需最小 dTy")
    parser.add_argument("--wiggle-entry-fx-drop-n", type=float, default=10.0, help="摆动进洞所需最小 dFx 下降量")
    parser.add_argument("--wiggle-entry-ty-drop-nm", type=float, default=3.0, help="摆动进洞所需最小 dTy 下降量")
    parser.add_argument("--min-entry-back-mm", type=float, default=8.0, help="至少后拉该距离后才允许判进洞")
    parser.add_argument(
        "--post-entry-enabled", action="store_true",
        help="执行配置的进洞后后退、下压辅助动作",
    )
    parser.add_argument(
        "--post-entry-relief-mm", type=float, default=0.0,
        help="确认进洞后沿搜索反方向固定卸力距离；0 表示关闭",
    )
    parser.add_argument(
        "--post-entry-relief-opposing-force-n", type=float, default=0.0,
        help="触发卸力所需的前后向反力阈值；当前 X− 搜索对应 dFz >= 此值，0 表示关闭",
    )
    parser.add_argument(
        "--post-entry-relief-confirm-frames", type=int, default=3,
        help="进洞后观察窗口内触发卸力所需的反力帧数，默认 3",
    )
    parser.add_argument(
        "--post-entry-relief-observation-frames", type=int, default=3,
        help="进洞后用于确认反向力的连续采样帧数，默认 3",
    )
    parser.add_argument(
        "--post-entry-relief-speed-mm-s", type=float, default=1.0,
        help="确认进洞后的卸力线速度，默认 1 mm/s",
    )
    parser.add_argument(
        "--post-entry-descend-mm", type=float, default=10.0,
        help="进洞后沿世界 Z− 的最大下降距离，默认 10 mm；0 表示关闭",
    )
    parser.add_argument(
        "--post-entry-descend-fx-rise-n", type=float, default=0.0,
        help="进洞后下降时，相对下降前基线 raw Fx 上升至该值即停止；0 表示不读取力传感器",
    )
    parser.add_argument(
        "--post-entry-descend-speed-mm-s", type=float, default=1.0,
        help="进洞后 Z− 下降线速度，默认 1 mm/s",
    )
    parser.add_argument("--post-entry-linear-accel-mm-s2", type=float, default=20.0,
                        help="进洞后辅助动作的线加速度，单位 mm/s²")
    parser.add_argument("--first-hole-settle-down-mm", type=float, default=10.0,
                        help="第一洞确认后沿世界 Z− 下压距离，默认 10 mm")
    parser.add_argument("--first-hole-lift-mm", type=float, default=40.0,
                        help="第一洞下压完成后沿世界 Z+ 提起距离，默认 40 mm")
    parser.add_argument("--second-hole-y-offset-mm", type=float, default=30.0,
                        help="第一洞确认后沿世界 Y+ 移至第二洞的距离，默认 30 mm")
    parser.add_argument("--second-hole-descend-mm", type=float, default=48.0,
                        help="第二洞沿世界 Z− 下降距离，默认 48 mm")
    parser.add_argument("--second-hole-lift-mm", type=float, default=40.0,
                        help="第二洞下降完成后沿世界 Z+ 提起距离，默认 40 mm")
    parser.add_argument("--second-hole-final-axis", choices=("x", "y", "z"), default="x",
                        help="第二洞完成后最终平移所使用的世界坐标轴")
    parser.add_argument("--second-hole-final-direction", choices=("negative", "positive"), default="negative",
                        help="第二洞完成后最终平移方向")
    parser.add_argument("--second-hole-final-distance-mm", type=float, default=110.0,
                        help="第二洞完成后最终平移距离，默认 110 mm")
    parser.add_argument("--second-hole-lateral-speed-mm-s", type=float, default=10.0,
                        help="第二洞上提与 Y+ 平移速度，默认 10 mm/s")
    parser.add_argument("--second-hole-final-speed-mm-s", type=float, default=25.0,
                        help="第二洞完成后最终平移速度，单位 mm/s")
    parser.add_argument("--second-hole-descend-speed-mm-s", type=float, default=5.0,
                        help="第二洞 Z− 下降速度，默认 5 mm/s")
    parser.add_argument("--second-hole-linear-accel-mm-s2", type=float, default=20.0,
                        help="第二洞路径线加速度，单位 mm/s²")
    parser.add_argument("--second-hole-near-target-mm", type=float, default=2.0,
                        help="第二洞非下降段的断连恢复近目标阈值，单位 mm")
    parser.add_argument("--second-hole-descend-near-target-mm", type=float, default=3.0,
                        help="第二洞下降段的断连恢复近目标阈值，单位 mm")
    parser.add_argument("--second-hole-descend-recovery-speed-mm-s", type=float, default=3.0,
                        help="第二洞下降段断连恢复速度，单位 mm/s")
    parser.add_argument("--second-hole-pause-after-first-down-s", type=float, default=0.5,
                        help="第一洞下压后停留时间，单位 s")
    parser.add_argument("--second-hole-pause-after-second-down-s", type=float, default=0.5,
                        help="第二洞下降后停留时间，单位 s")
    parser.add_argument("--second-hole-enabled", action="store_true",
                        help="第一洞确认后执行配置的第二洞路径")
    parser.add_argument("--pre-search-initial-descend-mm", type=float, default=10.0,
                        help="规划到起始关节位后先沿世界 Z− 下降距离，默认 10 mm")
    parser.add_argument("--pre-search-forward-mm", type=float, default=142.0,
                        help="初始下降后沿世界 X+ 前进距离，默认 142 mm")
    parser.add_argument("--pre-search-descend-mm", type=float, default=20.0,
                        help="前进后沿世界 Z− 下降距离，默认 20 mm")
    parser.add_argument("--pre-search-approach-speed-mm-s", type=float, default=10.0,
                        help="前进和下降预接近速度，默认 10 mm/s")
    parser.add_argument("--output", type=Path, help="运行日志路径；默认 logs/hole_a_wiggle_<时间>.log")
    args = parser.parse_args(argv)
    if args.live != args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    legs = None
    if args.pre_search_leg:
        legs = []
        for raw_leg in args.pre_search_leg:
            fields = raw_leg.split("|")
            if len(fields) != 5:
                parser.error("--pre-search-leg 格式必须为 name|axis|direction|distance_mm|speed_mm_s")
            name, axis, direction, distance_text, speed_text = fields
            try:
                distance_mm, speed_mm_s = float(distance_text), float(speed_text)
            except ValueError:
                parser.error("--pre-search-leg 的距离和速度必须为数值")
            legs.append({"name": name, "axis": axis, "direction": direction, "distance_mm": distance_mm, "speed_mm_s": speed_mm_s})
    if legs is not None:
        for index, leg in enumerate(legs, start=1):
            if not isinstance(leg, dict):
                parser.error(f"第 {index} 个搜索前运动段必须是对象")
            required = {"name", "axis", "direction", "distance_mm", "speed_mm_s"}
            if required - set(leg):
                parser.error(f"第 {index} 个搜索前运动段缺少字段：{', '.join(sorted(required - set(leg)))}")
            if not isinstance(leg["name"], str) or not leg["name"]:
                parser.error(f"第 {index} 个搜索前运动段 name 必须是非空字符串")
            if leg["axis"] not in ("x", "y", "z") or leg["direction"] not in ("negative", "positive"):
                parser.error(f"第 {index} 个搜索前运动段 axis/direction 无效")
            for key in ("distance_mm", "speed_mm_s"):
                value = leg[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                    parser.error(f"第 {index} 个搜索前运动段 {key} 必须为正有限数")
    args.pre_search_legs = legs
    for name in ("pre_search_press_speed_mm_s", "fast_linear_speed_mm_s", "fast_linear_accel_mm_s2", "max_back_mm", "sample_timeout", "max_force_delta_n", "max_torque_delta_nm", "min_entry_back_mm", "entry_pre_fx_n", "entry_pre_ty_nm", "entry_fx_drop_n", "entry_ty_drop_nm", "entry_ty_abs_nm", "entry_relaxed_pre_fx_n", "entry_relaxed_pre_ty_nm", "entry_relaxed_fx_drop_n", "entry_relaxed_ty_drop_nm", "entry_release_pre_force_n", "entry_release_pre_torque_nm", "entry_release_force_n", "entry_release_torque_nm", "entry_release_force_drop_n", "entry_release_torque_drop_nm", "entry_release_confirm_interval_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    for name in ("pre_search_press_mm", "retry_search_press_increment_mm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须为非负有限数")
    if not all(math.isfinite(value) for value in args.initial_right_joints_deg):
        parser.error("--initial-right-joints-deg 必须全部为有限数")
    if args.continuous_force_monitor_hz <= 0.0:
        parser.error("--continuous-force-monitor-hz 必须为正数")
    if not math.isfinite(args.continuous_force_monitor_hz) or not 0.0 <= args.continuous_force_monitor_hz <= 50.0:
        parser.error("--continuous-force-monitor-hz 必须在 0 到 50 之间")
    if not math.isfinite(args.continuous_force_output_hz) or not 0.0 <= args.continuous_force_output_hz <= 50.0:
        parser.error("--continuous-force-output-hz 必须在 0 到 50 之间")
    if args.continuous_release_frames < 2:
        parser.error("--continuous-release-frames 必须至少为 2")
    for name in ("continuous_release_fx_rise_tolerance_n", "continuous_release_ty_rise_tolerance_nm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须为非负数")
    if args.baseline_samples < 1:
        parser.error("--baseline-samples 必须至少为 1")
    if args.search_attempt < 1:
        parser.error("--search-attempt 必须至少为 1")
    for name in ("baseline_settle_s", "baseline_interval_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须为非负有限数")
    if args.entry_release_confirm_samples < 1:
        parser.error("--entry-release-confirm-samples 必须至少为 1")
    resultant_values = (
        "entry_resultant_pre_force_n", "entry_resultant_pre_torque_nm",
        "entry_resultant_force_n", "entry_resultant_torque_nm",
        "entry_resultant_force_drop_n", "entry_resultant_torque_drop_nm",
    )
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0 for name in resultant_values):
        parser.error("合力连续卸载判据的数值必须为非负有限数")
    if args.entry_resultant_release_frames < 0:
        parser.error("--entry-resultant-release-frames 不得为负数")
    if args.release_events_required < 1:
        parser.error("--release-events-required 必须至少为 1")
    if args.post_entry_relief_mm < 0.0 or args.post_entry_relief_mm > 2.0:
        parser.error("--post-entry-relief-mm 必须在 0 到 2 mm 之间")
    if args.post_entry_relief_mm > 0.0 and not (
        math.isfinite(args.post_entry_relief_speed_mm_s)
        and args.post_entry_relief_speed_mm_s > 0.0
    ):
        parser.error("启用进洞后固定后拉时：速度必须为正")
    if not math.isfinite(args.post_entry_descend_mm) or not 0.0 <= args.post_entry_descend_mm <= 10.0:
        parser.error("--post-entry-descend-mm 必须在 0 到 10 mm 之间")
    if args.post_entry_descend_mm > 0.0 and not (
        math.isfinite(args.post_entry_descend_fx_rise_n)
        and args.post_entry_descend_fx_rise_n >= 0.0
        and math.isfinite(args.post_entry_descend_speed_mm_s)
        and args.post_entry_descend_speed_mm_s > 0.0
    ):
        parser.error("启用进洞后下降时：Fx 阈值必须为非负数，速度必须为正")
    for name in (
        "fast_linear_accel_mm_s2", "pre_search_leg_accel_mm_s2", "pre_search_press_accel_mm_s2",
        "first_hole_lift_mm",
        "second_hole_lift_mm", "second_hole_y_offset_mm", "second_hole_descend_mm",
        "second_hole_final_distance_mm",
        "second_hole_lateral_speed_mm_s", "second_hole_final_speed_mm_s",
        "second_hole_descend_speed_mm_s", "second_hole_linear_accel_mm_s2",
        "second_hole_near_target_mm", "second_hole_descend_near_target_mm",
        "second_hole_descend_recovery_speed_mm_s",
        "pre_search_forward_mm", "pre_search_descend_mm", "pre_search_approach_speed_mm_s",
        "pre_search_initial_descend_mm",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    for name in ("second_hole_pause_after_first_down_s", "second_hole_pause_after_second_down_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须为非负有限数")
    if not math.isfinite(args.first_hole_settle_down_mm) or args.first_hole_settle_down_mm < 0.0:
        parser.error("--first-hole-settle-down-mm 必须为非负有限数")
    fx_reversal_enabled = args.entry_fx_reversal_confirm_frames > 0
    if fx_reversal_enabled and not (
        math.isfinite(args.entry_fx_reversal_pre_n)
        and math.isfinite(args.entry_fx_reversal_current_n)
        and math.isfinite(args.entry_fx_reversal_drop_n)
        and args.entry_fx_reversal_pre_n > 0.0
        and args.entry_fx_reversal_current_n < 0.0
        and args.entry_fx_reversal_drop_n > 0.0
        and args.entry_fx_reversal_confirm_frames >= 2
        and math.isfinite(args.entry_fx_reversal_peak_window_mm)
        and args.entry_fx_reversal_peak_window_mm > 0.0
    ):
        parser.error("启用 Fx 翻转进洞判据时：pre>0、current<0、drop>0、peak-window>0，且 confirm-frames 至少为 2")
    return args

def write_control_snapshot(arm: object, output_file: object, event: str) -> None:
    """Record both arms so a right-servo failure can be correlated with left-arm tasks."""
    state = arm.get_state() or {}  # type: ignore[attr-defined]
    arms = state.get("state") or {}
    record: dict[str, object] = {
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "monotonic_s": time.monotonic(),
        "event": event,
        "system_state": state.get("system_state"),
    }
    for arm_key in ("left", "right"):
        per_arm = arms.get(arm_key) or {}
        velocity = per_arm.get("velocity")
        velocity_values = (
            [float(value) for value in velocity]
            if isinstance(velocity, (list, tuple)) and len(velocity) == 7
            else None
        )
        record[arm_key] = {
            "task_state": per_arm.get("task_state"),
            "enabled": per_arm.get("enabled"),
            "position_deg": per_arm.get("position"),
            "velocity_deg_s": velocity_values,
            "max_abs_velocity_deg_s": (
                max(abs(value) for value in velocity_values)
                if velocity_values is not None
                else None
            ),
            "eef_pose": per_arm.get("eef_pose"),
        }
    output_file.write("[CONTROL] " + json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[attr-defined]
    output_file.flush()

# The workpiece-profile entry point is intentionally kept in this file.  It
# translates the minimal JSON profile directly into this module's arguments;
# no subprocess and no call to place_hole_a_wiggle.py is involved.
WORKPIECE_CONFIGS = {
    1: Path(__file__).resolve().parent / "config/workpiece_1_gzhu.toml",
    2: Path(__file__).resolve().parent / "config/workpiece_2.toml",
}

STOP_KEYS = {
    "max_back_mm", "max_force_delta_n", "max_torque_delta_nm",
    "continuous_release_frames", "continuous_release_fx_rise_tolerance_n",
    "continuous_release_ty_rise_tolerance_nm", "min_entry_back_mm",
    "entry_pre_fx_n", "entry_pre_ty_nm", "entry_fx_drop_n", "entry_ty_drop_nm",
    "entry_ty_abs_nm", "entry_relaxed_pre_fx_n", "entry_relaxed_pre_ty_nm",
    "entry_relaxed_fx_drop_n", "entry_relaxed_ty_drop_nm",
    "entry_release_pre_force_n", "entry_release_pre_torque_nm",
    "entry_release_force_n", "entry_release_torque_nm",
    "entry_release_force_drop_n", "entry_release_torque_drop_nm",
    "entry_release_confirm_samples", "entry_release_confirm_interval_s",
    "entry_resultant_pre_force_n", "entry_resultant_pre_torque_nm",
    "entry_resultant_force_n", "entry_resultant_torque_nm",
    "entry_resultant_force_drop_n", "entry_resultant_torque_drop_nm",
    "entry_resultant_release_frames", "release_events_required",
    "entry_fx_reversal_pre_n", "entry_fx_reversal_current_n",
    "entry_fx_reversal_drop_n", "entry_fx_reversal_confirm_frames",
    "entry_fx_reversal_peak_window_mm",
}



def read_profile(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        raw = load_toml(path)
    except ValueError as exc:
        raise ValueError(f"无法读取 TOML 配置 {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("配置必须是对象，且 schema_version 必须为 2")
    workpiece = raw.get("workpiece_serial_number")
    if isinstance(workpiece, bool) or not isinstance(workpiece, int) or workpiece < 1:
        raise ValueError("workpiece_serial_number 必须是正整数")
    missing = {"initial_right_joints_deg", "arm_host", "force_sensor_host", "search_axis", "search_direction", "pre_search", "search_motion", "retry", "post_entry", "second_hole"} - set(raw)
    if missing:
        raise ValueError(f"配置缺少必填字段: {', '.join(sorted(missing))}")
    for key in ("arm_host", "force_sensor_host"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{key} 必须是非空字符串")
    stop_conditions = raw.get("stop_conditions", {})
    if not isinstance(stop_conditions, dict):
        raise ValueError("stop_conditions 必须是对象")
    unknown = set(stop_conditions) - STOP_KEYS
    if unknown:
        raise ValueError(f"stop_conditions 包含不支持的字段: {', '.join(sorted(unknown))}")
    pre_search = raw.get("pre_search")
    search_motion = raw.get("search_motion")
    retry = raw.get("retry")
    post_entry = raw.get("post_entry")
    second_hole = raw.get("second_hole")
    if not isinstance(pre_search, dict):
        raise ValueError("配置必须包含 pre_search 对象")
    for label, table, keys in (
        ("search_motion", search_motion, {"speed_mm_s", "accel_mm_s2"}),
        ("retry", retry, {"max_attempts", "lift_mm", "speed_mm_s", "accel_mm_s2"}),
        ("post_entry", post_entry, {"enabled", "relief_mm", "relief_opposing_force_n", "relief_confirm_frames", "relief_observation_frames", "relief_speed_mm_s", "descend_mm", "descend_fx_rise_n", "descend_speed_mm_s", "linear_accel_mm_s2"}),
    ):
        if not isinstance(table, dict) or keys - set(table):
            raise ValueError(f"{label} 缺少字段: {', '.join(sorted(keys - set(table or {})))}")
    if not isinstance(post_entry["enabled"], bool):
        raise ValueError("post_entry.enabled 必须是布尔值")
    if not isinstance(second_hole, dict) or not isinstance(second_hole.get("enabled"), bool):
        raise ValueError("second_hole 必须是包含布尔 enabled 的对象")
    required = {"frame", "legs", "press", "baseline", "leg_accel_mm_s2"}
    if required - set(pre_search):
        raise ValueError(f"pre_search 缺少字段: {', '.join(sorted(required - set(pre_search)))}")
    if pre_search["frame"] != "world":
        raise ValueError("pre_search.frame 目前只支持 world")
    if not isinstance(pre_search["legs"], list) or not pre_search["legs"]:
        raise ValueError("pre_search.legs 必须是非空数组")
    if not isinstance(pre_search["press"], dict) or not isinstance(pre_search["baseline"], dict):
        raise ValueError("pre_search.press 和 pre_search.baseline 必须是对象")
    press_required = {"axis", "direction", "first_attempt_mm", "retry_increment_mm", "speed_mm_s", "accel_mm_s2"}
    baseline_required = {"settle_s", "samples", "interval_s"}
    if press_required - set(pre_search["press"]):
        raise ValueError(f"pre_search.press 缺少字段: {', '.join(sorted(press_required - set(pre_search['press'])))}")
    if baseline_required - set(pre_search["baseline"]):
        raise ValueError(f"pre_search.baseline 缺少字段: {', '.join(sorted(baseline_required - set(pre_search['baseline'])))}")
    if second_hole["enabled"]:
        second_required = {
            "first_hole_settle_down_mm", "first_hole_lift_mm",
            "y_offset_mm", "descend_mm", "lift_after_second_hole_mm",
            "final_axis", "final_direction", "final_distance_mm", "lateral_speed_mm_s", "final_speed_mm_s",
            "descend_speed_mm_s", "linear_accel_mm_s2",
            "near_target_mm", "descend_near_target_mm", "descend_recovery_speed_mm_s",
            "pause_after_first_down_s", "pause_after_second_down_s",
        }
        if second_required - set(second_hole):
            raise ValueError(f"second_hole 缺少字段: {', '.join(sorted(second_required - set(second_hole)))}")
    return workpiece, raw

def append_scalar(argv: list[str], key: str, value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label}.{key} 必须是字符串或数字")
    argv.extend(["--" + key.replace("_", "-"), str(value)])

def profile_argv(
    profile: dict[str, Any], workpiece: int, *, live: bool,
    arm_host: str | None, reset_right_task_before_servo: bool, attempt: int,
) -> list[str]:
    joints = profile.get("initial_right_joints_deg")
    if (
        not isinstance(joints, list) or len(joints) != 7
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in joints)
    ):
        raise ValueError("initial_right_joints_deg 必须是 7 个数值组成的数组")
    pre_search = profile["pre_search"]
    search_motion = profile["search_motion"]
    post_entry = profile["post_entry"]
    second_hole = profile["second_hole"]
    if not isinstance(pre_search, dict) or not isinstance(search_motion, dict) or not isinstance(post_entry, dict):
        raise ValueError("配置运动参数必须为对象")
    press = pre_search["press"]
    baseline = pre_search["baseline"]
    if not isinstance(press, dict) or not isinstance(baseline, dict):
        raise ValueError("pre_search.press 和 pre_search.baseline 必须是对象")
    pre_search_argv = [
            "--pre-search-press-axis", str(press["axis"]),
            "--pre-search-press-direction", str(press["direction"]),
            "--pre-search-press-mm", str(press["first_attempt_mm"]),
            "--retry-search-press-increment-mm", str(press["retry_increment_mm"]),
            "--pre-search-press-speed-mm-s", str(press["speed_mm_s"]),
            "--pre-search-press-accel-mm-s2", str(press["accel_mm_s2"]),
            "--pre-search-leg-accel-mm-s2", str(pre_search["leg_accel_mm_s2"]),
            "--baseline-settle-s", str(baseline["settle_s"]),
            "--baseline-samples", str(baseline["samples"]),
            "--baseline-interval-s", str(baseline["interval_s"]),
    ]
    for leg in pre_search["legs"]:
        pre_search_argv.extend(["--pre-search-leg", "|".join(str(leg[key]) for key in ("name", "axis", "direction", "distance_mm", "speed_mm_s"))])
    argv = [
        "--initial-joint-planner", "ompl",
        "--trajectory-workpiece", str(workpiece),
        "--fast-linear-speed-mm-s", str(search_motion["speed_mm_s"]),
        "--fast-linear-accel-mm-s2", str(search_motion["accel_mm_s2"]),
        *pre_search_argv,
        "--search-attempt", str(attempt),
        "--return-search-exhausted",
        "--post-entry-relief-mm", str(post_entry["relief_mm"]),
        "--post-entry-relief-opposing-force-n", str(post_entry["relief_opposing_force_n"]),
        "--post-entry-relief-confirm-frames", str(post_entry["relief_confirm_frames"]),
        "--post-entry-relief-observation-frames", str(post_entry["relief_observation_frames"]),
        "--post-entry-relief-speed-mm-s", str(post_entry["relief_speed_mm_s"]),
        "--post-entry-descend-mm", str(post_entry["descend_mm"]),
        "--post-entry-descend-fx-rise-n", str(post_entry["descend_fx_rise_n"]),
        "--post-entry-descend-speed-mm-s", str(post_entry["descend_speed_mm_s"]),
        "--post-entry-linear-accel-mm-s2", str(post_entry["linear_accel_mm_s2"]),
        "--initial-right-joints-deg", *(str(value) for value in joints),
        "--arm-host", str(profile["arm_host"]),
        "--force-host", str(profile["force_sensor_host"]),
    ]
    if post_entry["enabled"]:
        argv.append("--post-entry-enabled")
    if second_hole["enabled"]:
        argv.extend([
            "--second-hole-enabled",
            "--first-hole-settle-down-mm", str(second_hole["first_hole_settle_down_mm"]),
            "--first-hole-lift-mm", str(second_hole["first_hole_lift_mm"]),
            "--second-hole-y-offset-mm", str(second_hole["y_offset_mm"]),
            "--second-hole-descend-mm", str(second_hole["descend_mm"]),
            "--second-hole-lift-mm", str(second_hole["lift_after_second_hole_mm"]),
            "--second-hole-final-axis", str(second_hole["final_axis"]),
            "--second-hole-final-direction", str(second_hole["final_direction"]),
            "--second-hole-final-distance-mm", str(second_hole["final_distance_mm"]),
            "--second-hole-lateral-speed-mm-s", str(second_hole["lateral_speed_mm_s"]),
            "--second-hole-final-speed-mm-s", str(second_hole["final_speed_mm_s"]),
            "--second-hole-descend-speed-mm-s", str(second_hole["descend_speed_mm_s"]),
            "--second-hole-linear-accel-mm-s2", str(second_hole["linear_accel_mm_s2"]),
            "--second-hole-pause-after-first-down-s", str(second_hole["pause_after_first_down_s"]),
            "--second-hole-pause-after-second-down-s", str(second_hole["pause_after_second_down_s"]),
            "--second-hole-near-target-mm", str(second_hole["near_target_mm"]),
            "--second-hole-descend-near-target-mm", str(second_hole["descend_near_target_mm"]),
            "--second-hole-descend-recovery-speed-mm-s", str(second_hole["descend_recovery_speed_mm_s"]),
        ])
    for key in ("search_axis", "search_direction"):
        append_scalar(argv, key, profile[key], label="配置")
    for key in sorted(profile.get("stop_conditions", {})):
        append_scalar(argv, key, profile["stop_conditions"][key], label="stop_conditions")
    log_path = Path(__file__).resolve().parent / "logs" / (
        f"hole_a_workpiece_{workpiece}_attempt_{attempt}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    )
    argv.extend(["--output", str(log_path)])
    if arm_host is not None:
        argv.extend(["--arm-host", arm_host])
    if reset_right_task_before_servo:
        argv.append("--reset-right-task-before-servo")
    if live:
        argv.extend(["--live", "--confirm-live"])
    return argv
