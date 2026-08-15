#!/usr/bin/env python3
"""Continuous 15-mm/s straight-line place search with 50-Hz force monitoring.

The taught start is a fixed right-arm joint vector.  After that joint move,
the actual SDK end-effector pose is used as the world-frame centre.  By
default, a cycle backs 1 mm in world X and samples at that centre point; no
lateral motion is performed.  Force is checked at every centre endpoint.
Hole entry is declared when a previously high Fx/Ty load abruptly unloads.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUSH_T_DIR = Path(__file__).resolve().parent / "vendor/push_t"
if str(PUSH_T_DIR) not in sys.path:
    sys.path.insert(0, str(PUSH_T_DIR))

from move_by_joints import (  # noqa: E402
    current_pose,
    ensure_ready,
    import_sdk,
    minimum_scurve_duration,
    orientation_error_rad,
    position_error_m,
    quat_to_rotvec,
    rotvec_to_quat,
    rotvec_to_rpy,
    rpy_to_rotvec,
    slerp,
    smoothstep5,
)
from loop_config import load_config  # noqa: E402
from tool.planner_runtime import PlannerRuntime  # noqa: E402
from tool.left_motion_right_servo_lock import right_servo_guard  # noqa: E402


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


class PlacementContactReached(RuntimeError):
    """A validated insertion-release signature stopped the search successfully."""


class PostEntryFxRiseReached(RuntimeError):
    """The post-insertion descent reached its raw sensor-Fx stop threshold."""


SEARCH_DISTANCE_EXHAUSTED_EXIT_CODE = 42
MAX_SERVO_RECOVERY_ATTEMPTS = 20


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
    parser.add_argument("--live", action="store_true", help="连接真实硬件")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--force-host", default=os.environ.get("PARM_FORCE_SENSOR_HOST", "192.168.3.22"))
    parser.add_argument(
        "--reset-right-task-before-servo",
        action="store_true",
        help="规划回起点前调用 cancel_task(2) 清理右臂残留任务；总控串联时使用",
    )
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
    parser.add_argument(
        "--search-attempt", type=int, default=1,
        help="当前搜索轮次；第 N 轮从初始点累计下压 N×pre-search-press-mm，默认第 1 轮",
    )
    parser.add_argument(
        "--return-search-exhausted", action="store_true",
        help="达到最大搜索距离时返回内部重试状态码，而非直接报错；仅供工件配置入口使用",
    )
    parser.add_argument(
        "--max-consecutive-back-target-timeouts", type=int, default=2,
        help="连续搜索目标未到位的最大次数；每次原地采力后继续下一搜索点，默认 2 次",
    )
    parser.add_argument(
        "--same-target-retries", type=int, default=1,
        help="搜索目标首次未到位时，保持同一目标完整重发次数；默认 1 次",
    )
    parser.add_argument("--search-axis", choices=("x", "y"), default="x", help="直线搜索使用的世界坐标轴，默认 X")
    parser.add_argument("--search-direction", choices=("negative", "positive"), default="negative", help="沿搜索轴的方向，默认负方向")
    parser.add_argument("--back-step-mm", type=float, default=1.0, help="每周期沿搜索轴的移动距离，默认 1 mm")
    parser.add_argument(
        "--wiggle-half-amplitude-mm", type=float, default=1.0,
        help="世界 Y 相对中心的单侧摆幅，默认 1 mm（左右总摆幅 2 mm）",
    )
    parser.add_argument(
        "--straight-line", action="store_true", default=True,
        help="只沿配置的世界轴直线搜索并在每个中心点采样；这是默认模式，不执行横向摆动",
    )
    parser.add_argument(
        "--with-wiggle", action="store_false", dest="straight_line",
        help="恢复旧的每轮 Y+/Y− 摆动搜索，仅用于已验证的横向修正场景",
    )
    parser.add_argument(
        "--continuous-straight-search", action="store_true", default=True,
        help="直线搜索时连续匀速推送；默认启用，仍每 1 mm 采样，但不在每个采样点等待末端到位",
    )
    parser.add_argument(
        "--no-continuous-straight-search", action="store_false", dest="continuous_straight_search",
        help="关闭连续推送，恢复每 1 mm 的完整末端到位确认",
    )
    parser.add_argument(
        "--single-continuous-search", action="store_true",
        help="将直线搜索作为一条连续轨迹推送到最大距离；仅保留高频力监测，不做每毫米端点采样",
    )
    parser.add_argument(
        "--continuous-force-monitor-hz", type=float, default=50.0,
        help="连续直线搜索中的力监测频率；默认 50 Hz，0 表示仅保留每毫米采样",
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
    parser.add_argument("--second-hole-lift-mm", type=float, default=40.0,
                        help="第一洞确认后沿世界 Z+ 提起距离，默认 40 mm")
    parser.add_argument("--second-hole-y-offset-mm", type=float, default=30.0,
                        help="第一洞确认后沿世界 Y+ 移至第二洞的距离，默认 30 mm")
    parser.add_argument("--second-hole-descend-mm", type=float, default=48.0,
                        help="第二洞沿世界 Z− 下降距离，默认 48 mm")
    parser.add_argument("--second-hole-lateral-speed-mm-s", type=float, default=10.0,
                        help="第二洞上提与 Y+ 平移速度，默认 10 mm/s")
    parser.add_argument("--second-hole-x-return-speed-mm-s", type=float, default=25.0,
                        help="第二洞后沿世界 X−110 mm 的速度，默认 25 mm/s")
    parser.add_argument("--second-hole-descend-speed-mm-s", type=float, default=5.0,
                        help="第二洞 Z− 下降速度，默认 5 mm/s")
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
    for name in ("pre_search_press_mm", "retry_search_press_increment_mm", "pre_search_press_speed_mm_s", "back_step_mm", "wiggle_half_amplitude_mm", "fast_linear_speed_mm_s", "fast_linear_accel_mm_s2", "slow_after_back_mm", "slow_linear_speed_mm_s", "slow_linear_accel_mm_s2", "quick_endpoint_tolerance_mm", "quick_endpoint_wait_s", "max_back_mm", "sample_timeout", "max_force_delta_n", "max_torque_delta_nm", "min_entry_back_mm", "entry_pre_fx_n", "entry_pre_ty_nm", "entry_fx_drop_n", "entry_ty_drop_nm", "entry_ty_abs_nm", "entry_relaxed_pre_fx_n", "entry_relaxed_pre_ty_nm", "entry_relaxed_fx_drop_n", "entry_relaxed_ty_drop_nm", "entry_release_pre_force_n", "entry_release_pre_torque_nm", "entry_release_force_n", "entry_release_torque_nm", "entry_release_force_drop_n", "entry_release_torque_drop_nm", "entry_release_confirm_interval_s", "wiggle_entry_pre_fx_n", "wiggle_entry_pre_ty_nm", "wiggle_entry_fx_drop_n", "wiggle_entry_ty_drop_nm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    if not all(math.isfinite(value) for value in args.initial_right_joints_deg):
        parser.error("--initial-right-joints-deg 必须全部为有限数")
    if args.back_step_mm > 2.0 or args.wiggle_half_amplitude_mm > 2.0:
        parser.error("为安全起见，单步后拉和单侧摆幅均不得超过 2 mm")
    if not args.straight_line:
        args.continuous_straight_search = False
        if args.search_axis == "y":
            parser.error("Y 轴搜索不能与 --with-wiggle 同用；摆动同样占用 Y 轴")
    if args.single_continuous_search and not args.continuous_straight_search:
        parser.error("--single-continuous-search 需要 --continuous-straight-search")
    if args.single_continuous_search and args.continuous_force_monitor_hz <= 0.0:
        parser.error("--single-continuous-search 需要正的 --continuous-force-monitor-hz")
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
    if args.max_consecutive_back_target_timeouts < 1:
        parser.error("--max-consecutive-back-target-timeouts 必须至少为 1")
    if args.same_target_retries < 0:
        parser.error("--same-target-retries 不得为负数")
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
        "second_hole_lift_mm", "second_hole_y_offset_mm", "second_hole_descend_mm",
        "second_hole_lateral_speed_mm_s", "second_hole_x_return_speed_mm_s",
        "second_hole_descend_speed_mm_s",
        "pre_search_forward_mm", "pre_search_descend_mm", "pre_search_approach_speed_mm_s",
        "pre_search_initial_descend_mm",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
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


def vector(frame: dict, key: str) -> list[float]:
    values = frame.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError(f"无效力传感器帧 {key}: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"力传感器 {key} 存在非有限值")
    return result


def read_wrench(sensor: object, timeout: float) -> tuple[list[float], list[float]]:
    """Return the SDK subscriber's currently cached force frame."""
    frame = sensor.wait_first(timeout=timeout)  # type: ignore[attr-defined]
    return vector(frame, "filtered_force"), vector(frame, "filtered_torque")


def mean(rows: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(3)]


def norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("无法计算空采样集的中位数")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def command_for(target_rpy: list[float]) -> list[float]:
    return target_rpy[:3] + rpy_to_rotvec(target_rpy[3:])


class TargetNotReached(RuntimeError):
    """A commanded servo target remained outside its endpoint tolerance."""


class ServoWorkerTimeout(RuntimeError):
    """The isolated public-SDK ServoPose worker did not acknowledge in time."""


def _servo_worker_main(host: str, commands: object, replies: object) -> None:
    """Own one public SDK Arm client for one ServoPose session at a time."""
    arm = None
    try:
        from parm_sdk import Arm

        arm = Arm(host=host)
        while True:
            message = commands.recv()  # type: ignore[attr-defined]
            request_id = message["id"]
            operation = message["operation"]
            try:
                if operation == "start":
                    arm.start_servo_pose(**message["kwargs"])
                elif operation == "push":
                    arm.push_servo_pose(2, message["waypoint"])
                elif operation == "stop":
                    arm.stop_servo_pose(2)
                elif operation == "close":
                    replies.send({"id": request_id, "ok": True})  # type: ignore[attr-defined]
                    return
                else:
                    raise ValueError(f"unknown servo worker operation: {operation!r}")
                replies.send({"id": request_id, "ok": True})  # type: ignore[attr-defined]
            except Exception as exc:
                replies.send({  # type: ignore[attr-defined]
                    "id": request_id, "ok": False,
                    "exception_type": type(exc).__name__, "exception": str(exc),
                })
    except Exception as exc:
        # A start request waiting in the parent will time out if startup fails
        # before the first command can be consumed.
        try:
            replies.send({  # type: ignore[attr-defined]
                "id": None, "ok": False,
                "exception_type": type(exc).__name__, "exception": str(exc),
            })
        except Exception:
            pass
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass


class ServoWorker:
    """Parent-side RPC wrapper with a killable boundary around public SDK calls."""

    def __init__(self, host: str) -> None:
        self._context = mp.get_context("spawn")
        # A Pipe gives the parent a real ``poll(timeout)`` operation.  The
        # queue feeder thread used previously could itself wedge after the SDK
        # worker stalled, leaving the parent blocked before its watchdog ran.
        self._command_recv, self._command_send = self._context.Pipe(duplex=False)
        self._reply_recv, self._reply_send = self._context.Pipe(duplex=False)
        self._next_id = 0
        self._process = self._context.Process(
            target=_servo_worker_main,
            args=(host, self._command_recv, self._reply_send),
            daemon=True,
        )
        self._process.start()
        # The parent only sends commands and receives replies.  Closing its
        # duplicate child ends makes worker death visible as EOF/BrokenPipe.
        self._command_recv.close()
        self._reply_send.close()

    def _call(self, operation: str, *, timeout_s: float, **payload: object) -> None:
        if not self._process.is_alive():
            raise ServoWorkerTimeout(f"ServoPose worker exited before {operation}")
        self._next_id += 1
        request_id = self._next_id
        try:
            self._command_send.send({"id": request_id, "operation": operation, **payload})
            if not self._reply_recv.poll(timeout_s):
                raise TimeoutError
            reply = self._reply_recv.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.terminate()
            raise ServoWorkerTimeout(
                f"ServoPose worker connection failed during {operation}: {exc}"
            ) from exc
        except TimeoutError:
            self.terminate()
            raise ServoWorkerTimeout(
                f"ServoPose worker {operation} exceeded {timeout_s:.2f}s deadline"
            ) from None
        if reply.get("id") != request_id:
            self.terminate()
            raise ServoWorkerTimeout(
                f"ServoPose worker returned an unexpected reply during {operation}: {reply!r}"
            )
        if reply.get("ok"):
            return
        raise RuntimeError(
            f"ServoPose worker {operation} failed: "
            f"{reply.get('exception_type')}: {reply.get('exception')}"
        )

    def start(self, **kwargs: object) -> None:
        self._call("start", timeout_s=5.0, kwargs=kwargs)

    def push(self, waypoint: list[float]) -> None:
        self._call("push", timeout_s=0.5, waypoint=waypoint)

    def stop(self) -> None:
        self._call("stop", timeout_s=3.0)

    def terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=1.0)
        for connection in (
            self._command_send, self._reply_recv,
        ):
            try:
                connection.close()
            except OSError:
                pass

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._call("close", timeout_s=0.5)
            except Exception:
                pass
        self.terminate()


class ServoArmProxy:
    """Delegate normal Arm API calls while isolating public ServoPose calls."""

    def __init__(self, arm: object, host: str) -> None:
        self._arm = arm
        self._host = host
        self._servo_worker: ServoWorker | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self._arm, name)

    def start_servo_pose(self, arm_id: int, **kwargs: object) -> None:
        if arm_id != 2:
            raise ValueError("此放置流程的隔离 ServoPose 仅允许右臂 arm_id=2")
        self.dispose_servo_worker()
        worker = ServoWorker(self._host)
        try:
            worker.start(arm_id=2, **kwargs)
        except Exception:
            worker.close()
            raise
        self._servo_worker = worker

    def push_servo_pose(self, arm_id: int, waypoint: list[float]) -> None:
        if arm_id != 2 or self._servo_worker is None:
            raise RuntimeError("ServoPose push requires an active isolated right-arm session")
        self._servo_worker.push(waypoint)

    def stop_servo_pose(self, arm_id: int) -> None:
        if arm_id != 2:
            raise ValueError("此放置流程的隔离 ServoPose 仅允许右臂 arm_id=2")
        worker = self._servo_worker
        self._servo_worker = None
        if worker is None:
            return
        try:
            worker.stop()
        finally:
            worker.close()

    def dispose_servo_worker(self) -> None:
        worker = self._servo_worker
        self._servo_worker = None
        if worker is not None:
            worker.close()

    def close(self) -> None:
        self.dispose_servo_worker()
        self._arm.close()  # type: ignore[attr-defined]


def push_servo_pose_with_timeout(
    arm: object, waypoint: list[float], *, timeout_s: float = 0.5,
) -> None:
    """Validate and send one ServoPose frame through the public SDK API."""
    if len(waypoint) != 6 or not all(math.isfinite(value) for value in waypoint):
        raise ValueError(f"ServoPose target must contain six finite values: {waypoint!r}")
    # ``timeout_s`` is retained in the signature so callers keep one common
    # interface; the SDK owns its command transport and ServoPose semantics.
    _ = timeout_s
    arm.push_servo_pose(2, waypoint)  # type: ignore[attr-defined]


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
    output_file.flush()  # type: ignore[attr-defined]


def wait_for_right_task_stopped(
    arm: object,
    output_file: object | None,
    *,
    timeout_s: float = 3.0,
    confirmed_event: str = "right_task_stopped_confirmed",
    timeout_event: str = "right_task_stop_wait_timeout",
    cancel_if_still_running_after_s: float | None = None,
    event_logger: object | None = None,
) -> None:
    """Wait until the controller has actually retired the active right-arm task.

    ServoPose stop is asynchronous on the controller.  A short grace period
    is allowed first; callers starting a new ServoPose may request a
    cancel_task fallback so the old stream cannot overlap the new one.
    """
    deadline = time.monotonic() + timeout_s
    cancel_deadline = (
        time.monotonic() + cancel_if_still_running_after_s
        if cancel_if_still_running_after_s is not None else None
    )
    cancel_sent = False
    last_task_state = "unavailable"

    def record(event: str, task_state: str) -> None:
        if output_file is not None:
            write_control_snapshot(arm, output_file, event)
        if event_logger is not None:
            event_logger(event, task_state=task_state)  # type: ignore[operator]

    while time.monotonic() < deadline:
        state = arm.get_state() or {}  # type: ignore[attr-defined]
        task_state = str(
            (((state.get("state") or {}).get("right") or {}).get("task_state") or "")
        ).strip().lower()
        last_task_state = task_state or "unavailable"
        if task_state in {"completed", "cancelled", "idle", "stopped"}:
            record(confirmed_event, task_state)
            return
        if (
            cancel_deadline is not None
            and not cancel_sent
            and time.monotonic() >= cancel_deadline
        ):
            arm.cancel_task(2)  # type: ignore[attr-defined]
            cancel_sent = True
            record(confirmed_event + "_cancel_sent", task_state)
        time.sleep(0.05)
    record(timeout_event, last_task_state)
    raise RuntimeError(
        f"右臂任务未在 {timeout_s:.1f}s 内进入已停止状态；"
        f"最后 task_state={last_task_state!r}"
    )


def local_z_in_world(rpy: list[float]) -> list[float]:
    """Unit +Z of right_hand_transfer_end expressed in SDK world coordinates."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Third column of Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr]


def current_right_joints_deg(arm: object) -> list[float]:
    state = arm.get_state()  # type: ignore[attr-defined]
    values = ((state or {}).get("state") or {}).get("right", {}).get("position")
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise RuntimeError(f"右臂关节状态不可用: {values!r}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"右臂关节状态含非有限值: {values!r}")
    return result


def wait_for_right_joint_target(
    arm: object, target_deg: list[float], label: str, *, timeout_s: float = 90.0,
) -> None:
    """Wait until the planner task has ended and the physical arm is settled."""
    deadline = time.monotonic() + timeout_s
    last_error_deg = float("inf")
    last_velocity_deg_s = float("inf")
    last_task_state = "unknown"
    stable_samples = 0
    required_stable_samples = 20  # 1 second at the 20 Hz polling rate below.
    while time.monotonic() < deadline:
        state = arm.get_state()  # type: ignore[attr-defined]
        per_arm = ((state or {}).get("state") or {}).get("right") or {}
        actual = per_arm.get("position")
        velocity = per_arm.get("velocity")
        if not isinstance(actual, (list, tuple)) or len(actual) != 7:
            stable_samples = 0
            time.sleep(0.05)
            continue
        actual_deg = [float(value) for value in actual]
        # Compare wrapped angles so equivalent values around +/-180 deg do not fail.
        errors_deg = [
            abs((actual - target + 180.0) % 360.0 - 180.0)
            for actual, target in zip(actual_deg, target_deg)
        ]
        last_error_deg = max(errors_deg)
        if isinstance(velocity, (list, tuple)) and len(velocity) == 7:
            last_velocity_deg_s = max(abs(float(value)) for value in velocity)
        else:
            last_velocity_deg_s = float("inf")
        last_task_state = str(per_arm.get("task_state", "")).strip().lower()
        settled = (
            last_error_deg <= 0.5
            and last_velocity_deg_s <= 0.05
            and last_task_state != "running"
        )
        stable_samples = stable_samples + 1 if settled else 0
        if stable_samples >= required_stable_samples:
            print(
                f"{label}已物理到位并稳定1秒：最大关节误差={last_error_deg:.3f}°，"
                f"最大关节速度={last_velocity_deg_s:.4f}°/s，"
                f"task_state={last_task_state!r}。"
            )
            return
        time.sleep(0.05)
    raise TargetNotReached(
        f"{label}未在 {timeout_s:.1f}s 内物理到位并稳定："
        f"最大关节误差={last_error_deg:.3f}°，"
        f"最大关节速度={last_velocity_deg_s:.4f}°/s，"
        f"task_state={last_task_state!r}。"
    )


def move_to_saved_joints_servo(arm: object, target_deg: list[float]) -> None:
    """Continuously interpolate the recorded seven joint angles through servo_j."""
    start_deg = current_right_joints_deg(arm)
    largest_delta = max(abs(end - start) for start, end in zip(start_deg, target_deg))
    print(
        "关节 servo 回起点："
        f"当前={[round(value, 3) for value in start_deg]}，"
        f"目标={[round(value, 3) for value in target_deg]}，"
        f"最大关节差={largest_delta:.2f}°。"
    )
    if largest_delta <= 0.5:
        return
    hz, max_speed_deg_s = 50.0, 3.0
    duration = max(1.0 / hz, 1.875 * largest_delta / max_speed_deg_s)
    steps = max(1, math.ceil(duration * hz))
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        alpha = smoothstep5(index / float(steps))
        target = [start + alpha * (end - start) for start, end in zip(start_deg, target_deg)]
        arm.servo_j(2, target)  # type: ignore[attr-defined]
        deadline += 1.0 / hz
        time.sleep(max(0.0, deadline - time.monotonic()))
    settle_deadline = time.monotonic() + 8.0
    final = start_deg
    while time.monotonic() < settle_deadline:
        arm.servo_j(2, target_deg)  # type: ignore[attr-defined]
        final = current_right_joints_deg(arm)
        if max(abs(end - actual) for end, actual in zip(target_deg, final)) <= 0.5:
            print("关节 servo 已到保存起点。")
            return
        time.sleep(1.0 / hz)
    raise TargetNotReached(
        "关节 servo 未在超时内到位："
        f"最大关节误差={max(abs(end - actual) for end, actual in zip(target_deg, final)):.2f}°。"
    )


def move_to_saved_joints_planned(
    arm: object, config_path: Path, target_deg: list[float], *, planner_name: str,
) -> None:
    """Plan against the configured scene (including the table), then execute once."""
    config = load_config(config_path, live=True)
    planner = PlannerRuntime(config)
    try:
        planner.open(required_frames={config.base_frame, config.right_tool_frame})
        print(
            f"Planner {planner_name.upper()} 回起点：先使用当前规划场景做右臂碰撞检查和绕障规划，"
            f"目标={[round(value, 3) for value in target_deg]}。"
        )
        planner.move_right_joints(target_deg, planner=planner_name)
        print("Planner move_j 已下发，等待右臂物理到固定关节起点。")
        wait_for_right_joint_target(arm, target_deg, "固定关节起点")
    finally:
        planner.close()


def move_in_active_session(
    arm: object,
    target_rpy: list[float],
    *,
    initial: bool = False,
    position_tolerance_m: float | None = None,
    linear_speed_mps: float | None = None,
    linear_accel_mps2: float | None = None,
    settle_timeout_s: float | None = None,
    diagnostic: object | None = None,
) -> None:
    """Move within the caller-owned continuous servo-pose session."""
    target = command_for(target_rpy)
    start = current_pose(arm, 2)
    distance = position_error_m(target, start)
    angle = orientation_error_rad(target, start)
    if diagnostic is not None:
        diagnostic("servo_move_begin")  # type: ignore[operator]
    position_tolerance = position_tolerance_m if position_tolerance_m is not None else (0.0005 if initial else 0.0003)
    orientation_tolerance = 0.02 if initial else 0.01
    if distance <= position_tolerance and angle <= orientation_tolerance:
        return
    hz = 50.0
    period = 1.0 / hz
    speed = 0.01 if initial else (linear_speed_mps if linear_speed_mps is not None else 0.003)
    accel = 0.02 if initial else (linear_accel_mps2 if linear_accel_mps2 is not None else 0.01)
    linear_duration = minimum_scurve_duration(
        distance, speed, accel
    )
    angular_duration = minimum_scurve_duration(
        angle, 0.10 if initial else 0.05, 0.20 if initial else 0.10
    )
    steps = max(1, math.ceil(max(linear_duration, angular_duration, period) * hz))
    q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        alpha = smoothstep5(index / float(steps))
        waypoint = [start[axis] + alpha * (target[axis] - start[axis]) for axis in range(3)]
        waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
        push_servo_pose_with_timeout(arm, waypoint)
        if diagnostic is not None and (index == 1 or index % 10 == 0 or index == steps):
            diagnostic(f"servo_push_waypoint_{index}_of_{steps}")  # type: ignore[operator]
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))
    settle_deadline = time.monotonic() + (
        settle_timeout_s if settle_timeout_s is not None else (8.0 if initial else 4.0)
    )
    final_pose = start
    settle_push_count = 0
    while time.monotonic() < settle_deadline:
        settle_push_count += 1
        push_servo_pose_with_timeout(arm, target)
        final_pose = current_pose(arm, 2)
        if (
            position_error_m(target, final_pose) <= position_tolerance
            and orientation_error_rad(target, final_pose) <= orientation_tolerance
        ):
            if diagnostic is not None:
                diagnostic("servo_target_reached")  # type: ignore[operator]
            return
        if diagnostic is not None and settle_push_count % 10 == 0:
            diagnostic(f"servo_settle_push_{settle_push_count}")  # type: ignore[operator]
        time.sleep(period)
    if diagnostic is not None:
        diagnostic("servo_target_timeout")  # type: ignore[operator]
    raise TargetNotReached(
        "末端未在超时内到位："
        f"位置误差={position_error_m(target, final_pose) * 1000.0:.1f}mm，"
        f"姿态误差={math.degrees(orientation_error_rad(target, final_pose)):.2f}°。"
    )


def move_continuous_linear_step(
    arm: object,
    command_start_rpy: list[float],
    target_rpy: list[float],
    *,
    linear_speed_mps: float,
    linear_accel_mps2: float,
    initial_speed_mps: float,
    force_monitor: object | None = None,
    push_diagnostic: object | None = None,
    decelerate_to_stop: bool = False,
) -> float:
    """Stream one search increment at constant command velocity.

    This intentionally performs no endpoint settle/readback.  Consecutive
    calls share their command endpoint, so the controller receives a single
    linear stream instead of a sequence of one-millimetre S-curves.  The
    caller samples only after the stream has crossed each 1-mm checkpoint.
    Acceleration is carried across checkpoints, so a low configured
    acceleration produces one smooth ramp rather than 1-mm stop/start ramps.
    ``decelerate_to_stop`` is for a standalone approach leg: it uses a
    trapezoidal (or triangular) profile and reaches zero command velocity at
    the endpoint.
    """
    start = command_for(command_start_rpy)
    target = command_for(target_rpy)
    distance = position_error_m(target, start)
    if distance <= 0.0:
        return initial_speed_mps
    hz = 50.0
    period = 1.0 / hz
    speed0 = min(max(initial_speed_mps, 0.0), linear_speed_mps)
    if decelerate_to_stop:
        # Each side of a full-speed ramp consumes v²/(2a).  If the leg is
        # shorter, use a triangular profile whose peak exactly fits distance.
        full_ramp_distance = linear_speed_mps * linear_speed_mps / linear_accel_mps2
        speed1 = (
            linear_speed_mps
            if distance >= full_ramp_distance
            else math.sqrt(distance * linear_accel_mps2)
        )
        accel_time = speed1 / linear_accel_mps2
        accel_distance = speed1 * accel_time / 2.0
        cruise_distance = max(0.0, distance - 2.0 * accel_distance)
        cruise_time = cruise_distance / speed1 if speed1 > 0.0 else 0.0
        duration = 2.0 * accel_time + cruise_time
    else:
        speed1 = min(linear_speed_mps, math.sqrt(speed0 * speed0 + 2.0 * linear_accel_mps2 * distance))
        accel_time = (speed1 - speed0) / linear_accel_mps2
        accel_distance = (speed0 + speed1) * accel_time / 2.0
        cruise_distance = max(0.0, distance - accel_distance)
        duration = accel_time + cruise_distance / linear_speed_mps
    steps = max(1, math.ceil(duration * hz))
    q0, q1 = rotvec_to_quat(start[3:]), rotvec_to_quat(target[3:])
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        elapsed = min(index * duration / steps, duration)
        if decelerate_to_stop:
            if elapsed <= accel_time:
                travelled = linear_accel_mps2 * elapsed * elapsed / 2.0
            elif elapsed <= accel_time + cruise_time:
                travelled = accel_distance + speed1 * (elapsed - accel_time)
            else:
                decel_elapsed = elapsed - accel_time - cruise_time
                travelled = (
                    accel_distance + cruise_distance
                    + speed1 * decel_elapsed
                    - linear_accel_mps2 * decel_elapsed * decel_elapsed / 2.0
                )
        elif elapsed <= accel_time:
            travelled = speed0 * elapsed + linear_accel_mps2 * elapsed * elapsed / 2.0
        else:
            travelled = accel_distance + linear_speed_mps * (elapsed - accel_time)
        alpha = min(1.0, travelled / distance)
        waypoint = [start[axis] + alpha * (target[axis] - start[axis]) for axis in range(3)]
        waypoint += quat_to_rotvec(slerp(q0, q1, alpha))
        if push_diagnostic is not None:
            push_diagnostic(  # type: ignore[operator]
                "servo_push_before", waypoint_index=index, waypoint_count=steps,
                waypoint_xyz_rotvec=waypoint,
            )
        try:
            push_servo_pose_with_timeout(arm, waypoint)
        except Exception as exc:
            if push_diagnostic is not None:
                push_diagnostic(  # type: ignore[operator]
                    "servo_push_failed", waypoint_index=index, waypoint_count=steps,
                    waypoint_xyz_rotvec=waypoint,
                    exception_type=type(exc).__name__, exception=str(exc),
                )
            raise
        if force_monitor is not None:
            force_monitor(alpha)  # type: ignore[operator]
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))
    return 0.0 if decelerate_to_stop else speed1


def move_wiggle_quick(
    arm: object,
    target_rpy: list[float],
    *,
    linear_speed_mps: float,
    linear_accel_mps2: float,
    endpoint_tolerance_m: float,
    endpoint_wait_s: float,
    diagnostic: object | None = None,
) -> None:
    """Stream one fast search leg, confirming its endpoint before sampling.

    The servo-pose session is never restarted.  Unlike the standard leg, this
    waits only a short time for a 0.5-mm endpoint confirmation.  If the arm is
    still outside that envelope, it falls back to the original 0.3-mm / 4-s
    confirmation before force is read; a quick run therefore never samples an
    unconfirmed endpoint merely to save time.
    """
    try:
        move_in_active_session(
            arm, target_rpy,
            position_tolerance_m=endpoint_tolerance_m,
            linear_speed_mps=linear_speed_mps,
            linear_accel_mps2=linear_accel_mps2,
            settle_timeout_s=endpoint_wait_s,
            diagnostic=diagnostic,
        )
    except TargetNotReached:
        print("快速运动段未在短确认窗口到位；退回完整到位确认后再采样。")
        move_in_active_session(
            arm, target_rpy,
            linear_speed_mps=linear_speed_mps,
            linear_accel_mps2=linear_accel_mps2,
            diagnostic=diagnostic,
        )


def placement_main(argv: list[str] | None = None) -> int:
    run_started = time.monotonic()
    args = parse_args(argv)
    # Recorded UI joint values are the sole taught start definition.
    saved_joints_deg = list(args.initial_right_joints_deg)
    if not args.live:
        print(
            "dry-run：不会连接硬件；固定右臂起始关节角="
            f"{saved_joints_deg}，初始回位方式={args.initial_motion}"
        )
        if args.straight_line:
            if args.single_continuous_search:
                print(
                    f"单段连续直线：沿世界 {args.search_axis.upper()}"
                    f"{'+' if args.search_direction == 'positive' else '−'} 从 0 连续移动至 "
                    f"{args.max_back_mm:.1f}mm；速度 {args.fast_linear_speed_mm_s:.1f}mm/s；"
                    f"仅以 {args.continuous_force_monitor_hz:.1f}Hz（每 "
                    f"{1000.0 / args.continuous_force_monitor_hz:.0f}ms）监测力/扭矩。"
                )
            else:
                print(
                    f"纯直线：每周期仅沿世界 {args.search_axis.upper()}{'+' if args.search_direction == 'positive' else '−'} "
                    f"移动 {args.back_step_mm:.3f}mm 并在中心采样；"
                    f"{args.slow_after_back_mm:.1f}mm 前 {args.fast_linear_speed_mm_s:.1f}mm/s，之后 "
                    f"{args.slow_linear_speed_mm_s:.1f}mm/s；最大后拉 {args.max_back_mm:.1f}mm。"
                )
        else:
            print(
                f"每周期：后拉中心后 Y=中心±{args.wiggle_half_amplitude_mm:.3f}mm；"
                f"Y- 到下一后拉中心合并为对角线（X -= {args.back_step_mm:.3f}mm 且 Y 回中心）；"
                f"{args.slow_after_back_mm:.1f}mm 前 {args.fast_linear_speed_mm_s:.1f}mm/s，之后 "
                f"{args.slow_linear_speed_mm_s:.1f}mm/s；最大后拉 {args.max_back_mm:.1f}mm。"
            )
        return 0

    output_path = args.output or Path("logs") / f"hole_a_wiggle_{datetime.now():%Y%m%d_%H%M%S}.log"
    output_path = output_path.expanduser().resolve().with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"运行日志将实时写入：{output_path}", flush=True)
    sdk = import_sdk()
    arm = sensor = None
    servo_started = False
    continuous_servo_stopped = False
    servo_guard = None
    search_started_at: float | None = None
    log_file = output_path.open("w", encoding="utf-8")

    def log_event(event: str, **fields: object) -> None:
        record = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.monotonic() - run_started, 3),
            "event": event,
            **fields,
        }
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_file.flush()

    def is_servo_connection_loss(exc: Exception) -> bool:
        """Only recover the transient command-port loss, never motion/contact errors."""
        message = str(exc)
        return (
            isinstance(exc, ServoWorkerTimeout)
            or
            ("tcp://" in message and ":5600 unavailable" in message)
            or message == "timeout waiting for response"
            or message.startswith("socket error:")
        )

    def reconnect_servo_arm(label: str, recovery_index: int) -> None:
        """Replace the ZMQ client after its command port disappeared."""
        nonlocal arm
        assert arm is not None
        # A killed worker only drops its client connection; it does not prove
        # the controller retired the previous ServoPose task.  Issue an
        # explicit public-SDK stop from a fresh worker and observe task end
        # before a recovery session is ever allowed to start.
        if isinstance(arm, ServoArmProxy):
            arm.dispose_servo_worker()
        stop_worker = ServoWorker(args.arm_host)
        try:
            log_event(
                "servo_recovery_stop_stale_session_sent", label=label,
                recovery_index=recovery_index,
            )
            stop_worker.stop()
        except Exception as stop_exc:
            log_event(
                "servo_recovery_stop_stale_session_error", label=label,
                recovery_index=recovery_index,
                exception_type=type(stop_exc).__name__, exception=str(stop_exc),
            )
            raise RuntimeError(
                f"{label} 恢复前无法停止旧 ServoPose 会话：{stop_exc}"
            ) from stop_exc
        finally:
            stop_worker.close()
        wait_for_right_task_stopped(
            arm, log_file,
            confirmed_event="servo_recovery_stale_session_stopped",
            timeout_event="servo_recovery_stale_session_stop_timeout",
            cancel_if_still_running_after_s=0.5,
            event_logger=log_event,
        )
        deadline = time.monotonic() + 10.0
        last_error: Exception | None = None
        try:
            arm.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        while time.monotonic() < deadline:
            candidate = None
            try:
                candidate = sdk.Arm(host=args.arm_host)
                ensure_ready(candidate, 2)
                arm = ServoArmProxy(candidate, args.arm_host)
                log_event(
                    "servo_reconnected", label=label,
                    recovery_index=recovery_index,
                )
                return
            except Exception as exc:
                last_error = exc
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                time.sleep(0.25)
        raise RuntimeError(
            f"{label} 的 ServoPose 控制端在 10 秒内未恢复：{last_error}"
        )

    def hold_servo_target_until_settled(
        label: str, target_rpy: list[float], *, timeout_s: float = 15.0,
        position_tolerance_m: float = 0.001,
        orientation_tolerance_rad: float = math.radians(1.0),
        stable_samples_required: int = 10,
    ) -> None:
        """Keep streaming an endpoint until the physical TCP is stably there."""
        assert arm is not None
        target = command_for(target_rpy)
        deadline = time.monotonic() + timeout_s
        stable_samples = 0
        last_position_error_m = float("inf")
        last_orientation_error_rad = float("inf")
        while time.monotonic() < deadline:
            push_servo_pose_with_timeout(arm, target)
            actual = current_pose(arm, 2)
            last_position_error_m = position_error_m(target, actual)
            last_orientation_error_rad = orientation_error_rad(target, actual)
            settled = (
                last_position_error_m <= position_tolerance_m
                and last_orientation_error_rad <= orientation_tolerance_rad
            )
            stable_samples = stable_samples + 1 if settled else 0
            if stable_samples >= stable_samples_required:
                log_event(
                    "servo_endpoint_physically_confirmed", label=label,
                    position_error_mm=round(last_position_error_m * 1000.0, 3),
                    orientation_error_deg=round(math.degrees(last_orientation_error_rad), 3),
                    stable_samples=stable_samples,
                )
                return
            time.sleep(0.02)
        raise TargetNotReached(
            f"{label} 未在 {timeout_s:.1f}s 内物理到位："
            f"位置误差={last_position_error_m * 1000.0:.1f}mm，"
            f"姿态误差={math.degrees(last_orientation_error_rad):.2f}°。"
        )

    def move_continuous_with_recovery(
        label: str,
        start_rpy: list[float],
        target_rpy: list[float],
        *,
        linear_speed_mps: float,
        linear_accel_mps2: float,
        initial_speed_mps: float,
        force_monitor: object | None = None,
        push_diagnostic: object | None = None,
        near_target_mm: float = 1.0,
        recovery_speed_mps: float | None = None,
        decelerate_to_stop: bool = False,
        confirm_physical_endpoint: bool = False,
    ) -> float:
        """Finish one absolute world-frame leg across transient 5600 disconnects.

        A resumed stream always begins from the newly measured TCP pose, never
        from the last command waypoint.  This prevents a stale command from
        producing a jump after the controller reconnects.
        """
        nonlocal arm, continuous_servo_stopped, servo_started
        assert arm is not None
        segment_start = list(start_rpy)
        speed0 = initial_speed_mps
        full_distance_m = position_error_m(
            command_for(target_rpy), command_for(start_rpy),
        )
        for recovery_index in range(MAX_SERVO_RECOVERY_ATTEMPTS + 1):
            try:
                remaining_before_m = position_error_m(
                    command_for(target_rpy), command_for(segment_start),
                )
                completed_fraction = (
                    min(1.0, max(0.0, 1.0 - remaining_before_m / full_distance_m))
                    if full_distance_m > 1e-9 else 1.0
                )
                segment_force_monitor = None
                if force_monitor is not None:
                    segment_force_monitor = lambda alpha: force_monitor(  # type: ignore[operator]
                        completed_fraction + alpha * (1.0 - completed_fraction)
                    )
                final_speed = move_continuous_linear_step(
                    arm, segment_start, target_rpy,
                    linear_speed_mps=(
                        recovery_speed_mps
                        if recovery_index and recovery_speed_mps is not None
                        else linear_speed_mps
                    ),
                    linear_accel_mps2=linear_accel_mps2,
                    initial_speed_mps=speed0,
                    force_monitor=segment_force_monitor,
                    push_diagnostic=push_diagnostic,
                    decelerate_to_stop=decelerate_to_stop,
                )
                if confirm_physical_endpoint:
                    hold_servo_target_until_settled(label, target_rpy)
                return final_speed
            except Exception as exc:
                if (
                    not is_servo_connection_loss(exc)
                    or recovery_index >= MAX_SERVO_RECOVERY_ATTEMPTS
                ):
                    raise
                log_event(
                    "servo_disconnect", label=label,
                    recovery_index=recovery_index + 1,
                    exception_type=type(exc).__name__, exception=str(exc),
                    last_command_start_xyz_rpy=segment_start,
                    target_xyz_rpy=target_rpy,
                )
                print(
                    f"{label} ServoPose 连接短暂断开；读取实际位姿后恢复剩余路径 "
                    f"({recovery_index + 1}/{MAX_SERVO_RECOVERY_ATTEMPTS})。",
                    flush=True,
                )
                reconnect_servo_arm(label, recovery_index + 1)
                actual = current_pose(arm, 2)
                actual_rpy = actual[:3] + rotvec_to_rpy(actual[3:])
                remaining_mm = position_error_m(command_for(target_rpy), actual) * 1000.0
                log_event(
                    "servo_recovery_pose", label=label,
                    recovery_index=recovery_index + 1,
                    actual_xyz_rpy=actual_rpy, target_xyz_rpy=target_rpy,
                    remaining_mm=round(remaining_mm, 3),
                )
                arm.start_servo_pose(  # type: ignore[attr-defined]
                    2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                    input_frequency_hz=50.0,
                )
                servo_started = True
                continuous_servo_stopped = False
                if remaining_mm <= near_target_mm:
                    log_event(
                        "servo_resume_skipped_near_target", label=label,
                        recovery_index=recovery_index + 1,
                        remaining_mm=round(remaining_mm, 3),
                        threshold_mm=near_target_mm,
                    )
                    if confirm_physical_endpoint:
                        hold_servo_target_until_settled(label, target_rpy)
                    return 0.0
                segment_start = actual_rpy
                speed0 = 0.0
                log_event(
                    "servo_resume", label=label, recovery_index=recovery_index + 1,
                    restart_xyz_rpy=segment_start, target_xyz_rpy=target_rpy,
                    remaining_mm=round(remaining_mm, 3),
                    speed_mm_s=round((recovery_speed_mps or linear_speed_mps) * 1000.0, 3),
                )
        raise AssertionError("unreachable")

    log_event(
        "run_started",
        initial_motion=args.initial_motion,
        straight_line=args.straight_line,
        search_axis=args.search_axis,
        search_direction=args.search_direction,
        back_step_mm=args.back_step_mm,
        wiggle_half_amplitude_mm=args.wiggle_half_amplitude_mm,
        max_back_mm=args.max_back_mm,
        quick_endpoint_tolerance_mm=args.quick_endpoint_tolerance_mm,
        quick_endpoint_wait_s=args.quick_endpoint_wait_s,
    )
    try:
        raw_arm = sdk.Arm(host=args.arm_host)
        ensure_ready(raw_arm, 2)
        arm = ServoArmProxy(raw_arm, args.arm_host)
        write_control_snapshot(arm, log_file, "script_connected")
        sensor = sdk.ForceSensor(host=args.force_host, hand=2)
        if args.reset_right_task_before_servo:
            print(
                "清理右臂残留任务后再规划回起点。",
                flush=True,
            )
            arm.cancel_task(2)  # type: ignore[attr-defined]
            wait_for_right_task_stopped(
                arm, log_file,
                confirmed_event="right_cancel_confirmed",
                timeout_event="right_cancel_wait_timeout",
            )
        print("正在移动到保存的起始位姿……")
        if args.initial_motion == "planner-movej":
            move_to_saved_joints_planned(
                arm, args.planner_config, saved_joints_deg,
                planner_name=args.initial_joint_planner,
            )
        else:
            move_to_saved_joints_servo(arm, saved_joints_deg)
        # The taught joint pose is a safe landmark.  Start the legacy Hole-A
        # search only after the requested world-frame approach from that pose.
        def pre_search_approach_leg(
            label: str, axis: int, distance_mm: float,
        ) -> None:
            actual = current_pose(arm, 2)
            start = actual[:3] + rotvec_to_rpy(actual[3:])
            target = list(start)
            target[axis] += distance_mm / 1000.0
            axis_name = "XYZ"[axis]
            direction = "+" if distance_mm > 0.0 else "−"
            print(
                f"搜索前接近 {label}：沿世界 {axis_name}{direction} "
                f"{abs(distance_mm):.1f}mm，"
                f"{args.pre_search_approach_speed_mm_s:.1f}mm/s。"
            )
            log_event(
                "pre_search_approach_leg_start", label=label, axis=axis_name,
                distance_mm=distance_mm,
                speed_mm_s=args.pre_search_approach_speed_mm_s,
                start_xyz_rpy=start, target_xyz_rpy=target,
                force_monitoring_enabled=False,
            )
            def push_diagnostic(event: str, **fields: object) -> None:
                """Persist every approach push and the exact failed waypoint."""
                log_event(
                    f"pre_search_{event}", label=label, axis=axis_name,
                    **fields,
                )
                if event == "servo_push_failed":
                    try:
                        write_control_snapshot(
                            arm, log_file,
                            f"pre_search_{label}_servo_push_failed",
                        )
                    except Exception as snapshot_exc:
                        log_event(
                            "pre_search_failure_snapshot_failed", label=label,
                            exception_type=type(snapshot_exc).__name__,
                            exception=str(snapshot_exc),
                        )

            move_continuous_with_recovery(
                f"pre_search_{label}", start, target,
                linear_speed_mps=args.pre_search_approach_speed_mm_s / 1000.0,
                linear_accel_mps2=0.02, initial_speed_mps=0.0,
                push_diagnostic=push_diagnostic,
                decelerate_to_stop=True,
                confirm_physical_endpoint=True,
            )
            finished = current_pose(arm, 2)
            log_event(
                "pre_search_approach_leg_completed", label=label,
                actual_end_xyz_rpy=finished[:3] + rotvec_to_rpy(finished[3:]),
            )

        servo_guard = right_servo_guard()
        servo_guard.__enter__()
        try:
            log_event("pre_search_servo_session_starting")
            arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                input_frequency_hz=50.0,
            )
            log_event("pre_search_servo_session_started")
            pre_search_approach_leg(
                "initial_descend", 2, -args.pre_search_initial_descend_mm
            )
            pre_search_approach_leg("forward", 0, args.pre_search_forward_mm)
            pre_search_approach_leg("descend", 2, -args.pre_search_descend_mm)
        except Exception as exc:
            log_event(
                "pre_search_servo_session_error",
                exception_type=type(exc).__name__, exception=str(exc),
            )
            try:
                write_control_snapshot(arm, log_file, "pre_search_servo_session_error")
            except Exception as snapshot_exc:
                log_event(
                    "pre_search_failure_snapshot_failed",
                    exception_type=type(snapshot_exc).__name__, exception=str(snapshot_exc),
                )
            raise
        finally:
            primary_error_active = sys.exc_info()[0] is not None
            try:
                arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                log_event("pre_search_servo_session_stop_sent")
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_pre_search_approach_stop_confirmed",
                    timeout_event="right_pre_search_approach_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                    event_logger=log_event,
                )
            except Exception as stop_exc:
                log_event(
                    "pre_search_servo_session_stop_error",
                    exception_type=type(stop_exc).__name__, exception=str(stop_exc),
                )
                if not primary_error_active:
                    raise
            servo_guard.__exit__(None, None, None)
            servo_guard = None
        # The taught pose is the recovery landmark, not the force-baseline
        # pose.  Every retry returns here, then presses the configured extra
        # distance deeper than the preceding attempt before rebuilding the
        # baseline at that new search start.
        initial_pose = current_pose(arm, 2)
        press_start = initial_pose[:3] + rotvec_to_rpy(initial_pose[3:])
        press_depth_mm = (
            args.pre_search_press_mm
            + (args.search_attempt - 1) * args.retry_search_press_increment_mm
        )
        press_increment_mm = (
            args.pre_search_press_mm
            if args.search_attempt == 1 else args.retry_search_press_increment_mm
        )
        press_target = list(press_start)
        press_target[2] -= press_depth_mm / 1000.0
        print(
            "已到放置初始位姿；沿世界 Z− LIN 下压 "
            f"{press_depth_mm:.3f}mm（第 {args.search_attempt} 轮累计深度）后再采集力基线。"
        )
        log_event(
            "pre_search_press_start",
            attempt=args.search_attempt,
            distance_mm=press_depth_mm,
            incremental_depth_mm=press_increment_mm,
            axis="z",
            direction="negative",
            start_xyz_rpy=press_start,
            target_xyz_rpy=press_target,
        )
        servo_guard = right_servo_guard()
        servo_guard.__enter__()
        try:
            arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                input_frequency_hz=50.0,
            )
            servo_started = True
            continuous_servo_stopped = False
            move_in_active_session(
                arm,
                press_target,
                linear_speed_mps=args.pre_search_press_speed_mm_s / 1000.0,
                linear_accel_mps2=0.01,
            )
        finally:
            if servo_started and not continuous_servo_stopped:
                arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_pre_search_press_stop_confirmed",
                    timeout_event="right_pre_search_press_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                )
                continuous_servo_stopped = True
            servo_started = False
            servo_guard.__exit__(None, None, None)
            servo_guard = None
        log_event(
            "pre_search_press_completed",
            attempt=args.search_attempt,
            distance_mm=press_depth_mm,
            incremental_depth_mm=press_increment_mm,
        )
        # Contact limits must be relative to the actual red-dot start state,
        # not the state before this script moved the arm to that point.
        print("下压完成；请勿接触工件，正在稳定并采集右腕力基线……")
        time.sleep(0.5)
        baseline_samples = []
        for _ in range(args.baseline_samples):
            baseline_samples.append(read_wrench(sensor, args.sample_timeout))
            time.sleep(0.05)
        force_base = mean([force for force, _ in baseline_samples])
        torque_base = mean([torque for _, torque in baseline_samples])
        print(f"起始位姿力基线 F={force_base} N，T={torque_base} N*m")
        log_event("force_baseline", force_n=force_base, torque_nm=torque_base)

        # Read the servo centre only after the loaded arm has settled during
        # baseline collection.  Reading it before this delay leaves a stale
        # target when the held workpiece causes sub-millimetre elastic settling.
        initial_actual = current_pose(arm, 2)
        centre = initial_actual[:3] + rotvec_to_rpy(initial_actual[3:])
        print(
            "力基线采集完成；以后拉中心使用稳定后的实际末端 xyz+rpy="
            f"{[round(value, 6) for value in centre]}。"
        )
        search_axis_index = 0 if args.search_axis == "x" else 1
        search_direction_sign = 1.0 if args.search_direction == "positive" else -1.0
        search_motion_name = (
            f"world_{args.search_axis}_{args.search_direction}_search"
        )
        start_search_coordinate = centre[search_axis_index]
        log_event("search_centre_ready", pose_xyz_rpy=centre)

        def post_entry_relief_if_needed() -> None:
            """After insertion, back off, then descend under raw-Fx monitoring."""
            nonlocal continuous_servo_stopped
            # The original search stream has already been stopped on the
            # continuous-monitor success paths.  If a non-continuous success
            # reached this handler, stop it before making the post-entry moves.
            if servo_started:
                if not continuous_servo_stopped:
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    continuous_servo_stopped = True
                # A high-rate force detector may already have issued stop;
                # in either case do not start the relief stream until the
                # controller has retired that old ServoPose task.
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_pre_relief_stop_confirmed",
                    timeout_event="right_pre_relief_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                )
            if args.post_entry_relief_mm > 0.0:
                actual = current_pose(arm, 2)
                relief_start = actual[:3] + rotvec_to_rpy(actual[3:])
                relief_target = list(relief_start)
                # Always move opposite to the configured search direction.
                relief_target[search_axis_index] -= (
                    search_direction_sign * args.post_entry_relief_mm / 1000.0
                )
                print(
                    f"进洞确认后，沿世界 {args.search_axis.upper()}"
                    f"{'−' if search_direction_sign > 0.0 else '+'} 固定后拉 "
                    f"{args.post_entry_relief_mm:.3f}mm。"
                )
                arm.start_servo_pose(  # type: ignore[attr-defined]
                    2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0
                )
                continuous_servo_stopped = False
                try:
                    move_continuous_with_recovery(
                        "post_entry_relief", relief_start, relief_target,
                        linear_speed_mps=args.post_entry_relief_speed_mm_s / 1000.0,
                        linear_accel_mps2=0.02, initial_speed_mps=0.0,
                    )
                finally:
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    wait_for_right_task_stopped(
                        arm, log_file,
                        confirmed_event="right_post_entry_relief_stop_confirmed",
                        timeout_event="right_post_entry_relief_stop_wait_timeout",
                        cancel_if_still_running_after_s=0.5,
                    )
                    continuous_servo_stopped = True
                log_event("post_entry_relief_completed", distance_mm=args.post_entry_relief_mm)

            if args.post_entry_descend_mm <= 0.0:
                return
            descend_start_raw = current_pose(arm, 2)
            descend_start = descend_start_raw[:3] + rotvec_to_rpy(descend_start_raw[3:])
            descend_target = list(descend_start)
            descend_target[2] -= args.post_entry_descend_mm / 1000.0
            force_monitoring_enabled = args.post_entry_descend_fx_rise_n > 0.0
            descend_fx_base: float | None = None
            if force_monitoring_enabled:
                time.sleep(0.25)
                descend_fx_samples: list[float] = []
                for index in range(args.baseline_samples):
                    descend_fx_samples.append(read_wrench(sensor, args.sample_timeout)[0][0])
                    if index + 1 < args.baseline_samples:
                        time.sleep(0.05)
                descend_fx_base = sum(descend_fx_samples) / len(descend_fx_samples)
            print(
                f"进洞确认后，直接沿世界 Z− 下降 {args.post_entry_descend_mm:.3f}mm；"
                + (
                    f"raw Fx 上升 {args.post_entry_descend_fx_rise_n:.2f}N 即停止。"
                    if force_monitoring_enabled else "不读取力传感器。"
                )
            )
            log_event(
                "post_entry_descend_start",
                max_distance_mm=args.post_entry_descend_mm,
                raw_fx_monitoring_enabled=force_monitoring_enabled,
                raw_fx_baseline_n=descend_fx_base,
                raw_fx_rise_threshold_n=args.post_entry_descend_fx_rise_n if force_monitoring_enabled else None,
                start_xyz_rpy=descend_start,
                target_xyz_rpy=descend_target,
            )
            last_descend_output_at = 0.0

            def monitor_descend_raw_fx(alpha: float) -> None:
                nonlocal continuous_servo_stopped, last_descend_output_at
                assert descend_fx_base is not None
                force, _ = read_wrench(sensor, args.sample_timeout)
                delta_fx = force[0] - descend_fx_base
                descended_mm = alpha * args.post_entry_descend_mm
                now = time.monotonic()
                if now - last_descend_output_at >= 0.1:
                    last_descend_output_at = now
                    print(f"[post_entry_descend] down≈{descended_mm:.2f}mm raw dFx={delta_fx:.2f}N")
                if delta_fx >= args.post_entry_descend_fx_rise_n:
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    continuous_servo_stopped = True
                    log_event(
                        "post_entry_descend_fx_stop",
                        descended_mm=round(descended_mm, 3), raw_fx_n=force[0],
                        raw_fx_baseline_n=descend_fx_base, raw_delta_fx_n=delta_fx,
                        threshold_n=args.post_entry_descend_fx_rise_n,
                    )
                    raise PostEntryFxRiseReached()

            arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0
            )
            continuous_servo_stopped = False
            stopped_by_fx = False
            try:
                move_continuous_with_recovery(
                    "post_entry_descend", descend_start, descend_target,
                    linear_speed_mps=args.post_entry_descend_speed_mm_s / 1000.0,
                    linear_accel_mps2=0.02, initial_speed_mps=0.0,
                    force_monitor=monitor_descend_raw_fx if force_monitoring_enabled else None,
                )
            except PostEntryFxRiseReached:
                stopped_by_fx = True
                print("进洞后下降：raw Fx 上升达到阈值，已停止。")
            finally:
                arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_post_entry_descend_stop_confirmed",
                    timeout_event="right_post_entry_descend_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                )
                continuous_servo_stopped = True
            log_event(
                "post_entry_descend_completed", max_distance_mm=args.post_entry_descend_mm,
                stopped_by_raw_fx_rise=stopped_by_fx,
            )

        def move_from_first_hole_to_second_hole() -> None:
            """Move from the confirmed first hole to the taught second-hole offset.

            Each leg starts from the measured TCP pose, intentionally has no
            force read and does not assert endpoint arrival: a contact-limited
            final Z- descent therefore ends quietly rather than failing solely
            because the full 50-mm command was not physically reached.
            """
            nonlocal continuous_servo_stopped
            if servo_started:
                if not continuous_servo_stopped:
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    continuous_servo_stopped = True
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_pre_second_hole_stop_confirmed",
                    timeout_event="right_pre_second_hole_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                )

            def run_leg(label: str, axis: int, distance_mm: float, speed_mm_s: float) -> None:
                nonlocal continuous_servo_stopped
                actual = current_pose(arm, 2)
                start = actual[:3] + rotvec_to_rpy(actual[3:])
                target = list(start)
                target[axis] += distance_mm / 1000.0
                axis_name = "XYZ"[axis]
                direction = "+" if distance_mm > 0.0 else "−"
                print(
                    f"第二洞路径 {label}：沿世界 {axis_name}{direction} "
                    f"{abs(distance_mm):.1f}mm，{speed_mm_s:.1f}mm/s。"
                )
                log_event(
                    "second_hole_leg_start", label=label, axis=axis_name,
                    distance_mm=distance_mm, speed_mm_s=speed_mm_s,
                    start_xyz_rpy=start, target_xyz_rpy=target,
                    force_monitoring_enabled=False,
                )
                arm.start_servo_pose(  # type: ignore[attr-defined]
                    2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
                    input_frequency_hz=50.0,
                )
                continuous_servo_stopped = False
                try:
                    move_continuous_with_recovery(
                        f"second_hole_{label}", start, target,
                        linear_speed_mps=speed_mm_s / 1000.0,
                        linear_accel_mps2=0.02, initial_speed_mps=0.0,
                        near_target_mm=(3.0 if label == "descend_into_second_hole" else 2.0),
                        recovery_speed_mps=(0.003 if label == "descend_into_second_hole" else None),
                    )
                finally:
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    wait_for_right_task_stopped(
                        arm, log_file,
                        confirmed_event=f"right_second_hole_{label}_stop_confirmed",
                        timeout_event=f"right_second_hole_{label}_stop_wait_timeout",
                        cancel_if_still_running_after_s=0.5,
                    )
                    continuous_servo_stopped = True
                completed = current_pose(arm, 2)
                completed_rpy = completed[:3] + rotvec_to_rpy(completed[3:])
                log_event(
                    "second_hole_leg_completed", label=label,
                    actual_end_xyz_rpy=completed_rpy,
                )

            print("第一洞确认成功；执行后续路径（不读取力传感器）。")
            run_leg("settle_down_after_first_hole", 2, -10.0,
                    args.second_hole_descend_speed_mm_s)
            print("第一洞后下探完成；停留 0.5 秒。")
            log_event("second_hole_pause", label="after_first_hole_down", duration_s=0.5)
            time.sleep(0.5)
            run_leg("lift_from_first_hole", 2, args.second_hole_lift_mm,
                    args.second_hole_lateral_speed_mm_s)
            run_leg("shift_to_second_hole", 1, args.second_hole_y_offset_mm,
                    args.second_hole_lateral_speed_mm_s)
            run_leg("descend_into_second_hole", 2, -args.second_hole_descend_mm,
                    args.second_hole_descend_speed_mm_s)
            print("第二洞下降完成；停留 0.5 秒。")
            log_event("second_hole_pause", label="after_second_hole_down", duration_s=0.5)
            time.sleep(0.5)
            run_leg("lift_after_second_hole", 2, args.second_hole_lift_mm,
                    args.second_hole_lateral_speed_mm_s)
            run_leg("shift_x_negative_after_second_hole", 0, -110.0,
                    args.second_hole_x_return_speed_mm_s)
            print("第二洞后续路径结束。")

        # Keep the single log open until the ServoPose session has stopped.
        if True:
            previous_back_wrench: tuple[list[float], list[float]] | None = None
            def sample(event: str, cycle: int, back_mm: float, target: list[float]) -> tuple[list[float], list[float]]:
                force, torque = read_wrench(sensor, args.sample_timeout)
                dforce = [value - base for value, base in zip(force, force_base)]
                dtorque = [value - base for value, base in zip(torque, torque_base)]
                actual = current_pose(arm, 2)
                actual_rpy = rotvec_to_rpy(actual[3:])
                force_norm, torque_norm = norm(dforce), norm(dtorque)
                motion = {
                    "initial": "initial",
                    "back_centre": search_motion_name,
                    "wiggle_y_plus": "left_y_positive",
                    "wiggle_y_minus": "right_y_negative",
                    "wiggle_return_centre": "return_to_centre",
                }.get(event, event)
                log_event(
                    "force_sample",
                    motion=motion,
                    cycle=cycle,
                    back_mm=round(back_mm, 3),
                    target_xyz_rpy=target,
                    actual_xyz_rpy=actual[:3] + actual_rpy,
                    force_n=force,
                    torque_nm=torque,
                    delta_force_n=dforce,
                    delta_torque_nm=dtorque,
                    delta_force_norm_n=round(force_norm, 6),
                    delta_torque_norm_nm=round(torque_norm, 6),
                    insertion_decision="pending",
                )
                print(
                    f"[{event}] back={back_mm:.1f}mm dF={dforce} |dF|={force_norm:.2f}N "
                    f"dT={dtorque} |dT|={torque_norm:.2f}N*m"
                )
                if force_norm > args.max_force_delta_n or torque_norm > args.max_torque_delta_nm:
                    if event != "initial":
                        log_event(
                            "insertion_decision",
                            result="entered_by_contact_limit",
                            motion=motion,
                            cycle=cycle,
                            back_mm=round(back_mm, 3),
                            delta_force_norm_n=round(force_norm, 6),
                            delta_torque_norm_nm=round(torque_norm, 6),
                        )
                        print(
                            "SUCCESS：插孔搜索阶段达到相对力/扭矩接触限；"
                            "按已进入孔位处理，停止后续 ServoPose 推送。"
                            f" event={event} back={back_mm:.1f}mm "
                            f"|dF|={force_norm:.2f}N（限 {args.max_force_delta_n:.2f}），"
                            f"|dT|={torque_norm:.2f}N*m（限 {args.max_torque_delta_nm:.2f}）。"
                        )
                        raise PlacementContactReached()
                    raise RuntimeError(
                        "安全停止：初始位姿相对力/扭矩超限。"
                        f"|dF|={force_norm:.2f}N（限 {args.max_force_delta_n:.2f}），"
                        f"|dT|={torque_norm:.2f}N*m（限 {args.max_torque_delta_nm:.2f}）。"
                    )
                return dforce, dtorque

            # The SDK subscriber receives force frames at 100 Hz.  During a
            # continuous move, inspect its latest cached frame at the 50-Hz
            # ServoPose cadence; do not emit a full force_sample log per tick.
            # The normal 1-mm samples below remain the durable trace.
            # Each 50-Hz check compares the current frame to precisely the
            # preceding 20-ms frame.  Logging is independently rate-limited
            # to 10 Hz so it cannot obscure the control output.
            previous_continuous_wrench: tuple[list[float], list[float]] | None = None
            # A release must start from a *local high-contact* frame.  Do not
            # let a long, mildly changing contact plateau become the release
            # window; lock the most recent local peak once a real descent
            # starts and count only the following descending frames.
            continuous_contact_peak: tuple[list[float], list[float], float] | None = None
            continuous_release_frames = 0
            # Workpieces whose contact direction is not aligned with SDK Fx/Ty
            # need a sign-independent release test.  It is disabled by default
            # and enabled only by a reviewed workpiece profile.
            continuous_resultant_peak: tuple[float, float, float] | None = None
            continuous_resultant_release_frames = 0
            release_events_seen = 0
            fx_reversal_confirmed_frames = 0
            fx_reversal_peak: tuple[float, float] | None = None
            last_continuous_monitor_output_at = 0.0
            continuous_monitor_tick = 0
            continuous_monitor_stride = (
                max(1, round(50.0 / args.continuous_force_monitor_hz))
                if args.continuous_force_monitor_hz > 0.0
                else 0
            )

            def release_event_should_stop(signature: str, **details: object) -> bool:
                """Count independently validated unloads; a profile may require repeats."""
                nonlocal release_events_seen
                release_events_seen += 1
                should_stop = release_events_seen >= args.release_events_required
                log_event(
                    "insertion_release_event",
                    signature=signature,
                    occurrence=release_events_seen,
                    required_occurrences=args.release_events_required,
                    stop=should_stop,
                    **details,
                )
                if not should_stop:
                    print(
                        f"检测到第 {release_events_seen}/{args.release_events_required} 次合格卸载（{signature}）；"
                        "继续连续搜索，等待下一次卸载。"
                    )
                return should_stop

            def monitor_continuous_wrench(back_mm: float) -> None:
                nonlocal previous_continuous_wrench, continuous_contact_peak, continuous_release_frames
                nonlocal continuous_resultant_peak, continuous_resultant_release_frames
                nonlocal fx_reversal_confirmed_frames, fx_reversal_peak
                nonlocal last_continuous_monitor_output_at
                nonlocal continuous_monitor_tick, continuous_servo_stopped
                if continuous_monitor_stride == 0:
                    return
                continuous_monitor_tick += 1
                if continuous_monitor_tick % continuous_monitor_stride:
                    return
                frame = sensor.get_state()  # type: ignore[attr-defined]
                if frame is None:
                    return
                force = vector(frame, "filtered_force")
                torque = vector(frame, "filtered_torque")
                dforce = [value - base for value, base in zip(force, force_base)]
                dtorque = [value - base for value, base in zip(torque, torque_base)]
                force_norm, torque_norm = norm(dforce), norm(dtorque)
                now = time.monotonic()
                # Persist every 50-Hz frame for post-run verification of the
                # consecutive-release rule.  Keep console output at 10 Hz.
                log_event(
                    "continuous_force_monitor_tick",
                    back_mm=round(back_mm, 3),
                    monitor_hz=args.continuous_force_monitor_hz,
                    delta_force_n=dforce,
                    delta_torque_nm=dtorque,
                    delta_force_norm_n=round(force_norm, 6),
                    delta_torque_norm_nm=round(torque_norm, 6),
                )
                output_period_s = (
                    1.0 / args.continuous_force_output_hz
                    if args.continuous_force_output_hz > 0.0 else math.inf
                )
                if now - last_continuous_monitor_output_at >= output_period_s:
                    last_continuous_monitor_output_at = now
                    log_event(
                        "continuous_force_monitor",
                        back_mm=round(back_mm, 3),
                        monitor_hz=args.continuous_force_monitor_hz,
                        delta_force_n=dforce,
                        delta_torque_nm=dtorque,
                        delta_force_norm_n=round(force_norm, 6),
                        delta_torque_norm_nm=round(torque_norm, 6),
                    )
                    print(
                        f"[force_100ms] back≈{back_mm:.2f}mm |dF|={force_norm:.2f}N "
                        f"|dT|={torque_norm:.2f}N*m"
                    )
                if force_norm > args.max_force_delta_n or torque_norm > args.max_torque_delta_nm:
                    # Do not wait for logging or a control-state RPC before
                    # ending the active stream: this path protects the part
                    # after a high-rate force trigger.
                    arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                    continuous_servo_stopped = True
                    log_event(
                        "insertion_decision",
                        result="entered_by_contact_limit_continuous_monitor",
                        back_mm=round(back_mm, 3),
                        monitor_hz=args.continuous_force_monitor_hz,
                        delta_force_norm_n=round(force_norm, 6),
                        delta_torque_norm_nm=round(torque_norm, 6),
                    )
                    print(
                        "SUCCESS：连续力监测达到相对力/扭矩接触限；"
                        "已立即停止后续 ServoPose 推送。"
                        f" back≈{back_mm:.2f}mm |dF|={force_norm:.2f}N，|dT|={torque_norm:.2f}N*m"
                    )
                    raise PlacementContactReached()
                fx_reversal_enabled = args.entry_fx_reversal_confirm_frames > 0
                if fx_reversal_enabled:
                    if (
                        fx_reversal_peak is not None
                        and back_mm - fx_reversal_peak[1] > args.entry_fx_reversal_peak_window_mm
                    ):
                        fx_reversal_peak = None
                    if dforce[0] >= args.entry_fx_reversal_pre_n:
                        if fx_reversal_peak is None:
                            fx_reversal_peak = (dforce[0], back_mm)
                        else:
                            # Retain the strongest positive Fx in this contact
                            # lobe, but measure the window from its final
                            # positive frame so a gradual sign reversal is not
                            # discarded as a stale peak.
                            fx_reversal_peak = (
                                max(fx_reversal_peak[0], dforce[0]), back_mm,
                            )
                    if fx_reversal_confirmed_frames == 0:
                        fx_reversal_candidate = (
                            back_mm >= args.min_entry_back_mm
                            and fx_reversal_peak is not None
                            and dforce[0] <= args.entry_fx_reversal_current_n
                            and fx_reversal_peak[0] - dforce[0] >= args.entry_fx_reversal_drop_n
                        )
                        if fx_reversal_candidate:
                            fx_reversal_confirmed_frames = 1
                    elif dforce[0] <= args.entry_fx_reversal_current_n:
                        fx_reversal_confirmed_frames += 1
                    else:
                        fx_reversal_confirmed_frames = 0
                    if fx_reversal_confirmed_frames >= args.entry_fx_reversal_confirm_frames:
                        arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                        continuous_servo_stopped = True
                        log_event(
                            "insertion_decision",
                            result="entered_by_fx_negative_reversal_continuous_monitor",
                            back_mm=round(back_mm, 3),
                            confirm_frames=fx_reversal_confirmed_frames,
                            peak_delta_fx_n=fx_reversal_peak[0] if fx_reversal_peak else None,
                            peak_back_mm=round(fx_reversal_peak[1], 3) if fx_reversal_peak else None,
                            delta_fx_n=dforce[0],
                            fx_drop_n=round((fx_reversal_peak[0] - dforce[0]) if fx_reversal_peak else 0.0, 6),
                        )
                        print(
                            "SUCCESS：连续力监测检测到并确认 Fx 快速翻转为负载；"
                            "立即停止后续 ServoPose 推送。"
                            f" back≈{back_mm:.2f}mm frames={fx_reversal_confirmed_frames} "
                            f"dFx_peak={fx_reversal_peak[0] if fx_reversal_peak else 0.0:.2f}"
                            f"→{dforce[0]:.2f}N"
                        )
                        raise PlacementContactReached()
                resultant_enabled = args.entry_resultant_release_frames >= 2 and all((
                    args.entry_resultant_pre_force_n > 0.0,
                    args.entry_resultant_pre_torque_nm > 0.0,
                    args.entry_resultant_force_n > 0.0,
                    args.entry_resultant_torque_nm > 0.0,
                    args.entry_resultant_force_drop_n > 0.0,
                    args.entry_resultant_torque_drop_nm > 0.0,
                ))
                if resultant_enabled and back_mm >= args.min_entry_back_mm:
                    if continuous_resultant_peak is None:
                        if (force_norm >= args.entry_resultant_pre_force_n
                                and torque_norm >= args.entry_resultant_pre_torque_nm):
                            continuous_resultant_peak = (force_norm, torque_norm, back_mm)
                            continuous_resultant_release_frames = 0
                    else:
                        peak_force_norm, peak_torque_norm, peak_back_mm = continuous_resultant_peak
                        if force_norm >= peak_force_norm and torque_norm >= peak_torque_norm:
                            continuous_resultant_peak = (force_norm, torque_norm, back_mm)
                            continuous_resultant_release_frames = 0
                        elif (previous_continuous_wrench is not None
                              # Keep a valid unload streak through normal
                              # sensor-scale rebound; use the same tolerances
                              # as the Fx/Ty continuous-release detector.
                              and force_norm <= (
                                  norm(previous_continuous_wrench[0])
                                  + args.continuous_release_fx_rise_tolerance_n
                              )
                              and torque_norm <= (
                                  norm(previous_continuous_wrench[1])
                                  + args.continuous_release_ty_rise_tolerance_nm
                              )):
                            continuous_resultant_release_frames += 1
                        else:
                            continuous_resultant_peak = None
                            continuous_resultant_release_frames = 0
                    if continuous_resultant_peak is not None:
                        peak_force_norm, peak_torque_norm, peak_back_mm = continuous_resultant_peak
                        resultant_entry = (
                            continuous_resultant_release_frames >= args.entry_resultant_release_frames
                            and force_norm <= args.entry_resultant_force_n
                            and torque_norm <= args.entry_resultant_torque_nm
                            and peak_force_norm - force_norm >= args.entry_resultant_force_drop_n
                            and peak_torque_norm - torque_norm >= args.entry_resultant_torque_drop_nm
                        )
                        if resultant_entry:
                            if not release_event_should_stop(
                                "resultant_release",
                                back_mm=round(back_mm, 3),
                                release_frames=continuous_resultant_release_frames,
                                release_start_back_mm=round(peak_back_mm, 3),
                                release_start_force_norm_n=round(peak_force_norm, 6),
                                release_start_torque_norm_nm=round(peak_torque_norm, 6),
                                delta_force_norm_n=round(force_norm, 6),
                                delta_torque_norm_nm=round(torque_norm, 6),
                            ):
                                continuous_resultant_peak = None
                                continuous_resultant_release_frames = 0
                                previous_continuous_wrench = (dforce, dtorque)
                                return
                            arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                            continuous_servo_stopped = True
                            log_event(
                                "insertion_decision",
                                result="entered_by_resultant_release_continuous_monitor",
                                back_mm=round(back_mm, 3),
                                release_frames=continuous_resultant_release_frames,
                                release_start_back_mm=round(peak_back_mm, 3),
                                release_start_force_norm_n=round(peak_force_norm, 6),
                                release_start_torque_norm_nm=round(peak_torque_norm, 6),
                                delta_force_norm_n=round(force_norm, 6),
                                delta_torque_norm_nm=round(torque_norm, 6),
                            )
                            print(
                                "SUCCESS：检测到合力/合扭矩连续卸载，已停止后续 ServoPose 推送。"
                                f" back≈{back_mm:.2f}mm frames={continuous_resultant_release_frames} "
                                f"|dF|={peak_force_norm:.2f}→{force_norm:.2f}N "
                                f"|dT|={peak_torque_norm:.2f}→{torque_norm:.2f}N*m"
                            )
                            raise PlacementContactReached()
                if previous_continuous_wrench is not None and back_mm >= args.min_entry_back_mm:
                    previous_force, previous_torque = previous_continuous_wrench
                    strong_entry = (
                        previous_force[0] >= args.entry_pre_fx_n
                        and previous_torque[1] >= args.entry_pre_ty_nm
                        and previous_force[0] - dforce[0] >= args.entry_fx_drop_n
                        and (
                            previous_torque[1] - dtorque[1] >= args.entry_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    relaxed_entry = (
                        previous_force[0] >= args.entry_relaxed_pre_fx_n
                        and previous_torque[1] >= args.entry_relaxed_pre_ty_nm
                        and previous_force[0] - dforce[0] >= args.entry_relaxed_fx_drop_n
                        and (
                            previous_torque[1] - dtorque[1] >= args.entry_relaxed_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    if strong_entry or relaxed_entry:
                        if not release_event_should_stop(
                            "fx_ty_single_frame_release",
                            back_mm=round(back_mm, 3),
                            delta_force_n=dforce,
                            delta_torque_nm=dtorque,
                            previous_delta_force_n=previous_force,
                            previous_delta_torque_nm=previous_torque,
                        ):
                            previous_continuous_wrench = (dforce, dtorque)
                            return
                        arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                        continuous_servo_stopped = True
                        log_event(
                            "insertion_decision",
                            result=("entered_by_strong_fx_ty_release_continuous_monitor"
                                    if strong_entry else "entered_by_relaxed_fx_ty_release_continuous_monitor"),
                            back_mm=round(back_mm, 3),
                            monitor_hz=args.continuous_force_monitor_hz,
                            previous_delta_force_n=previous_force,
                            previous_delta_torque_nm=previous_torque,
                            delta_force_n=dforce,
                            delta_torque_nm=dtorque,
                        )
                        print(
                            "SUCCESS：连续力监测检测到进洞的 Fx/Ty 同步突变卸载；"
                            "立即停止后续 ServoPose 推送。"
                            f" back≈{back_mm:.2f}mm dFx={previous_force[0]:.2f}→{dforce[0]:.2f}N "
                            f"dTy={previous_torque[1]:.2f}→{dtorque[1]:.2f}N*m"
                        )
                        raise PlacementContactReached()
                contact_level = (
                    dforce[0] >= args.entry_relaxed_pre_fx_n
                    and dtorque[1] >= args.entry_relaxed_pre_ty_nm
                )
                if continuous_contact_peak is None:
                    if contact_level:
                        continuous_contact_peak = (dforce, dtorque, back_mm)
                        continuous_release_frames = 0
                else:
                    peak_force, peak_torque, peak_back_mm = continuous_contact_peak
                    near_peak = (
                        dforce[0] >= peak_force[0] - args.continuous_release_fx_rise_tolerance_n
                        and dtorque[1] >= peak_torque[1] - args.continuous_release_ty_rise_tolerance_nm
                    )
                    if near_peak and contact_level:
                        # Refresh the reference while contact remains at its
                        # local maximum, so release is measured from the last
                        # loaded frame rather than an old approach frame.
                        continuous_contact_peak = (dforce, dtorque, back_mm)
                        continuous_release_frames = 0
                    elif previous_continuous_wrench is not None:
                        previous_force, previous_torque = previous_continuous_wrench
                        descending = (
                            dforce[0] <= previous_force[0] + args.continuous_release_fx_rise_tolerance_n
                            and dtorque[1] <= previous_torque[1] + args.continuous_release_ty_rise_tolerance_nm
                        )
                        if descending:
                            continuous_release_frames += 1
                        elif contact_level:
                            continuous_contact_peak = (dforce, dtorque, back_mm)
                            continuous_release_frames = 0
                        else:
                            continuous_contact_peak = None
                            continuous_release_frames = 0
                log_event(
                    "continuous_release_state",
                    back_mm=round(back_mm, 3),
                    release_frames=continuous_release_frames,
                    peak_back_mm=(round(continuous_contact_peak[2], 3)
                                  if continuous_contact_peak is not None else None),
                    peak_delta_force_n=(continuous_contact_peak[0]
                                        if continuous_contact_peak is not None else None),
                    peak_delta_torque_nm=(continuous_contact_peak[1]
                                         if continuous_contact_peak is not None else None),
                )
                if (
                    continuous_contact_peak is not None
                    and continuous_release_frames >= args.continuous_release_frames
                    and back_mm >= args.min_entry_back_mm
                ):
                    release_start_force, release_start_torque, release_start_back_mm = continuous_contact_peak
                    strong_entry = (
                        release_start_force[0] >= args.entry_pre_fx_n
                        and release_start_torque[1] >= args.entry_pre_ty_nm
                        and release_start_force[0] - dforce[0] >= args.entry_fx_drop_n
                        and (
                            release_start_torque[1] - dtorque[1] >= args.entry_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    relaxed_entry = (
                        release_start_force[0] >= args.entry_relaxed_pre_fx_n
                        and release_start_torque[1] >= args.entry_relaxed_pre_ty_nm
                        and release_start_force[0] - dforce[0] >= args.entry_relaxed_fx_drop_n
                        and (
                            release_start_torque[1] - dtorque[1] >= args.entry_relaxed_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    if strong_entry or relaxed_entry:
                        if not release_event_should_stop(
                            "fx_ty_multi_frame_release",
                            back_mm=round(back_mm, 3),
                            release_frames=continuous_release_frames,
                            release_start_back_mm=round(release_start_back_mm, 3),
                            release_start_delta_force_n=release_start_force,
                            release_start_delta_torque_nm=release_start_torque,
                            delta_force_n=dforce,
                            delta_torque_nm=dtorque,
                        ):
                            continuous_contact_peak = None
                            continuous_release_frames = 0
                            previous_continuous_wrench = (dforce, dtorque)
                            return
                        arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                        continuous_servo_stopped = True
                        log_event(
                            "insertion_decision",
                            result=("entered_by_strong_fx_ty_release_continuous_streak"
                                    if strong_entry else "entered_by_relaxed_fx_ty_release_continuous_streak"),
                            back_mm=round(back_mm, 3),
                            monitor_hz=args.continuous_force_monitor_hz,
                            release_frames=continuous_release_frames,
                            release_start_back_mm=round(release_start_back_mm, 3),
                            release_start_delta_force_n=release_start_force,
                            release_start_delta_torque_nm=release_start_torque,
                            delta_force_n=dforce,
                            delta_torque_nm=dtorque,
                        )
                        print(
                            "SUCCESS：连续力监测检测到多帧 Fx/Ty 同步卸载；"
                            "立即停止后续 ServoPose 推送。"
                            f" back≈{back_mm:.2f}mm frames={continuous_release_frames} "
                            f"dFx={release_start_force[0]:.2f}→{dforce[0]:.2f}N "
                            f"dTy={release_start_torque[1]:.2f}→{dtorque[1]:.2f}N*m"
                        )
                        raise PlacementContactReached()
                previous_continuous_wrench = (dforce, dtorque)

            sample("initial", 0, 0.0, centre)
            # Arm immediately before the first non-zero command, then keep
            # feeding the same session through all back/wiggle segments.
            servo_guard = right_servo_guard()
            servo_guard.__enter__()
            arm.start_servo_pose(  # type: ignore[attr-defined]
                2, pose_type="absolute", frame="world", tcp=[0.0] * 6, input_frequency_hz=50.0
            )
            servo_started = True
            # The preparatory Z-down LIN used its own ServoPose session and
            # stopped it before baseline collection.  This is a new search
            # session, so final cleanup must stop and confirm *this* task.
            continuous_servo_stopped = False
            # The pure-search clock excludes joint return, force-baseline
            # collection, task-reset handling, and any inter-arm lock wait.
            search_started_at = time.monotonic()
            write_control_snapshot(arm, log_file, "after_start_servo_pose")
            diagnostic = lambda event: write_control_snapshot(arm, log_file, event)
            cycles = math.ceil(args.max_back_mm / args.back_step_mm)
            # After a non-final Y- sample, the next back-centre command starts
            # at Y- and returns Y to centre while moving X backward.  This
            # replaces the old Y- -> centre -> next-X pair with one diagonal.
            arriving_from_right_y_negative = False
            consecutive_back_target_timeouts = 0
            # The previous commanded endpoint, rather than a measured pose.
            # Keeping this across checkpoints is what avoids synthesising a
            # fresh accelerate/decelerate profile at every millimetre.
            continuous_command_target = list(centre)
            continuous_command_speed_mps = 0.0
            if args.single_continuous_search:
                single_target = list(centre)
                single_target[search_axis_index] = (
                    start_search_coordinate
                    + search_direction_sign * args.max_back_mm / 1000.0
                )
                log_event(
                    "motion_start",
                    motion=search_motion_name,
                    mode="single_continuous_segment",
                    start_back_mm=0.0,
                    end_back_mm=round(args.max_back_mm, 3),
                    linear_speed_mm_s=round(args.fast_linear_speed_mm_s, 3),
                    linear_accel_mm_s2=round(args.fast_linear_accel_mm_s2, 3),
                    insertion_decision="searching",
                )
                print(
                    f"[motion_profile] 单段连续 {args.search_axis.upper()}"
                    f"{'+' if args.search_direction == 'positive' else '−'} 搜索："
                    f"0→{args.max_back_mm:.1f}mm，"
                    f"{args.fast_linear_speed_mm_s:.1f}mm/s；力监测 "
                    f"{args.continuous_force_monitor_hz:.1f}Hz。"
                )
                move_continuous_with_recovery(
                    "first_hole_single_continuous_search",
                    centre,
                    single_target,
                    linear_speed_mps=args.fast_linear_speed_mm_s / 1000.0,
                    linear_accel_mps2=args.fast_linear_accel_mm_s2 / 1000.0,
                    initial_speed_mps=0.0,
                    force_monitor=lambda alpha: monitor_continuous_wrench(
                        alpha * args.max_back_mm,
                    ),
                )
                log_event(
                    "insertion_decision",
                    result="not_entered_max_search_distance_reached",
                    back_mm=round(args.max_back_mm, 3),
                )
                if args.return_search_exhausted:
                    print(
                        "本轮搜索达到最大距离，交由工件配置入口执行抬升、回初始点并重试。"
                    )
                    return SEARCH_DISTANCE_EXHAUSTED_EXIT_CODE
                raise RuntimeError(
                    f"在最大搜索距离 {args.max_back_mm:.1f}mm 内未检测到进洞特征。"
                )
            for cycle in range(1, cycles + 1):
                back_mm = min(cycle * args.back_step_mm, args.max_back_mm)
                if back_mm >= args.slow_after_back_mm:
                    linear_speed_mps = args.slow_linear_speed_mm_s / 1000.0
                    linear_accel_mps2 = args.slow_linear_accel_mm_s2 / 1000.0
                    speed_label = "slow"
                else:
                    linear_speed_mps = args.fast_linear_speed_mm_s / 1000.0
                    linear_accel_mps2 = args.fast_linear_accel_mm_s2 / 1000.0
                    speed_label = "fast"
                print(
                    f"[motion_profile] back={back_mm:.1f}mm {speed_label} "
                    f"v={linear_speed_mps * 1000.0:.1f}mm/s a={linear_accel_mps2 * 1000.0:.1f}mm/s²"
                )
                log_event(
                    "motion_start",
                    motion=(
                        f"{search_motion_name}_and_return_y_to_centre"
                        if arriving_from_right_y_negative
                        else search_motion_name
                    ),
                    cycle=cycle,
                    back_mm=round(back_mm, 3),
                    speed_profile=speed_label,
                    linear_speed_mm_s=round(linear_speed_mps * 1000.0, 3),
                    linear_accel_mm_s2=round(linear_accel_mps2 * 1000.0, 3),
                    insertion_decision="searching",
                )
                centre[search_axis_index] = (
                    start_search_coordinate
                    + search_direction_sign * back_mm / 1000.0
                )
                back_target_timed_out = False
                if args.continuous_straight_search:
                    continuous_command_speed_mps = move_continuous_with_recovery(
                        "first_hole_continuous_search",
                        continuous_command_target,
                        centre,
                        linear_speed_mps=linear_speed_mps,
                        linear_accel_mps2=linear_accel_mps2,
                        initial_speed_mps=continuous_command_speed_mps,
                        force_monitor=(
                            lambda alpha, start_mm=min(
                                (cycle - 1) * args.back_step_mm, args.max_back_mm,
                            ), end_mm=back_mm: monitor_continuous_wrench(
                                start_mm + alpha * (end_mm - start_mm),
                            )
                            if continuous_monitor_stride
                            else None
                        ),
                    )
                    continuous_command_target = list(centre)
                    back_event = "back_centre_continuous"
                else:
                    try:
                        # Straight-line centre samples must use the strict
                        # endpoint check; do not begin the next push from a
                        # merely short-confirmed position.
                        move_in_active_session(
                            arm, centre,
                            linear_speed_mps=linear_speed_mps,
                            linear_accel_mps2=linear_accel_mps2,
                            diagnostic=diagnostic,
                        )
                        back_event = (
                            "back_centre_from_right_y_negative"
                            if arriving_from_right_y_negative
                            else "back_centre"
                        )
                        consecutive_back_target_timeouts = 0
                    except TargetNotReached:
                        back_event = ""
                        # A single exact-target resend handles transient tracking
                        # lag without changing the workpiece pose.  Unlike the
                        # retired Z+ recovery, it cannot pull a seated part out.
                        for retry_index in range(1, args.same_target_retries + 1):
                            print(
                                "后拉目标首次未到位；保持同一中心目标完整重发 "
                                f"({retry_index}/{args.same_target_retries})。"
                            )
                            log_event(
                                "motion_retry",
                                motion=f"{search_motion_name}_same_target",
                                cycle=cycle,
                                back_mm=round(back_mm, 3),
                                retry_index=retry_index,
                            )
                            try:
                                move_in_active_session(
                                    arm, centre,
                                    linear_speed_mps=linear_speed_mps,
                                    linear_accel_mps2=linear_accel_mps2,
                                    diagnostic=diagnostic,
                                )
                                back_event = f"back_centre_after_same_target_retry_{retry_index}"
                                consecutive_back_target_timeouts = 0
                                break
                            except TargetNotReached:
                                print("同一目标重发后仍未到位。")
                        if not back_event:
                            # Do not lift: when the part is already seated, a Z+
                            # recovery can pull it out.  Sample the actual stopped
                            # pose, then let the next cycle issue the next search point.
                            back_target_timed_out = True
                            consecutive_back_target_timeouts += 1
                            back_event = "back_centre_target_timeout_continue_x_negative"
                            print(
                                "后拉目标重发后仍未到位；保持当前位置，不抬高。"
                                "先采集力/扭矩，再继续下一搜索点 "
                                f"({consecutive_back_target_timeouts}/"
                                f"{args.max_consecutive_back_target_timeouts})。"
                            )
                dforce, dtorque = sample(back_event, cycle, back_mm, centre)
                arriving_from_right_y_negative = False
                if previous_back_wrench is not None and back_mm >= args.min_entry_back_mm:
                    previous_force, previous_torque = previous_back_wrench
                    strong_entry = (
                        previous_force[0] >= args.entry_pre_fx_n
                        and previous_torque[1] >= args.entry_pre_ty_nm
                        and previous_force[0] - dforce[0] >= args.entry_fx_drop_n
                        and (
                            previous_torque[1] - dtorque[1] >= args.entry_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    relaxed_entry = (
                        previous_force[0] >= args.entry_relaxed_pre_fx_n
                        and previous_torque[1] >= args.entry_relaxed_pre_ty_nm
                        and previous_force[0] - dforce[0] >= args.entry_relaxed_fx_drop_n
                        and (
                            previous_torque[1] - dtorque[1] >= args.entry_relaxed_ty_drop_nm
                            or abs(dtorque[1]) <= args.entry_ty_abs_nm
                        )
                    )
                    previous_force_norm = norm(previous_force)
                    previous_torque_norm = norm(previous_torque)
                    force_norm = norm(dforce)
                    torque_norm = norm(dtorque)
                    release_entry_candidate = (
                        previous_force_norm >= args.entry_release_pre_force_n
                        and previous_torque_norm >= args.entry_release_pre_torque_nm
                        and force_norm <= args.entry_release_force_n
                        and torque_norm <= args.entry_release_torque_nm
                        and previous_force_norm - force_norm >= args.entry_release_force_drop_n
                        and previous_torque_norm - torque_norm >= args.entry_release_torque_drop_nm
                    )
                    if release_entry_candidate:
                        print(
                            "候选进洞：检测到中心位姿接触载荷同步卸载；"
                            "保持当前位置进行原地确认。"
                            f" back={back_mm:.1f}mm "
                            f"|dF|={previous_force_norm:.2f}→{force_norm:.2f}N "
                            f"|dT|={previous_torque_norm:.2f}→{torque_norm:.2f}N*m"
                        )
                        confirm_force_norms: list[float] = []
                        confirm_torque_norms: list[float] = []
                        for confirm_index in range(1, args.entry_release_confirm_samples + 1):
                            time.sleep(args.entry_release_confirm_interval_s)
                            confirm_force, confirm_torque = sample(
                                f"entry_release_confirm_{confirm_index}", cycle, back_mm, centre,
                            )
                            confirm_force_norms.append(norm(confirm_force))
                            confirm_torque_norms.append(norm(confirm_torque))
                        confirm_force_median = median(confirm_force_norms)
                        confirm_torque_median = median(confirm_torque_norms)
                        release_confirmed = (
                            confirm_force_median <= args.entry_release_force_n
                            and confirm_torque_median <= args.entry_release_torque_nm
                        )
                        if not release_confirmed:
                            print(
                                "卸载候选未确认：原地确认载荷中位数仍偏高；继续搜索。"
                                f" |dF|_median={confirm_force_median:.2f}N "
                                f"|dT|_median={confirm_torque_median:.2f}N*m"
                            )
                        if release_confirmed:
                            log_event("insertion_decision", result="entered_by_centre_release_confirmed",
                                      cycle=cycle, back_mm=round(back_mm, 3))
                            print(
                                "SUCCESS：检测到并确认中心位姿接触载荷同步卸载；"
                                "停止在当前后拉中心位姿，不执行本轮左右摆动。"
                                f" back={back_mm:.1f}mm "
                                f"|dF|={previous_force_norm:.2f}→{force_norm:.2f}N "
                                f"|dT|={previous_torque_norm:.2f}→{torque_norm:.2f}N*m"
                            )
                            return 0
                    entered = strong_entry or relaxed_entry
                    if entered:
                        log_event("insertion_decision",
                                  result=("entered_by_strong_fx_ty_release" if strong_entry
                                          else "entered_by_relaxed_fx_ty_release"),
                                  cycle=cycle, back_mm=round(back_mm, 3))
                        print(
                            "SUCCESS：检测到进洞的 Fx/Ty 同步突变卸载"
                            f"（{'强突变' if strong_entry else '中等接触卸载'}）；"
                            "立即停止在后拉中心位姿，不执行本轮左右摆动。"
                            f" back={back_mm:.1f}mm "
                            f"dFx={previous_force[0]:.2f}→{dforce[0]:.2f}N "
                            f"(drop={previous_force[0] - dforce[0]:.2f}N) "
                            f"dTy={previous_torque[1]:.2f}→{dtorque[1]:.2f}N*m "
                            f"(drop={previous_torque[1] - dtorque[1]:.2f}N*m, "
                            f"|current|={abs(dtorque[1]):.2f}N*m)"
                        )
                        return 0
                if (
                    back_target_timed_out
                    and consecutive_back_target_timeouts >= args.max_consecutive_back_target_timeouts
                ):
                    raise TargetNotReached(
                        "连续搜索目标未到位，已保持当前位置采样且未执行抬高恢复；"
                        f"次数={consecutive_back_target_timeouts}，"
                        f"上限={args.max_consecutive_back_target_timeouts}。"
                    )
                previous_back_wrench = (dforce, dtorque)
                if args.straight_line:
                    log_event(
                        "insertion_decision",
                        result="not_entered_continue_straight_search",
                        cycle=cycle,
                        back_mm=round(back_mm, 3),
                    )
                    continue
                centre_force, centre_torque = dforce, dtorque

                def wiggle_entered(force: list[float], torque: list[float]) -> bool:
                    return (
                        back_mm >= args.min_entry_back_mm
                        and centre_force[0] >= args.wiggle_entry_pre_fx_n
                        and centre_torque[1] >= args.wiggle_entry_pre_ty_nm
                        and force[0] < 0.0
                        and torque[1] < 0.0
                        and centre_force[0] - force[0] >= args.wiggle_entry_fx_drop_n
                        and centre_torque[1] - torque[1] >= args.wiggle_entry_ty_drop_nm
                    )

                def report_wiggle_entry(event: str, force: list[float], torque: list[float]) -> None:
                    log_event("insertion_decision", result="entered_by_wiggle_fx_ty_sign_reversal",
                              motion=event, cycle=cycle, back_mm=round(back_mm, 3),
                              delta_force_n=force, delta_torque_nm=torque)
                    print(
                        "SUCCESS：检测到摆动进洞的 Fx/Ty 正负突变；"
                        "立即停止，不再执行后续摆动或后拉。"
                        f" event={event} back={back_mm:.1f}mm "
                        f"dFx={centre_force[0]:.2f}→{force[0]:.2f}N "
                        f"(drop={centre_force[0] - force[0]:.2f}N) "
                        f"dTy={centre_torque[1]:.2f}→{torque[1]:.2f}N*m "
                        f"(drop={centre_torque[1] - torque[1]:.2f}N*m)"
                    )

                def wiggle_release_entered(
                    event: str,
                    force: list[float],
                    torque: list[float],
                    target: list[float],
                ) -> bool:
                    """Confirm a centre-contact to lateral-search unload in place."""
                    if back_mm < args.min_entry_back_mm:
                        return False
                    centre_force_norm = norm(centre_force)
                    centre_torque_norm = norm(centre_torque)
                    force_norm = norm(force)
                    torque_norm = norm(torque)
                    if not (
                        centre_force_norm >= args.entry_release_pre_force_n
                        and centre_torque_norm >= args.entry_release_pre_torque_nm
                        and force_norm <= args.entry_release_force_n
                        and torque_norm <= args.entry_release_torque_nm
                        and centre_force_norm - force_norm >= args.entry_release_force_drop_n
                        and centre_torque_norm - torque_norm >= args.entry_release_torque_drop_nm
                    ):
                        return False
                    print(
                        "候选进洞：摆动后检测到中心接触载荷同步卸载；"
                        "保持当前位置进行原地确认。"
                        f" event={event} back={back_mm:.1f}mm "
                        f"|dF|={centre_force_norm:.2f}→{force_norm:.2f}N "
                        f"|dT|={centre_torque_norm:.2f}→{torque_norm:.2f}N*m"
                    )
                    confirm_force_norms: list[float] = []
                    confirm_torque_norms: list[float] = []
                    for confirm_index in range(1, args.entry_release_confirm_samples + 1):
                        time.sleep(args.entry_release_confirm_interval_s)
                        confirm_force, confirm_torque = sample(
                            f"entry_release_confirm_{event}_{confirm_index}",
                            cycle,
                            back_mm,
                            target,
                        )
                        confirm_force_norms.append(norm(confirm_force))
                        confirm_torque_norms.append(norm(confirm_torque))
                    confirm_force_median = median(confirm_force_norms)
                    confirm_torque_median = median(confirm_torque_norms)
                    if (
                        confirm_force_median > args.entry_release_force_n
                        or confirm_torque_median > args.entry_release_torque_nm
                    ):
                        print(
                            "摆动卸载候选未确认：原地确认载荷中位数仍偏高；继续搜索。"
                            f" |dF|_median={confirm_force_median:.2f}N "
                            f"|dT|_median={confirm_torque_median:.2f}N*m"
                        )
                        return False
                    print(
                        "SUCCESS：检测到并确认摆动后的接触载荷同步卸载；"
                        "停止在当前位置，不再执行后续摆动或后拉。"
                        f" event={event} back={back_mm:.1f}mm"
                    )
                    log_event("insertion_decision", result="entered_by_wiggle_release_confirmed",
                              motion=event, cycle=cycle, back_mm=round(back_mm, 3))
                    return True

                left = list(centre)
                left[1] += args.wiggle_half_amplitude_mm / 1000.0
                log_event("motion_start", motion="left_y_positive", cycle=cycle,
                          back_mm=round(back_mm, 3), insertion_decision="searching")
                move_wiggle_quick(
                    arm, left,
                    linear_speed_mps=linear_speed_mps,
                    linear_accel_mps2=linear_accel_mps2,
                    endpoint_tolerance_m=args.quick_endpoint_tolerance_mm / 1000.0,
                    endpoint_wait_s=args.quick_endpoint_wait_s,
                    diagnostic=diagnostic,
                )
                dforce, dtorque = sample("wiggle_y_plus", cycle, back_mm, left)
                if wiggle_release_entered("wiggle_y_plus", dforce, dtorque, left):
                    return 0
                if wiggle_entered(dforce, dtorque):
                    report_wiggle_entry("wiggle_y_plus", dforce, dtorque)
                    return 0

                right = list(centre)
                right[1] -= args.wiggle_half_amplitude_mm / 1000.0
                log_event("motion_start", motion="right_y_negative", cycle=cycle,
                          back_mm=round(back_mm, 3), insertion_decision="searching")
                move_wiggle_quick(
                    arm, right,
                    linear_speed_mps=linear_speed_mps,
                    linear_accel_mps2=linear_accel_mps2,
                    endpoint_tolerance_m=args.quick_endpoint_tolerance_mm / 1000.0,
                    endpoint_wait_s=args.quick_endpoint_wait_s,
                    diagnostic=diagnostic,
                )
                dforce, dtorque = sample("wiggle_y_minus", cycle, back_mm, right)
                if wiggle_release_entered("wiggle_y_minus", dforce, dtorque, right):
                    return 0
                if wiggle_entered(dforce, dtorque):
                    report_wiggle_entry("wiggle_y_minus", dforce, dtorque)
                    return 0

                if cycle == cycles:
                    # On the final unsuccessful cycle, preserve the original
                    # return-centre behaviour before reporting search failure.
                    log_event("motion_start", motion="return_to_centre", cycle=cycle,
                              back_mm=round(back_mm, 3), insertion_decision="searching")
                    move_wiggle_quick(
                        arm, centre,
                        linear_speed_mps=linear_speed_mps,
                        linear_accel_mps2=linear_accel_mps2,
                        endpoint_tolerance_m=args.quick_endpoint_tolerance_mm / 1000.0,
                        endpoint_wait_s=args.quick_endpoint_wait_s,
                        diagnostic=diagnostic,
                    )
                    dforce, dtorque = sample("wiggle_return_centre", cycle, back_mm, centre)
                    if wiggle_release_entered("wiggle_return_centre", dforce, dtorque, centre):
                        return 0
                    if wiggle_entered(dforce, dtorque):
                        report_wiggle_entry("wiggle_return_centre", dforce, dtorque)
                        return 0
                else:
                    log_event(
                        "motion_chain_armed",
                        motion="right_y_negative_to_next_back_centre_diagonal",
                        cycle=cycle,
                        next_back_mm=round(min((cycle + 1) * args.back_step_mm, args.max_back_mm), 3),
                    )
                    arriving_from_right_y_negative = True
                log_event("insertion_decision", result="not_entered_continue_search",
                          cycle=cycle, back_mm=round(back_mm, 3))
        log_event("insertion_decision", result="not_entered_max_back_reached",
                  back_mm=round(args.max_back_mm, 3))
        raise RuntimeError(f"在最大后拉 {args.max_back_mm:.1f}mm 内未检测到进洞特征。")
    except PlacementContactReached:
        log_event("insertion_decision", result="entered_by_validated_release")
        move_from_first_hole_to_second_hole()
        return 0
    except Exception as exc:
        log_event("run_error", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        search_ended_at = time.monotonic()
        pure_search_elapsed_s = (
            search_ended_at - search_started_at
            if search_started_at is not None
            else None
        )
        stop_servo_error: Exception | None = None
        if servo_started and arm is not None and not continuous_servo_stopped:
            try:
                try:
                    write_control_snapshot(arm, log_file, "before_stop_servo_pose")
                except Exception as exc:
                    print(f"警告：停止前控制状态记录失败：{exc}", file=sys.stderr)
                arm.stop_servo_pose(2)
                wait_for_right_task_stopped(
                    arm, log_file,
                    confirmed_event="right_servo_stop_confirmed",
                    timeout_event="right_servo_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                )
                try:
                    write_control_snapshot(arm, log_file, "after_stop_servo_pose")
                except Exception as exc:
                    print(f"警告：停止后控制状态记录失败：{exc}", file=sys.stderr)
            except Exception as exc:
                # A placement must never be reported as successful unless the
                # active ServoPose session was confirmed stopped.
                stop_servo_error = exc
        if servo_guard is not None:
            servo_guard.__exit__(None, None, None)
        if sensor is not None:
            sensor.close()
        if arm is not None:
            arm.close()
        log_event(
            "run_finished",
            elapsed_s=round(time.monotonic() - run_started, 3),
            pure_search_elapsed_s=(round(pure_search_elapsed_s, 3) if pure_search_elapsed_s is not None else None),
        )
        log_file.close()
        print(f"运行日志：{output_path}")
        if pure_search_elapsed_s is not None:
            print(f"纯搜索耗时（ServoPose 启动至搜索结束）：{pure_search_elapsed_s:.2f} 秒")
        if stop_servo_error is not None:
            raise RuntimeError("停止右臂 ServoPose 失败，拒绝报告放置成功") from stop_servo_error


# The workpiece-profile entry point is intentionally kept in this file.  It
# translates the minimal JSON profile directly into this module's arguments;
# no subprocess and no call to place_hole_a_wiggle.py is involved.
SINGLE_SEARCH_SPEED_MM_S = 15.0
MAX_SEARCH_ATTEMPTS = 6
PRE_SEARCH_PRESS_MM = 1.0
RETRY_SEARCH_PRESS_INCREMENT_MM = 1.5
PRE_SEARCH_PRESS_SPEED_MM_S = 1.0
RETRY_LIFT_MM = 30.0
RETRY_LIFT_SPEED_MM_S = 10.0
POST_ENTRY_RELIEF_MM = 0.0
POST_ENTRY_RELIEF_OPPOSING_FORCE_N = 1.5
POST_ENTRY_RELIEF_CONFIRM_FRAMES = 2
POST_ENTRY_RELIEF_OBSERVATION_FRAMES = 3
POST_ENTRY_RELIEF_SPEED_MM_S = 1.0
POST_ENTRY_DESCEND_MM = 10.0
POST_ENTRY_DESCEND_FX_RISE_N = 0.0
POST_ENTRY_DESCEND_SPEED_MM_S = 1.0
PRE_SEARCH_INITIAL_DESCEND_MM = 10.0
PRE_SEARCH_FORWARD_MM = 142.0
PRE_SEARCH_DESCEND_MM = 20.0
PRE_SEARCH_APPROACH_SPEED_MM_S = 25.0
SECOND_HOLE_LIFT_MM = 40.0
SECOND_HOLE_Y_OFFSET_MM = 30.0
SECOND_HOLE_DESCEND_MM = 48.0
SECOND_HOLE_LATERAL_SPEED_MM_S = 10.0
SECOND_HOLE_X_RETURN_SPEED_MM_S = 25.0
SECOND_HOLE_DESCEND_SPEED_MM_S = 5.0
WORKPIECE_CONFIGS = {
    1: Path(__file__).resolve().parent / "place_hole_a_wiggle_new_workpiece_1.json",
    2: Path(__file__).resolve().parent / "place_hole_a_wiggle_new_workpiece_2.json",
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
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 配置 {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("配置必须是对象，且 schema_version 必须为 1")
    workpiece = raw.get("workpiece_serial_number")
    if isinstance(workpiece, bool) or not isinstance(workpiece, int) or workpiece < 1:
        raise ValueError("workpiece_serial_number 必须是正整数")
    missing = {"initial_right_joints_deg", "search_axis", "search_direction"} - set(raw)
    if missing:
        raise ValueError(f"配置缺少必填字段: {', '.join(sorted(missing))}")
    stop_conditions = raw.get("stop_conditions", {})
    if not isinstance(stop_conditions, dict):
        raise ValueError("stop_conditions 必须是对象")
    unknown = set(stop_conditions) - STOP_KEYS
    if unknown:
        raise ValueError(f"stop_conditions 包含不支持的字段: {', '.join(sorted(unknown))}")
    return workpiece, raw


def append_scalar(argv: list[str], key: str, value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label}.{key} 必须是字符串或数字")
    argv.extend(["--" + key.replace("_", "-"), str(value)])


def lift_after_exhausted_search(
    *, arm_host: str | None, session_log_event: object | None = None,
) -> None:
    """Safely clear the fixture before the next attempt re-plans to the taught pose."""
    if session_log_event is not None:
        session_log_event(  # type: ignore[operator]
            "retry_lift_recovery_connecting",
            distance_mm=RETRY_LIFT_MM,
            speed_mm_s=RETRY_LIFT_SPEED_MM_S,
            axis="z",
            direction="positive",
        )
    sdk = import_sdk()
    arm = None
    servo_guard = None
    servo_started = False
    try:
        arm = sdk.Arm(host=arm_host or os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
        ensure_ready(arm, 2)
        start = current_pose(arm, 2)
        start_rpy = start[:3] + rotvec_to_rpy(start[3:])
        target_rpy = list(start_rpy)
        target_rpy[2] += RETRY_LIFT_MM / 1000.0
        if session_log_event is not None:
            session_log_event(  # type: ignore[operator]
                "retry_lift_motion_start",
                distance_mm=RETRY_LIFT_MM,
                speed_mm_s=RETRY_LIFT_SPEED_MM_S,
                start_xyz_rpy=start_rpy,
                target_xyz_rpy=target_rpy,
            )
        print(
            "本轮未进洞：沿世界 Z+ LIN 抬升 "
            f"{RETRY_LIFT_MM:.1f}mm，随后重新规划回放置初始点。"
        )
        servo_guard = right_servo_guard()
        servo_guard.__enter__()
        arm.start_servo_pose(  # type: ignore[attr-defined]
            2, pose_type="absolute", frame="world", tcp=[0.0] * 6,
            input_frequency_hz=50.0,
        )
        servo_started = True
        move_in_active_session(
            arm,
            target_rpy,
            linear_speed_mps=RETRY_LIFT_SPEED_MM_S / 1000.0,
            linear_accel_mps2=0.02,
        )
        print("30mm 抬升完成。")
        if session_log_event is not None:
            session_log_event(  # type: ignore[operator]
                "retry_lift_motion_completed", distance_mm=RETRY_LIFT_MM,
            )
    finally:
        try:
            if servo_started and arm is not None:
                arm.stop_servo_pose(2)  # type: ignore[attr-defined]
                wait_for_right_task_stopped(
                    arm,
                    None,
                    confirmed_event="retry_lift_stop_confirmed",
                    timeout_event="retry_lift_stop_wait_timeout",
                    cancel_if_still_running_after_s=0.5,
                    event_logger=session_log_event,
                )
        finally:
            if servo_guard is not None:
                servo_guard.__exit__(None, None, None)
            if arm is not None:
                arm.close()


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
    argv = [
        "--initial-joint-planner", "ompl", "--straight-line",
        "--continuous-straight-search", "--single-continuous-search",
        "--fast-linear-speed-mm-s", str(SINGLE_SEARCH_SPEED_MM_S),
        "--pre-search-press-mm", str(PRE_SEARCH_PRESS_MM),
        "--retry-search-press-increment-mm", str(RETRY_SEARCH_PRESS_INCREMENT_MM),
        "--pre-search-press-speed-mm-s", str(PRE_SEARCH_PRESS_SPEED_MM_S),
        "--search-attempt", str(attempt),
        "--return-search-exhausted",
        "--post-entry-relief-mm", str(POST_ENTRY_RELIEF_MM),
        "--post-entry-relief-opposing-force-n", str(POST_ENTRY_RELIEF_OPPOSING_FORCE_N),
        "--post-entry-relief-confirm-frames", str(POST_ENTRY_RELIEF_CONFIRM_FRAMES),
        "--post-entry-relief-observation-frames", str(POST_ENTRY_RELIEF_OBSERVATION_FRAMES),
        "--post-entry-relief-speed-mm-s", str(POST_ENTRY_RELIEF_SPEED_MM_S),
        "--post-entry-descend-mm", str(POST_ENTRY_DESCEND_MM),
        "--post-entry-descend-fx-rise-n", str(POST_ENTRY_DESCEND_FX_RISE_N),
        "--post-entry-descend-speed-mm-s", str(POST_ENTRY_DESCEND_SPEED_MM_S),
        "--pre-search-initial-descend-mm", str(PRE_SEARCH_INITIAL_DESCEND_MM),
        "--pre-search-forward-mm", str(PRE_SEARCH_FORWARD_MM),
        "--pre-search-descend-mm", str(PRE_SEARCH_DESCEND_MM),
        "--pre-search-approach-speed-mm-s", str(PRE_SEARCH_APPROACH_SPEED_MM_S),
        "--second-hole-lift-mm", str(SECOND_HOLE_LIFT_MM),
        "--second-hole-y-offset-mm", str(SECOND_HOLE_Y_OFFSET_MM),
        "--second-hole-descend-mm", str(SECOND_HOLE_DESCEND_MM),
        "--second-hole-lateral-speed-mm-s", str(SECOND_HOLE_LATERAL_SPEED_MM_S),
        "--second-hole-x-return-speed-mm-s", str(SECOND_HOLE_X_RETURN_SPEED_MM_S),
        "--second-hole-descend-speed-mm-s", str(SECOND_HOLE_DESCEND_SPEED_MM_S),
        "--initial-right-joints-deg", *(str(value) for value in joints),
    ]
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="以工件序号运行内置的 Hole-A 连续搜索逻辑。"
    )
    parser.add_argument("--workpiece", type=int, choices=sorted(WORKPIECE_CONFIGS), required=True)
    parser.add_argument("--live", action="store_true", help="允许连接真实硬件")
    parser.add_argument("--confirm-live", action="store_true", help="确认现场安全，允许真实运动")
    parser.add_argument("--arm-host", help="右臂控制器地址；省略时使用执行器默认值")
    parser.add_argument(
        "--reset-right-task-before-servo",
        action="store_true",
        help="开始 ServoPose 前发送 cancel_task(2) 清理右臂残留任务",
    )
    parser.add_argument("--print-command", action="store_true", help="仅打印内置执行参数，不执行")
    wrapper_args = parser.parse_args()
    if wrapper_args.live != wrapper_args.confirm_live:
        parser.error("真实运动必须同时指定 --live --confirm-live")
    config_path = WORKPIECE_CONFIGS[wrapper_args.workpiece]
    try:
        workpiece, profile = read_profile(config_path)
        if workpiece != wrapper_args.workpiece:
            raise ValueError(f"工件号与配置不一致：{config_path}")
    except ValueError as exc:
        parser.error(str(exc))
    if wrapper_args.print_command:
        argv = profile_argv(
            profile, workpiece, live=wrapper_args.live,
            arm_host=wrapper_args.arm_host,
            reset_right_task_before_servo=wrapper_args.reset_right_task_before_servo,
            attempt=1,
        )
        print(f"执行工件 {workpiece} 配置：{config_path}")
        print("第 1 轮内置执行参数：", shlex.join(argv))
        return 0
    session_log_path = Path(__file__).resolve().parent / "logs" / (
        f"hole_a_workpiece_{workpiece}_retry_session_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    )
    session_log_path.parent.mkdir(parents=True, exist_ok=True)
    with session_log_path.open("w", encoding="utf-8") as session_log:
        def session_log_event(event: str, **fields: object) -> None:
            session_log.write(json.dumps({
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
                "event": event,
                **fields,
            }, ensure_ascii=False) + "\n")
            session_log.flush()

        session_log_event(
            "retry_session_started",
            workpiece=workpiece,
            max_attempts=MAX_SEARCH_ATTEMPTS,
            first_search_press_mm=PRE_SEARCH_PRESS_MM,
            retry_search_press_increment_mm=RETRY_SEARCH_PRESS_INCREMENT_MM,
            retry_lift_mm=RETRY_LIFT_MM,
        )
        print(f"重试会话日志将实时写入：{session_log_path}")
        try:
            for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):
                argv = profile_argv(
                    profile, workpiece, live=wrapper_args.live,
                    arm_host=wrapper_args.arm_host,
                    reset_right_task_before_servo=wrapper_args.reset_right_task_before_servo,
                    attempt=attempt,
                )
                session_log_event(
                    "attempt_started", attempt=attempt,
                    press_depth_mm=(
                        PRE_SEARCH_PRESS_MM
                        + (attempt - 1) * RETRY_SEARCH_PRESS_INCREMENT_MM
                    ),
                    command=argv,
                )
                print(f"执行工件 {workpiece} 配置：{config_path}（第 {attempt}/{MAX_SEARCH_ATTEMPTS} 轮）")
                print("内置执行参数：", shlex.join(argv))
                result = placement_main(argv)
                session_log_event("attempt_finished", attempt=attempt, exit_code=result)
                if result == 0:
                    session_log_event("retry_session_succeeded", attempt=attempt)
                    return 0
                if result != SEARCH_DISTANCE_EXHAUSTED_EXIT_CODE:
                    return result
                if attempt == MAX_SEARCH_ATTEMPTS:
                    raise RuntimeError(
                        f"已完成 {MAX_SEARCH_ATTEMPTS} 轮搜索，每轮最大距离 40mm，仍未检测到进洞特征。"
                    )
                session_log_event("retry_lift_recovery_requested", attempt=attempt)
                lift_after_exhausted_search(
                    arm_host=wrapper_args.arm_host,
                    session_log_event=session_log_event,
                )
        except Exception as exc:
            session_log_event(
                "retry_session_error",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        finally:
            session_log_event("retry_session_finished")
    raise AssertionError("搜索重试循环意外结束")


if __name__ == "__main__":
    script_started = time.monotonic()
    try:
        raise SystemExit(main())
    finally:
        print(f"本次执行总耗时：{time.monotonic() - script_started:.2f} 秒", flush=True)
