#!/usr/bin/env python3
"""反面分支：左手持件观察、ICP定位并移动到右手交接位。

执行前提是 Module 5-1 已完成左手反抓，且工件相对左夹爪保持不动。本模块
只负责左侧准备工作，不移动右臂、不闭合右夹爪，也不释放左夹爪。

成功后会输出本轮 ``result.json``。传入 ``--handoff-result-file`` 时，还会
原子写入一个供 Module 8-2 或兼容总控读取的 JSON 握手文件。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules.reverse_grasp_handoff_positions import (
    load_workpiece_handoff_profile,
)
PYTHON = ROOT / ".venv/bin/python"
if not PYTHON.is_file():
    raise RuntimeError(f"handoff runtime environment does not exist: {PYTHON}")
if __name__ == "__main__" and Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(
        str(PYTHON),
        [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

# 新总控会显式导出 PARM_BODY_HEAD_CALIBRATION。保留按 mode 推导的回退，
# 让已经启动的总控在人工确认后重试交接时，也不会错误退回 xiangxin 标定。
_DEPLOYMENT_MODE = os.environ.get("PARM_DEPLOYMENT_MODE", "xiangxin")
_DEFAULT_ALIGNMENT_BY_MODE = {
    "gzhu": ROOT / "tools/calibration/head_d435/gzhu_reverse_observation.json",
    "xiangxin": ROOT / "tools/calibration/head_d435/xiangxin_reverse_observation.json",
}
DEFAULT_ALIGNMENT = Path(
    os.environ.get(
        "PARM_BODY_HEAD_CALIBRATION",
        _DEFAULT_ALIGNMENT_BY_MODE.get(
            _DEPLOYMENT_MODE,
            _DEFAULT_ALIGNMENT_BY_MODE["xiangxin"],
        ),
    )
).expanduser().resolve()
# Backward-compatible exports used by the reversed-head observation contract.
# The per-workpiece profile remains the runtime source of truth.
HANDOFF_HEAD_PITCH_DEG = 25.0
HANDOFF_HEAD_YAW_DEG = 35.0
DEFAULT_OUTPUT_DIR = ROOT / "output/reversed_head_pose_d435"
DEFAULT_AUTO_MASK_CONFIG = (
    ROOT
    / "tools/handoff_observation/head_d435_auto_mask_config.json"
)
DEPLOYMENT_OBSERVATION_PROFILES = (
    Path(__file__).resolve().parent / "deployment_observation_profiles.json"
)
WORKPIECE_2_CONFIG_DIR = Path(__file__).resolve().parent / "workpiece_2_config"
WORKPIECE_2_POSE_CONFIG = WORKPIECE_2_CONFIG_DIR / "head_d435_pose_config.json"
WORKPIECE_2_AUTO_MASK_CONFIG = (
    WORKPIECE_2_CONFIG_DIR / "head_d435_auto_mask_config.json"
)
WORKPIECE_2_HANDOFF_CONFIG = WORKPIECE_2_CONFIG_DIR / "handoff_config.json"
WORKPIECE_2_OUTPUT_DIR = ROOT / "output/module_5_2_workpiece_2_head_d435"
HEAD_MOTION = ROOT / "tools/move_head_gimbal.py"
NO_GRASP_EXIT_CODE = 20
LEFT_GRIPPER_HELD_POSITION_MIN_PERCENT = 20.0


def head_feedback_pitch_sign() -> float:
    """返回当前部署中「驱动反馈 -> 云台命令角」的 pitch 符号。"""
    return -1.0 if os.environ.get("PARM_DEPLOYMENT_MODE") == "gzhu" else 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment",
        type=Path,
        default=DEFAULT_ALIGNMENT,
        help="头部D435参考外参；运行时按实际交接观察角动态传播",
    )
    parser.add_argument(
        "--sam2-url",
        default=os.environ.get("PARM_SAM2_URL"),
        help="SAM2 服务地址；未指定时使用 Module 0 已验证的项目内服务 socket",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "interactive"),
        default=None,
        help=(
            "SAM2分割方式；默认auto使用固定观察位的自动标点（1、2号正点）；"
            "interactive使用全手工ROI/正负点"
        ),
    )
    parser.add_argument(
        "--auto-mask-config",
        type=Path,
        help="覆盖当前部署的固定观察位自动 ROI/SAM2 正负点配置",
    )
    parser.add_argument("--planner-host", default=os.environ.get("PARM_PLANNER_HOST", "127.0.0.1"))
    # The legacy observation helpers name this endpoint ``driver-host``, but
    # they use it exclusively to read the head pose.  Xiangxin hosts it on the
    # head controller rather than localhost.
    parser.add_argument("--driver-host", default=os.environ.get("PARM_HEAD_HOST", "192.168.3.22"))
    parser.add_argument("--arm-host", default=os.environ.get("PARM_ARM_HOST", "192.168.3.11"))
    parser.add_argument("--gripper-host", default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"))
    parser.add_argument(
        "--head-host",
        default=os.environ.get("PARM_HEAD_HOST", "192.168.3.22"),
        help="云台 driver 主机；不复用历史 --driver-host",
    )
    parser.add_argument(
        "--workpiece",
        choices=("1", "2"),
        default=os.environ.get("PARM_LOOP_WORKPIECE", "1"),
        help="工件编号；总控默认从 PARM_LOOP_WORKPIECE 传入，缺省为原工件1流程",
    )
    parser.add_argument(
        "--mode",
        choices=("xiangxin", "gzhu"),
        default=os.environ.get("PARM_DEPLOYMENT_MODE", "xiangxin"),
        help="现场部署；必须与 Module 4-1 的 mode 相同",
    )
    parser.add_argument(
        "--handoff-result-file",
        type=Path,
        help="可选：写入包含本轮result.json路径的Module 5-2握手JSON",
    )
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=2.0 * float(os.environ.get("PARM_LEFT_SPEED_SCALE", "1.0")),
        help="相对 0.05 基准速度的倍率；默认使观察位速度为 0.1",
    )
    parser.add_argument(
        "--require-confirmation",
        action="store_true",
        help="恢复各运动阶段人工确认；默认自动确认",
    )
    return parser.parse_args()


def run(
    command: list[str], *, recoverable_return_codes: set[int] | None = None
) -> int:
    print("\n[module-5-2] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    recoverable = recoverable_return_codes or set()
    if completed.returncode != 0 and completed.returncode not in recoverable:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return int(completed.returncode)


def cancel_left_arm(arm_host: str) -> None:
    try:
        from parm_sdk import Arm
    except Exception as exc:
        print(
            f"[stop] 无法导入parm_sdk，未能停止左臂：{exc}",
            flush=True,
        )
        return
    arm = None
    try:
        arm = Arm(host=arm_host, status_port=None)
        arm.cancel_task(1)
        print("[stop] left arm cancel_task(1) 已发送", flush=True)
    except Exception as exc:
        print(f"[stop] left arm cancel_task失败：{exc}", flush=True)
    finally:
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass


def newest_result(output_dir: Path, previous: set[Path]) -> Path:
    results = [
        path
        for path in output_dir.glob("*/result.json")
        if path not in previous
    ]
    if not results:
        raise RuntimeError(
            f"D435位姿估计没有在{output_dir}生成新的result.json"
        )
    return max(results, key=lambda path: path.stat().st_mtime).resolve()


def camera_serial_from_alignment(alignment: Path) -> str:
    """读取与 body_base 外参同属一次现场标定的头部相机序列号。"""
    payload = json.loads(alignment.read_text(encoding="utf-8"))
    serial = str(payload.get("camera_serial", "")).strip()
    if not serial:
        raise ValueError(f"头部标定缺少 camera_serial：{alignment}")
    return serial


def deployment_observation_profile() -> dict:
    """读取现场专属观察角与可用的 SAM2 自动点配置。"""
    mode = os.environ.get("PARM_DEPLOYMENT_MODE", "xiangxin").strip()
    payload = json.loads(DEPLOYMENT_OBSERVATION_PROFILES.read_text(encoding="utf-8"))
    try:
        profile = dict(payload[mode])
        head_pose = dict(profile["head_pose_deg"])
        profile["head_pose_deg"] = {
            "pitch": float(head_pose["pitch"]),
            "yaw": float(head_pose["yaw"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"反抓交接观察配置无效或缺少部署 {mode!r}: "
            f"{DEPLOYMENT_OBSERVATION_PROFILES}"
        ) from exc
    profile["mode"] = mode
    return profile


def publish_handoff_result(
    destination: Path,
    result: Path,
    *,
    workpiece: str,
    profile_id: str | None = None,
    handoff_center_body_m: list[float] | None = None,
) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "module_5_2_left_handoff_ready",
        "workpiece": workpiece,
        "result": str(result),
        "output_frame": "body_base",
        "left_gripper_holds_object": True,
        "created_at": time.time(),
    }
    if profile_id is not None:
        payload["profile_id"] = profile_id
    if handoff_center_body_m is not None:
        payload["handoff_center_body_m"] = handoff_center_body_m
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"[module-5-2] 已写入交接握手：{destination}", flush=True)


def main() -> int:
    args = parse_args()
    # 后续 profile 加载、云台反馈符号及右臂交接都从同一个环境读取 mode。
    os.environ["PARM_DEPLOYMENT_MODE"] = args.mode
    if args.workpiece == "2":
        if args.mask_mode is None:
            args.mask_mode = "auto"
        return main_workpiece_2(args)
    profile = load_workpiece_handoff_profile("1", mode=args.mode)
    observation = dict(profile["left_observation"])
    deployment_observation = deployment_observation_profile()
    # 左手关节观察位和交接中心仍由工件 profile 管理；云台角、自动点必须
    # 按现场管理，不能把 xiangxin 的 25°/35° 点位套到 gzhu 的 2°/26.5° 画面。
    observation["head_pose_deg"] = deployment_observation["head_pose_deg"]
    # TOML 是交接链的最终部署配置；JSON 保留给已有工具使用。启动时比对二者，
    # 防止有人只改了一侧而把广珠自动标点套到祥鑫观察角（或反过来）。
    if observation["head_pose_deg"] != profile["left_observation"]["head_pose_deg"]:
        raise RuntimeError(
            "交接部署配置不一致：reverse_grasp_handoff_positions.toml 与 "
            "deployment_observation_profiles.json 的云台观察角不同"
        )
    if args.mask_mode is None:
        args.mask_mode = str(deployment_observation["default_mask_mode"])
    handoff_center = list(profile["handoff"]["center_body_m"])
    from modules.module_0_prepare.ensure_sam2_service import (
        require_sam2_service,
        sam2_url,
    )
    require_sam2_service()
    if args.sam2_url is None:
        args.sam2_url = sam2_url()
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    if not HEAD_MOTION.is_file():
        raise FileNotFoundError(f"云台动作工具不存在：{HEAD_MOTION}")
    alignment = args.alignment.expanduser().resolve()
    if not alignment.is_file():
        raise FileNotFoundError(f"D435外参不存在：{alignment}")
    camera_serial = camera_serial_from_alignment(alignment)
    output_dir = args.output_dir.expanduser().resolve()
    auto_mask_config: Path | None = args.auto_mask_config
    if auto_mask_config is None:
        auto_mask_config = Path(observation["auto_mask_config"]).resolve()
    else:
        auto_mask_config = auto_mask_config.expanduser().resolve()
    if args.mask_mode == "auto":
        if auto_mask_config is None:
            raise RuntimeError(
                f"{deployment_observation['mode']} 在观察角 "
                f"{observation['head_pose_deg']} 尚无已标定的固定 SAM2 自动点；"
                "请使用 --mask-mode interactive 完成交互分割后再标定自动点。"
            )
        if not auto_mask_config.is_file():
            raise FileNotFoundError(f"自动分割配置不存在：{auto_mask_config}")
    feedback_sign = head_feedback_pitch_sign()

    # 1. 左手经共用高位准备点到共用观察位。
    observation_exit = run([
        sys.executable,
        "-m",
        "tools.handoff_observation.execute_plan",
        "--planner-host",
        args.planner_host,
        "--driver-host",
        args.driver_host,
        "--arm-host",
        args.arm_host,
        "--gripper-host",
        args.gripper_host,
        "--speed",
        str(0.05 * args.speed_multiplier),
        "--middle-joints-deg",
        *(str(value) for value in observation["prepare_joints_deg"]),
        "--observation-joints-deg",
        *(str(value) for value in observation["observation_joints_deg"]),
        "--skip-head",
        "--verify-left-gripper-held",
        "--held-position-min-percent",
        str(LEFT_GRIPPER_HELD_POSITION_MIN_PERCENT),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ], recoverable_return_codes={NO_GRASP_EXIT_CODE})
    if observation_exit == NO_GRASP_EXIT_CODE:
        print(
            "[module-5-2] 观察位夹爪复检未确认持件；左臂已原路返回，"
            "通知上层重新执行反抓。",
            flush=True,
        )
        return NO_GRASP_EXIT_CODE

    # 2. 反手交接专用流程：云台到标定角度，再运行 D435、SAM2 和 ICP。
    previous = set(output_dir.glob("*/result.json"))
    run([
        str(PYTHON), str(HEAD_MOTION), "--execute", "--host", args.head_host,
        "--pitch", str(observation["head_pose_deg"]["pitch"]),
        "--yaw", str(observation["head_pose_deg"]["yaw"]),
        "--feedback-pitch-sign", str(feedback_sign),
        "--timeout-s", "60",
    ])
    run([
        sys.executable,
        "-m",
        "tools.handoff_observation.estimate_pose_head_d435",
        "--alignment",
        str(alignment),
        "--camera-serial",
        camera_serial,
        "--propagate-head-extrinsic",
        "--output-frame",
        "body_base",
        "--sam2-url",
        args.sam2_url,
        "--mask-mode",
        args.mask_mode,
        "--no-auto-gripper-prompt",
        "--auto-mask-config",
        str(auto_mask_config or DEFAULT_AUTO_MASK_CONFIG),
        "--output-dir",
        str(output_dir),
        "--planner-host",
        args.planner_host,
        "--driver-host",
        args.driver_host,
        "--expected-head-pitch",
        str(observation["head_pose_deg"]["pitch"]),
        "--expected-head-yaw",
        str(observation["head_pose_deg"]["yaw"]),
        "--head-feedback-pitch-sign", str(feedback_sign),
    ])
    result = newest_result(output_dir, previous)
    print(
        f"[module-5-2] 反手交接观察完成；SAM2分割模式={args.mask_mode}；"
        f"自动配置={auto_mask_config if args.mask_mode == 'auto' else '未使用'}；"
        f"部署={deployment_observation['mode']}；"
        f"云台={observation['head_pose_deg']}",
        flush=True,
    )
    print(f"[module-5-2] 本轮D435结果：{result}", flush=True)

    # 3. 复用拍摄时左TCP到工件的刚性关系，将工件送到交接位。
    run([
        sys.executable,
        "-m",
        "tools.handoff_observation.move_object_center",
        "--result",
        str(result),
        "--target-center",
        *(str(value) for value in handoff_center),
        "--fixed-handoff-orientation",
        "--reuse-grasp-relation",
        "--execute",
        "--planner-host",
        args.planner_host,
        "--speed-multiplier",
        str(args.speed_multiplier),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])

    if args.handoff_result_file:
        publish_handoff_result(
            args.handoff_result_file,
            result,
            workpiece="1",
            handoff_center_body_m=handoff_center,
        )
    print(
        "[module-5-2] 左手已持工件1到共用高位交接中心="
        f"{handoff_center}；左夹爪保持夹紧，右臂尚未运动。",
        flush=True,
    )
    print(f"MODULE_5_2_RESULT={result}", flush=True)
    return 0


def main_workpiece_2(args: argparse.Namespace) -> int:
    """工件2左手交接准备：只观察和送交接位，绝不驱动右臂。"""
    from modules.module_0_prepare.ensure_sam2_service import (
        require_sam2_service,
        sam2_url,
    )

    profile = load_workpiece_handoff_profile("2", mode=args.mode)
    observation = profile["left_observation"]
    handoff = profile["handoff"]
    handoff_center = list(handoff["center_body_m"])
    head_pose = observation["head_pose_deg"]
    feedback_sign = head_feedback_pitch_sign()
    if args.speed_multiplier <= 0.0:
        raise ValueError("--speed-multiplier must be positive")
    for path, label in (
        (HEAD_MOTION, "云台动作工具"),
        (WORKPIECE_2_POSE_CONFIG, "工件2 D435位姿配置"),
        (WORKPIECE_2_AUTO_MASK_CONFIG, "工件2自动分割配置"),
        (WORKPIECE_2_HANDOFF_CONFIG, "工件2交接配置"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")

    require_sam2_service()
    if args.sam2_url is None:
        args.sam2_url = sam2_url()
    output_dir = WORKPIECE_2_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = set(output_dir.glob("*/result.json"))

    # 工件2复用已验证的左臂头部观察位；该动作不涉及右臂。
    observation_exit = run([
        sys.executable,
        "-m",
        "tools.handoff_observation.execute_plan",
        "--planner-host", args.planner_host,
        "--driver-host", args.driver_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
        "--speed", str(0.05 * args.speed_multiplier),
        "--middle-joints-deg",
        *(str(value) for value in observation["prepare_joints_deg"]),
        "--observation-joints-deg",
        *(str(value) for value in observation["observation_joints_deg"]),
        "--skip-head",
        "--verify-left-gripper-held",
        "--held-position-min-percent", str(LEFT_GRIPPER_HELD_POSITION_MIN_PERCENT),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ], recoverable_return_codes={NO_GRASP_EXIT_CODE})
    if observation_exit == NO_GRASP_EXIT_CODE:
        print(
            "[module-5-2/workpiece2] 观察位夹爪复检未确认持件；"
            "左臂已原路返回，通知上层重新执行反抓。",
            flush=True,
        )
        return NO_GRASP_EXIT_CODE
    run([
        str(PYTHON), str(HEAD_MOTION), "--execute", "--host", args.head_host,
        "--pitch", str(head_pose["pitch"]),
        "--yaw", str(head_pose["yaw"]),
        "--feedback-pitch-sign", str(feedback_sign),
        "--timeout-s", "60",
    ])
    run([
        sys.executable,
        "-m",
        "tools.handoff_observation.estimate_pose_head_d435",
        "--config", str(WORKPIECE_2_POSE_CONFIG),
        "--alignment", str(observation["alignment"]),
        "--propagate-head-extrinsic",
        "--output-frame", "body_base",
        "--sam2-url", args.sam2_url,
        "--mask-mode", args.mask_mode,
        "--no-auto-gripper-prompt",
        "--auto-mask-config", str(WORKPIECE_2_AUTO_MASK_CONFIG),
        "--output-dir", str(output_dir),
        "--planner-host", args.planner_host,
        "--driver-host", args.driver_host,
        "--expected-head-pitch", str(head_pose["pitch"]),
        "--expected-head-yaw", str(head_pose["yaw"]),
        "--head-feedback-pitch-sign", str(feedback_sign),
    ])
    result = newest_result(output_dir, previous)
    run([
        sys.executable,
        "-m",
        "tools.handoff_observation.move_object_center",
        "--result", str(result),
        "--config", str(WORKPIECE_2_HANDOFF_CONFIG),
        "--target-center", *(str(value) for value in handoff_center),
        "--cartesian-planner", "rrtstar",
        "--fixed-handoff-orientation",
        "--reuse-grasp-relation",
        "--execute",
        "--planner-host", args.planner_host,
        "--speed-multiplier", str(args.speed_multiplier),
        *([] if args.require_confirmation else ["--auto-confirm"]),
    ])
    if args.handoff_result_file:
        publish_handoff_result(
            args.handoff_result_file,
            result,
            workpiece="2",
            profile_id=str(profile["profile_id"]),
            handoff_center_body_m=handoff_center,
        )
    print(
        "[module-5-2/workpiece2] 左手已将工件2送到"
        f"body_base交接中心={handoff_center}；"
        "左夹爪保持夹紧，右臂尚未运动。",
        flush=True,
    )
    print(f"MODULE_5_2_RESULT={result}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stop] 收到Ctrl+C，停止左臂当前任务。", flush=True)
        cancel_left_arm(parse_args().arm_host)
        raise SystemExit(130)
