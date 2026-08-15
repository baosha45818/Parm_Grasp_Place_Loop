#!/usr/bin/env python3
"""串行启动独立 Inspect 和 Left Place，并保持旧总控的文件握手协议。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
INSPECT = Path(__file__).with_name("standalone_inspect.py")
LEFT_PLACE = ROOT / "modules/module_3_left_place/standalone_left_place.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loop_types import ResultStatus
from modules.module_status import (
    report as report_module_status,
    report_exception as report_module_exception,
)
from modules.module_3_left_place.placement_config import load_workpiece_parameters
from modules.stage_result import write_stage_result
from tool.runtime_log import RuntimeLogCapture
from tool.workpiece_timer import module_timer


WORKPIECE_PARAMETERS = {
    "1": load_workpiece_parameters("1"),
    "2": load_workpiece_parameters("2"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workpiece",
        choices=("1", "2"),
        required=True,
        help="本次调用明确指定的工件类型；不从环境或历史默认值推断",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("auto", "manual", "auto-then-manual"),
        default=os.environ.get("PARM_INSPECT_PROMPT_MODE"),
        help="显式转发给通用 standalone_inspect；缺省时统一使用 auto",
    )
    parser.add_argument(
        "--planner-host",
        default=os.environ.get("PARM_PLANNER_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--arm-host",
        default=os.environ.get("PARM_ARM_HOST", os.environ.get("GRASPNET_V7_ARM_HOST", "192.168.3.11")),
    )
    parser.add_argument(
        "--gripper-host",
        default=os.environ.get("PARM_GRIPPER_HOST", "192.168.3.11"),
    )
    parser.add_argument("--gripper-hand", type=int, default=1)
    parser.add_argument(
        "--execute-url",
        default="http://127.0.0.1:5667/middle/waypoint/execute",
    )
    left_scale = float(os.environ.get("PARM_LEFT_SPEED_SCALE", "2.0"))
    parser.add_argument("--preset-speed", type=float, default=0.2 * left_scale)
    parser.add_argument("--motion-speed", type=float, default=0.1 * left_scale)
    parser.add_argument("--point-threshold", type=int, default=700)
    parser.add_argument(
        "--post-graspup-planner",
        choices=("rrtconnect", "rrtstar"),
        default="rrtstar",
    )
    parser.add_argument("--offline-start-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--graspup-lift-m", type=float, default=0.10)
    parser.add_argument("--target-xy-radius-m", type=float, default=0.0)
    parser.add_argument(
        "--branch1-front-target-object-xyz",
        type=float,
        nargs=3,
        default=[0.517668, 0.008779, -0.300],
    )
    parser.add_argument(
        "--branch1-front-target-z-candidates-m",
        type=float,
        nargs="+",
        default=[-0.38, -0.37, -0.36, -0.35, -0.34],
    )
    parser.add_argument(
        "--branch1-front-release-object-z-candidates-m",
        type=float,
        nargs="+",
        default=[-0.40, -0.39, -0.38],
    )
    parser.add_argument(
        "--branch1-reversed-target-object-xyz",
        type=float,
        nargs=3,
        default=[0.437668, 0.238779, -0.300],
    )
    parser.add_argument(
        "--branch1-reversed-target-z-candidates-m",
        type=float,
        nargs="+",
        default=[-0.38, -0.37, -0.36, -0.35, -0.34],
    )
    parser.add_argument(
        "--branch1-reversed-release-object-z-candidates-m",
        type=float,
        nargs="+",
        default=[-0.39, -0.38, -0.37, -0.36, -0.35],
    )
    parser.add_argument("--weight-handoff-file", type=Path)
    parser.add_argument("--placement-ready-file", type=Path)
    parser.add_argument("--placement-permit-file", type=Path)
    parser.add_argument("--auto-confirm-motion", action="store_true")
    parser.add_argument("--skip-motion", action="store_true")
    parser.add_argument(
        "--skip-inspect-direct",
        action="store_true",
        help="跳过PC1/PC2与ICP，直接以Module 3固定 left_eef DIRECT放置。",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    # 该包装器会把放置参数显式转交给 Module 3；因此必须按照所选工件替换
    # 未在命令行给出的默认值，避免包装器旧值覆盖工件参数文件。
    workpiece_parameters = WORKPIECE_PARAMETERS[args.workpiece]
    supplied = set(sys.argv[1:])

    def was_supplied(option: str) -> bool:
        return any(token == option or token.startswith(f"{option}=") for token in supplied)

    defaults = (
        ("target_xy_radius_m", "--target-xy-radius-m", workpiece_parameters.TARGET_XY_RADIUS_M),
        ("branch1_front_target_object_xyz", "--branch1-front-target-object-xyz", workpiece_parameters.FRONT_TARGET_OBJECT_XYZ_M),
        ("branch1_front_target_z_candidates_m", "--branch1-front-target-z-candidates-m", workpiece_parameters.FRONT_TARGET_Z_CANDIDATES_M),
        ("branch1_front_release_object_z_candidates_m", "--branch1-front-release-object-z-candidates-m", workpiece_parameters.FRONT_RELEASE_Z_CANDIDATES_M),
        ("branch1_reversed_target_object_xyz", "--branch1-reversed-target-object-xyz", workpiece_parameters.REVERSED_TARGET_OBJECT_XYZ_M),
        ("branch1_reversed_target_z_candidates_m", "--branch1-reversed-target-z-candidates-m", workpiece_parameters.REVERSED_TARGET_Z_CANDIDATES_M),
        ("branch1_reversed_release_object_z_candidates_m", "--branch1-reversed-release-object-z-candidates-m", workpiece_parameters.REVERSED_RELEASE_Z_CANDIDATES_M),
    )
    for attribute, option, value in defaults:
        if not was_supplied(option):
            setattr(args, attribute, list(value) if isinstance(value, list) else value)
    return args


def run(command: list[str], label: str) -> None:
    module = {
        "Module 2 Inspect": "module_2_left_inspect",
        "Module 3 Left Place": "module_3_left_place",
    }.get(label, label.lower().replace(" ", "_"))
    report_module_status(module, ResultStatus.WAIT, f"开始：{label}")
    print(f"\n[apart] {label}: {' '.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        if module == "module_2_left_inspect":
            report_module_status(
                module,
                ResultStatus.RETRY,
                (
                    f"Module 2 失败，exit={exc.returncode}；"
                    "返回相机重试层，保持 graspup 交接状态后重试"
                ),
                reason_code="MODULE_2_RETRY",
            )
        else:
            report_module_exception(
                module,
                exc,
                action="终止当前 Place 子链并交由总控安全停机，保留现场",
                target_module="initial_state",
                recovery_level="SAFE_STOP",
                reason_code=f"EXIT_{exc.returncode}",
            )
        raise
    except KeyboardInterrupt:
        report_module_status(
            module,
            ResultStatus.ABORTED,
            f"{label}收到中断，停止当前模块",
            reason_code="INTERRUPTED",
        )
        raise
    except BaseException as exc:
        report_module_exception(
            module,
            exc,
            action="终止当前子链并交由上层执行安全停机",
            target_module="initial_state",
            recovery_level="SAFE_STOP",
        )
        raise
    else:
        report_module_status(module, ResultStatus.SUCCESS, f"完成：{label}")


def main() -> int:
    args = parse_args()
    if (args.placement_ready_file is None) != (args.placement_permit_file is None):
        raise ValueError("placement ready/permit 文件必须同时提供")
    work_dir = (
        args.output_dir
        or (
            args.placement_ready_file.parent
            if args.placement_ready_file is not None
            else ROOT / "output/module_2_left_inspect_apart/manual"
        )
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PARM_PLANNING_RECORD_DIR", str(work_dir / "planning_trajectories")
    )
    attempt_tag = f"{os.getpid()}"
    inspection_result = work_dir / f"left_inspect_result_{attempt_tag}.json"
    placement_result = work_dir / f"left_place_result_{attempt_tag}.json"

    inspect_command = [
        str(PYTHON), str(INSPECT),
        "--planner-host", args.planner_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
        "--gripper-hand", str(args.gripper_hand),
        "--execute-url", args.execute_url,
        "--preset-speed", str(args.preset_speed),
        "--motion-speed", str(args.motion_speed),
        "--point-threshold", str(args.point_threshold),
        "--workpiece", args.workpiece,
        "--result-file", str(inspection_result),
    ]
    if args.prompt_mode is not None:
        inspect_command.extend(["--prompt-mode", args.prompt_mode])
    if args.weight_handoff_file is not None:
        inspect_command.extend(["--weight-handoff-file", str(args.weight_handoff_file)])
    if args.skip_motion:
        inspect_command.append("--skip-motion")
    if args.skip_inspect_direct:
        inspection_payload = {
            "stage": "left_inspect",
            "outcome": "success",
            "direct_only_without_inspect": True,
            "inspection_skipped_reason": "loop_direct_place_after_grasp",
            "workpiece": args.workpiece,
            "weight_handoff_file": (
                str(args.weight_handoff_file) if args.weight_handoff_file else None
            ),
        }
        write_stage_result(inspection_result, inspection_payload)
        report_module_status(
            "module_2_left_inspect",
            ResultStatus.SUCCESS,
            "总控DIRECT模式：跳过PC1/PC2、SAM2和ICP；已发布无ICP DIRECT交接",
        )
        print("[apart] 总控DIRECT模式：跳过 Module 2 Inspect，直接启动 Module 3 DIRECT。", flush=True)
    else:
        with module_timer("module_2_left_inspect"):
            run(inspect_command, "Module 2 Inspect")

    inspection_payload = json.loads(inspection_result.read_text(encoding="utf-8"))
    inspection_outcome = inspection_payload.get("outcome")
    if inspection_outcome in {
        "held_object_dropped",
        "insufficient_point_cloud_discarded",
        "pc1_pc2_registration_failed_recovered",
    }:
        # These are successful safety recoveries from Module 2, not inputs to
        # Module 3.  Preserve the existing ready-file protocol so the parent
        # scheduler resets this workpiece and restarts Module 1.
        if args.placement_ready_file is not None:
            args.placement_ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.placement_ready_file.write_text(
                json.dumps(inspection_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        report_module_status(
            "module_2_left_inspect",
            ResultStatus.RETRY,
            f"Inspect 正常恢复：{inspection_outcome}；跳过 Module 3 并返回总控",
            reason_code=inspection_outcome.upper(),
        )
        print(
            f"[apart] Inspect outcome={inspection_outcome}；"
            "已跳过 Module 3 Left Place。",
            flush=True,
        )
        return 0

    place_command = [
        str(PYTHON), str(LEFT_PLACE),
        "--inspection-result", str(inspection_result),
        "--workpiece", args.workpiece,
        "--result-file", str(placement_result),
        "--planner-host", args.planner_host,
        "--arm-host", args.arm_host,
        "--gripper-host", args.gripper_host,
        "--gripper-hand", str(args.gripper_hand),
        "--execute-url", args.execute_url,
        "--preset-speed", str(args.preset_speed),
        "--motion-speed", str(args.motion_speed),
        "--post-graspup-planner", args.post_graspup_planner,
        "--offline-start-tolerance-deg", str(args.offline_start_tolerance_deg),
        "--graspup-lift-m", str(args.graspup_lift_m),
        "--target-xy-radius-m", str(args.target_xy_radius_m),
        "--branch1-front-target-object-xyz",
        *[str(value) for value in args.branch1_front_target_object_xyz],
        "--branch1-front-target-z-candidates-m",
        *[str(value) for value in args.branch1_front_target_z_candidates_m],
        "--branch1-front-release-object-z-candidates-m",
        *[str(value) for value in args.branch1_front_release_object_z_candidates_m],
        "--branch1-reversed-target-object-xyz",
        *[str(value) for value in args.branch1_reversed_target_object_xyz],
        "--branch1-reversed-target-z-candidates-m",
        *[str(value) for value in args.branch1_reversed_target_z_candidates_m],
        "--branch1-reversed-release-object-z-candidates-m",
        *[str(value) for value in args.branch1_reversed_release_object_z_candidates_m],
    ]
    if args.weight_handoff_file is not None:
        place_command.extend(["--weight-handoff-file", str(args.weight_handoff_file)])
    if args.auto_confirm_motion:
        place_command.append("--auto-confirm-motion")
    if args.skip_motion:
        place_command.append("--skip-motion")
    if args.placement_ready_file is not None:
        place_command.extend([
            "--placement-ready-file", str(args.placement_ready_file),
            "--placement-permit-file", str(args.placement_permit_file),
        ])
    with module_timer("module_3_left_place"):
        run(place_command, "Module 3 Left Place")
    return 0


if __name__ == "__main__":
    log_dir = ROOT / "output/module_2_left_inspect_apart/flows" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    log_path = log_dir / "run.log"
    with RuntimeLogCapture(log_path):
        print(f"[log] 本次 Inspect+Place 日志: {log_path}", flush=True)
        print(f"[log] command: {' '.join(sys.argv)}", flush=True)
        try:
            raise SystemExit(main())
        except SystemExit:
            raise
        except KeyboardInterrupt:
            print("[log] 收到 Ctrl+C，Inspect+Place 中断。", flush=True)
            raise SystemExit(130)
        except BaseException:
            traceback.print_exc()
            raise SystemExit(1)
