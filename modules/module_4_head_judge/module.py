"""使用头部 RGB-D、SAM2 和 CAD/ICP 发布工件正反与位姿。

该文件只组织视觉流程，不导入规划器，也不控制机械臂或夹爪。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


LOOP_ROOT = Path(__file__).resolve().parents[2]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))
MODULE_ROOT = Path(__file__).resolve().parent
MODULE_4_1_ROOT = MODULE_ROOT / "module_4_1_buffer_state_estimation"
VISION_ENTRY = MODULE_4_1_ROOT / "run_orientation.py"
CAPTURE_ENTRY = MODULE_4_1_ROOT / "vision/capture_rgbd.py"
CONFIG_ROOT = MODULE_4_1_ROOT / "configs"
SUPPORTED_WORKPIECES = {"1", "2"}
SOURCE_PYTHON = LOOP_ROOT / ".venv/bin/python"
from modules.module_0_prepare.ensure_sam2_service import require_sam2_service, sam2_url
from modules.module_4_head_judge.module_4_1_buffer_state_estimation.deployment_modes import (
    HeadD435Mode,
    apply_mode_environment,
    load_mode,
)


class ObjectOrientation(str, Enum):
    FRONT = "FRONT"
    BACK = "BACK"
    OTHER = "OTHER"


@dataclass(frozen=True)
class ObjectResult:
    object_key: str
    orientation: ObjectOrientation
    score_margin_m: float | None
    tilt_recovery_candidate: bool
    reason: str
    diagnostics: dict


@dataclass(frozen=True)
class OrientationResult:
    capture_dir: Path
    objects: tuple[ObjectResult, ...]
    report_path: Path
    workpiece_id: str = "1"
    mode: str = ""
    camera_serial: str = ""
    body_head_calibration: Path | None = None

    @property
    def orientation(self) -> ObjectOrientation:
        """单个目标返回其状态；没有目标或多个目标统一返回 OTHER。"""
        if len(self.objects) == 1:
            return self.objects[0].orientation
        return ObjectOrientation.OTHER


TiltRecoveryHandler = Callable[[OrientationResult], None]


def _require_paths(workpiece_id: str, mode: HeadD435Mode) -> None:
    required = [
        SOURCE_PYTHON,
        CAPTURE_ENTRY,
        VISION_ENTRY,
        MODULE_4_1_ROOT / "vision/mask.py",
        MODULE_4_1_ROOT / f"vision/workpiece_{workpiece_id}_icp.py",
        CONFIG_ROOT / f"workpiece_{workpiece_id}.json",
        mode.calibration_path,
        mode.workspace_path(workpiece_id),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Head Judge 依赖不存在: " + ", ".join(str(path) for path in missing)
        )


def _to_object_result(diagnostic: dict) -> ObjectResult:
    raw = str(diagnostic.get("classification", "unknown"))
    orientation = {
        "positive": ObjectOrientation.FRONT,
        "reverse": ObjectOrientation.BACK,
    }.get(raw, ObjectOrientation.OTHER)
    tilt_candidate = (
        orientation is ObjectOrientation.OTHER
        and raw in {"unknown", "rejected_geometry"}
    )
    reasons = {
        "unknown": "CAD 正反假设不明确，工件可能侧放或被遮挡。",
        "rejected_geometry": "反面假设未通过 ICP 或深度质量检查。",
        "rejected_outside_safe_roi": "SAM2 mask 位于桌面安全工作区之外。",
    }
    return ObjectResult(
        object_key=str(diagnostic.get("object_key", "unknown")),
        orientation=orientation,
        score_margin_m=diagnostic.get("score_margin_m"),
        tilt_recovery_candidate=tilt_candidate,
        reason=reasons.get(raw, "CAD 正反面 ICP 假设已通过检查。"),
        diagnostics=diagnostic,
    )


def _read_result(
    capture_dir: Path, workpiece_id: str, mode: HeadD435Mode,
) -> OrientationResult:
    diagnostics_path = (
        capture_dir / "sam2_masks/orientation_diagnostics.json"
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    result = OrientationResult(
        capture_dir=capture_dir,
        objects=tuple(_to_object_result(item) for item in diagnostics),
        report_path=(
            capture_dir / "sam2_masks/tabletop_orientation_result.json"
        ),
        workpiece_id=workpiece_id,
        mode=mode.name,
        camera_serial=mode.camera_serial,
        body_head_calibration=mode.calibration_path,
    )
    payload = {
        "schema_version": 2,
        "event": "module_4_1_tabletop_state_estimated",
        "mode": mode.name,
        "workpiece_id": workpiece_id,
        "camera_serial": mode.camera_serial,
        "body_head_calibration": str(mode.calibration_path),
        "overall_orientation": result.orientation.value,
        "capture_dir": str(result.capture_dir),
        "orientation_diagnostics": str(diagnostics_path),
        "objects": [
            {**asdict(item), "orientation": item.orientation.value}
            for item in result.objects
        ],
        "tilt_recovery_contract": (
            "OTHER 且 tilt_recovery_candidate=true 时，调用方可以摆正工件，"
            "随后必须重新采集并再次运行 Head Judge。"
        ),
    }
    result.report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def inspect_tabletop_orientation(
    capture_dir: Path | None = None,
    *,
    workpiece_id: str | int = "1",
    mode: str | None = None,
    handle_tilt_candidate: TiltRecoveryHandler | None = None,
) -> OrientationResult:
    """采集或读取 RGB-D，完成分割和正反面判断并返回结构化结果。"""
    workpiece_id = str(workpiece_id)
    if workpiece_id not in SUPPORTED_WORKPIECES:
        raise ValueError(f"不支持的工件编号: {workpiece_id}")
    mode_name = mode or os.environ.get("PARM_DEPLOYMENT_MODE")
    if not mode_name:
        raise ValueError("必须明确指定 mode='xiangxin' 或 mode='gzhu'")
    deployment = load_mode(mode_name)
    apply_mode_environment(deployment, workpiece_id)
    _require_paths(workpiece_id, deployment)
    require_sam2_service()
    if capture_dir is None:
        width, height = deployment.image_size(workpiece_id)
        completed = subprocess.run(
            [
                str(SOURCE_PYTHON), str(CAPTURE_ENTRY),
                "--serial", deployment.camera_serial,
                "--width", str(width), "--height", str(height),
                "--fps", str(deployment.fps),
            ],
            cwd=MODULE_4_1_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("Capture complete: ")
        ]
        if not lines:
            raise RuntimeError("RGB-D 采集完成，但未返回输出目录")
        capture_dir = Path(lines[-1].removeprefix("Capture complete: "))

    capture_dir = Path(capture_dir).resolve()
    if not (capture_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"不是有效 RGB-D 采集目录: {capture_dir}")
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    capture_serial = str(metadata.get("serial", ""))
    if capture_serial != deployment.camera_serial:
        raise RuntimeError(
            f"采集相机与 {deployment.name} 标定不匹配: "
            f"capture={capture_serial!r}, calibration={deployment.camera_serial!r}"
        )

    subprocess.run(
        [
            str(SOURCE_PYTHON), str(VISION_ENTRY),
            "--workpiece", workpiece_id,
            "--mode", deployment.name,
            "--capture-dir", str(capture_dir),
            "--socket", sam2_url(),
        ],
        cwd=MODULE_4_1_ROOT,
        check=True,
    )
    result = _read_result(capture_dir, workpiece_id, deployment)
    if handle_tilt_candidate and any(
        item.tilt_recovery_candidate for item in result.objects
    ):
        handle_tilt_candidate(result)
    return result


def inspect_tabletop_state(
    capture_dir: Path | None = None,
    *,
    workpiece_id: str | int = "1",
    mode: str | None = None,
    handle_tilt_candidate: TiltRecoveryHandler | None = None,
) -> OrientationResult:
    """统一发布工件正反与 ICP 位姿；保留旧入口作为兼容别名。"""
    return inspect_tabletop_orientation(
        capture_dir,
        workpiece_id=workpiece_id,
        mode=mode,
        handle_tilt_candidate=handle_tilt_candidate,
    )
