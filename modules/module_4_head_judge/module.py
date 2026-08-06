"""使用头部 RGB-D、SAM2 和受桌面约束的 ICP 判断工件正反面。

该文件只组织视觉流程，不导入规划器，也不控制机械臂或夹爪。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


LOOP_ROOT = Path(__file__).resolve().parents[2]
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))
VISION_ROOT = (
    LOOP_ROOT
    / "modules/module_5_left_pick/module_5_1_reverse_grasp/implementation"
)
SOURCE_PYTHON = LOOP_ROOT / ".venv/bin/python"
from modules.module_0_prepare.ensure_sam2_service import require_sam2_service, sam2_url


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

    @property
    def orientation(self) -> ObjectOrientation:
        """单个目标返回其状态；没有目标或多个目标统一返回 OTHER。"""
        if len(self.objects) == 1:
            return self.objects[0].orientation
        return ObjectOrientation.OTHER


TiltRecoveryHandler = Callable[[OrientationResult], None]


def _require_paths() -> None:
    required = [
        SOURCE_PYTHON,
        VISION_ROOT / "capture_head_rgbd.py",
        VISION_ROOT / "sam2_tabletop_masks.py",
        VISION_ROOT / "sam2_icp_grasp.py",
        VISION_ROOT.parent / "config/cad_orientation.json",
        VISION_ROOT.parent / "config/tabletop_workspace.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Head Judge 依赖不存在: " + ", ".join(str(path) for path in missing)
        )


def _run_visual_script(script: str, *args: object) -> None:
    subprocess.run(
        [str(SOURCE_PYTHON), str(VISION_ROOT / script), *(str(arg) for arg in args)],
        cwd=VISION_ROOT,
        check=True,
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


def _read_result(capture_dir: Path) -> OrientationResult:
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
    )
    payload = {
        "schema_version": 1,
        "overall_orientation": result.orientation.value,
        "capture_dir": str(result.capture_dir),
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
    handle_tilt_candidate: TiltRecoveryHandler | None = None,
) -> OrientationResult:
    """采集或读取 RGB-D，完成分割和正反面判断并返回结构化结果。"""
    _require_paths()
    require_sam2_service()
    if capture_dir is None:
        completed = subprocess.run(
            [str(SOURCE_PYTHON), str(VISION_ROOT / "capture_head_rgbd.py")],
            cwd=VISION_ROOT,
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

    _run_visual_script("sam2_tabletop_masks.py", "--capture-dir", capture_dir, "--socket", sam2_url())
    _run_visual_script("sam2_icp_grasp.py", "--capture-dir", capture_dir)
    result = _read_result(capture_dir)
    if handle_tilt_candidate and any(
        item.tilt_recovery_candidate for item in result.objects
    ):
        handle_tilt_candidate(result)
    return result
