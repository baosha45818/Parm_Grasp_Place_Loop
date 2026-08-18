#!/usr/bin/env python3
"""一次采集同时检查 Buffer 工件正反并发布 ICP 工件位姿。

本工具不会修改 Grasp-and-Place Loop 的任何代码或状态。它只在本目录保存
采集、SAM2/ICP 中间结果和最终 JSON；当工件为反面、目标数不为一，或视觉
结果不确定时，以非零退出码返回。调用方必须显式处理 BACK；loop 可进入已验证
的反抓恢复链，其他调用方可选择安全停止。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOOP_ROOT = HERE.parents[2]
PYTHON = LOOP_ROOT / ".venv/bin/python"
VISION_ROOT = HERE / "vision"
CAPTURE = VISION_ROOT / "capture_rgbd.py"
QUALITY_GATE_CONFIG = HERE / "quality_gate.json"

# Module 4 is the project-wide owner of SAM2/ICP execution and the
# FRONT/BACK/OTHER mapping.  Buffer Check only adds a stricter release gate.
if str(LOOP_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOP_ROOT))
from modules.module_4_head_judge import (  # noqa: E402
    ObjectOrientation,
    inspect_tabletop_state,
)
from modules.module_4_head_judge.module_4_1_buffer_state_estimation.deployment_modes import (  # noqa: E402
    apply_mode_environment,
    load_mode,
)

EXIT_OK = 0
EXIT_BACK = 20
EXIT_UNCERTAIN = 21
EXIT_CONFIGURATION = 22


def _inside_here(path: Path) -> Path:
    """Resolve a writable target and reject paths outside this isolated tool."""
    resolved = path.resolve()
    try:
        resolved.relative_to(HERE)
    except ValueError as exc:
        raise ValueError(
            f"为确保不写入现有流程目录，路径必须位于 {HERE}: {resolved}"
        ) from exc
    return resolved


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, check=False, capture_output=True)
    if completed.returncode:
        # Do not hide camera/SDK stderr behind the generic exit-22 wrapper.
        _forward_output(completed)
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout, stderr=completed.stderr
        )
    return completed


def _forward_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)


def _capture(args: argparse.Namespace) -> Path:
    before = set(args.capture_root.glob("head_rgbd_*"))
    completed = _run(
        [
            str(PYTHON), str(CAPTURE), "--serial", args.serial,
            "--output-dir", str(args.capture_root),
            "--width", str(args.width), "--height", str(args.height),
            "--depth-fuse-frames", str(args.depth_fuse_frames),
            "--fps", str(args.fps),
        ],
        cwd=VISION_ROOT,
    )
    _forward_output(completed)
    created = sorted(set(args.capture_root.glob("head_rgbd_*")) - before)
    if len(created) != 1:
        raise RuntimeError(f"D435i 采集应创建一个目录，实际为: {created}")
    return created[0]


def _load_quality_gate(path: Path = QUALITY_GATE_CONFIG) -> dict[str, float]:
    """Load strict FRONT acceptance criteria local to this isolated checker."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "minimum_positive_icp_fitness",
        "minimum_positive_depth_match_fraction",
    )
    try:
        gate = {key: float(payload[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"质量门槛配置无效: {path}") from exc
    if (
        not all(math.isfinite(value) for value in gate.values())
        or not 0.0 <= gate["minimum_positive_icp_fitness"] <= 1.0
        or not 0.0 <= gate["minimum_positive_depth_match_fraction"] <= 1.0
    ):
        raise ValueError(f"质量门槛超出有效范围: {path}")
    return gate


def _positive_quality_reason(diagnostic: dict, gate: dict[str, float]) -> str | None:
    """Apply Module 4-style ICP/depth checks before a positive result can pass."""
    try:
        positive = diagnostic["positive"]
        fitness = float(positive["icp_fitness"])
        depth_match = float(positive["render"]["match_fraction"])
    except (KeyError, TypeError, ValueError):
        return "positive 结果缺少完整的 ICP/深度质量数据"
    values = (fitness, depth_match)
    if not all(math.isfinite(value) for value in values):
        return "positive 结果包含非有限 ICP/深度质量数据"
    failures = []
    if fitness < gate["minimum_positive_icp_fitness"]:
        failures.append(
            f"positive_icp_fitness={fitness:.3f} < {gate['minimum_positive_icp_fitness']:.3f}"
        )
    if depth_match < gate["minimum_positive_depth_match_fraction"]:
        failures.append(
            "positive_depth_match_fraction="
            f"{depth_match:.3f} < {gate['minimum_positive_depth_match_fraction']:.3f}"
        )
    return "; ".join(failures) if failures else None


def _reverse_quality_reason(diagnostic: dict, gate: dict[str, float]) -> str | None:
    """Require the reverse hypothesis to independently meet the same gates."""
    try:
        reverse = diagnostic["reverse"]
        fitness = float(reverse["icp_fitness"])
        depth_match = float(reverse["render"]["match_fraction"])
    except (KeyError, TypeError, ValueError):
        return "reverse 结果缺少完整的 ICP/深度质量数据"
    if not all(math.isfinite(value) for value in (fitness, depth_match)):
        return "reverse 结果包含非有限 ICP/深度质量数据"
    failures = []
    if fitness < gate["minimum_positive_icp_fitness"]:
        failures.append(
            f"reverse_icp_fitness={fitness:.3f} < {gate['minimum_positive_icp_fitness']:.3f}"
        )
    if depth_match < gate["minimum_positive_depth_match_fraction"]:
        failures.append(
            "reverse_depth_match_fraction="
            f"{depth_match:.3f} < {gate['minimum_positive_depth_match_fraction']:.3f}"
        )
    return "; ".join(failures) if failures else None


def _reverse_fallback_reason(diagnostic: dict, gate: dict[str, float]) -> str | None:
    """Return a safe BACK fallback explanation, otherwise ``None``.

    This handles a conflicting Module 4 FRONT classification: its positive
    geometry fails the Buffer release gate while reverse geometry independently
    passes it.  The score margin is retained for diagnostics but is not a
    safety criterion: a rejected positive hypothesis must never block a
    verified reverse-grasp recovery.
    """
    positive_failure = _positive_quality_reason(diagnostic, gate)
    if not positive_failure or not (
        "positive_icp_fitness=" in positive_failure
        or "positive_depth_match_fraction=" in positive_failure
    ):
        return None
    reverse_failure = _reverse_quality_reason(diagnostic, gate)
    if reverse_failure:
        return None
    return (
        "正面 ICP/深度质量门失败，而反面 ICP/深度质量门完整通过"
    )


def _classify(
    module4_result: object, gate: dict[str, float], *, workpiece_id: str = "1",
) -> tuple[str, str]:
    """Apply the Buffer release policy to Module 4's public orientation result."""
    objects = module4_result.objects
    if len(objects) != 1:
        return "UNCERTAIN", f"Buffer 必须恰好检测到一个工件，实际为 {len(objects)} 个"
    item = objects[0]
    if workpiece_id == "2":
        pose = item.diagnostics.get("pose") or {}
        confidence = item.diagnostics.get("pose_confidence") or {}
        if not pose.get("valid", False):
            reasons = confidence.get("low_reasons") or ["工件2位姿质量未通过"]
            return "UNCERTAIN", f"工件2粗配准/ICP位姿不可靠（{', '.join(reasons)}）"
        if item.orientation is ObjectOrientation.FRONT and pose.get("orientation_valid", False):
            return "FRONT", "工件2全局粗配准和全6DoF ICP确认正面；位姿已同时发布"
        if item.orientation is ObjectOrientation.BACK and pose.get("orientation_valid", False):
            return "BACK", "工件2全局粗配准和全6DoF ICP确认反面；位姿已同时发布"
        reasons = confidence.get("low_reasons") or [item.reason]
        return "UNCERTAIN", f"工件2正反或位姿不确定（{', '.join(reasons)}）"
    if item.orientation is ObjectOrientation.FRONT:
        quality_failure = _positive_quality_reason(item.diagnostics, gate)
        if quality_failure:
            reverse_fallback = _reverse_fallback_reason(item.diagnostics, gate)
            if reverse_fallback:
                return "BACK", f"{reverse_fallback}；按安全反抓回退处理"
            return "UNCERTAIN", f"正面假设未通过质量门槛（{quality_failure}）；必须停止后续流程"
        return "FRONT", "CAD 正面假设胜出；允许继续后续流程"
    if item.orientation is ObjectOrientation.BACK:
        return "BACK", "检测到工件已翻至反面；必须停止后续流程"
    reverse_fallback = _reverse_fallback_reason(item.diagnostics, gate)
    if reverse_fallback:
        return "BACK", f"{reverse_fallback}；按安全反抓回退处理"
    return "UNCERTAIN", f"Module 4 判定不可靠（{item.reason}）；必须停止后续流程"


def _all_objects_both_face_quality_rejected(
    module4_result: object, gate: dict[str, float], *, workpiece_id: str = "1",
) -> bool:
    """Whether this uncertainty is safe to resolve by taking a fresh capture.

    Zero or multiple detections are observation-count failures, not a final
    face decision: retry them regardless of the individual classifications.
    For exactly one detection, retry only when its face geometry is unreliable.

    Module 4 uses both ``rejected_geometry`` and ``unknown`` when its two
    ICP hypotheses cannot be released reliably.  The latter includes a score
    margin below the orientation threshold, so excluding it would turn a
    re-capturable observation into an immediate loop fault.
    """
    objects = module4_result.objects
    if len(objects) != 1:
        return True
    if workpiece_id == "2":
        item = objects[0]
        pose = item.diagnostics.get("pose") or {}
        return (
            item.orientation is ObjectOrientation.OTHER
            or not pose.get("valid", False)
            or not pose.get("orientation_valid", False)
        )
    for item in objects:
        diagnostic = item.diagnostics
        if not isinstance(diagnostic, dict):
            return False
        classification = diagnostic.get("classification")
        # ``unknown`` is emitted solely because the positive/reverse score
        # margin did not reach Module 4's decision threshold.  It is safe to
        # obtain a fresh observation, even when there are several candidates.
        if classification == "unknown":
            continue
        if classification != "rejected_geometry":
            return False
        reverse_quality = diagnostic.get("reverse_geometry_quality") or {}
        reverse_rejected = reverse_quality.get("accepted") is False
        if not (reverse_rejected and _positive_quality_reason(diagnostic, gate) is not None):
            return False
    return True


def _write_report(
    capture_dir: Path, module4_result: object, decision: str, reason: str,
    *, workpiece_id: str, mode: object, capture_attempt: int,
    max_uncertain_retries: int,
) -> Path:
    diagnostics_path = capture_dir / "sam2_masks/orientation_diagnostics.json"
    report = {
        "schema_version": 3,
        "event": "module_4_1_buffer_state_estimated",
        "mode": mode.name,
        "workpiece_id": workpiece_id,
        "camera_serial": mode.camera_serial,
        "body_head_calibration": str(mode.calibration_path),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "capture_dir": str(capture_dir),
        "orientation_diagnostics": str(diagnostics_path),
        "decision": decision,
        "continue_allowed": decision == "FRONT",
        "reason": reason,
        "capture_attempt": capture_attempt,
        "max_uncertain_retries": max_uncertain_retries,
        "module_4_report": str(module4_result.report_path),
        "objects": [
            {
                "object_key": item.object_key,
                "orientation": item.orientation.value,
                "reason": item.reason,
                "pose": item.diagnostics.get("pose"),
                "diagnostics": item.diagnostics,
            }
            for item in module4_result.objects
        ],
        "contract": {
            "FRONT": "exit 0; 工件正面朝上，可以继续。",
            "BACK": f"exit {EXIT_BACK}; 工件反面朝上，调用方必须显式进入反抓恢复或安全停止。",
            "UNCERTAIN": f"exit {EXIT_UNCERTAIN}; 不得猜测，调用方必须报错并停止。",
        },
    }
    report_path = capture_dir / "buffer_face_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workpiece", choices=("1", "2"), default="1",
        help="工件编号；1保持双面多yaw约束ICP，2使用FPFH/RANSAC粗配准和全6DoF ICP",
    )
    parser.add_argument(
        "--mode", choices=("xiangxin", "gzhu"), required=True,
        help="原子选择现场头部相机序列号、手眼标定、分辨率和桌面工作区",
    )
    parser.add_argument("--capture-dir", type=Path, help="复用本工具目录下已有的 RGB-D 采集；省略时新采集")
    parser.add_argument("--capture-root", type=Path, default=HERE / "captures")
    parser.add_argument(
        "--result-file", type=Path,
        help="将本次最终报告额外复制到调用方指定路径（用于 loop 的精确握手）",
    )
    parser.add_argument(
        "--depth-fuse-frames", type=int, default=60,
        help="每次 Buffer Check 取中位数融合的对齐深度帧数，默认60",
    )
    parser.add_argument(
        "--max-uncertain-retries", type=int, default=3,
        help="正反两面 ICP/深度质量门均失败时的额外重新采集次数，默认3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = load_mode(args.mode)
    apply_mode_environment(mode, args.workpiece)
    args.serial = mode.camera_serial
    args.width, args.height = mode.image_size(args.workpiece)
    args.fps = mode.fps
    for dependency in (PYTHON, CAPTURE, QUALITY_GATE_CONFIG):
        if not dependency.is_file():
            raise FileNotFoundError(f"缺少依赖: {dependency}")
    if min(args.width, args.height, args.depth_fuse_frames) < 1 or args.max_uncertain_retries < 0:
        raise ValueError("--width、--height、--depth-fuse-frames 必须至少为 1，--max-uncertain-retries 必须非负")
    args.capture_root = _inside_here(args.capture_root)
    args.capture_root.mkdir(parents=True, exist_ok=True)

    gate = _load_quality_gate()
    capture_attempt = 0
    while True:
        capture_attempt += 1
        capture_dir = _inside_here(args.capture_dir) if args.capture_dir else _capture(args)
        if not (capture_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"不是有效的 D435i RGB-D 采集目录: {capture_dir}")
        # Do not duplicate SAM2/ICP behavior here.  Module 4 owns the project
        # perception chain and returns its validated FRONT/BACK/OTHER contract.
        module4_result = inspect_tabletop_state(
            capture_dir, workpiece_id=args.workpiece, mode=mode.name
        )
        decision, reason = _classify(module4_result, gate, workpiece_id=args.workpiece)
        can_retry = (
            args.capture_dir is None
            and decision == "UNCERTAIN"
            and _all_objects_both_face_quality_rejected(
                module4_result, gate, workpiece_id=args.workpiece
            )
            and capture_attempt <= args.max_uncertain_retries
        )
        if not can_retry:
            break
        print(
            "[buffer-check] 正反两面 ICP/深度质量门均未通过；"
            f"重新采集并判定（第 {capture_attempt}/{args.max_uncertain_retries} 次重试）。",
            flush=True,
        )
    report_path = _write_report(
        capture_dir, module4_result, decision, reason,
        workpiece_id=args.workpiece, mode=mode,
        capture_attempt=capture_attempt,
        max_uncertain_retries=args.max_uncertain_retries,
    )
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(json.dumps({"mode": mode.name, "workpiece_id": args.workpiece, "decision": decision, "reason": reason, "report": str(report_path)}, ensure_ascii=False))
    return {"FRONT": EXIT_OK, "BACK": EXIT_BACK, "UNCERTAIN": EXIT_UNCERTAIN}[decision]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"BUFFER_FACE_CHECK_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CONFIGURATION)
