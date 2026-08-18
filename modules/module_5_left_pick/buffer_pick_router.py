#!/usr/bin/env python3
"""Module 5 的 Buffer 抓取路由：只消费 Module 4-1 已发布的状态与 ICP TF。

这个文件不采集图像、不运行 ICP，也不直接驱动机械臂。它把 Module 4-1 的
唯一工件报告收敛为三个稳定的 Module 5 分支：

* ``FRONT_TO_INTERMEDIATE``：正面抓取后送到中间 Buffer；
* ``BACK_TO_HANDOFF``：反面抓取后送到双臂交接位；
* ``OTHER``：拒绝猜测，交回总控重试/人工处理。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FRONT_TO_INTERMEDIATE = "FRONT_TO_INTERMEDIATE"
BACK_TO_HANDOFF = "BACK_TO_HANDOFF"
OTHER = "OTHER"


def route_module4_result(payload: dict) -> tuple[str, str]:
    """Validate the Module 4-1 handoff and return the sole Module 5 branch."""
    decision = str(payload.get("decision") or "")
    objects = payload.get("objects") or []
    if decision not in {"FRONT", "BACK"}:
        return OTHER, f"Module 4-1 decision={decision or 'missing'}"
    if len(objects) != 1:
        return OTHER, f"Module 4-1 必须发布唯一工件，实际={len(objects)}"
    object_orientation = str(objects[0].get("orientation") or "")
    if object_orientation != decision:
        return OTHER, (
            "Module 4-1 总体正反面与唯一工件不一致："
            f"decision={decision}, object={object_orientation or 'missing'}"
        )
    diagnostic = objects[0].get("diagnostics") or {}
    # 工件1使用 positive/reverse ICP；工件2统一位姿写在 pose 内。两者都
    # 必须有有效位姿，Module 5 才能继续。具体抓取适配器由后续分支选择。
    face_key = "positive" if decision == "FRONT" else "reverse"
    face_pose = (diagnostic.get(face_key) or {}).get("T_camera_object")
    pose = diagnostic.get("pose") or {}
    generic_pose = (
        pose.get("T_camera_object")
        or pose.get("T_head_camera_workpiece")
        or pose.get("T_body_base_workpiece")
    )
    if face_pose is None and generic_pose is None:
        return OTHER, f"Module 4-1 {decision} 结果缺少 ICP T_camera_object"
    return (
        FRONT_TO_INTERMEDIATE if decision == "FRONT" else BACK_TO_HANDOFF,
        f"接受 Module 4-1 {decision} ICP TF",
    )


def route_result_file(result_file: Path) -> tuple[str, str]:
    return route_module4_result(json.loads(result_file.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module4-result", type=Path, required=True)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    branch, reason = route_result_file(args.module4_result)
    report = {
        "event": "module_5_buffer_pick_routed",
        "module4_result": str(args.module4_result.resolve()),
        "branch": branch,
        "reason": reason,
    }
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if branch == FRONT_TO_INTERMEDIATE else (20 if branch == BACK_TO_HANDOFF else 21)


if __name__ == "__main__":
    raise SystemExit(main())
