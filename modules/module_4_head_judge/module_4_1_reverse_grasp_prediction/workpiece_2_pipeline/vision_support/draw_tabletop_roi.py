#!/usr/bin/env python3
"""手动画出工件 2 头部相机安全 ROI，并按确认结果更新本流程配置。

该工具只读取给定的 RGB 图像；不会连接相机、云台、机械臂或夹爪。
在 OpenCV 窗口拖拽矩形，按 Enter/Space 确认选择，按 C 取消。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
DEFAULT_CONFIG = PIPELINE_ROOT / "config/tabletop_workspace.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="要标注的头部 RGB 图片")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="要更新的本流程 tabletop_workspace.json")
    parser.add_argument("--apply", action="store_true", help="终端确认后写入配置；省略时仅打印结果")
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取 RGB 图片: {image_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"配置不存在: {config_path}")

    height, width = image.shape[:2]
    print("在弹出的窗口拖拽框选工件活动范围；按 Enter/Space 确认，按 C 取消。")
    x, y, roi_width, roi_height = (int(value) for value in cv2.selectROI("Draw tabletop ROI", image, False, True))
    cv2.destroyAllWindows()
    if roi_width <= 0 or roi_height <= 0:
        print("未选择有效 ROI，配置未修改。")
        return 1
    roi = [x, y, roi_width, roi_height]
    print(f"safe_roi_xywh = {roi}  (image={width}x{height})")
    if not args.apply:
        print("预览模式：未写入配置。确认无误后加 --apply 再运行。")
        return 0
    answer = input("确认写入本流程配置？输入 y 后回车确认: ").strip().lower()
    if answer != "y":
        print("未确认，配置未修改。")
        return 0
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = payload.get("reference_color_size_wh")
    if expected is not None and [width, height] != [int(value) for value in expected]:
        raise RuntimeError(f"图片尺寸 {width}x{height} 与配置参考尺寸 {expected} 不一致，拒绝写入")
    payload["safe_roi_xywh"] = roi
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已更新: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
