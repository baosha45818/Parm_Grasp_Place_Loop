#!/usr/bin/env python3
"""在当前左手反抓观察位，为 gzhu 的头部 D435i 重新标定 SAM2 固定提示点。

本脚本只读取一次头部相机，不会移动云台、机械臂或夹爪。应在总控已经把左
手送到反抓观察位、云台送到 pitch=2°/yaw=26.5° 后运行。保存后，M5.2 的下一
次重试会自动使用新配置。

操作：
  b  重新框选 SAM2 ROI（框选完按 Enter/Space 确认，c 取消）
  左键  添加工件正点
  右键  添加背景负点
  u  撤销最后添加的点
  p  用当前 ROI/正负点调用 SAM2，刷新掩码预览
  s  保存 gzhu 自动点配置（会先备份旧配置）
  q/Esc 退出，不保存
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
if __name__ == "__main__" and PYTHON.is_file() and Path(sys.executable).resolve() != PYTHON.resolve():
    os.execv(str(PYTHON), [str(PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from tools.inspection_support.robot_camera import D405Camera
from tools.inspection_support.sam2_icp import prompt_segment


DEFAULT_POSE_CONFIG = Path(__file__).with_name("head_d435_pose_config.json")
DEFAULT_OUTPUT_CONFIG = Path(__file__).with_name("gzhu_head_d435_auto_mask_config.json")
DEFAULT_OUTPUT_ROOT = ROOT / "output/handoff_prompt_annotation"
WINDOW = "Gzhu D435 SAM2 标点"
DEPTH_WINDOW = "Gzhu D435 ICP有效深度（白色可用）"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-serial", default=os.environ.get("PARM_HEAD_D435_SERIAL", "261222076212"))
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG,
                        help="读取 D435 图像流与深度采集参数的 JSON；序列号由 --camera-serial 覆盖")
    parser.add_argument("--sam2-url", help="SAM2 服务地址；省略时确保项目 UDS 服务已启动")
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG,
                        help="按 s 后写入的 gzhu 固定标点配置")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="保存本次标注 RGB、预览图和标注记录的目录")
    parser.add_argument("--capture-frames", type=int,
                        help="覆盖采集帧数；默认沿用位姿配置（通常为 15）")
    parser.add_argument("--ui", choices=("web", "opencv"), default="web",
                        help="标注界面；默认 web，适用于无 X11/OpenCV 窗口的终端")
    parser.add_argument("--serve-host", default="127.0.0.1",
                        help="web 标注服务监听地址（默认只允许本机浏览器访问）")
    parser.add_argument("--serve-port", type=int, default=0,
                        help="web 标注服务端口；0 表示自动选择空闲端口")
    return parser.parse_args()


def _normalized(point: tuple[int, int], width: int, height: int) -> list[float]:
    return [round(point[0] / max(width - 1, 1), 8), round(point[1] / max(height - 1, 1), 8)]


def _roi_normalized(box: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    x, y, w, h = box
    return _normalized((x, y), width, height) + _normalized((x + w, y + h), width, height)


def _save_config(destination: Path, *, roi: tuple[int, int, int, int], positives: list[tuple[int, int]],
                 negatives: list[tuple[int, int]], width: int, height: int) -> Path | None:
    if not positives:
        raise ValueError("至少标记一个工件正点后才能保存")
    x, y, w, h = roi
    if w <= 1 or h <= 1:
        raise ValueError("ROI 无效；请按 b 重新框选")
    payload = {
        "schema_version": 1,
        "description": "Gzhu 反抓交接固定观察位的 SAM2 提示点；头部命令 pitch=2 deg、yaw=26.5 deg。由 annotate_gzhu_handoff_prompts.py 重新人工标注。",
        "roi_xyxy_normalized": _roi_normalized(roi, width, height),
        "positive_points_normalized": [_normalized(point, width, height) for point in positives],
        "negative_points_normalized": [_normalized(point, width, height) for point in negatives],
        "mask_color_bgr": [0, 255, 0],
        "mask_alpha": 0.45,
    }
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = destination.with_name(f"{destination.stem}.{stamp}.bak.json")
        backup.write_text(destination.read_text(encoding="utf-8"), encoding="utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return backup


def _draw(image: np.ndarray, roi: tuple[int, int, int, int], points: list[tuple[Literal["positive", "negative"], tuple[int, int]]],
          mask: np.ndarray | None, message: str) -> np.ndarray:
    canvas = image.copy()
    if mask is not None:
        selected = mask.astype(bool)
        canvas[selected] = np.rint(canvas[selected].astype(np.float32) * 0.55 + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
    x, y, w, h = roi
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
    positive_index = negative_index = 0
    for kind, point in points:
        if kind == "positive":
            positive_index += 1
            cv2.circle(canvas, point, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"+{positive_index}", (point[0] + 9, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            negative_index += 1
            cv2.drawMarker(canvas, point, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(canvas, f"-{negative_index}", (point[0] + 9, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
    help_line = "b ROI | 左键 正点 | 右键 负点 | u 撤销 | p 预览SAM2 | s 保存 | q 退出"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (20, 20, 20), -1)
    cv2.putText(canvas, help_line, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, message, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
    return canvas


def _capture(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    cfg = json.loads(args.pose_config.expanduser().resolve().read_text(encoding="utf-8"))
    count = int(args.capture_frames if args.capture_frames is not None else cfg["capture_frames"])
    if count < 1:
        raise ValueError("--capture-frames 必须至少为 1")
    cfg["camera_serial"] = str(args.camera_serial)
    print(f"[capture] 打开 D435 serial={cfg['camera_serial']}，采集 {count} 帧；不会移动机器人。", flush=True)
    with D405Camera(
        serial=cfg["camera_serial"], width=int(cfg["color_width"]), height=int(cfg["color_height"]),
        depth_width=int(cfg["depth_width"]), depth_height=int(cfg["depth_height"]), fps=int(cfg["fps"]),
        warmup_frames=int(cfg["warmup_frames"]), enable_postprocess=False, camera_name="head_d435i",
        color_balance_mode="neutral_roi", color_balance_roi_xyxy=(730, 90, 1020, 315),
        depth_visual_preset=4.0, laser_power=360.0,
    ) as camera:
        frames = [camera.read() for _ in range(count)]
    frame = frames[-1]
    return (
        frame.color,
        frame.depth,
        float(frame.depth_scale),
        float(cfg["min_depth_m"]),
        float(cfg["max_depth_m"]),
    )


def _png_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("无法编码标注页面图像")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _validate_annotation(payload: dict, width: int, height: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    raw_roi = payload.get("roi")
    if not isinstance(raw_roi, list) or len(raw_roi) != 4:
        raise ValueError("ROI 必须是 [x, y, width, height]")
    x, y, w, h = (int(value) for value in raw_roi)
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 1 or h <= 1:
        raise ValueError("ROI 无效")

    def points(key: str) -> list[tuple[int, int]]:
        values = payload.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} 必须是点列表")
        result = []
        for point in values:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{key} 包含无效坐标")
            px, py = (int(value) for value in point)
            if not (0 <= px < width and 0 <= py < height):
                raise ValueError(f"{key} 坐标超出图像范围: {point}")
            result.append((px, py))
        return result

    positives, negatives = points("positives"), points("negatives")
    if not positives:
        raise ValueError("至少标记一个工件正点")
    return (x, y, w, h), positives, negatives


def _web_page(image_url: str, depth_url: str, width: int, height: int) -> str:
    """返回无外部依赖的浏览器标点页面。"""
    return """<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>Gzhu D435 SAM2 标点</title>
<style>body{font-family:sans-serif;margin:16px;background:#202124;color:#eee}button{margin:3px;padding:8px 12px}canvas{max-width:96vw;height:auto;border:1px solid #777;display:block;margin:8px 0}.hint{color:#ffd54f}.row{display:flex;gap:16px;flex-wrap:wrap}.panel{max-width:96vw}</style>
<h2>Gzhu 反抓交接：SAM2 重新标点</h2>
<p class='hint'>先框选 ROI；再在左图上标正/负点。右图白色区域才有 ICP 可用深度，正点必须落在工件且对应白色区域。</p>
<button onclick="mode='roi'">框选 ROI</button><button onclick="mode='positive'">标工件正点</button><button onclick="mode='negative'">标背景负点</button><button onclick='undo()'>撤销</button><button onclick='preview()'>SAM2 预览</button><button onclick='save()'>保存配置</button>
<span id='status'>等待操作</span><div class='row'><div class='panel'><b>RGB / SAM2 掩码</b><canvas id='main' width='__W__' height='__H__'></canvas></div><div class='panel'><b>ICP 有效深度</b><canvas id='depth' width='__W__' height='__H__'></canvas></div></div>
<script>
const W=__W__,H=__H__, canvas=document.getElementById('main'), ctx=canvas.getContext('2d'), dcanvas=document.getElementById('depth'), dctx=dcanvas.getContext('2d');
const base=new Image(), depth=new Image(), overlay=new Image(); base.src='__IMAGE__'; depth.src='__DEPTH__';
let roi=[0,0,W-1,H-1], positives=[], negatives=[], history=[], mode='positive', drag=null, hasMask=false;
function setStatus(s){document.getElementById('status').textContent=s}
function draw(){ctx.clearRect(0,0,W,H);ctx.drawImage(base,0,0);if(hasMask)ctx.drawImage(overlay,0,0);ctx.strokeStyle='#ffff00';ctx.lineWidth=3;ctx.strokeRect(roi[0],roi[1],roi[2],roi[3]);ctx.font='20px sans-serif';positives.forEach((p,i)=>{ctx.fillStyle='#ff2020';ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.fill();ctx.fillText('+'+(i+1),p[0]+10,p[1]-8)});negatives.forEach((p,i)=>{ctx.strokeStyle='#2080ff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(p[0]-7,p[1]-7);ctx.lineTo(p[0]+7,p[1]+7);ctx.moveTo(p[0]+7,p[1]-7);ctx.lineTo(p[0]-7,p[1]+7);ctx.stroke();ctx.fillStyle='#2080ff';ctx.fillText('-'+(i+1),p[0]+10,p[1]-8)});if(drag){ctx.strokeStyle='#ff00ff';ctx.strokeRect(drag[0],drag[1],drag[2],drag[3])}}
function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(W-1,Math.round((e.clientX-r.left)*W/r.width))),Math.max(0,Math.min(H-1,Math.round((e.clientY-r.top)*H/r.height)))]}
canvas.addEventListener('mousedown',e=>{const p=point(e);if(mode==='roi'){drag=[p[0],p[1],0,0]}else{const target=mode==='positive'?positives:negatives;target.push(p);history.push(mode);hasMask=false;setStatus('已添加点；请点击 SAM2 预览');draw()}});
canvas.addEventListener('mousemove',e=>{if(!drag)return;const p=point(e);drag[2]=p[0]-drag[0];drag[3]=p[1]-drag[1];draw()});
canvas.addEventListener('mouseup',e=>{if(!drag)return;const p=point(e);const x=Math.min(drag[0],p[0]),y=Math.min(drag[1],p[1]),w=Math.abs(p[0]-drag[0]),h=Math.abs(p[1]-drag[1]);drag=null;if(w>1&&h>1){roi=[x,y,w,h];hasMask=false;setStatus('ROI 已更新')}draw()});
function undo(){const k=history.pop();if(k==='positive')positives.pop();if(k==='negative')negatives.pop();hasMask=false;draw()}
async function preview(){if(!positives.length){setStatus('至少需要一个正点');return}setStatus('SAM2 计算中…');const r=await fetch('/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({roi,positives,negatives})});const j=await r.json();if(!r.ok){setStatus('失败：'+j.error);return}overlay.onload=()=>{hasMask=true;draw()};overlay.src=j.mask;setStatus('SAM2 掩码：'+j.pixels+' 像素')}
async function save(){if(!positives.length){setStatus('至少需要一个正点');return}const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({roi,positives,negatives})});const j=await r.json();setStatus(r.ok?'已保存：可关闭本页并回总控输入 y':'保存失败：'+j.error)}
base.onload=draw;depth.onload=()=>dctx.drawImage(depth,0,0);</script></html>""".replace("__W__", str(width)).replace("__H__", str(height)).replace("__IMAGE__", image_url).replace("__DEPTH__", depth_url)


def _run_web_annotator(args: argparse.Namespace, image: np.ndarray, depth_preview: np.ndarray, output: Path) -> int:
    """在本机浏览器提供交互标点，并把保存结果写回活动 gzhu 配置。"""
    height, width = image.shape[:2]
    page = _web_page(_png_data_url(image), _png_data_url(depth_preview), width, height).encode("utf-8")
    saved = {"done": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._payload()
                roi, positives, negatives = _validate_annotation(payload, width, height)
                if self.path == "/preview":
                    x, y, w, h = roi
                    mask = prompt_segment(image, [x, y, x + w, y + h], [list(p) for p in positives], [list(p) for p in negatives], args.sam2_url)
                    green = np.zeros((height, width, 4), dtype=np.uint8)
                    green[mask.astype(bool)] = (0, 255, 0, 120)
                    self._json(200, {"mask": _png_data_url(green), "pixels": int(np.count_nonzero(mask))})
                    return
                if self.path == "/save":
                    backup = _save_config(args.output_config, roi=roi, positives=positives, negatives=negatives, width=width, height=height)
                    record = {"image": str((output / "annotated_source_rgb.png").resolve()), "roi_xywh": roi, "positive_points": positives, "negative_points": negatives, "sam2_url": args.sam2_url, "output_config": str(args.output_config.expanduser().resolve()), "backup": None if backup is None else str(backup)}
                    (output / "annotation_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    self._json(200, {"saved": str(args.output_config.expanduser().resolve())})
                    saved["done"] = True
                    return
                self._json(404, {"error": "未知接口"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

    server = HTTPServer((args.serve_host, args.serve_port), Handler)
    host, port = server.server_address[:2]
    print(f"[annotate] 请在本机浏览器打开：http://{host}:{port}/", flush=True)
    print("[annotate] 保存后脚本会自动退出；Ctrl+C 可不保存退出。", flush=True)
    server.timeout = 0.5
    try:
        while not saved["done"]:
            server.handle_request()
    finally:
        server.server_close()
    print(f"[annotate] 已保存标注记录：{output / 'annotation_record.json'}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.sam2_url is None:
        from modules.module_0_prepare.ensure_sam2_service import ensure_sam2_service, sam2_url
        ensure_sam2_service()
        args.sam2_url = sam2_url()
    image, depth, depth_scale, min_depth_m, max_depth_m = _capture(args)
    height, width = image.shape[:2]
    if depth.shape != (height, width):
        raise RuntimeError(
            f"RGB/depth 未对齐：RGB={image.shape[:2]}，depth={depth.shape}"
        )
    output = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output / "annotated_source_rgb.png"), image)
    depth_m = depth.astype(np.float32) * depth_scale
    valid_depth = (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    depth_preview = np.zeros_like(image)
    depth_preview[valid_depth] = (255, 255, 255)
    cv2.putText(depth_preview, f"white = {min_depth_m:.2f}-{max_depth_m:.2f} m valid for ICP", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output / "icp_valid_depth.png"), depth_preview)
    if args.ui == "web":
        return _run_web_annotator(args, image, depth_preview, output)

    roi = (0, 0, width - 1, height - 1)
    points: list[tuple[Literal["positive", "negative"], tuple[int, int]]] = []
    mask: np.ndarray | None = None
    message = "先按 b 框选工件区域，再添加正/负点。"

    def click(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        nonlocal mask, message
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(("positive", (x, y)))
            mask = None
            message = f"已添加正点 {sum(kind == 'positive' for kind, _ in points)}；按 p 预览。"
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append(("negative", (x, y)))
            mask = None
            message = f"已添加负点 {sum(kind == 'negative' for kind, _ in points)}；按 p 预览。"

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, click)
    while True:
        cv2.imshow(WINDOW, _draw(image, roi, points, mask, message))
        cv2.imshow(DEPTH_WINDOW, depth_preview)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            print("[annotate] 未保存，退出。", flush=True)
            break
        if key == ord("b"):
            selected = cv2.selectROI("框选 SAM2 ROI（Enter/Space确认，c取消）", image, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("框选 SAM2 ROI（Enter/Space确认，c取消）")
            if selected[2] > 1 and selected[3] > 1:
                roi = tuple(int(value) for value in selected)
                mask = None
                message = "ROI 已更新；请添加正/负点。"
            else:
                message = "ROI 未改变。"
        elif key == ord("u"):
            if points:
                points.pop()
                mask = None
                message = "已撤销最后一个点。"
        elif key == ord("p"):
            positives = [list(point) for kind, point in points if kind == "positive"]
            negatives = [list(point) for kind, point in points if kind == "negative"]
            if not positives:
                message = "至少添加一个正点才能预览。"
                continue
            x, y, w, h = roi
            try:
                mask = prompt_segment(image, [x, y, x + w, y + h], positives, negatives, args.sam2_url)
                message = f"SAM2 掩码已刷新：{int(np.count_nonzero(mask))} 像素。"
            except Exception as exc:
                mask = None
                message = f"SAM2 预览失败：{exc}"
        elif key == ord("s"):
            positives = [point for kind, point in points if kind == "positive"]
            negatives = [point for kind, point in points if kind == "negative"]
            try:
                backup = _save_config(args.output_config, roi=roi, positives=positives, negatives=negatives, width=width, height=height)
                record = {
                    "image": str((output / "annotated_source_rgb.png").resolve()),
                    "roi_xywh": roi,
                    "positive_points": positives,
                    "negative_points": negatives,
                    "sam2_url": args.sam2_url,
                    "output_config": str(args.output_config.expanduser().resolve()),
                    "backup": None if backup is None else str(backup),
                }
                (output / "annotation_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if mask is not None:
                    cv2.imwrite(str(output / "sam2_preview.png"), _draw(image, roi, points, mask, "已保存"))
                print(f"[annotate] 已保存：{args.output_config.expanduser().resolve()}", flush=True)
                if backup is not None:
                    print(f"[annotate] 旧配置备份：{backup}", flush=True)
                print(f"[annotate] 本次标注记录：{output / 'annotation_record.json'}", flush=True)
                break
            except Exception as exc:
                message = f"保存失败：{exc}"
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
