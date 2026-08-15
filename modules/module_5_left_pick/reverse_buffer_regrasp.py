#!/usr/bin/env python3
"""Execute a reverse-side table regrasp from Buffer Check's accepted ICP pose.

This adapter deliberately does not run SAM2 or ICP again.  It consumes the
unique BACK result from Buffer Check, reconstructs the same CAD wide-head TCP
that Module 5-1 would create, then delegates motion and hold verification to
Module 5-1's reviewed executor.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REVERSE_IMPL = ROOT / 'modules/module_5_left_pick/module_5_1_reverse_grasp/implementation'
sys.path[:0] = [str(ROOT), str(REVERSE_IMPL)]

from sam2_icp_grasp import reverse_grasp_record_from_icp
from tool.inspection_support.stl_model import load_stl_triangles_m


BODY_HEAD = ROOT / 'modules/module_4_head_judge/module_4_2_pose_estimation/calibration/head_charuco_13x5_calibration_head.json'
ORIENTATION_CONFIG = ROOT / 'modules/module_5_left_pick/module_5_1_reverse_grasp/config/cad_orientation.json'
DEFAULT_STL = ROOT / 'tool/welding_up_side_1118_only_2_small_screws_removed.stl'
EXECUTOR = REVERSE_IMPL / 'execute_confirmed_grasp.py'
OUTPUT_ROOT = ROOT / 'output/module_5_left_pick/reverse_buffer_regrasp'
MAX_ICP_ROTATION_PROJECTION_ERROR = .01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--buffer-check-result', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--planner-host', default='127.0.0.1')
    parser.add_argument('--arm-host', default='192.168.3.11')
    parser.add_argument('--gripper-host', default='192.168.3.11')
    parser.add_argument('--speed', type=float, default=.20)
    return parser.parse_args()


def _nearest_proper_rotation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    raw = np.asarray(matrix, dtype=float)
    u, _, vt = np.linalg.svd(raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, float(np.linalg.norm(raw - rotation))


def _load_back_icp(result_path: Path) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    payload = json.loads(result_path.read_text(encoding='utf-8'))
    objects = payload.get('objects') or []
    # ``continue_allowed`` means the caller may continue its normal FRONT
    # workflow, so a correct BACK result intentionally sets it to false.
    if payload.get('decision') != 'BACK':
        raise RuntimeError('Buffer Check 未放行唯一 BACK 工件，拒绝使用 ICP 反抓')
    if len(objects) != 1 or objects[0].get('orientation') != 'BACK':
        raise RuntimeError('Buffer Check 必须返回唯一 BACK 工件，拒绝使用 ICP 反抓')
    diagnostic = objects[0].get('diagnostics') or {}
    try:
        T = np.asarray(diagnostic['reverse']['T_camera_object'], dtype=float)
        table_normal = np.asarray(diagnostic['table_normal_camera'], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Buffer Check 缺少反面 ICP 位姿或桌面法向量') from exc
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise RuntimeError('Buffer Check 的 reverse.T_camera_object 不是有效 4x4 矩阵')
    if table_normal.shape != (3,) or not np.all(np.isfinite(table_normal)) or np.linalg.norm(table_normal) < 1e-8:
        raise RuntimeError('Buffer Check 的 table_normal_camera 无效')
    rotation, correction_error = _nearest_proper_rotation(T[:3, :3])
    if correction_error > MAX_ICP_ROTATION_PROJECTION_ERROR:
        raise RuntimeError(f'ICP 旋转矩阵偏离正交矩阵过大: projection_error={correction_error:.6f}')
    T[:3, :3] = rotation
    return payload, diagnostic, T, table_normal / np.linalg.norm(table_normal)


def _camera_matrix(capture_dir: Path) -> np.ndarray:
    metadata = json.loads((capture_dir / 'metadata.json').read_text(encoding='utf-8'))
    intrinsics = metadata.get('color_intrinsics') or {}
    try:
        return np.array([[intrinsics['fx'], 0, intrinsics['ppx']], [0, intrinsics['fy'], intrinsics['ppy']], [0, 0, 1]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'Buffer Check capture 缺少 color_intrinsics: {capture_dir}') from exc


def main() -> int:
    args = parse_args()
    payload, diagnostic, T, table_normal = _load_back_icp(args.buffer_check_result)
    capture_dir = Path(payload.get('capture_dir') or '')
    if not capture_dir.is_dir():
        raise RuntimeError(f'Buffer Check capture_dir 不存在: {capture_dir}')
    quality = diagnostic.get('reverse') or {}
    record = reverse_grasp_record_from_icp(
        T, table_normal, _camera_matrix(capture_dir),
        load_stl_triangles_m(DEFAULT_STL).reshape(-1, 3),
        json.loads(ORIENTATION_CONFIG.read_text(encoding='utf-8')),
        object_key=str(diagnostic.get('object_key') or 'buffer-check-0'),
        quality={key: quality[key] for key in ('icp_fitness', 'icp_rmse_m') if key in quality} | {
            'depth_match_fraction': (quality.get('render') or {}).get('match_fraction'),
            'buffer_check_result': str(args.buffer_check_result),
            'buffer_check_capture_dir': str(capture_dir),
            'icp_pose_source': 'buffer_check.diagnostics.reverse.T_camera_object',
        },
    )
    output_dir = OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir.mkdir(parents=True, exist_ok=False)
    grasp_json = output_dir / 'grasp_poses.json'
    grasp_json.write_text(json.dumps([record], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'buffer_face_check.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'BUFFER_REVERSE_ICP_GRASP_POSES={grasp_json}', flush=True)
    if not args.execute:
        print('ICP reverse grasp pose generated; --execute was not supplied.', flush=True)
        return 0
    command = [sys.executable, str(EXECUTOR), '--grasp-json', str(grasp_json), '--body-head', str(BODY_HEAD), '--planner-host', args.planner_host, '--arm-host', args.arm_host, '--gripper-host', args.gripper_host, '--speed', str(args.speed)]
    command.append('--execute')
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
