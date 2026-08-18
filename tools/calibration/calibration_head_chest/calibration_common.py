"""Shared chessboard, robot-pose, and offline hand-eye calibration helpers."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BOARD = {
    "inner_cols": 11,
    "inner_rows": 8,
    "square_size_m": 0.03,
    "top_left_color": "white",
    "corner_colors": {
        "top_left": "white",
        "bottom_left": "white",
        "top_right": "black",
    },
}

# 17-marker board photographed in 68e3c8ceff7a670cafc22f71b04d15e8.jpg.
# It is a 5 x 7 checker layout: the black cells contain DICT_5X5_100 tags
# 0--16.  Row 0 is the physical top row when the printed board is upright.
HEAD_ARUCO_5X7_BOARD = {
    "type": "aruco_grid",
    "dictionary": "DICT_5X5_100",
    "marker_length_m": 0.03725,
    "grid_cell_size_m": 0.05000,
    "grid_cols": 5,
    "grid_rows": 7,
    "marker_layout": {
        16: [1, 0], 15: [3, 0],
        14: [0, 1], 13: [2, 1], 12: [4, 1],
        11: [1, 2], 10: [3, 2],
        9: [0, 3], 8: [2, 3], 7: [4, 3],
        6: [1, 4], 5: [3, 4],
        4: [0, 5], 3: [2, 5], 2: [4, 5],
        1: [1, 6], 0: [3, 6],
    },
}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def board_object_points(board: dict) -> np.ndarray:
    cols = int(board.get("inner_cols", DEFAULT_BOARD["inner_cols"]))
    rows = int(board.get("inner_rows", DEFAULT_BOARD["inner_rows"]))
    size = float(board.get("square_size_m", DEFAULT_BOARD["square_size_m"]))
    if cols < 2 or rows < 2 or size <= 0:
        raise ValueError("board must contain valid inner_cols, inner_rows, square_size_m")
    points = np.zeros((rows * cols, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * size
    return points


def canonicalize_checkerboard_corners(
    gray: np.ndarray, corners: np.ndarray, board: dict, return_diagnostics: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Orient corners by the printed text strip above the board's top edge."""
    cols = int(board["inner_cols"])
    rows = int(board["inner_rows"])
    points = corners.reshape(-1, 2).astype(np.float32)
    if len(points) != cols * rows:
        diagnostics = {"best_score": float("-inf"), "margin": 0.0, "relative_margin": 0.0}
        return (corners, diagnostics) if return_diagnostics else corners
    grid = points.reshape(rows, cols, 2)
    candidates = (
        grid,
        grid[:, ::-1, :],
        grid[::-1, :, :],
        grid[::-1, ::-1, :],
    )
    inner_object = np.stack(
        np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1)), axis=-1
    ).astype(np.float32).reshape(-1, 2)
    marker_scale = 60
    marker_x_min, marker_x_max = 0.0, min(6.0, float(cols + 1))
    marker_y_min, marker_y_max = -0.60, 0.0
    marker_transform = np.array(
        [
            [marker_scale, 0.0, -marker_x_min * marker_scale],
            [0.0, marker_scale, -marker_y_min * marker_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    marker_size = (
        int((marker_x_max - marker_x_min) * marker_scale),
        int((marker_y_max - marker_y_min) * marker_scale),
    )
    best_score = -np.inf
    second_score = -np.inf
    best_points = points
    for candidate in candidates:
        homography, _ = cv2.findHomography(inner_object, candidate.reshape(-1, 2))
        if homography is None:
            continue
        marker = cv2.warpPerspective(
            gray,
            marker_transform @ np.linalg.inv(homography),
            marker_size,
            borderValue=255,
        )
        # Text strokes are local dark details; high-pass filtering rejects the
        # broad lighting gradient of the white board margin.
        local_mean = cv2.GaussianBlur(marker, (0, 0), 3)
        ink = np.maximum(local_mean.astype(np.float32) - marker.astype(np.float32), 0.0)
        score = float(np.mean(ink))
        if score > best_score:
            second_score = best_score
            best_score = score
            best_points = candidate.reshape(-1, 2).copy()
        elif score > second_score:
            second_score = score
    margin = float(best_score - second_score) if np.isfinite(second_score) else 0.0
    diagnostics = {
        "best_score": float(best_score),
        "second_score": float(second_score),
        "margin": margin,
        "relative_margin": margin / max(abs(float(best_score)), 1e-6),
    }
    oriented = best_points.reshape(-1, 1, 2)
    return (oriented, diagnostics) if return_diagnostics else oriented


def detect_board_observation(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    board: dict,
    distortion: np.ndarray | None = None,
    require_marker_confidence: bool = False,
    min_marker_score: float = 0.0,
    min_marker_relative_margin: float = 0.0,
    corners_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if board.get("type") == "charuco":
        if corners_override is not None:
            raise RuntimeError("manual chessboard corners are not valid for a ChArUco board")
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
            raise RuntimeError("OpenCV contrib ArUco/ChArUco support is required for this board")
        dictionary_name = str(board["dictionary"])
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise RuntimeError(f"unknown OpenCV ArUco dictionary: {dictionary_name}")
        squares_x = int(board["squares_x"])
        squares_y = int(board["squares_y"])
        square_size = float(board["square_size_m"])
        marker_size = float(board["marker_size_m"])
        if squares_x < 2 or squares_y < 2 or square_size <= 0 or marker_size <= 0 or marker_size >= square_size:
            raise RuntimeError("invalid ChArUco board geometry")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_size, marker_size, dictionary
        )
        charuco_parameters = cv2.aruco.CharucoParameters()
        # With homography interpolation disabled, interpolate corners with the
        # active color-profile intrinsics and distortion model. This is the
        # OpenCV equivalent of the board-generator setting.
        if not bool(board.get("homography_interpolation", False)):
            charuco_parameters.cameraMatrix = np.asarray(camera_matrix, dtype=np.float64)
            charuco_parameters.distCoeffs = (
                np.zeros(5, dtype=np.float64)
                if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1)
            )
        detector = cv2.aruco.CharucoDetector(charuco_board, charuco_parameters)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
        # Four points are mathematically sufficient for PnP, but six makes a
        # one-bad-corner solution much less likely during hand-eye calibration.
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 6:
            return None
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        object_corners = np.asarray(charuco_board.getChessboardCorners(), dtype=np.float64)
        if np.any(ids < 0) or np.any(ids >= len(object_corners)):
            raise RuntimeError("ChArUco detector returned an invalid corner ID")
        object_points = object_corners[ids]
        image_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        error = np.linalg.norm(
            projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
        )
        return (
            rvec.reshape(3), tvec.reshape(3), image_points,
            float(np.sqrt(np.mean(error ** 2))),
        )
    if board.get("type") == "aruco_grid":
        if corners_override is not None:
            raise RuntimeError("manual chessboard corners are not valid for an ArUco board")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module is required for this board")
        dictionary_name = str(board["dictionary"])
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise RuntimeError(f"unknown OpenCV ArUco dictionary: {dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        marker_corners, marker_ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(image)
        if marker_ids is None:
            return None
        layout = {int(key): value for key, value in board["marker_layout"].items()}
        marker_size = float(board["marker_length_m"])
        cell_size = float(board["grid_cell_size_m"])
        if marker_size <= 0 or cell_size <= 0 or marker_size > cell_size:
            raise RuntimeError("invalid ArUco board marker_length_m/grid_cell_size_m")
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        used_corners: list[np.ndarray] = []
        for marker_id, corners in zip(marker_ids.reshape(-1), marker_corners):
            cell = layout.get(int(marker_id))
            if cell is None:
                continue
            col, row = map(int, cell)
            # Each tag sits in the black square of a 5 cm checker grid. The
            # tag's black outer edge is smaller (measured 3.7--3.75 cm), so
            # it must be centred in the grid cell rather than treated as the
            # grid pitch itself. The printed IDs are all rotated 180 degrees
            # relative to the board's +X/+Y grid convention.
            x0, y0 = col * cell_size, row * cell_size
            half = marker_size / 2.0
            object_points.append(np.array([
                [x0 + half, y0 + half, 0.0], [x0 - half, y0 + half, 0.0],
                [x0 - half, y0 - half, 0.0], [x0 + half, y0 - half, 0.0],
            ], dtype=np.float64))
            image_points.append(np.asarray(corners, dtype=np.float32).reshape(4, 2))
            used_corners.append(np.asarray(corners, dtype=np.float32).reshape(4, 2))
        if len(object_points) < 2:
            return None
        object_points_array = np.concatenate(object_points, axis=0)
        image_points_array = np.concatenate(image_points, axis=0).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            object_points_array, image_points_array, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        projected, _ = cv2.projectPoints(object_points_array, rvec, tvec, camera_matrix, distortion)
        error = np.linalg.norm(
            projected.reshape(-1, 2) - image_points_array.reshape(-1, 2), axis=1
        )
        return (
            rvec.reshape(3), tvec.reshape(3),
            np.concatenate(used_corners, axis=0).reshape(-1, 1, 2),
            float(np.sqrt(np.mean(error ** 2))),
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (int(board["inner_cols"]), int(board["inner_rows"]))

    def evaluate(corners: np.ndarray):
        ok, rvec, tvec = cv2.solvePnP(
            board_object_points(board), corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        projected, _ = cv2.projectPoints(
            board_object_points(board), rvec, tvec, camera_matrix, distortion
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        return rvec.reshape(3), tvec.reshape(3), corners, float(np.sqrt(np.mean(error ** 2)))

    if corners_override is not None:
        corners = np.asarray(corners_override, dtype=np.float32).reshape(-1, 1, 2)
        expected = pattern[0] * pattern[1]
        if len(corners) != expected or not np.all(np.isfinite(corners)):
            raise RuntimeError(f"invalid manually reviewed corners; expected {expected} points")
        return evaluate(corners)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_variants = [gray, clahe.apply(gray), cv2.bitwise_not(gray)]
    corner_candidates = []
    for gray_variant in gray_variants:
        if hasattr(cv2, "findChessboardCornersSB"):
            for sb_flags in (
                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                cv2.CALIB_CB_EXHAUSTIVE,
                0,
            ):
                found, corners = cv2.findChessboardCornersSB(gray_variant, pattern, flags=sb_flags)
                if found and corners is not None:
                    corner_candidates.append((True, corners))
                    break
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray_variant, pattern, flags)
        if found and corners is not None:
            corner_candidates.append((False, corners))

    if not corner_candidates:
        return None

    best = None
    for used_sb, corners in corner_candidates:
        corners = corners.reshape(-1, 1, 2).astype(np.float32)
        if not used_sb:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners, marker = canonicalize_checkerboard_corners(
            gray, corners, board, return_diagnostics=True
        )
        if require_marker_confidence and (
            marker["best_score"] < min_marker_score
            or marker["relative_margin"] < min_marker_relative_margin
        ):
            continue
        observation = evaluate(corners)
        if observation is not None and (best is None or observation[3] < best[3]):
            best = observation
    return best


def detect_board_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    board: dict,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    observation = detect_board_observation(image, camera_matrix, board, distortion)
    if observation is None:
        if board.get("type") == "aruco_grid":
            raise RuntimeError("ArUco board not detected; at least two known tags must be visible")
        if board.get("type") == "charuco":
            raise RuntimeError("ChArUco board not detected; at least six interpolated corners must be visible")
        pattern = (int(board["inner_cols"]), int(board["inner_rows"]))
        raise RuntimeError(f"chessboard not detected; expected inner corners {pattern}")
    rvec, tvec, _, _ = observation
    return rvec, tvec


def _import_parm_sdk():
    try:
        return importlib.import_module("parm_sdk")
    except Exception:
        pass
    candidates = [
        REPO_ROOT / ".venv" / "lib" / "python3.10" / "site-packages",
        Path.home() / "Public" / "data_collector" / ".venv" / "lib" / "python3.10" / "site-packages",
    ]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            return importlib.import_module("parm_sdk")
        except Exception:
            continue
    raise RuntimeError("parm_sdk is not importable; use the robot SDK environment")


def read_robot_base_to_eef(arm, arm_id: int) -> np.ndarray:
    state = arm.get_state()
    key = "left" if int(arm_id) == 1 else "right"
    data = (state or {}).get("state", {}).get(key, {})
    raw = data.get("eef_pose")
    if isinstance(raw, dict):
        xyz = raw.get("xyz")
        if xyz is None:
            raise RuntimeError(f"unsupported eef_pose dictionary: {raw}")
        if raw.get("rpy") is not None:
            # The live SDK state currently exposes dictionary-form RPY in radians.
            values = np.asarray(raw["rpy"], dtype=np.float64).reshape(-1)
            if values.size != 3:
                raise RuntimeError(f"invalid eef_pose.rpy: {raw}")
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = Rotation.from_euler("xyz", values).as_matrix()
            transform[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
            return transform
        rotvec = next(
            (raw[name] for name in ("rotvec", "rotation_vector", "rvec") if raw.get(name) is not None),
            None,
        )
        if rotvec is None:
            raise RuntimeError(f"unsupported eef_pose dictionary: {raw}")
        raw = list(xyz) + list(rotvec)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size < 6 or not np.all(np.isfinite(values[:6])):
        raise RuntimeError(f"invalid arm{arm_id} eef_pose: {raw}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[3:6]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def transform_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    transform[:3, 3] = tvec
    return transform


def transform_to_json(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=float).round(12).tolist()


def _residual_transform(transform: np.ndarray) -> np.ndarray:
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]


def solve_wrist(samples: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve T_eef_wrist1 and T_base_board for a stationary board."""
    base_eef = [np.asarray(s["T_base_eef"]) for s in samples]
    camera_board = [np.asarray(s["T_wrist1_board"]) for s in samples]

    def residual(x):
        flange_camera = pose_vector_to_transform(x[:6])
        base_board = pose_vector_to_transform(x[6:12])
        return np.concatenate([
            _residual_transform(np.linalg.inv(base_board) @ a @ flange_camera @ b)
            for a, b in zip(base_eef, camera_board)
        ])

    # A zero-pose nonlinear start has local minima for wide-angle hand-eye
    # data. OpenCV's closed-form hand-eye result provides a stable initializer.
    rotation, translation = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in base_eef],
        [transform[:3, 3] for transform in base_eef],
        [transform[:3, :3] for transform in camera_board],
        [transform[:3, 3] for transform in camera_board],
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    flange_camera = np.eye(4, dtype=np.float64)
    flange_camera[:3, :3] = rotation
    flange_camera[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    base_board = base_eef[0] @ flange_camera @ camera_board[0]
    x0 = np.r_[
        flange_camera[:3, 3],
        Rotation.from_matrix(flange_camera[:3, :3]).as_rotvec(),
        base_board[:3, 3],
        Rotation.from_matrix(base_board[:3, :3]).as_rotvec(),
    ]
    result = least_squares(residual, x0, loss="soft_l1", f_scale=0.01, max_nfev=3000)
    return pose_vector_to_transform(result.x[:6]), pose_vector_to_transform(result.x[6:12]), float(np.sqrt(np.mean(result.fun ** 2)))


def wrist_residuals(samples: list[dict], T_eef_wrist1: np.ndarray, T_base_board: np.ndarray) -> list[dict]:
    rows = []
    for sample in samples:
        residual = _residual_transform(
            np.linalg.inv(T_base_board)
            @ np.asarray(sample["T_base_eef"])
            @ T_eef_wrist1
            @ np.asarray(sample["T_wrist1_board"])
        )
        rows.append(
            {
                "index": int(sample["index"]),
                "translation_mm": float(np.linalg.norm(residual[:3]) * 1000.0),
                "rotation_deg": float(np.linalg.norm(residual[3:]) * 180.0 / np.pi),
                "reprojection_rms_px": float(sample.get("reprojection_rms_px", float("nan"))),
            }
        )
    return rows


def solve_wrist_with_outlier_rejection(
    samples: list[dict],
    max_translation_mm: float = 2.0,
    max_rotation_deg: float = 0.5,
    min_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, float, list[dict], list[dict], list[dict]]:
    """Solve once, remove explicit residual outliers, then solve again."""
    T_eef_wrist1, T_base_board, _ = solve_wrist(samples)
    initial_rows = wrist_residuals(samples, T_eef_wrist1, T_base_board)
    rejected = [
        row for row in initial_rows
        if row["translation_mm"] > max_translation_mm or row["rotation_deg"] > max_rotation_deg
    ]
    rejected_indices = {row["index"] for row in rejected}
    used_samples = [sample for sample in samples if sample["index"] not in rejected_indices]
    if rejected and len(used_samples) < min_samples:
        raise RuntimeError(
            f"outlier rejection would leave only {len(used_samples)} samples; need at least {min_samples}"
        )
    if rejected:
        T_eef_wrist1, T_base_board, rms = solve_wrist(used_samples)
    else:
        rms = float(np.sqrt(np.mean(np.asarray([
            row["translation_mm"] / 1000.0 for row in initial_rows
        ]) ** 2)))
    final_rows = wrist_residuals(used_samples, T_eef_wrist1, T_base_board)
    return T_eef_wrist1, T_base_board, rms, used_samples, rejected, final_rows


def solve_laser(
    samples: list[dict], camera_name: str = "laser"
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a fixed camera pose and the gripper-held board pose.

    ``camera_name`` selects the per-sample ``T_<camera_name>_board`` field.
    The historical function name is retained for compatibility with the Femto
    Bolt calibration script.
    """
    base_eef = [np.asarray(s["T_base_eef"]) for s in samples]
    camera_board = [np.asarray(s[f"T_{camera_name}_board"]) for s in samples]

    def residual(x):
        base_camera = pose_vector_to_transform(x[:6])
        flange_board = pose_vector_to_transform(x[6:12])
        return np.concatenate([
            _residual_transform(np.linalg.inv(base_camera) @ a @ flange_board @ np.linalg.inv(b))
            for a, b in zip(base_eef, camera_board)
        ])

    # The fixed-camera pose has several local minima when the board is viewed
    # from a limited range. Use deterministic starts and retain the best fit.
    starts = []
    for camera_z in (0.5, 1.0, 1.5):
        for camera_rotvec in (
            np.zeros(3),
            np.array([np.pi, 0.0, 0.0]),
            np.array([0.0, np.pi, 0.0]),
            np.array([0.0, 0.0, np.pi]),
        ):
            x0 = np.zeros(12)
            x0[2] = camera_z
            x0[3:6] = camera_rotvec
            starts.append(x0)
    results = [
        least_squares(residual, x0, loss="soft_l1", f_scale=0.01, max_nfev=2000)
        for x0 in starts
    ]
    result = min(results, key=lambda item: float(np.mean(item.fun ** 2)))
    return (
        pose_vector_to_transform(result.x[:6]),
        pose_vector_to_transform(result.x[6:12]),
        float(np.sqrt(np.mean(result.fun ** 2))),
    )


def laser_residuals(
    samples: list[dict], T_base_laser: np.ndarray, T_eef_board: np.ndarray,
    camera_name: str = "laser",
) -> list[dict]:
    """Report fixed-camera/board-held-by-EEF consistency for every sample."""
    rows = []
    for sample in samples:
        residual = _residual_transform(
            np.linalg.inv(T_base_laser)
            @ np.asarray(sample["T_base_eef"])
            @ T_eef_board
            @ np.linalg.inv(np.asarray(sample[f"T_{camera_name}_board"]))
        )
        rows.append(
            {
                "index": int(sample["index"]),
                "translation_mm": float(np.linalg.norm(residual[:3]) * 1000.0),
                "rotation_deg": float(np.degrees(np.linalg.norm(residual[3:]))),
                "reprojection_rms_px": float(sample.get("reprojection_rms_px", float("nan"))),
            }
        )
    return rows


def solve_laser_robust(
    samples: list[dict], max_outlier_fraction: float = 0.25, min_samples: int = 8,
    camera_name: str = "laser",
) -> tuple[np.ndarray, np.ndarray, float, list[dict], list[int], list[dict]]:
    """Fit eye-to-hand extrinsics while explicitly rejecting gross outliers.

    Rejection is bounded and reported; it never silently turns a mostly bad
    calibration session into an accepted result.
    """
    if not 0.0 <= max_outlier_fraction < 0.5:
        raise ValueError("max_outlier_fraction must be in [0, 0.5)")
    used = list(samples)
    rejected: list[int] = []
    max_rejected = int(np.floor(len(samples) * max_outlier_fraction))
    while True:
        T_base_laser, T_eef_board, rms = solve_laser(used, camera_name=camera_name)
        rows = laser_residuals(used, T_base_laser, T_eef_board, camera_name=camera_name)
        if len(used) <= min_samples or len(rejected) >= max_rejected:
            return T_base_laser, T_eef_board, rms, used, rejected, rows
        translation = np.asarray([row["translation_mm"] for row in rows])
        rotation = np.asarray([row["rotation_deg"] for row in rows])
        median_t, median_r = float(np.median(translation)), float(np.median(rotation))
        mad_t = float(np.median(np.abs(translation - median_t)))
        mad_r = float(np.median(np.abs(rotation - median_r)))
        # Keep the thresholds above sensor/robot noise. A sample must be a
        # clear outlier in translation or rotation before it is removed.
        limit_t = median_t + max(3.5 * mad_t, 10.0)
        limit_r = median_r + max(3.5 * mad_r, 2.0)
        scores = np.maximum(translation / max(limit_t, 1e-6), rotation / max(limit_r, 1e-6))
        worst = int(np.argmax(scores))
        if scores[worst] <= 1.0:
            return T_base_laser, T_eef_board, rms, used, rejected, rows
        rejected.append(int(used[worst]["index"]))
        used.pop(worst)


def pose_vector_to_transform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def capture_samples(
    camera,
    camera_name: str,
    arm_host: str,
    arm_id: int,
    board: dict,
    output_dir: Path,
    count: int,
    teach_free: bool = False,
    require_marker_confidence: bool = False,
    min_marker_score: float = 0.0,
    min_marker_relative_margin: float = 0.0,
    gripper_host: str | None = None,
    gripper_hand: int = 1,
    gripper_force: float = 60.0,
) -> None:
    sdk = _import_parm_sdk()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use a command-only connection for cancellation, matching the existing
    # tool calibration scripts. A completed/interrupted teach task can remain
    # latched remotely even after its local Python process has exited.
    with sdk.Arm(host=arm_host) as arm, sdk.Arm(host=arm_host, status_port=None) as cancel_arm:
        teach_started = False
        gripper = None
        gripper_closed: bool | None = None
        gripper_busy = False
        gripper_lock = threading.Lock()
        gripper_thread = None

        def toggle_gripper() -> None:
            """Use the same sync-in-worker pattern as parm_workpiece_collector."""
            nonlocal gripper, gripper_closed, gripper_busy
            with gripper_lock:
                if gripper_busy:
                    print(f"[{camera_name}] gripper is busy; wait for the current command")
                    return
                gripper_busy = True
            try:
                if gripper is None:
                    gripper = sdk.Gripper(host=gripper_host)
                    print(f"[{camera_name}] initializing gripper hand={gripper_hand}...")
                    try:
                        gripper.init(int(gripper_hand))
                    except Exception as exc:
                        # Keep the collector's behavior: an init failure is
                        # reported, but an already-homed gripper may still move.
                        print(f"[{camera_name}] gripper init/homing failed (continuing): {exc}")
                    state = (gripper.get_state() or {}).get(int(gripper_hand)) or {}
                    position = state.get("position")
                    gripper_closed = False if position is None else float(position) <= 50.0
                    print(
                        f"[{camera_name}] gripper hand={gripper_hand} state after init: "
                        f"position={position!r}"
                    )
                target = 100.0 if gripper_closed else 0.0
                # Do not send speed: the deployed AG gripper rejects that
                # optional parameter. Sync gives a confirmed completed action;
                # this worker thread keeps the preview responsive.
                gripper.move(
                    int(gripper_hand), target, force=float(gripper_force),
                    mode="sync", preempt=True,
                )
                gripper_closed = target == 0.0
                print(
                    f"[{camera_name}] gripper hand={gripper_hand} "
                    f"is now {'closed' if gripper_closed else 'open'}"
                )
            except Exception as exc:
                print(f"[{camera_name}] gripper toggle failed: {exc}")
            finally:
                with gripper_lock:
                    gripper_busy = False
        try:
            try:
                cancel_arm.cancel_task(arm_id=arm_id)
                time.sleep(0.15)
                print(f"[{camera_name}] cleared any previous arm{arm_id} task")
            except Exception as exc:
                # Some arm-server revisions return an error when idle; starting
                # teach below remains the authoritative operation.
                print(f"[{camera_name}] warning: pre-teach cancel_task failed: {exc}")
            arm.start_teach(arm_id=arm_id, free=bool(teach_free))
            teach_started = True
            mode = "Free" if teach_free else "drag"
            print(f"[{camera_name}] arm{arm_id} teach mode started: {mode}")
            with camera as camera_instance:
                if cv2 is None:
                    raise RuntimeError("OpenCV is required for the calibration preview window")
                window_name = f"calibration_{camera_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                samples: list[dict] = []
                print(f"[{camera_name}] preview ready: c=capture d=delete-last q=quit")
                while len(samples) < count:
                    frame = camera_instance.read()
                    observation = detect_board_observation(
                        frame.color, frame.color_intrinsics, board, frame.color_distortion,
                        require_marker_confidence, min_marker_score, min_marker_relative_margin,
                    )
                    vis = np.array(frame.color, copy=True)
                    if board.get("type") == "aruco_grid":
                        target_corners = len(board["marker_layout"]) * 4
                    elif board.get("type") == "charuco":
                        target_corners = (int(board["squares_x"]) - 1) * (int(board["squares_y"]) - 1)
                    else:
                        target_corners = int(board["inner_cols"]) * int(board["inner_rows"])
                    current_corners = 0 if observation is None else len(observation[2])
                    reproj = None if observation is None else observation[3]
                    if observation is not None:
                        rvec, tvec, corners, reproj = observation
                        if board.get("type") == "aruco_grid":
                            for quad in corners.reshape(-1, 4, 2).astype(np.int32):
                                cv2.polylines(vis, [quad], True, (0, 255, 0), 2)
                        elif board.get("type") == "charuco":
                            cv2.aruco.drawDetectedCornersCharuco(vis, corners)
                        else:
                            cv2.drawChessboardCorners(
                                vis,
                                (int(board["inner_cols"]), int(board["inner_rows"])),
                                corners,
                                True,
                            )
                        cv2.drawFrameAxes(
                            vis,
                            frame.color_intrinsics,
                            frame.color_distortion,
                            rvec,
                            tvec,
                            0.06,
                        )
                    bar = np.zeros((105, vis.shape[1], 3), dtype=np.uint8)
                    line1 = (
                        f"{camera_name}  samples={len(samples)}/{count}  "
                        f"corners={current_corners}/{target_corners}  "
                        f"reproj={'%.2f px' % reproj if reproj is not None else '--'}"
                    )
                    cv2.putText(bar, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
                    gripper_label = (
                        "gripper=disabled" if gripper_host is None else
                        "gripper=unknown" if gripper_closed is None else
                        "gripper=closed" if gripper_closed else "gripper=open"
                    )
                    cv2.putText(
                        bar,
                        "c=capture  SPACE=toggle-gripper  d=delete-last  q/ESC=quit",
                        (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1,
                    )
                    cv2.putText(
                        bar,
                        ("POSE OK" if observation is not None else "NO BOARD POSE") + f"  {gripper_label}",
                        (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 0) if observation is not None else (0, 0, 255),
                        2,
                    )
                    cv2.imshow(window_name, np.vstack([vis, bar]))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("d"):
                        if samples:
                            samples.pop()
                            print(f"[{camera_name}] deleted last sample; remaining={len(samples)}")
                        continue
                    if key == ord(" "):
                        if gripper_host is None:
                            print(f"[{camera_name}] gripper toggle disabled")
                            continue
                        gripper_thread = threading.Thread(target=toggle_gripper, daemon=True)
                        gripper_thread.start()
                        continue
                    if key != ord("c"):
                        continue
                    # Capture a fresh image between two robot reads. The preview
                    # frame may be old by the time the operator presses c.
                    before = read_robot_base_to_eef(arm, arm_id)
                    time.sleep(0.05)
                    capture_frame = camera_instance.read()
                    observation = detect_board_observation(
                        capture_frame.color,
                        capture_frame.color_intrinsics,
                        board,
                        capture_frame.color_distortion,
                        require_marker_confidence,
                        min_marker_score,
                        min_marker_relative_margin,
                    )
                    time.sleep(0.05)
                    after = read_robot_base_to_eef(arm, arm_id)
                    translation_delta_mm = float(np.linalg.norm(before[:3, 3] - after[:3, 3]) * 1000.0)
                    rotation_delta_deg = float(
                        np.degrees(
                            (Rotation.from_matrix(after[:3, :3])
                            * Rotation.from_matrix(before[:3, :3]).inv()).magnitude()
                        )
                    )
                    if observation is None:
                        print(f"[{camera_name}] rejected capture: board not detected in fresh frame")
                        continue
                    if translation_delta_mm > 0.5 or rotation_delta_deg > 0.25:
                        print(
                            f"[{camera_name}] rejected capture: arm is still moving "
                            f"(translation={translation_delta_mm:.2f}mm rotation={rotation_delta_deg:.2f}deg)"
                        )
                        continue
                    rvec, tvec, _, reproj = observation
                    index = len(samples)
                    image_path = output_dir / f"sample_{index:03d}.png"
                    cv2.imwrite(str(image_path), capture_frame.color)
                    sample = {
                        "index": index,
                        "image": image_path.name,
                        "T_base_eef": transform_to_json(after),
                        "camera_matrix": np.asarray(capture_frame.color_intrinsics).tolist(),
                        "distortion_coeffs": None if capture_frame.color_distortion is None else np.asarray(capture_frame.color_distortion).tolist(),
                        "image_size": [int(capture_frame.color.shape[1]), int(capture_frame.color.shape[0])],
                        "camera_serial": capture_frame.serial,
                        "teach_mode": mode,
                        "reprojection_rms_px": reproj,
                        "eef_stability": {
                            "translation_delta_mm": translation_delta_mm,
                            "rotation_delta_deg": rotation_delta_deg,
                        },
                        f"T_{camera_name}_board": transform_to_json(transform_from_pose(rvec, tvec)),
                    }
                    (output_dir / f"sample_{index:03d}.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
                    samples.append(sample)
                    print(
                        f"[{camera_name}] captured {len(samples)}/{count}: corners={len(observation[2])}, "
                        f"reproj={reproj:.2f}px stability={translation_delta_mm:.2f}mm/{rotation_delta_deg:.2f}deg"
                    )
                cv2.destroyWindow(window_name)
            save_json(output_dir / "session.json", {
                "camera_name": camera_name,
                "sample_files": [f"sample_{i:03d}.json" for i in range(len(samples))],
                "sample_count": len(samples),
                "teach_mode": mode,
                "gripper_toggle": {
                    "enabled": gripper_host is not None,
                    "host": gripper_host,
                    "hand": int(gripper_hand),
                    "force": float(gripper_force),
                },
            })
        finally:
            if gripper_thread is not None and gripper_thread.is_alive():
                print(f"[{camera_name}] waiting for gripper command to finish")
                gripper_thread.join(timeout=5.0)
            if gripper is not None:
                try:
                    gripper.close()
                except Exception as exc:
                    print(f"[{camera_name}] warning: failed to close gripper client: {exc}")
            if teach_started:
                try:
                    arm.stop_teach(arm_id=arm_id)
                    print(f"[{camera_name}] arm{arm_id} teach mode stopped")
                except Exception as exc:
                    print(f"[{camera_name}] warning: failed to stop teach mode: {exc}")
                    try:
                        cancel_arm.cancel_task(arm_id=arm_id)
                        print(f"[{camera_name}] cleared arm{arm_id} task after stop_teach failure")
                    except Exception as cancel_exc:
                        print(f"[{camera_name}] warning: cancel_task fallback failed: {cancel_exc}")


def load_samples(data_dir: Path) -> list[dict]:
    samples = []
    manifest = data_dir / "session.json"
    if manifest.exists():
        paths = [data_dir / str(name) for name in load_json(manifest).get("sample_files", [])]
    else:
        paths = sorted(data_dir.glob("sample_*.json"))
    for path in paths:
        if path.exists():
            samples.append(load_json(path))
    if len(samples) < 8:
        raise RuntimeError(f"at least 8 calibration samples are required, found {len(samples)}")
    return samples


def _corner_orientations(corners: np.ndarray, board: dict) -> tuple[np.ndarray, ...]:
    """Return the four possible physical assignments for detected corners."""
    rows = int(board["inner_rows"])
    cols = int(board["inner_cols"])
    grid = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 2)
    return (
        grid,
        grid[:, ::-1, :],
        grid[::-1, :, :],
        grid[::-1, ::-1, :],
    )


def _draw_corner_review(image: np.ndarray, corners: np.ndarray, board: dict, orientation: int) -> np.ndarray:
    vis = np.array(image, copy=True)
    rows = int(board["inner_rows"])
    cols = int(board["inner_cols"])
    grid = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 2)
    cv2.drawChessboardCorners(vis, (cols, rows), grid.reshape(-1, 1, 2), True)
    origin = tuple(np.round(grid[0, 0]).astype(int))
    x_axis = tuple(np.round(grid[0, -1]).astype(int))
    y_axis = tuple(np.round(grid[-1, 0]).astype(int))
    cv2.circle(vis, origin, 8, (0, 0, 255), -1)
    cv2.circle(vis, x_axis, 7, (0, 255, 0), -1)
    cv2.circle(vis, y_axis, 7, (255, 0, 0), -1)
    cv2.putText(vis, "O=(0,0)", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(vis, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(vis, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    bar = np.zeros((95, vis.shape[1], 3), dtype=np.uint8)
    mapping = (
        "1: O top-left, X right, Y down",
        "2: O top-right, X left, Y down",
        "3: O bottom-left, X right, Y up",
        "4: O bottom-right, X left, Y up",
    )
    cv2.putText(bar, mapping[orientation], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
    cv2.putText(bar, "1-4 select  x=flip X  y=flip Y  SPACE/ENTER confirm  q/ESC abort", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
    return np.vstack([vis, bar])


def review_board_corners(
    samples: list[dict], data_dir: Path, board: dict, camera_name: str = "laser"
) -> list[dict]:
    """Require an operator to confirm the corner-to-board coordinate assignment."""
    if cv2 is None:
        raise RuntimeError("OpenCV GUI is required for manual corner review")
    window_name = "laser_manual_corner_review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    reviewed = []
    try:
        for position, sample in enumerate(samples, start=1):
            image_path = data_dir / str(sample["image"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read calibration image: {image_path}")
            camera_matrix = np.asarray(sample["camera_matrix"], dtype=np.float64)
            distortion_raw = sample.get("distortion_coeffs")
            distortion = None if distortion_raw is None else np.asarray(distortion_raw, dtype=np.float64)
            detected = detect_board_observation(image, camera_matrix, board, distortion)
            if detected is None:
                raise RuntimeError(f"chessboard not detected for manual review: {image_path}")
            options = _corner_orientations(detected[2], board)
            selected = 0
            print(
                f"[{camera_name}] review {position}/{len(samples)} {image_path.name}: "
                "select 1-4, then SPACE/ENTER"
            )
            while True:
                cv2.imshow(window_name, _draw_corner_review(image, options[selected], board, selected))
                key = cv2.waitKey(0) & 0xFF
                if ord("1") <= key <= ord("4"):
                    selected = key - ord("1")
                    continue
                if key == ord("x"):
                    # Reversing +X also moves the origin to the opposite X edge.
                    selected ^= 1
                    continue
                if key == ord("y"):
                    # Reversing +Y also moves the origin to the opposite Y edge.
                    selected ^= 2
                    continue
                if key in (ord(" "), 13):
                    updated = dict(sample)
                    updated["manual_corners_px"] = options[selected].reshape(-1, 2).astype(float).tolist()
                    updated["corner_review"] = {
                        "status": "MANUALLY_CONFIRMED",
                        "orientation": selected + 1,
                    }
                    save_json(data_dir / f"sample_{int(sample['index']):03d}.json", updated)
                    reviewed.append(updated)
                    break
                if key in (ord("q"), 27):
                    raise RuntimeError("manual corner review aborted; no calibration was solved")
    finally:
        cv2.destroyWindow(window_name)
    return reviewed


def refresh_board_poses(
    samples: list[dict],
    data_dir: Path,
    camera_name: str,
    board: dict,
    require_marker_confidence: bool = False,
    min_marker_score: float = 0.0,
    min_marker_relative_margin: float = 0.0,
) -> list[dict]:
    """Recompute target poses from saved images using the current orientation rules."""
    refreshed = []
    for sample in samples:
        image_path = data_dir / str(sample["image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read calibration image: {image_path}")
        camera_matrix = np.asarray(sample["camera_matrix"], dtype=np.float64)
        distortion_raw = sample.get("distortion_coeffs")
        distortion = None if distortion_raw is None else np.asarray(distortion_raw, dtype=np.float64)
        manual_corners = sample.get("manual_corners_px")
        observation = detect_board_observation(
            image,
            camera_matrix,
            board,
            distortion,
            False if manual_corners is not None else require_marker_confidence,
            min_marker_score,
            min_marker_relative_margin,
            None if manual_corners is None else np.asarray(manual_corners, dtype=np.float32),
        )
        if observation is None:
            raise RuntimeError(f"chessboard not detected while refreshing {image_path}")
        rvec, tvec, _, reprojection_rms = observation
        updated = dict(sample)
        updated[f"T_{camera_name}_board"] = transform_to_json(transform_from_pose(rvec, tvec))
        updated["reprojection_rms_px"] = reprojection_rms
        refreshed.append(updated)
    return refreshed
