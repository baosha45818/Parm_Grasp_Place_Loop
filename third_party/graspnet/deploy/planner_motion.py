"""Shared planner-only arm motion adapter for GraspNet deployment code.

Every physical arm move follows the production path used by
``Parm_Grasp_Place_Loop``:

``Planner.plan_*_trajectory(...) -> POST /middle/waypoint/execute``.

This module deliberately never calls ``Arm.move_j``, ``Arm.move_l`` or any
servo API.  The execute endpoint is responsible for sending the approved
planned trajectory to the robot controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_PLANNER_HOST = "127.0.0.1"
# GraspNet uses its own calibrated Planner instance.  Keep all three SDK
# sub-services together; changing only trajectory_port would mix TF/waypoints
# from the normal Planner with the calibrated robot model.
DEFAULT_TRAJECTORY_PORT = 5605
DEFAULT_TF_PORT = 5614
DEFAULT_WAYPOINT_PORT = 5615
DEFAULT_EXECUTE_URL = "http://127.0.0.1:5667/middle/waypoint/execute"
LEFT_GROUP = "left_arm"
LEFT_ARM_ID = 1
LEFT_JOINT_NAMES = [f"left_arm_joint{index}" for index in range(1, 8)]
DEFAULT_BASE_FRAME = "body_base"
DEFAULT_LEFT_TOOL_FRAME = "left_hand_tool_end_link"
DEFAULT_LEGACY_ARM1_TF_BRIDGE = Path(
    "/home/dainanjun/qiuguolin/graspnet/deploy/calibration/output/planner_tf_bridge.json"
)


def _validate_speed(speed: float) -> float:
    speed = float(speed)
    if not math.isfinite(speed) or not 0.0 < speed <= 1.0:
        raise ValueError("planner max_vel/max_acc must be finite and in (0, 1]")
    return speed


def _planned_trajectory(response: dict, expected_arm_id: int = LEFT_ARM_ID) -> tuple[list[list[float]], int, float]:
    data = response.get("data") or {}
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError(f"planner did not return a non-empty trajectory: {response}")
    if any(not isinstance(point, list) or len(point) != 7 for point in trajectory):
        raise RuntimeError("planner returned an invalid single-arm trajectory")
    arm_id = int(data.get("arm_id", -1))
    if arm_id != expected_arm_id:
        raise RuntimeError(f"planner returned arm_id={arm_id}; expected arm_id={expected_arm_id}")
    frequency = float(data.get("sample_frequency", 10.0))
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise RuntimeError(f"planner returned invalid sample_frequency={frequency}")
    return trajectory, arm_id, frequency


def _execute(arm_id: int, trajectory: list[list[float]], sample_frequency: float, execute_url: str) -> None:
    payload = json.dumps({
        "arm_id": int(arm_id),
        "trajectory": trajectory,
        "sample_frequency": float(sample_frequency),
    }).encode("utf-8")
    request = Request(
        str(execute_url), data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(request, timeout=300.0) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"planned-trajectory execute returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"cannot reach planned-trajectory execute endpoint {execute_url}; "
            "start /home/dainanjun/Public/plucky_hjt_parm with docker compose"
        ) from exc
    try:
        body = json.loads(response_text)
    except json.JSONDecodeError:
        body = {"raw_response": response_text}
    if body.get("success") is False:
        raise RuntimeError(f"planned trajectory execution failed: {body}")


def _transform_from_planner_response(response: dict, source: str, target: str) -> np.ndarray:
    """Return ``T_target_source`` from the deployed Planner TF response.

    The running Planner service serializes the inverse of the transform named
    by ``lookup_transform(source, target)``.  This is the convention used when
    the accepted bridge calibration was captured; normalize it here so every
    GraspNet Cartesian command uses the same convention.
    """
    data = response.get("data") or response
    if data.get("source_frame") != source or data.get("target_frame") != target:
        raise RuntimeError(
            f"unexpected Planner TF response for {source} -> {target}: {data}"
        )
    translation = data.get("translation") or {}
    rotation = data.get("rotation") or {}
    xyz = [translation.get(axis) for axis in "xyz"]
    xyzw = [rotation.get(axis) for axis in "xyzw"]
    if any(value is None for value in xyz + xyzw):
        raise RuntimeError(f"planner returned an invalid TF response: {response}")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(xyzw, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return np.linalg.inv(T)


def _target_transform(position: Sequence[float], orientation_xyzw: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(orientation_xyzw, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def _load_legacy_arm1_bridge(bridge_path: str | Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load the accepted ``arm1_base/eef_pose`` to Planner-TCP bridge."""
    path = Path(bridge_path)
    if not path.is_file():
        raise RuntimeError(f"legacy arm1-base TF bridge is missing: {path}")
    bridge = json.loads(path.read_text(encoding="utf-8"))
    if bridge.get("status") != "ACCEPTED":
        raise RuntimeError(f"legacy arm1-base TF bridge is not ACCEPTED: {path}")
    T_planner_base_arm1_base = np.asarray(bridge.get("T_planner_base_base"), dtype=np.float64)
    T_eef_bridge_tool = np.asarray(bridge.get("T_eef_planner_tool"), dtype=np.float64)
    if T_planner_base_arm1_base.shape != (4, 4) or not np.all(np.isfinite(T_planner_base_arm1_base)):
        raise RuntimeError(f"{path}: T_planner_base_base must be a finite 4x4 matrix")
    if T_eef_bridge_tool.shape != (4, 4) or not np.all(np.isfinite(T_eef_bridge_tool)):
        raise RuntimeError(f"{path}: T_eef_planner_tool must be a finite 4x4 matrix")
    planner_base_frame = str(bridge.get("planner_base_frame") or "").strip()
    bridge_tool_frame = str(bridge.get("planner_tool_frame") or "").strip()
    if not planner_base_frame or not bridge_tool_frame:
        raise RuntimeError(f"{path}: planner_base_frame and planner_tool_frame are required")
    return T_planner_base_arm1_base, T_eef_bridge_tool, planner_base_frame, bridge_tool_frame


def _transform_cartesian_target_to_planner_base(
    planner, position: Sequence[float], orientation_xyzw: Sequence[float], *,
    source_frame: str, base_frame: str, tool_frame: str, legacy_tf_bridge: str | Path,
) -> tuple[list[float], list[float]]:
    source = str(source_frame).strip()
    if not source:
        raise ValueError("Cartesian source_frame must be explicitly declared")
    T_source_target = _target_transform(position, orientation_xyzw)
    if source == base_frame:
        T_base_target = T_source_target
    elif source == "legacy_arm1_base":
        (
            T_bridge_base_legacy_base,
            T_legacy_eef_bridge_tool,
            bridge_base_frame,
            bridge_tool_frame,
        ) = _load_legacy_arm1_bridge(legacy_tf_bridge)
        if bridge_base_frame == base_frame:
            T_base_bridge_base = np.eye(4)
        else:
            T_base_bridge_base = _transform_from_planner_response(
                planner.lookup_transform(bridge_base_frame, base_frame), bridge_base_frame, base_frame
            )
        if bridge_tool_frame == tool_frame:
            T_bridge_tool_target_tool = np.eye(4)
        else:
            T_bridge_tool_target_tool = _transform_from_planner_response(
                planner.lookup_transform(tool_frame, bridge_tool_frame), tool_frame, bridge_tool_frame
            )
        # The legacy command is T_arm1_base_eef.  Planner receives a target
        # for ``tool_frame``, so both fixed bridge transforms are mandatory.
        T_base_target = (
            T_base_bridge_base
            @ T_bridge_base_legacy_base
            @ T_source_target
            @ T_legacy_eef_bridge_tool
            @ T_bridge_tool_target_tool
        )
    else:
        T_base_target = _transform_from_planner_response(
            planner.lookup_transform(source, base_frame), source, base_frame
        ) @ T_source_target
    return T_base_target[:3, 3].tolist(), Rotation.from_matrix(T_base_target[:3, :3]).as_quat().tolist()


def plan_joint_rrtstar(
    target_joints_deg: Sequence[float], *, planner_host: str = DEFAULT_PLANNER_HOST,
    trajectory_port: int = DEFAULT_TRAJECTORY_PORT, tf_port: int = DEFAULT_TF_PORT,
    waypoint_port: int = DEFAULT_WAYPOINT_PORT,
    speed: float = 0.1, execute_url: str = DEFAULT_EXECUTE_URL, execute: bool = True,
) -> dict:
    """Plan left-arm joint RRTStar motion, optionally send the resulting trajectory."""
    target = [float(value) for value in target_joints_deg]
    if len(target) != 7 or not all(math.isfinite(value) for value in target):
        raise ValueError("left joint target must contain seven finite values in degrees")
    speed = _validate_speed(speed)
    from parm_sdk import Planner

    with Planner(
        host=str(planner_host), trajectory_port=int(trajectory_port),
        tf_port=int(tf_port), waypoint_port=int(waypoint_port),
    ) as planner:
        response = planner._trajectory.send_payload({
            "waypoints": [{
                "name": "graspnet_joint_rrtstar",
                "type": "joint",
                "group": LEFT_GROUP,
                "planner": "rrtstar",
                "max_velocity_scaling_factor": speed,
                "max_acceleration_scaling_factor": speed,
                "joint_names": LEFT_JOINT_NAMES,
                "joint_values": target,
            }],
            "topp": {"max_vel_factor": speed, "max_acc_factor": speed},
        }, timeout=30.0)
    trajectory, arm_id, sample_frequency = _planned_trajectory(response)
    print(f"-> planner joint RRTStar: {len(trajectory)} points at {sample_frequency:.2f} Hz")
    print("   final joints(deg):", [round(value, 3) for value in trajectory[-1]])
    if execute:
        _execute(arm_id, trajectory, sample_frequency, execute_url)
        print("-> planned left-arm trajectory executed")
    return response


def plan_cartesian(
    position_m: Sequence[float], orientation_xyzw: Sequence[float], *, planner_type: str = "lin",
    planner_host: str = DEFAULT_PLANNER_HOST, trajectory_port: int = DEFAULT_TRAJECTORY_PORT,
    tf_port: int = DEFAULT_TF_PORT, waypoint_port: int = DEFAULT_WAYPOINT_PORT, speed: float = 0.1,
    execute_url: str = DEFAULT_EXECUTE_URL, execute: bool = True,
    base_frame: str = DEFAULT_BASE_FRAME, tool_frame: str = DEFAULT_LEFT_TOOL_FRAME,
    source_frame: str = DEFAULT_BASE_FRAME,
    legacy_tf_bridge: str | Path = DEFAULT_LEGACY_ARM1_TF_BRIDGE,
) -> dict:
    """Plan a left-arm Cartesian RRTStar/LIN move in the production base frame."""
    position = [float(value) for value in position_m]
    orientation = [float(value) for value in orientation_xyzw]
    if len(position) != 3 or len(orientation) != 4 or not all(math.isfinite(value) for value in position + orientation):
        raise ValueError("Cartesian target must contain finite xyz and quaternion xyzw")
    if np.linalg.norm(orientation) < 1e-9:
        raise ValueError("Cartesian target quaternion must be non-zero")
    if planner_type not in {"rrtstar", "lin"}:
        raise ValueError("Cartesian planner_type must be 'rrtstar' or 'lin'")
    speed = _validate_speed(speed)
    orientation = (np.asarray(orientation) / np.linalg.norm(orientation)).tolist()
    from parm_sdk import Planner

    with Planner(
        host=str(planner_host), trajectory_port=int(trajectory_port),
        tf_port=int(tf_port), waypoint_port=int(waypoint_port),
    ) as planner:
        position, orientation = _transform_cartesian_target_to_planner_base(
            planner, position, orientation, source_frame=source_frame, base_frame=base_frame,
            tool_frame=tool_frame, legacy_tf_bridge=legacy_tf_bridge,
        )
        print(f"-> Planner TF: {source_frame} -> {base_frame}")
        if planner_type == "rrtstar":
            response = planner._trajectory.send_payload({
                "waypoints": [{
                    "name": "graspnet_cartesian_rrtstar",
                    "type": "cart",
                    "group": LEFT_GROUP,
                    "planner": "rrtstar",
                    "max_velocity_scaling_factor": speed,
                    "max_acceleration_scaling_factor": speed,
                    "frame_id": str(base_frame),
                    "ik_frame": str(tool_frame),
                    "position": position,
                    "orientation": orientation,
                }],
                "topp": {"max_vel_factor": speed, "max_acc_factor": speed},
            }, timeout=30.0)
        else:
            response = planner.plan_cart_trajectory(
                group=LEFT_GROUP, planner="lin", frame_id=str(base_frame), ik_frame=str(tool_frame),
                position=position, orientation=orientation, max_vel=speed, max_acc=speed,
            )
    trajectory, arm_id, sample_frequency = _planned_trajectory(response)
    print(
        f"-> planner Cartesian {planner_type.upper()}: {len(trajectory)} points at "
        f"{sample_frequency:.2f} Hz, frame={base_frame}, tool={tool_frame}"
    )
    if execute:
        _execute(arm_id, trajectory, sample_frequency, execute_url)
        print("-> planned left-arm trajectory executed")
    return response


def axis_angle_pose_to_quaternion(pose6: Sequence[float]) -> tuple[list[float], list[float]]:
    """Convert legacy [x,y,z,rx,ry,rz] axis-angle metres/radians to planner xyzw."""
    values = [float(value) for value in pose6]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("legacy Cartesian pose must contain six finite values")
    return values[:3], Rotation.from_rotvec(values[3:]).as_quat().tolist()


def euler_xyz_pose_to_quaternion(pose6: Sequence[float]) -> tuple[list[float], list[float]]:
    """Convert legacy [x,y,z,roll,pitch,yaw] Euler pose to planner xyzw."""
    values = [float(value) for value in pose6]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("legacy Cartesian pose must contain six finite values")
    return values[:3], Rotation.from_euler("xyz", values[3:]).as_quat().tolist()


def plan_legacy_axis_angle_pose(
    pose6: Sequence[float], *, planner_type: str = "lin", planner_host: str = DEFAULT_PLANNER_HOST,
    trajectory_port: int = DEFAULT_TRAJECTORY_PORT, tf_port: int = DEFAULT_TF_PORT,
    waypoint_port: int = DEFAULT_WAYPOINT_PORT,
    speed: float = 0.1, execute_url: str = DEFAULT_EXECUTE_URL, execute: bool = True,
    base_frame: str = DEFAULT_BASE_FRAME, tool_frame: str = DEFAULT_LEFT_TOOL_FRAME,
    source_frame: str = "legacy_arm1_base",
    legacy_tf_bridge: str | Path = DEFAULT_LEGACY_ARM1_TF_BRIDGE,
) -> dict:
    """Compatibility adapter for legacy servo/move_l pose commands.

    The legacy pose convention was base-frame xyz in metres plus axis-angle
    radians.  It is converted to the Planner cartesian quaternion convention.
    """
    position, orientation = axis_angle_pose_to_quaternion(pose6)
    return plan_cartesian(
        position, orientation, planner_type=planner_type, planner_host=planner_host,
        trajectory_port=trajectory_port, tf_port=tf_port, waypoint_port=waypoint_port,
        speed=speed, execute_url=execute_url, execute=execute,
        base_frame=base_frame, tool_frame=tool_frame, source_frame=source_frame,
        legacy_tf_bridge=legacy_tf_bridge,
    )


def plan_legacy_euler_xyz_pose(
    pose6: Sequence[float], *, planner_type: str = "lin", planner_host: str = DEFAULT_PLANNER_HOST,
    trajectory_port: int = DEFAULT_TRAJECTORY_PORT, tf_port: int = DEFAULT_TF_PORT,
    waypoint_port: int = DEFAULT_WAYPOINT_PORT,
    speed: float = 0.1, execute_url: str = DEFAULT_EXECUTE_URL, execute: bool = True,
    base_frame: str = DEFAULT_BASE_FRAME, tool_frame: str = DEFAULT_LEFT_TOOL_FRAME,
    source_frame: str = "legacy_arm1_base",
    legacy_tf_bridge: str | Path = DEFAULT_LEGACY_ARM1_TF_BRIDGE,
) -> dict:
    """Compatibility adapter for legacy Euler-XYZ ``move_l`` flange poses."""
    position, orientation = euler_xyz_pose_to_quaternion(pose6)
    return plan_cartesian(
        position, orientation, planner_type=planner_type, planner_host=planner_host,
        trajectory_port=trajectory_port, tf_port=tf_port, waypoint_port=waypoint_port,
        speed=speed, execute_url=execute_url, execute=execute,
        base_frame=base_frame, tool_frame=tool_frame, source_frame=source_frame,
        legacy_tf_bridge=legacy_tf_bridge,
    )
