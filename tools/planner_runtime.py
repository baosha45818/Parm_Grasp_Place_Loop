"""封装规划器 TF 查询、base 坐标系轨迹规划和直接 SDK 轨迹执行。"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any

from loop_config import LoopConfig
from loop_types import ObjectPose
from tools.planner_task_retry import retry_task_creation


class TaskBusyError(RuntimeError):
    """The target arm already has a running SDK task."""


class PlannerRuntime:
    """Planner/TF plus direct ``Arm.follow_trajectory`` execution live here."""

    # This recorder is intentionally limited to the two left-place edges that
    # are needed for placement analysis.  Other planner traffic is not a
    # placement trajectory and must not make these logs noisy.
    _RECORDED_LOGICAL_MOTIONS = {
        ("left_graspup_top", "left_place_position"),
        ("left_place_position", "left_graspup_top"),
        ("left_place_position", "left_graspup"),
    }
    def __init__(self, config: LoopConfig) -> None:
        self.config = config
        self._planner: Any = None
        default_dir = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "left_place_trajectories"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self._record_dir = Path(
            os.environ.get("PARM_PLANNING_RECORD_DIR", str(default_dir))
        )
        self._record_lock = threading.Lock()
        self._next_plan_context: dict[str, Any] | None = None

    def set_next_plan_context(
        self,
        from_stage: str,
        to_stage: str,
        *,
        start_tool_pose: ObjectPose | None = None,
    ) -> None:
        """Attach one human-readable flow edge to the next planning request."""
        start_pose = None
        if start_tool_pose is not None:
            start_pose = {
                "frame_id": start_tool_pose.frame_id,
                "ik_frame": None,
                "position_m": [float(value) for value in start_tool_pose.position],
                "orientation_xyzw": [float(value) for value in start_tool_pose.orientation],
            }
        self._next_plan_context = {
            "from": str(from_stage),
            "to": str(to_stage),
            "start_tool_pose": start_pose,
        }

    def _record_plan(
        self,
        *,
        status: str,
        arm: str,
        planner: str,
        request_type: str,
        target: dict[str, Any],
        start_joints_deg: list[float] | None = None,
        response: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Append one planner request, including the returned joint trajectory when present."""
        data = (response or {}).get("data") or {}
        trajectory = data.get("trajectory") or []
        logical_context = self._next_plan_context
        self._next_plan_context = None
        if (
            logical_context is None
            or (logical_context["from"], logical_context["to"])
            not in self._RECORDED_LOGICAL_MOTIONS
        ):
            return

        from_joint_deg = start_joints_deg or (trajectory[0] if trajectory else None)
        # A Cartesian placement candidate can supply its intended graspup-top
        # start pose.  For an online replan, read the actual start from the
        # planner TF tree.  Joint-space records deliberately omit Cartesian data.
        from_pose = None
        if request_type != "joint":
            from_pose = (
                logical_context["start_tool_pose"]
                or self._recorded_tool_pose(arm)
            )
            if from_pose is not None:
                from_pose["ik_frame"] = target.get("ik_frame")
        if "joint_values_deg" in target:
            to_location = {
                "kind": "joint_target",
                "joint_angles_deg": target["joint_values_deg"],
            }
        elif "position_m" in target:
            to_location = {
                "kind": "tool_pose_target",
                "frame_id": target.get("frame_id"),
                "ik_frame": target.get("ik_frame"),
                "position_m": target.get("position_m"),
                "orientation_xyzw": target.get("orientation_xyzw"),
            }
        else:
            to_location = {"kind": "waypoint_sequence", **target}
        default_from = f"{arm}_arm_current_state"
        default_to = (
            f"{arm}_arm_joint_target"
            if "joint_values_deg" in target
            else f"{arm}_hand_tool_target"
        )
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "status": status,
            "arm": arm,
            "planner": planner,
            "request_type": request_type,
            "start_joints_deg": start_joints_deg,
            "target": target,
            "from": {
                "source": (
                    "explicit_offline_start_state"
                    if start_joints_deg is not None
                    else "live_state_before_planning"
                ),
                "joint_angles_deg": from_joint_deg,
                "tool_pose": from_pose,
            },
            "to": to_location,
            "logical_motion": {
                "from": logical_context["from"],
                "to": logical_context["to"],
            },
            "planned_start_joints_deg": trajectory[0] if trajectory else None,
            "planned_end_joints_deg": trajectory[-1] if trajectory else None,
            "trajectory_joint_deg": trajectory if status == "success" else None,
            "sample_frequency_hz": data.get("sample_frequency"),
            "error": (
                f"{type(error).__name__}: {error}" if error is not None else None
            ),
        }
        try:
            self._record_dir.mkdir(parents=True, exist_ok=True)
            # Separate files per process avoid interleaving records from workers.
            path = self._record_dir / f"left_place_attempts_pid_{os.getpid()}.jsonl"
            with self._record_lock, path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as record_error:
            # Diagnostics must never block a live safety-critical planning path.
            print(f"[planner-record] 写入失败: {type(record_error).__name__}: {record_error}", flush=True)

    def _require_trajectory(
        self,
        response: dict[str, Any],
        *,
        arm: str,
        planner: str,
        request_type: str,
        target: dict[str, Any],
        start_joints_deg: list[float] | None = None,
    ) -> dict[str, Any]:
        trajectory = ((response.get("data") or {}).get("trajectory") or [])
        if not trajectory:
            error = RuntimeError(f"规划器未返回{arm}臂{request_type}轨迹: {response}")
            self._record_plan(
                status="failure",
                arm=arm,
                planner=planner,
                request_type=request_type,
                target=target,
                start_joints_deg=start_joints_deg,
                response=response,
                error=error,
            )
            raise error
        self._record_plan(
            status="success",
            arm=arm,
            planner=planner,
            request_type=request_type,
            target=target,
            start_joints_deg=start_joints_deg,
            response=response,
        )
        return response

    def _recorded_tool_pose(self, arm: str) -> dict[str, Any] | None:
        """Best-effort body_base tool pose for human-readable trajectory provenance."""
        if self._planner is None:
            return None
        tool_frame = (
            self.config.left_tool_frame if arm == "left" else self.config.right_tool_frame
        )
        try:
            raw = self._planner.lookup_transform(self.config.base_frame, tool_frame)
            data = (raw or {}).get("data") or raw or {}
            translation = data.get("translation") or {}
            rotation = data.get("rotation") or {}
            position = [translation.get(axis) for axis in "xyz"]
            orientation = [rotation.get(axis) for axis in "xyzw"]
            if any(value is None for value in position + orientation):
                return None
            return {
                "frame_id": self.config.base_frame,
                "ik_frame": tool_frame,
                "position_m": [float(value) for value in position],
                "orientation_xyzw": [float(value) for value in orientation],
            }
        except Exception:
            return None

    def _recorded_start_joints(
        self, arm: str, supplied: list[float] | None = None
    ) -> list[float] | None:
        """Use an explicit offline start when supplied, otherwise best-effort live state."""
        if supplied is not None:
            return [float(value) for value in supplied]
        try:
            from parm_sdk import Arm

            with Arm(host=os.environ.get("PARM_ARM_HOST", "192.168.3.11")) as sdk_arm:
                state = sdk_arm.get_state() or {}
            values = ((state.get("state") or {}).get(arm) or {}).get("position")
            if (
                isinstance(values, list)
                and len(values) == 7
                and all(math.isfinite(float(value)) for value in values)
            ):
                return [float(value) for value in values]
        except Exception:
            # State recording is diagnostic only; never turn a planning request into a failure.
            pass
        return None

    def open(self, required_frames: set[str] | None = None) -> None:
        if not self.config.live:
            print("[planner] dry-run: 不连接规划器，不执行机械臂运动")
            return
        from parm_sdk import Planner

        self._planner = Planner(host=self.config.planner_host)
        frames = self._planner.list_frame_ids()
        required = required_frames or {
            self.config.base_frame,
            self.config.left_tool_frame,
            self.config.right_tool_frame,
        }
        missing = sorted(required.difference(frames))
        if missing:
            raise RuntimeError(f"规划器 TF 缺少坐标系: {missing}")

    def close(self) -> None:
        if self._planner is not None:
            self._planner.close()
            self._planner = None

    def get_tool_pose(self, arm: str) -> dict[str, Any] | None:
        tool_frame = (
            self.config.left_tool_frame if arm == "left" else self.config.right_tool_frame
        )
        if self._planner is None:
            print(f"[planner][dry-run] TF: {tool_frame} -> {self.config.base_frame}")
            return None
        return self._planner.lookup_transform(self.config.base_frame, tool_frame)

    def lookup_transform(self, source_frame: str, target_frame: str) -> dict[str, Any] | None:
        if self._planner is None:
            print(f"[planner][dry-run] TF: {source_frame} -> {target_frame}")
            return None
        return self._planner.lookup_transform(source_frame, target_frame)

    @staticmethod
    def _allow_right_hand_tall_contact(arm: str) -> bool:
        """Limit the Module 9 contact exception to live left-arm requests."""
        marker = os.environ.get("PARM_RIGHT_SERVO_TALL_CONTACT_FILE")
        return arm == "left" and bool(marker) and Path(marker).is_file()

    def move_left_joints(
        self,
        joints_deg: list[float],
        *,
        planner: str = "rrtstar",
        use_left_eef_collision_box: bool = False,
    ) -> None:
        self._move_arm_joints(
            "left",
            joints_deg,
            planner=planner,
            use_left_eef_collision_box=use_left_eef_collision_box,
        )

    def move_right_joints(
        self, joints_deg: list[float], *, planner: str = "rrtstar"
    ) -> None:
        """Plan and execute one bounded right-arm joint-space target."""
        self._move_arm_joints("right", joints_deg, planner=planner)

    def _move_arm_joints(
        self,
        arm: str,
        joints_deg: list[float],
        *,
        planner: str = "rrtstar",
        use_left_eef_collision_box: bool = False,
    ) -> None:
        response = self.plan_joint_trajectory(
            arm,
            joints_deg,
            planner=planner,
            use_left_eef_collision_box=use_left_eef_collision_box,
        )
        if response is not None:
            self.execute_planned_result(response)

    def plan_joint_trajectory(
        self,
        arm: str,
        joints_deg: list[float],
        *,
        planner: str = "rrtstar",
        use_left_eef_collision_box: bool = False,
    ) -> dict[str, Any] | None:
        """Plan one arm joint-space target without executing it."""
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm: {arm!r}")
        if len(joints_deg) != 7 or not all(math.isfinite(value) for value in joints_deg):
            raise ValueError(f"{arm} joint target must contain seven finite degree values")
        if planner not in {"ptp", "lin", "ompl", "rrtstar"}:
            raise ValueError(f"unsupported joint planner: {planner!r}")
        if self._planner is None:
            print(f"[planner][dry-run] {arm} {planner.upper()} joints(deg)={joints_deg}")
            return None
        group = self.config.left_group if arm == "left" else self.config.right_group
        if use_left_eef_collision_box and arm != "left":
            raise ValueError("use_left_eef_collision_box 仅支持左臂请求")
        target = {
            "joint_values_deg": [float(value) for value in joints_deg],
            "use_left_eef_collision_box": bool(use_left_eef_collision_box),
        }
        start_joints_deg = self._recorded_start_joints(arm)
        joint_names = [f"{arm}_arm_joint{index}" for index in range(1, 8)]
        def request() -> dict[str, Any]:
            allow_right_hand_tall_contact = self._allow_right_hand_tall_contact(arm)
            # The SDK convenience methods do not expose the right-gripper/tall-box
            # exception. Use the equivalent wire request only when either
            # request-local collision policy is needed; ordinary requests retain
            # the SDK path unchanged.
            if (
                planner == "rrtstar"
                or use_left_eef_collision_box
                or allow_right_hand_tall_contact
            ):
                # The server exposes RRTstar although this SDK's public joint
                # helper only whitelists ptp/lin/ompl.
                payload = {
                    "waypoints": [{
                        "name": f"runtime_{planner}_joint_target",
                        "type": "joint",
                        "group": group,
                        "planner": planner,
                        "max_velocity_scaling_factor": self.config.max_velocity,
                        "max_acceleration_scaling_factor": self.config.max_acceleration,
                        "joint_names": joint_names,
                        "joint_values": target["joint_values_deg"],
                    }],
                    "topp": {
                        "max_vel_factor": self.config.max_velocity,
                        "max_acc_factor": self.config.max_acceleration,
                    },
                }
                if use_left_eef_collision_box:
                    payload["use_left_eef_collision_box"] = True
                if allow_right_hand_tall_contact:
                    payload["allow_right_hand_tall_contact"] = True
                return self._planner._trajectory.send_payload(payload, timeout=30.0)
            return self._planner.plan_joint_trajectory(
                group=group,
                planner=planner,
                joint_names=joint_names,
                joint_values=target["joint_values_deg"],
                max_vel=self.config.max_velocity,
                max_acc=self.config.max_acceleration,
                use_left_eef_collision_box=use_left_eef_collision_box,
            )

        try:
            response = retry_task_creation(
                request,
                label=f"{arm}臂 {planner.upper()} 关节规划",
            )
        except BaseException as exc:
            self._record_plan(
                status="failure", arm=arm, planner=planner,
                request_type="joint", target=target,
                start_joints_deg=start_joints_deg, error=exc,
            )
            raise
        return self._require_trajectory(
            response, arm=arm, planner=planner, request_type="joint", target=target,
            start_joints_deg=start_joints_deg,
        )

    def move_cartesian(self, arm: str, pose_name: str, *, planner: str = "rrtstar") -> None:
        pose = self.config.poses.get(pose_name)
        if pose is None:
            print(f"[planner][占位] {arm}: {pose_name} 尚未配置，跳过运动")
            return
        self._validate_pose(pose)
        group = self.config.left_group if arm == "left" else self.config.right_group
        tool_frame = (
            self.config.left_tool_frame if arm == "left" else self.config.right_tool_frame
        )
        if self._planner is None:
            print(
                f"[planner][dry-run] {arm} {planner.upper()} -> {pose_name}, "
                f"frame={self.config.base_frame}, xyz={pose.position}, quaternion={pose.orientation}"
            )
            return

        self.move_cartesian_pose(arm, pose, planner=planner)

    def move_cartesian_pose(
        self,
        arm: str,
        pose: ObjectPose,
        *,
        planner: str = "rrtstar",
        use_left_eef_collision_box: bool = False,
    ) -> None:
        """Plan and execute one explicit Cartesian tool pose in body_base."""
        response = self.plan_cartesian_pose(
            arm,
            pose,
            planner=planner,
            use_left_eef_collision_box=use_left_eef_collision_box,
        )
        if response is not None:
            self.execute_planned_result(response)

    def plan_cartesian_pose(
        self,
        arm: str,
        pose: ObjectPose,
        *,
        planner: str = "rrtstar",
        start_joints_deg: list[float] | None = None,
        rrtstar_goal_tolerance: bool = False,
        use_left_eef_collision_box: bool = False,
    ) -> dict[str, Any] | None:
        """Plan one explicit Cartesian tool pose without executing it."""
        if planner not in {"ptp", "lin", "ompl", "rrtstar"}:
            raise ValueError(f"unsupported Cartesian planner: {planner!r}")
        if rrtstar_goal_tolerance and planner != "rrtstar":
            raise ValueError(
                "rrtstar_goal_tolerance is only valid with planner='rrtstar'"
            )
        if use_left_eef_collision_box and arm != "left":
            raise ValueError("use_left_eef_collision_box 仅支持左臂请求")
        self._validate_pose(pose)
        group = self.config.left_group if arm == "left" else self.config.right_group
        tool_frame = (
            self.config.left_tool_frame if arm == "left" else self.config.right_tool_frame
        )
        if self._planner is None:
            print(
                f"[planner][dry-run] {arm} {planner.upper()}, "
                f"frame={pose.frame_id}, xyz={pose.position}, quaternion={pose.orientation}"
            )
            return None
        target = {
            "frame_id": pose.frame_id,
            "ik_frame": tool_frame,
            "position_m": list(pose.position),
            "orientation_xyzw": list(pose.orientation),
            "use_left_eef_collision_box": bool(use_left_eef_collision_box),
        }
        recorded_start = self._recorded_start_joints(arm, start_joints_deg)
        def request() -> dict[str, Any]:
            if start_joints_deg is not None:
                if (
                    len(start_joints_deg) != 7
                    or not all(math.isfinite(value) for value in start_joints_deg)
                ):
                    raise ValueError(
                        "offline Cartesian start must contain seven finite degrees"
                    )
                joint_names = [
                    f"{arm}_arm_joint{index}" for index in range(1, 8)
                ]
                payload = {
                        "start_state": {
                            "joint_names": joint_names,
                            "joint_values": [float(value) for value in start_joints_deg],
                        },
                        "waypoints": [{
                            "name": "runtime_offline_cartesian_target",
                            "type": "cart",
                            "group": group,
                            "planner": planner,
                            "max_velocity_scaling_factor": self.config.max_velocity,
                            "max_acceleration_scaling_factor": self.config.max_acceleration,
                            "frame_id": pose.frame_id,
                            "ik_frame": tool_frame,
                            "position": list(pose.position),
                            "orientation": list(pose.orientation),
                        }],
                        "topp": {
                            "max_vel_factor": self.config.max_velocity,
                            "max_acc_factor": self.config.max_acceleration,
                            "final_phase_ratio": 0.05,
                            "final_phase_oversample": 2.0,
                        },
                    }
                if rrtstar_goal_tolerance:
                    payload["rrtstar_goal_tolerance"] = True
                if use_left_eef_collision_box:
                    payload["use_left_eef_collision_box"] = True
                if self._allow_right_hand_tall_contact(arm):
                    payload["allow_right_hand_tall_contact"] = True
                return self._planner._trajectory.send_payload(payload, timeout=30.0)
            else:
                allow_right_hand_tall_contact = self._allow_right_hand_tall_contact(arm)
                # See the joint-space counterpart above. PTP/LIN need the raw
                # request only while a request-local collision policy is active.
                if not (
                    planner == "rrtstar"
                    or use_left_eef_collision_box
                    or allow_right_hand_tall_contact
                ):
                    return self._planner.plan_cart_trajectory(
                        group=group,
                        planner=planner,
                        frame_id=pose.frame_id,
                        ik_frame=tool_frame,
                        position=pose.position,
                        orientation=pose.orientation,
                        max_vel=self.config.max_velocity,
                        max_acc=self.config.max_acceleration,
                        rrtstar_goal_tolerance=rrtstar_goal_tolerance,
                        use_left_eef_collision_box=use_left_eef_collision_box,
                    )
                payload = {
                        "waypoints": [{
                            "name": f"runtime_{planner}_target",
                            "type": "cart",
                            "group": group,
                            "planner": planner,
                            "max_velocity_scaling_factor": self.config.max_velocity,
                            "max_acceleration_scaling_factor": self.config.max_acceleration,
                            "frame_id": pose.frame_id,
                            "ik_frame": tool_frame,
                            "position": list(pose.position),
                            "orientation": list(pose.orientation),
                        }],
                        "topp": {
                            "max_vel_factor": self.config.max_velocity,
                            "max_acc_factor": self.config.max_acceleration,
                            "final_phase_ratio": 0.05,
                            "final_phase_oversample": 2.0,
                        },
                    }
                if rrtstar_goal_tolerance:
                    payload["rrtstar_goal_tolerance"] = True
                if use_left_eef_collision_box:
                    payload["use_left_eef_collision_box"] = True
                if allow_right_hand_tall_contact:
                    payload["allow_right_hand_tall_contact"] = True
                return self._planner._trajectory.send_payload(payload, timeout=30.0)

        try:
            response = retry_task_creation(
                request,
                label=f"{arm}臂 {planner.upper()} 笛卡尔规划",
            )
        except BaseException as exc:
            self._record_plan(
                status="failure", arm=arm, planner=planner,
                request_type="cartesian", target=target,
                start_joints_deg=recorded_start, error=exc,
            )
            raise
        return self._require_trajectory(
            response, arm=arm, planner=planner, request_type="cartesian",
            target=target, start_joints_deg=recorded_start,
        )

    def plan_cartesian_sequence(
        self, arm: str, poses: list[tuple[str, ObjectPose]]
    ) -> dict[str, Any] | None:
        """Plan an ordered Cartesian waypoint sequence without executing it."""
        if not poses:
            raise ValueError("Cartesian sequence must contain at least one pose")
        group = self.config.left_group if arm == "left" else self.config.right_group
        tool_frame = (
            self.config.left_tool_frame if arm == "left" else self.config.right_tool_frame
        )
        waypoints: list[dict[str, Any]] = []
        for index, (method, pose) in enumerate(poses):
            if method not in {"lin", "ompl", "rrtstar"}:
                raise ValueError(f"Unsupported Cartesian planner: {method!r}")
            self._validate_pose(pose)
            waypoints.append({
                "name": f"temp_waypoint_{index}",
                "type": "cart",
                "group": group,
                "planner": method,
                "max_velocity_scaling_factor": self.config.max_velocity,
                "max_acceleration_scaling_factor": self.config.max_acceleration,
                "frame_id": pose.frame_id,
                "ik_frame": tool_frame,
                "position": list(pose.position),
                "orientation": list(pose.orientation),
            })
        if self._planner is None:
            print(
                f"[planner][dry-run] {arm} Cartesian sequence: "
                f"{[method for method, _ in poses]}"
            )
            return None
        target = {"waypoints": waypoints}
        start_joints_deg = self._recorded_start_joints(arm)
        allow_right_hand_tall_contact = self._allow_right_hand_tall_contact(arm)

        def request() -> dict[str, Any]:
            if not allow_right_hand_tall_contact:
                return self._planner.plan_trajectory(
                    waypoints,
                    max_vel=self.config.max_velocity,
                    max_acc=self.config.max_acceleration,
                )
            payload = {
                "waypoints": waypoints,
                "topp": {
                    "max_vel_factor": self.config.max_velocity,
                    "max_acc_factor": self.config.max_acceleration,
                    "final_phase_ratio": 0.05,
                    "final_phase_oversample": 2.0,
                },
                "allow_right_hand_tall_contact": True,
            }
            return self._planner._trajectory.send_payload(payload, timeout=30.0)

        try:
            response = retry_task_creation(
                request,
                label=f"{arm}臂笛卡尔序列规划",
            )
        except BaseException as exc:
            self._record_plan(
                status="failure", arm=arm, planner="sequence",
                request_type="cartesian_sequence", target=target,
                start_joints_deg=start_joints_deg, error=exc,
            )
            raise
        return self._require_trajectory(
            response, arm=arm, planner="sequence",
            request_type="cartesian_sequence", target=target,
            start_joints_deg=start_joints_deg,
        )

    def plan_joint_then_cartesian_sequence(
        self,
        arm: str,
        first_joints_deg: list[float],
        final_pose: ObjectPose,
        *,
        first_planner: str = "rrtstar",
        final_planner: str = "lin",
    ) -> dict[str, Any] | None:
        """Plan an exact joint pregrasp followed by a Cartesian approach."""
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm: {arm!r}")
        if first_planner not in {"lin", "ompl", "rrtstar"}:
            raise ValueError(f"unsupported joint planner: {first_planner!r}")
        if final_planner != "lin":
            raise ValueError(f"unsupported Cartesian planner: {final_planner!r}")
        if (
            len(first_joints_deg) != 7
            or not all(math.isfinite(value) for value in first_joints_deg)
        ):
            raise ValueError(f"{arm} joint target must contain seven finite values")
        self._validate_pose(final_pose)
        group = self.config.left_group if arm == "left" else self.config.right_group
        tool_frame = (
            self.config.left_tool_frame if arm == "left" else self.config.right_tool_frame
        )
        waypoints = [
            {
                "name": "temp_joint_pregrasp",
                "type": "joint",
                "group": group,
                "planner": first_planner,
                "max_velocity_scaling_factor": self.config.max_velocity,
                "max_acceleration_scaling_factor": self.config.max_acceleration,
                "joint_names": [
                    f"{arm}_arm_joint{index}" for index in range(1, 8)
                ],
                "joint_values": [float(value) for value in first_joints_deg],
            },
            {
                "name": "temp_cartesian_grasp",
                "type": "cart",
                "group": group,
                "planner": final_planner,
                "max_velocity_scaling_factor": self.config.max_velocity,
                "max_acceleration_scaling_factor": self.config.max_acceleration,
                "frame_id": final_pose.frame_id,
                "ik_frame": tool_frame,
                "position": list(final_pose.position),
                "orientation": list(final_pose.orientation),
            },
        ]
        if self._planner is None:
            print(
                f"[planner][dry-run] {arm} mixed sequence: "
                f"{first_planner} joints -> {final_planner} cart"
            )
            return None
        target = {"waypoints": waypoints}
        start_joints_deg = self._recorded_start_joints(arm, first_joints_deg)
        allow_right_hand_tall_contact = self._allow_right_hand_tall_contact(arm)

        def request() -> dict[str, Any]:
            if not allow_right_hand_tall_contact:
                return self._planner.plan_trajectory(
                    waypoints,
                    max_vel=self.config.max_velocity,
                    max_acc=self.config.max_acceleration,
                )
            payload = {
                "waypoints": waypoints,
                "topp": {
                    "max_vel_factor": self.config.max_velocity,
                    "max_acc_factor": self.config.max_acceleration,
                    "final_phase_ratio": 0.05,
                    "final_phase_oversample": 2.0,
                },
                "allow_right_hand_tall_contact": True,
            }
            return self._planner._trajectory.send_payload(payload, timeout=30.0)

        try:
            response = retry_task_creation(
                request,
                label=f"{arm}臂关节-笛卡尔序列规划",
            )
        except BaseException as exc:
            self._record_plan(
                status="failure", arm=arm, planner="sequence",
                request_type="joint_then_cartesian_sequence", target=target,
                start_joints_deg=start_joints_deg,
                error=exc,
            )
            raise
        return self._require_trajectory(
            response, arm=arm, planner="sequence",
            request_type="joint_then_cartesian_sequence", target=target,
            start_joints_deg=start_joints_deg,
        )

    def execute_planned_result(self, response: dict[str, Any]) -> None:
        """Execute a trajectory response previously returned by the planner."""
        data = response.get("data") or {}
        trajectory = data.get("trajectory")
        if not trajectory:
            raise RuntimeError(f"规划结果没有可执行轨迹: {response}")
        self._execute_planned_trajectory(
            arm_id=int(data["arm_id"]),
            trajectory=trajectory,
            sample_frequency=float(data.get("sample_frequency", 10.0)),
        )

    def soft_stop_arm(self, arm: str) -> None:
        """Stop one arm's active direct-SDK trajectory task after operator approval."""
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm: {arm!r}")
        from parm_sdk import Arm

        arm_id = 1 if arm == "left" else 2
        with Arm(host=os.environ.get("PARM_ARM_HOST", "192.168.3.11")) as sdk_arm:
            sdk_arm.cancel_task(arm_id)

    def _execute_planned_trajectory(
        self, *, arm_id: int, trajectory: list[list[float]], sample_frequency: float
    ) -> None:
        if arm_id not in {1, 2}:
            raise ValueError(f"规划结果 arm_id 必须是 1 或 2，实际为 {arm_id}")
        if not trajectory or any(len(point) != 7 for point in trajectory):
            raise ValueError("规划轨迹必须是非空的单臂 7 关节角路点序列")
        if not math.isfinite(sample_frequency) or sample_frequency <= 0.0:
            raise ValueError("规划轨迹 sample_frequency 必须为正数")
        guard = nullcontext()
        if arm_id == 1 and os.environ.get("PARM_LEFT_RIGHT_SERVO_LOCK"):
            from tools.left_motion_right_servo_lock import left_arm_motion_guard

            guard = left_arm_motion_guard("PlannerRuntime left trajectory")
        # The planner output is already a degree-valued joint trajectory.
        # Send it directly to the arm SDK; no browser/backend HTTP bridge is used.
        from parm_sdk import Arm

        arm_key = "left" if arm_id == 1 else "right"
        with guard:
            with Arm(host=os.environ.get("PARM_ARM_HOST", "192.168.3.11")) as sdk_arm:
                state = sdk_arm.get_state() or {}
                task_state = str(
                    (((state.get("state") or {}).get(arm_key) or {}).get("task_state") or "")
                ).strip().lower()
                if task_state == "running":
                    raise TaskBusyError(f"{arm_key} arm 已有 SDK 任务正在运行")
                sdk_arm.follow_trajectory(
                    arm_id=arm_id,
                    trajectory=[[float(value) for value in point] for point in trajectory],
                    sample_frequency=float(sample_frequency),
                )

    @staticmethod
    def _validate_pose(pose: ObjectPose) -> None:
        if pose.frame_id != "body_base":
            raise ValueError("All Cartesian poses must use frame_id='body_base'")
        if len(pose.position) != 3 or len(pose.orientation) != 4:
            raise ValueError("Pose must contain xyz and quaternion xyzw")
