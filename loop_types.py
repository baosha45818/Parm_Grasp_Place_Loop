"""定义 Loop 共用的状态、枚举、模块返回值和状态清理函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from time import time
from typing import Any


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    WAIT = "WAIT"
    FAULT = "FAULT"
    ABORTED = "ABORTED"


class Holding(str, Enum):
    EMPTY = "EMPTY"
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"
    UNKNOWN = "UNKNOWN"


class Orientation(str, Enum):
    UPRIGHT = "UPRIGHT"
    REVERSED_SMALL_END = "REVERSED_SMALL_END"
    REVERSED_OTHER = "REVERSED_OTHER"
    UNKNOWN = "UNKNOWN"


class ObservationClass(str, Enum):
    FACE_VISIBLE = "FACE_VISIBLE"
    OBLIQUE = "OBLIQUE"
    UNKNOWN = "UNKNOWN"


class FaceSide(str, Enum):
    FRONT = "FRONT"
    BACK = "BACK"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ModuleResult:
    status: ResultStatus
    previous_module: str
    reason_code: str = ""
    message: str = ""
    attempt_id: int = 0
    started_at: float = field(default_factory=time)
    finished_at: float = field(default_factory=time)


def make_result(
    status: ResultStatus, module: str, started_at: float, message: str
) -> ModuleResult:
    return ModuleResult(
        status=status,
        previous_module=module,
        message=message,
        started_at=started_at,
        finished_at=time(),
    )


@dataclass
class ObjectPose:
    frame_id: str
    position: list[float]
    orientation: list[float]


@dataclass
class LeftArmState:
    current_module: str = "0"
    holding: Holding = Holding.EMPTY
    object_orientation: Orientation = Orientation.UNKNOWN
    observation_class: ObservationClass = ObservationClass.UNKNOWN
    face_side: FaceSide = FaceSide.UNKNOWN
    grasp_region: str = "UNKNOWN"
    object_pose: ObjectPose | None = None
    pose_confidence: float = 0.0
    orientation_confidence: float = 0.0
    observation_pose_used: str = ""
    pose_quality: dict[str, Any] = field(default_factory=dict)
    grasp_stable: bool = False
    grasp_strategy: str = "GRASPNET"
    motion_state: str = "IDLE"
    last_error: str = ""


@dataclass
class RightArmState:
    current_module: str = "0"
    holding: Holding = Holding.EMPTY
    object_pose: ObjectPose | None = None
    target_object_pose: ObjectPose | None = None
    pose_confidence: float = 0.0
    grasp_stable: bool = False
    grasp_source: str = "NONE"
    motion_state: str = "IDLE"
    last_error: str = ""


@dataclass
class BufferState:
    capacity: int = 1
    full: bool = False
    needs_alignment: bool = False
    operator: str = "NONE"
    lock: Lock = field(default_factory=Lock, repr=False)


@dataclass
class SystemState:
    ready: bool = False
    run_mode: str = "STARTING"
    shutdown_requested: bool = False
    last_fault: str = ""


def clear_left_object(state: LeftArmState) -> None:
    state.holding = Holding.EMPTY
    state.object_orientation = Orientation.UNKNOWN
    state.observation_class = ObservationClass.UNKNOWN
    state.face_side = FaceSide.UNKNOWN
    state.grasp_region = "UNKNOWN"
    state.object_pose = None
    state.pose_confidence = 0.0
    state.orientation_confidence = 0.0
    state.observation_pose_used = ""
    state.pose_quality.clear()
    state.grasp_stable = False


def clear_right_object(state: RightArmState) -> None:
    state.holding = Holding.EMPTY
    state.object_pose = None
    state.target_object_pose = None
    state.pose_confidence = 0.0
    state.grasp_stable = False
    state.grasp_source = "NONE"


PoseMap = dict[str, ObjectPose | None]
ModelMap = dict[str, Any]
