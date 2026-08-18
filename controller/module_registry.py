"""把现有模块命令集中声明；main.py 不出现模块路径和业务参数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    resources: tuple[str, ...]
    command: Callable[[dict[str, str], bool], list[str]]
    result_file_is_contract: bool = True


def _execute_args(enabled: bool, *args: str) -> list[str]:
    return list(args) if enabled else []


def registry() -> dict[str, ModuleSpec]:
    def action(action_name: str):
        return lambda c, live: [
            str(PYTHON), str(ROOT / "controller/robot_actions.py"), "--action", action_name,
            "--result-file", c["result"],
            "--planner-host", c["planner_host"], "--arm-host", c["arm_host"],
            "--gripper-host", c["gripper_host"], "--force-host", c["force_host"],
            "--head-host", c["head_host"],
            "--d405-serial", c["d405_serial"],
            "--speed", c["prepare_speed"],
        ]
    def m1(c: dict[str, str], live: bool) -> list[str]:
        return [str(ROOT / "modules/module_1_left_grasp/run.sh"), "--mode", c["mode"], "--weight-handoff-file", c["result"]] + _execute_args(live, "--execute", "--confirm-real-grasp")
    def m3(kind: str):
        return lambda c, live: [str(ROOT / "modules/module_3_left_place/run.sh"), "--workpiece", c["workpiece"], "--handoff-file", c["handoff"], "--placement-kind", kind, "--result-file", c["result"], "--planner-host", c["planner_host"], "--arm-host", c["arm_host"], "--gripper-host", c["gripper_host"]] + _execute_args(live, "--execute", "--confirm-real-place")
    def m4(c: dict[str, str], live: bool) -> list[str]:
        # 一次 Module 4 调用只采集一次；重复观察完全由控制器统一计数，
        # 避免「模块内 4 次 + 总控外再重启」形成难以解释的双层重试。
        return [str(PYTHON), str(ROOT / "modules/module_4_head_judge/module_4_1_buffer_state_estimation/check_buffer_face.py"), "--workpiece", c["workpiece"], "--mode", c["mode"], "--result-file", c["result"], "--max-uncertain-retries", "0"]
    def m5front(c: dict[str, str], live: bool) -> list[str]:
        return [str(PYTHON), str(ROOT / "modules/module_5_left_pick/module_5_1_front_pick_to_graspuptop/main.py"), "--buffer-check-result", c["buffer_result"], "--result-file", c["result"], "--planner-host", c["planner_host"], "--arm-host", c["arm_host"], "--gripper-host", c["gripper_host"]] + _execute_args(live, "--execute")
    def m5reverse(c: dict[str, str], live: bool) -> list[str]:
        return [str(PYTHON), str(ROOT / "controller/reverse_pipeline.py"), "--buffer-check-result", c["buffer_result"], "--workpiece", c["workpiece"], "--result-file", c["result"], "--planner-host", c["planner_host"], "--arm-host", c["arm_host"], "--gripper-host", c["gripper_host"]] + _execute_args(live, "--live")
    def m8handoff(c: dict[str, str], live: bool) -> list[str]:
        return [str(PYTHON), str(ROOT / "modules/module_8_right_grasp/module_8_2_left_handoff/main.py"), "--result", c["handoff_result"], "--workpiece", c["workpiece"], "--right-place-ready-file", c["handoff_ready"], "--planner-host", c["planner_host"], "--arm-host", c["arm_host"], "--gripper-host", c["gripper_host"]]
    def m9(c: dict[str, str], live: bool) -> list[str]:
        return [str(PYTHON), str(ROOT / "modules/module_9_right_place/place_hole_wiggle.py"), "--workpiece", c["workpiece"], "--arm-host", c["arm_host"]] + _execute_args(live, "--live", "--confirm-live")
    def m10(c: dict[str, str], live: bool) -> list[str]:
        return [str(PYTHON), str(ROOT / "modules/module_10_right_idle/main.py"), "--planner-host", c["planner_host"], "--arm-host", c["arm_host"], "--gripper-host", c["gripper_host"], "--execute"]
    def right_front(c: dict[str, str], live: bool) -> list[str]:
        return [str(PYTHON), str(ROOT / "modules/module_8_right_grasp/module_8_1_buffer_grasp/workpiece_pipeline.py"), "--workpiece", c["workpiece"], "--result-file", c["result"], "--planner-host", c["planner_host"]] + _execute_args(live, "--execute")
    return {
        "controller_prepare": ModuleSpec("controller_prepare", ("LEFT_ARM", "RIGHT_ARM", "HEAD_D435"), action("prepare")),
        "controller_observation_pose": ModuleSpec("controller_observation_pose", ("LEFT_ARM", "HEAD_D435"), action("observation")),
        "module_1_left_grasp": ModuleSpec("module_1_left_grasp", ("LEFT_ARM", "LEFT_D405"), m1),
        "module_3_buffer": ModuleSpec("module_3_buffer", ("LEFT_ARM", "BUFFER"), m3("buffer")),
        "module_4_buffer_state": ModuleSpec("module_4_buffer_state", ("BUFFER", "HEAD_D435"), m4),
        "module_5_front": ModuleSpec("module_5_front", ("LEFT_ARM", "BUFFER"), m5front),
        "module_5_reverse": ModuleSpec("module_5_reverse", ("LEFT_ARM", "BUFFER", "HANDOFF_ZONE"), m5reverse),
        "module_8_handoff": ModuleSpec("module_8_handoff", ("LEFT_ARM", "RIGHT_ARM", "HANDOFF_ZONE"), m8handoff, result_file_is_contract=False),
        "module_9_right_place": ModuleSpec("module_9_right_place", ("RIGHT_ARM",), m9, result_file_is_contract=False),
        "module_10_right_idle": ModuleSpec("module_10_right_idle", ("RIGHT_ARM",), m10, result_file_is_contract=False),
        "module_3_middle": ModuleSpec("module_3_middle", ("LEFT_ARM", "BUFFER"), m3("middle")),
        "module_8_right_front": ModuleSpec("module_8_right_front", ("RIGHT_ARM", "HEAD_D435"), right_front),
    }
