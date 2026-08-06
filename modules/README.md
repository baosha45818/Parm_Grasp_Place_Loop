# 模块目录

模块化大 Loop 入口：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
.venv/bin/python \
  modules/modular_dual_arm_loop.py \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1  
```

该入口固定使用 `module_2_left_inspect_apart`，并保持旧总控已验证的双臂
并发、状态 `0/1/2`、ServoPose 互斥和停止链路。

当前编号为 0 到 10：

```text
0 Prepare
1 Left Grasp
2 Left Inspect
3 Left Place
4 Head Judge
5 Left Pick
6 Left Idle
7 Right Align
8 Right Grasp
9 Right Place
10 Right Idle
```

`module_2_left_inspect_apart` 是当前唯一的 Module 2。旧实现的公共视觉辅助
代码位于 `tool/inspection_support/legacy_inspect`，模块化运行不再引用旧目录。

Module 4 Head Judge 已实现独立只读视觉判断，入口为
`module_4_head_judge/run.py`；模块化总控在反面重抓前通过
`module_5_left_pick/run_head_judge_then_pick.py` 调用它。

Module 4-2 统一负责右臂缓冲区观察位的云台控制与视觉估计（`38.5°/0°`）；
Module 4-3 统一负责左手反抓后的交接观察位云台控制与 D435/SAM2/ICP
（`15°/35°`）。Module 5-2 不再控制云台。

Module 3 子模块固定为：

- `3-1` 正面放。
- `3-2` 侧面放。
- `3-3` 反面放。

Module 5 Left Pick、Module 8-2 Right Handoff、Module 9 Right Place 和
Module 10 Right Idle 均有模块化总控入口。
