# PARM 双臂 Grasp-and-Place Loop 架构

## 当前模块编号

| 编号 | 模块 | 当前实现状态 |
|---|---|---|
| 0 | Prepare | 目录与说明 |
| 1 | Left Grasp | GraspNet 接入；DP-Grasp 占位 |
| 2 | Left Inspect | 原版与拆分版都保持编号 2 |
| 3 | Left Place | 3-1 正面放、3-2 侧面放、3-3 反面放 |
| 4 | Head Judge | 已实现，并接入模块化总控的反面重抓分支 |
| 5 | Left Pick | 5-1 反面抓取、5-2 交接准备 |
| 6 | Left Idle | 左臂安全 Idle |
| 7 | Right Align | 缓冲区工件摆正 |
| 8 | Right Grasp | 8-1 缓冲区抓、8-2 左右手交接 |
| 9 | Right Place | 右臂孔位放置 |
| 10 | Right Idle | 右臂安全 Idle |

`modules/module_2_left_inspect_apart` 是当前唯一的 Module 2 实现。原
`module_2_left_inspect` 已不在模块化运行链路中；其可复用视觉工具已迁至
`tool/inspection_support/legacy_inspect`，因此旧目录可在人工确认后独立删除。

## Module 2 与 Module 3 的边界

拆分版 Module 2 只负责选择 PC_1/PC_2、胸前相机采集、SAM2 分割、STL
配准、正/侧/反分类、位姿估计和回到 `graspup`。

Module 3 固定映射如下：

```text
FRONT    -> 3-1 正面放
SIDE     -> 3-2 侧面放
REVERSED -> 3-3 反面放
```

Left Pick 和 Right Handoff 原本就不属于
`module_2_left_inspect/standalone_inspect.py`，因此不属于本次 Inspect 拆分。

## 后续模块边界

- Module 4 `head_judge`：使用头部 D435i、SAM2 和受桌面约束的 CAD/ICP，
  输出 `FRONT/BACK/OTHER`；模块化总控在 Module 5 反面重抓前调用。
  4-2 负责缓冲区工件观察（`38.5°/0°`）；4-3 负责反抓后交接观察
  （`15°/35°`）并生成交接所需的 `result.json`。
- Module 5 `left_pick`：根据 Head Judge 结果抓取桌面工件；5-1 完成反面
  抓取，5-2 将左手持件送到交接观察位，消费 Module 4-3 的 D435/ICP 结果后
  再送到双臂交接位。
- Module 8-2 `left_handoff`：接收 5-2 的本轮 ICP `result.json`，右手从
  左手接取工件并撤离到 Module 9 上方准备位。
- 桌面反抓的云台观察、RGB-D 与唯一抓取点预测属于
  `modules/module_4_head_judge/module_4_1_reverse_grasp_prediction/`；
  `modules/module_5_left_pick/module_5_1_reverse_grasp/execute_prediction.py`
  只消费预测并执行左臂反抓。
- 交接实现已拆分到
  `module_5_left_pick/module_5_2_handoff_prepare/main.py` 和
  `module_8_right_grasp/module_8_2_left_handoff/main.py`；
  `tests/loop/reverse_regrasp_right_handoff2.py` 是兼容编排入口。

之前的 `legacy_reverse_regrasp` 和外部 `desktop_grasp` 均为历史来源，不是
总控执行源。当前总控调用 Module 5 入口，避免两份执行实现产生差异。

## 总控入口

模块化入口：

```text
modules/modular_dual_arm_loop.py
```

它固定使用 `module_2_left_inspect_apart`，业务入口全部位于 `modules/`，
并复用旧总控验证过的双进程调度、共享状态、ServoPose 互斥与停止链路。

调度器入口：

```text
tests/loop/simple_dual_arm_loop.py
```

`main.py` 与 `schedulers.py` 是早期线程骨架，不是当前实机总控入口。

## 目录结构

```text
modules/
├── module_0_prepare/
├── module_1_left_grasp/
├── module_2_left_inspect_apart/
├── module_3_left_place/
│   ├── module_3_1_front_place/
│   ├── module_3_2_side_place/
│   └── module_3_3_reversed_place/
├── module_4_head_judge/
│   ├── module_4_2_pose_estimation/
├── module_5_left_pick/
│   ├── module_5_1_reverse_grasp/
│   └── module_5_2_handoff_prepare/
├── module_6_left_idle/
├── module_7_right_align/
├── module_8_right_grasp/
│   ├── module_8_1_buffer_grasp/
│   └── module_8_2_left_handoff/
├── module_9_right_place/
└── module_10_right_idle/
```

## 拆分版阶段文件

```text
GraspNet
  -> Module 2 Inspect JSON
  -> Module 3 Left Place JSON
```

Inspect 结果保存分类、`T_body_base_object`、`T_tool_object` 和诊断信息。
Left Place 结果保存 `place_module=3-1|3-2|3-3`、规划候选与执行结果。
