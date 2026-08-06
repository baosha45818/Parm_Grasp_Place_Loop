# PARM Grasp-and-Place Loop

## 模块化总控

新的模块化总控只调用 `modules/` 下的业务入口，固定使用拆分版 Inspect：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
.venv/bin/python \
  modules/modular_dual_arm_loop.py \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1
```

它复用已经验证的双进程调度、共享状态、右臂 ServoPose 互斥和
`Ctrl+C -> cancel_task()` 链路，但将 Grasp、Inspect/Place、Head
Judge/Pick、Align/Grasp、Handoff、Right Place 和 Right Idle 的调度入口
全部切换到 `modules/`。Module 2 固定为
`module_2_left_inspect_apart/run_inspect_then_place.py`。

## 项目内依赖

模块化全流程不再引用 `qiuguolin`、`zhuyiming` 或 `data_collector`：

- `.venv/`：Loop 主流程、相机、SDK 和 SAM2 Python 环境。
- `third_party/graspnet/`：Version7 GraspNet、checkpoint、点云算子、标定与 CUDA 环境。
- `modules/module_7_right_align/vendor/push_t/`：Module 7 摆正所需实现与示教位姿。
- `modules/module_8_right_grasp/module_8_1_buffer_grasp/vendor/push_t/`：Module 8-1 缓冲区抓取所需实现与示教位姿。
- `modules/module_9_right_place/vendor/push_t/`：Module 9 ServoPose 放置所需关节/姿态工具。
- `third_party/push_T/`：历史备份；模块化全流程不再调用。
- `third_party/anchor_urdf/`：Version9 名义 TCP 实验所需 URDF。
- `calibration/`：反抓交接用的 `body_base` 对齐外参。
- `runtime/parm_sdk/`：本机硬件 driver 与 Planner SDK wheel。
- `planner/plucky_hjt_parm/`：规划器 Docker Compose 源码。

规划器 Docker、夹爪/力传感器/云台/踏板 driver 和 SAM2 服务仍是运行时服务，
但它们的项目文件均已在本目录内；硬件本体与 Docker 镜像当然不能复制进代码目录。

旧总控保持不变，兼容入口仍为：

```text
tests/loop/simple_dual_arm_loop.py
```

## 工件计时

模块化总控会在 `output/time/<本次运行ID>/` 下为每个左臂循环创建一个
`*.json`。文件记录 Module 1--10 的每次尝试、每个模块的累计耗时，以及从
Module 1 开始到 Module 10 松爪完成的总耗时。未抓到、点云不足或异常中止的
循环也会保留对应的结束原因和已执行模块耗时。

## 三种左臂流程

原版人工 SAM2、一体化 Inspect 与放置：

```bash
cd /home/dainanjun/Public
/home/dainanjun/Public/data_collector/.venv/bin/python \
  /home/dainanjun/Public/Parm_Grasp_Place_Loop/tests/loop/simple_dual_arm_loop.py \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --left-flow original
```

SAM2 C/S 自动提示版：

```bash
cd /home/dainanjun/Public
/home/dainanjun/Public/data_collector/.venv/bin/python \
  /home/dainanjun/Public/Parm_Grasp_Place_Loop/tests/loop/simple_dual_arm_loop.py \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --left-flow sam2-cs-auto
```

拆分版 Module 2 Inspect 与 Module 3 Left Place：

```bash
cd /home/dainanjun/Public
/home/dainanjun/Public/data_collector/.venv/bin/python \
  /home/dainanjun/Public/Parm_Grasp_Place_Loop/tests/loop/simple_dual_arm_loop.py \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --left-flow apart
```

`apart` 是唯一的模块化 Inspect 入口，要求 SAM2 服务已经监听
`unix:///tmp/sam2-service.sock`。

拆分版 Module 3 已恢复原版候选参数、侧面历史优先级、固定离线起点、
边运动边规划、10 秒超时恢复、重量复检、下降失败开夹爪和 graspup
实际到位验证。

## 拆分范围

本次只拆分原版
旧 Inspect 中的 Inspect 与 Left Place 逻辑：

```text
module_2_left_inspect_apart
  -> module_3_left_place
```

原版 `module_2_left_inspect` 已不在模块化运行链路中。Left Pick 和 Right
Handoff 不属于该文件：

- Left Pick 的反面抓取实现已迁入
  `modules/module_5_left_pick/module_5_1_reverse_grasp/main.py`。
- 反抓后的左侧交接准备位于
  `modules/module_5_left_pick/module_5_2_handoff_prepare/main.py`。
- Right Handoff 实现位于
  `modules/module_8_right_grasp/module_8_2_left_handoff/main.py`。
- `tests/loop/reverse_regrasp_right_handoff2.py` 保留为兼容入口，依次调用
  上述两个模块。

## 模块编号

```text
0 Prepare
1 Left Grasp
2 Left Inspect（原版与拆分版编号都为 2）
3 Left Place：3-1 正面、3-2 侧面、3-3 反面
4 Head Judge
5 Left Pick
6 Left Idle
7 Right Align
8 Right Grasp：8-1 Buffer、8-2 Handoff
9 Right Place
10 Right Idle
```

Module 4 Head Judge 已接入新的模块化总控反面重抓分支；旧总控行为不变。
入口和运行方式见 `modules/module_4_head_judge/README.md`。

详细边界见 `ARCHITECTURE.md`，调度关系见 `SCHEDULER_RULES.md`。

`main.py` 和 `schedulers.py` 是早期线程骨架，不是当前实机总控入口。
