# Module 3：左臂 Buffer 放置

Module 3 是独立的固定 Buffer 放置模块。它不判断工件正反面，也不读取
Module 2；唯一业务输入是 Module 1 写出的已验证 `SINGLE` handoff。

```text
Module 1 verified SINGLE handoff
→ 检查左夹爪仍在持件
→ 到达 graspup
→ 到达 graspup_top
→ 保持 graspup_top 的 left_eef 姿态
→ 规划固定 Buffer TCP
→ 移动到 Buffer
→ 打开左夹爪
→ 返回 graspup_top
→ 返回 graspup
→ 写出放置结果
```

## Buffer 配置

每种工件只保留一个 `[buffer]` 配置：

- `workpiece_1_placement_parameters.toml`
- `workpiece_2_placement_parameters.toml`

当前固定目标均位于 `body_base`：

| 工件 | `left_eef` Buffer XYZ（m） | 松爪下降 |
| --- | --- | --- |
| `1` | `[0.457668, 0.258779, -0.320000]` | 无，到目标直接松爪 |
| `2` | `[0.487668, 0.338779, -0.335000]` | 无，到目标直接松爪 |

Module 3 不估计工件相对夹爪的姿态。真实执行会先到达 `graspup_top`，读取
此时的 `left_eef` 旋转，只把平移改成配置中的 Buffer XYZ。

## 输入要求

必须提供 `--weight-handoff-file`。程序要求：

```text
weight_check_enabled = true
weight_check.classification = SINGLE
```

Module 3 在实际持件运动前还会重复检查左夹爪是否稳定报告 `gripping`。如果
检测到掉件，会禁止后续持件轨迹，执行回 `graspup` 和松爪恢复并记录结果。

## 运行

### 无动作规划

默认是纯规划，不连接 Arm/Gripper 执行动作：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json
```

纯规划会读取机械臂当前 `left_eef` 旋转作为目标旋转；为了让结果与真实执行
完全一致，建议在机械臂已经位于 `graspup_top` 时运行。

### 全自动真实放置

真实动作必须同时提供 `--execute` 和 `--confirm-real-place`。授权后完整放置与
回程连续自动执行，不进行终端交互：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json \
  --execute \
  --confirm-real-place
```

`xiangxin` 环境使用：

```text
--arm-host 192.168.3.11 --gripper-host 192.168.3.11
```

可同时设置 `--placement-ready-file` 与 `--placement-permit-file`，让外部控制器
在轨迹规划完成后放行；未设置时不会等待外部门禁。

## Module 1 + Module 3 串联

以下命令会真实控制机械臂和夹爪，并在 Module 1 成功写出 handoff 后才启动
Module 3：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_1_left_grasp/run.sh \
  --mode gzhu \
  --execute \
  --confirm-real-grasp \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json \
&& \
modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json \
  --execute \
  --confirm-real-place
```

## 失败处理

- Buffer 候选不可达：真实模式回 `graspup`，下降到安全丢弃位，松爪并返回。
- 离线轨迹起点与实际关节不一致：丢弃离线轨迹并在线重规划。
- 松爪下降候选失败：记录每次失败；如果配置要求下降且所有候选都失败，
  保持夹持并返回 `graspup`，写出
  `release_motion_failed_object_retained`，不会在未经验证的高度松爪。
- 放置成功：松爪后依次返回 `graspup_top` 和 `graspup`。

## 输出与代码结构

默认结果：

```text
output/module_3_left_place/buffer_place_<timestamp>.json
```

真实成功结果包含 `outcome="success"`、`released=true` 和
`returned_to_graspup=true`；纯规划结果包含 `outcome="planned"`。
松爪下降全部失败时返回码为非零，结果包含 `released=false` 和
`object_retained_in_left_gripper=true`。

- `run.sh`：独立启动入口。
- `standalone_left_place.py`：handoff、规划、执行、持件检查和失败恢复。
- `placement_config.py`：读取并校验 `[buffer]` 配置。
- `placement_parameters.py`：`graspup`、`graspup_top` 与恢复运动常量。
- `workpiece_*_placement_parameters.toml`：各工件固定 Buffer TCP。

## 重构后的函数边界

Module 3 已删除旧正反面分支和无关外壳。业务步骤均通过命名函数表达，
`main()` 保留从 handoff 到放置完成的线性编排，所有函数及类均带中文用途说明。

主要顶层函数：

| 函数 | 职责 |
| --- | --- |
| `parse_args()` | 声明独立运行参数 |
| `validate_runtime_args()` | 集中校验执行授权、门禁文件和规划参数 |
| `load_verified_handoff()` | 读取并验证 Module 1 的 `SINGLE` handoff |
| `build_planner_config()` | 构造本模块唯一的 PlannerRuntime 配置 |
| `planner_matrix()` / `matrix_pose()` | 在 Planner 响应、齐次矩阵与目标位姿间转换 |
| `read_left_joints_deg()` | 真实执行时读取左臂当前关节角 |
| `wait_for_left_joint_target()` | 等待左臂到达指定关节目标 |
| `wait_for_left_graspup()` | 等待返回固定 `graspup` |
| `validate_z_candidates()` | 校验 Buffer 目标和松爪下探候选 |
| `open_left_gripper()` | 统一执行松爪并返回最终开度 |
| `publish_ready()` / `wait_permit()` | 可选的文件式外部规划门禁 |

`main()` 内部按以下阶段组织：

1. 参数和 handoff 校验。
2. 建立 PlannerRuntime，检查固定 Buffer 目标。
3. 真实模式复检持件，依次到达 `graspup` 与 `graspup_top`。
4. 保持 `graspup_top` 旋转，并发规划各 Z 候选。
5. 动作前复检持件和离线轨迹起点；不一致时在线重规划。
6. 执行 Buffer 轨迹、松爪并返回 `graspup_top`、`graspup`。
7. 写入成功、纯规划、掉件或恢复结果。

## 安全边界

- `--execute` 与 `--confirm-real-place` 必须同时出现或同时不出现。
- plan-only 只连接 Planner 做可达性计算，不创建 Arm/Gripper 执行动作。
- 真实模式在每个关键持件运动前复检左夹爪 `motion_state=gripping`。
- 离线轨迹起点与当前关节超过容差时，轨迹会被丢弃并重新规划。
- 外部门禁只有在 ready/permit 文件同时配置时才启用；默认全自动执行。
- 运行过程中没有回车、`y/n` 或其他终端交互确认。
