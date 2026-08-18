# Module 3：左臂固定位置放置

Module 3 是独立的固定位置放置模块。它不判断工件正反面，也不读取 Module 2。
`buffer` 分支只消费 Module 1 写出的已验证 `SINGLE` handoff；`middle` 分支只消费
Module 5.1 在 `graspuptop` 写出的正面重抓 handoff。

```text
Module 1 verified SINGLE handoff
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

## 固定位置配置

每种工件配置有两个固定 `left_eef` 位置：

- `[buffer]`：Module 1 → Buffer；
- `[middle]`：Module 5.1 → 桌中位置。

当前固定目标均位于 `body_base`：

| 工件 | `left_eef` Buffer XYZ（m） | 松爪下降 |
| --- | --- | --- |
| `1` | `[0.457668, 0.258779, -0.320000]` | 无，到目标直接松爪 |
| `2` | `[0.487668, 0.338779, -0.335000]` | 无，到目标直接松爪 |

Module 3 不估计工件相对夹爪的姿态，也不在运输过程中按夹爪状态中断。上游 handoff
已在 `graspuptop`；它保持此时的 `left_eef` 旋转，只把平移改成所选配置段的 XYZ。
Buffer 上是否实际有有效工件，由后续 Module 4 的视觉/ICP 统一判定。

## 输入要求

必须提供 `--handoff-file`（旧参数 `--weight-handoff-file` 仍兼容）和
`--placement-kind buffer|middle`。`buffer` 要求 Module 1 的重量 `SINGLE`；
`middle` 要求 Module 5.1 的 `FRONT`/`graspuptop` handoff。

```text
weight_check_enabled = true
weight_check.classification = SINGLE
```

Module 3 完成固定轨迹、松爪并回 `graspup`；不再把夹爪 `gripping` 状态作为掉件
判据。若工件未留在 Buffer，Module 4 的重复观察耗尽后会清空本次 Buffer 状态并回
Module 1 抓下一件。

## 四条固定主路径的本地轨迹缓存

真实运行时，Module 3 只缓存以下四条固定工位的**主轨迹**：

```text
graspuptop → buffer      buffer → graspuptop
graspuptop → middle      middle → graspuptop
```

缓存目录固定为 `trajactory/workpiece_1/`、`trajactory/workpiece_2/`，每条路径一个
JSON 文件。首次运行仍会请求规划器；只有规划结果已真实执行、且机械臂已验证到达
轨迹终点后才写入。之后相同工件、相同固定位置配置会直接播放文件中的轨迹，不再为
这四条主路径请求规划器。

若工件配置、速度、规划器类型或 `graspuptop` 关节位改变，缓存自动失效并在下一次
真实成功执行后覆盖；若实际关节不在缓存首点容差内，也会为安全起见重新规划。松爪
下探/回升、抓取和异常恢复动作不在缓存范围内。

如需删除两种工件的全部缓存，可运行：

```bash
python3 modules/module_3_left_place/clear_trajactory.py --yes
```

不传 `--yes` 时脚本只显示待清理项，不会删除文件。

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
  回退为在已验证到达的主目标位直接松爪，再返回 `graspup`。结果会写出
  `release_fallback_at_target=true`，供后续调整下降高度。
- 放置成功：松爪后依次返回 `graspup_top` 和 `graspup`。

## 输出与代码结构

默认结果：

```text
output/module_3_left_place/buffer_place_<timestamp>.json
```

真实成功结果包含 `outcome="success"`、`released=true` 和
`returned_to_graspup=true`；纯规划结果包含 `outcome="planned"`。
若下降松爪位不可达，真实成功结果仍为 `released=true`，并额外包含
`release_fallback_at_target=true` 以及失败的下降尝试记录。

- `run.sh`：独立启动入口。
- `standalone_left_place.py`：稳定的独立入口，以及放置阶段的线性编排；总控仍只调用它。
- `placement_support.py`：命令行、handoff 校验、位姿转换、夹爪、关节到位和 Planner 配置。
- `trajectory_cache.py`：四条固定主路径的缓存校验、读取与原子写入；不控制机械臂。
- `placement_config.py`：读取并校验 `[buffer]` 或 `[middle]` 配置。
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
| `load_handoff()` | 按放置类型读取并验证 Module 1 或 Module 5.1 handoff |
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
3. 真实模式依次到达 `graspup` 与 `graspup_top`。
4. 保持 `graspup_top` 旋转，并发规划各 Z 候选。
5. 校验离线轨迹起点；不一致时在线重规划。
6. 执行 Buffer 轨迹、松爪并返回 `graspup_top`、`graspup`。
7. 写入成功、纯规划或恢复结果。

## 安全边界

- `--execute` 与 `--confirm-real-place` 必须同时出现或同时不出现。
- plan-only 只连接 Planner 做可达性计算，不创建 Arm/Gripper 执行动作。
- 工件是否实际留在 Buffer 不由 Module 3 推断，统一交给 Module 4 视觉观察。
- 离线轨迹起点与当前关节超过容差时，轨迹会被丢弃并重新规划。
- 外部门禁只有在 ready/permit 文件同时配置时才启用；默认全自动执行。
- 运行过程中没有回车、`y/n` 或其他终端交互确认。
