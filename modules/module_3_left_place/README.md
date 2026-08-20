# Module 3：左臂固定工位放置

`module_3_left_place` 的唯一职责是：左臂已经在 `graspuptop` 持有工件时，将工件放到一个**预先标定的固定位置**，松开夹爪，并让左臂回到 `graspup` 安全位。

它有两个参数化分支，但它们复用同一套实现：

| 分支 | `--placement-kind` | 合法上游 | 固定放置位置 | 完成后的含义 |
| --- | --- | --- | --- | --- |
| Buffer 放置 | `buffer`（默认） | Module 1 的重量确认 `SINGLE` handoff | Buffer 工位 | Buffer 内新增一件，后续由 Module 4 观察 |
| Middle 放置 | `middle` | Module 5.1 的正面重抓 handoff | Middle 工位 | Middle 内新增一件，后续由右臂小总控消费 |

它不负责抓取、不做视觉、不会判断工件正反面，也不根据夹爪状态判断运输途中是否掉件。Module 3 只对“已验证的上游 handoff”执行固定工位放置；工件是否最终仍留在 Buffer，由后续 Module 4 的实际视觉观察判断。

---

## 1. 模块边界与物理流程

### 1.1 两条业务路径

```text
Buffer 路径
Module 1 已抓起且重量确认 SINGLE
  → handoff_pose=graspuptop
  → Module 3(buffer)
  → 固定 Buffer 工位松爪
  → 左臂回 graspup
  → Module 4 观察 Buffer

Middle 路径
Module 4 判定为 FRONT
  → Module 5.1 正面重抓并到 graspuptop
  → Module 3(middle)
  → 固定 Middle 工位松爪
  → 左臂回 graspup
  → 右臂观察、调整、抓取 Middle 工件
```

两条路径的运动骨架相同：

```text
上游 handoff
  → 到达 graspup_top（离线放置轨迹的统一起点）
  → 读取此刻 left_eef 的真实旋转
  → 规划 / 复用 graspup_top 到目标工位的主轨迹
  → 到达目标工位并确认关节到位
  → 可选 LIN 下探
  → 打开左夹爪
  → 回到目标主位（仅实际下探过时）
  → 回 graspup_top
  → 回 graspup
  → 原子写出结果 JSON
```

`graspuptop` 和 `graspup_top` 在代码与现场说明中指同一个“放置规划起点”；代码中的固定关节常量名为 `GRASPUP_ABOVE_JOINTS_DEG`。最终安全等待位为 `GRASPUP_JOINTS_DEG`，也就是 `graspup`。

### 1.2 Module 3 明确不做的事情

- 不调用 D405/D435、SAM、ICP 或任何视觉判定；
- 不使用 Module 4 输出的工件 TF 来计算放置位置；放置位置是每种工件的固定 `left_eef` 目标；
- 不调整夹爪相对工件的抓取姿态；目标旋转直接继承 `graspuptop` 时的真实末端旋转；
- 不在运输过程中用“是否仍 gripping”作掉件判断；
- 不缓存抓取、松爪下探、回升或异常丢弃动作，只缓存四条固定主路径；
- 不管理 Buffer/Middle 的全局状态与左右臂并发。这些由 `controller/` 维护。

---

## 2. 输入契约

### 2.1 命令行必需输入

必须提供一个上游 handoff：

```text
--handoff-file <上游结果.json>
```

旧参数 `--weight-handoff-file` 保留为同义兼容参数；二者不能同时使用。`--placement-kind` 决定所走的分支，取值只能是 `buffer` 或 `middle`。

真实执行必须同时提供以下两个开关；少一个会在解析参数时直接拒绝：

```text
--execute --confirm-real-place
```

不带这两个参数时为 plan-only 模式：模块只使用 Planner 做可达性规划和 TF 查询，绝不会创建 Arm / Gripper 指令或松爪。

### 2.2 Buffer handoff 校验

`placement_kind=buffer` 只接收 Module 1 的、已确认单件的 handoff。`placement_support.load_handoff()` 会严格检查：

```json
{
  "weight_check_enabled": true,
  "weight_check": {
    "classification": "SINGLE"
  },
  "handoff_pose": "graspuptop"
}
```

任一字段不满足都不会开始运动。这避免把空抓、双件、未做重量检查的抓取结果送往 Buffer。

### 2.3 Middle handoff 校验

`placement_kind=middle` 不接受 Module 1 handoff，只接收 Module 5.1 的正面重抓结果：

```json
{
  "module": "module_5_1_front_pick_to_graspuptop",
  "outcome": "front_regrasp_handoff",
  "handoff_pose": "graspuptop",
  "classification": "FRONT"
}
```

这条约束保证只有被 Module 4 判为正面、且已经由 Module 5.1 成功重新抓取的工件才能放到 Middle。反面工件不经过此模块的 Middle 分支。

### 2.4 运行环境输入

| 输入 | 默认值 / 来源 | 用途 |
| --- | --- | --- |
| `--workpiece 1|2` | `1` | 选择对应 TOML 的 Buffer/Middle 固定坐标与轨迹缓存目录 |
| `--planner-host` | `127.0.0.1` | 规划器服务地址 |
| `--arm-host` | `PARM_ARM_HOST`，否则 `192.168.3.11` | 读取左臂实际关节、等待到位 |
| `--gripper-host` | `PARM_GRIPPER_HOST`，否则 `192.168.3.11` | 检查并打开左夹爪 |
| `--execute-url` | Planner 的 middle waypoint execute URL | 发送规划轨迹 |
| `--result-file` | `output/module_3_left_place/buffer_place_<时间>.json` | 模块最终结果 |

注意：默认结果文件名前缀历史上固定为 `buffer_place_`，即使实际是 `middle` 分支也不会自动改名；`stage`、`placement_kind` 等 JSON 字段仍正确。独立运行时建议显式指定 `--result-file`，以免混淆。

---

## 3. 固定工位配置与坐标含义

### 3.1 配置文件

| 工件 | 配置文件 |
| --- | --- |
| 工件 1 | `workpiece_1_placement_parameters.toml` |
| 工件 2 | `workpiece_2_placement_parameters.toml` |

每个文件必须且只能有 `[buffer]`、`[middle]` 两个区段；每个区段必须含有下列三个键：

```toml
[buffer] # 或 [middle]
target_left_eef_xyz_m = [x, y, z]
target_z_candidates_m = [z1, z2, ...]
release_z_candidates_m = [rz1, rz2, ...] # 可以为空 []
```

`placement_config.py` 不直接使用 TOML 库，而是逐行解析安全字面量，并额外拒绝：未知 section、重复 key、缺失 key、非数值、非有限数值以及错误的 XYZ 长度。这是为了让现场配置错误在机械臂动作前就显式报错。

### 3.2 当前部署坐标

坐标单位均为米，参考系均为 `body_base`。最终目标的 X/Y 始终来自 `target_left_eef_xyz_m`；实际放置 Z 使用 `target_z_candidates_m` 中第一个规划成功的候选。

| 工件 | 区域 | XY 来源与目标 Z 候选 | 松爪下探 Z 候选 |
| --- | --- | --- | --- |
| 1 | Buffer | `XYZ=[0.487668, 0.288779, -0.320000]`；`[-0.320000]` | `[]`，主目标直接松爪 |
| 1 | Middle | `XYZ=[0.537668, -0.028779, -0.320000]`；`[-0.340000]` | `[]`，主目标直接松爪 |
| 2 | Buffer | `XYZ=[0.487668, 0.338779, -0.335000]`；`[-0.335000]` | `[]`，主目标直接松爪 |
| 2 | Middle | `XYZ=[0.507668, 0.008779, -0.320000]`；`[-0.340000]` | `[-0.350000]`，先下探后松爪 |

`target_z_candidates_m` 必须是严格递增的非空列表；`release_z_candidates_m` 可以是空列表，但如果非空，也必须严格递增。运行时会再过滤掉高于主目标的 release Z，防止“下探”实际向上运动。

### 3.3 姿态如何获得

Module 3 不把工件的 ICP 位姿转为放置姿态。代码在左臂到达 `graspuptop` 后查询：

```text
body_base_T_left_eef
```

取其旋转矩阵 `R`，然后构造每一个候选目标：

```text
T_target = [ R  [target_x, target_y, candidate_z]^T ]
           [ 0                 1                 ]
```

也就是说，工件相对夹爪的姿态由上游抓取自然保持，Module 3 只换末端平移。代码还会拒绝 `left_eef` 的本地 `+Z` 在 `body_base` 中没有指向下方的候选（`target_tool[2,2] >= 0`），避免以错误的工具朝向去规划放置。

---

## 4. 代码级执行过程

主入口是 `standalone_left_place.py:main()`。它可以按下面十个阶段理解。

### 阶段 1：解释参数与运行模式

`placement_support.parse_args()` 声明全部 CLI 参数，同时强制 `--execute` 与 `--confirm-real-place` 成对出现。随后 `validate_runtime_args()` 校验：

- 预备速度、运动速度在 `(0, 1]` 内；
- 缓存首点容差 `--offline-start-tolerance-deg` 大于零；
- 并发规划数 `--placement-planning-inflight` 在 `[1, 16]`；
- 如果使用文件门禁，ready 文件与 permit 文件必须成对提供；
- `--graspup-lift-m` 当前只能是固定的 `0.10 m`；
- 可选 `--pre-place-middle-joints-deg` 必须为七个有限关节角。

入口还会检查 Python 解释器。若当前并非项目根目录的 `.venv/bin/python`，会用该解释器 `exec` 重新启动自己，避免错误使用系统 Python 或别的虚拟环境。

### 阶段 2：验证 handoff，选择配置与结果身份

`load_handoff()` 先按第 2 节的契约校验上游 JSON，再由 `load_workpiece_parameters(workpiece, placement_kind)` 读当前 TOML 区段。

此时生成的逻辑身份是：

| `placement_kind` | 输出 `stage` | `place_module` | `classification` | `placement_branch` |
| --- | --- | --- | --- | --- |
| `buffer` | `buffer_place` | `3-buffer-place` | `BUFFER` | `BUFFER_DIRECT` |
| `middle` | `middle_place` | `3-middle-place` | `MIDDLE` | `MIDDLE_DIRECT` |

### 阶段 3：运动到统一起点，并行启动超时恢复监视

真实模式下，独立的运动线程使用 `PlannerRuntime.move_left_joints()` 将左臂送到：

```python
GRASPUP_ABOVE_JOINTS_DEG = [
  -1.3115, -60.0925, 4.4041, 88.8464,
  -15.3849, 73.1908, 12.9378,
]
```

每次关节运动后，都由 `wait_for_left_joint_target()` 用 Arm SDK 轮询实际左臂七关节，90 秒内要求最大环绕角误差不超过 `0.5°`。角误差按 `[-180°, 180°)` 回绕计算，避免关节角跨 ±180° 时产生假大误差。

若显式传入 `--pre-place-middle-joints-deg`，运动线程会先走该七关节中间位，再走 `graspuptop`；默认不启用。

运动线程启动的同时，另一个恢复线程开始等待：只有左臂到达 `graspuptop` 后，才给“找到可行主放置轨迹”最多 `6.0 s`。这是常量 `GRASPUP_ABOVE_PLANNING_TIMEOUT_S`。

### 阶段 4：读取真实工具姿态

主线程通过 `PlannerRuntime.lookup_transform("body_base", "left_eef")` 取得当前末端 TF，提取旋转矩阵作为所有候选放置位的旋转。结果记录在 `planning_scheduler.orientation_source` 和 `tool_rpy_deg`，便于排查“固定坐标正确但工具朝向错误”的问题。

### 阶段 5：离线规划多个 Z 候选

`plan_candidate()` 为每个候选建立独立 Planner 连接，并调用：

```python
plan_cartesian_pose(
    "left", target_pose,
    planner="rrtstar" 或 "ompl",
    start_joints_deg=GRASPUP_ABOVE_JOINTS_DEG,
    rrtstar_goal_tolerance=(planner == "rrtstar"),
    use_left_eef_collision_box=False,
)
```

规划起点固定为 `graspuptop`，而不是提交请求瞬间的随机实时关节状态。`--post-graspup-planner rrtstar` 是默认值；若选择 `rrtconnect`，实际传给 Planner 的名称为 `ompl`。

候选按配置顺序投递，但使用滑动窗口并发：最多 `--placement-planning-inflight`（默认 8）条请求在飞。首个**完成且成功**的候选被选择，并设置 `plan_found`；已提交的其余请求会自然收尾，但不会替换已选目标。每一次提交、失败、成功都写入 `planning_attempts`，也逐行追加到 `placement_planning_attempts.jsonl`。

### 阶段 6：缓存命中和实际起点二次校验

首先尝试读取 `graspuptop_to_buffer` 或 `graspuptop_to_middle` 缓存。命中时直接获得原先的 Planner 轨迹和 `selected_candidate`；缺少目标元数据的旧缓存不会被盲目播放，而是重新规划。

真实执行前，无论轨迹来自缓存还是本次离线规划，都会读取左臂实际关节并与轨迹第一点比较：

```text
max(abs(wrapped(actual_joint - trajectory_first_joint)))
    <= --offline-start-tolerance-deg  # 默认 1.0°
```

通过则执行这条轨迹；不通过则从当前状态向同一笛卡尔目标在线重规划。这样既能复用固定路径，也不会从错误起点播放离线轨迹。

### 阶段 7：执行主放置轨迹

可选门禁的行为如下：如果同时传入 `--placement-ready-file` 与 `--placement-permit-file`，模块先原子写出 ready JSON，再轮询 permit 文件；文件内容必须是 `allow`、`y`、`yes` 或 `1` 才继续。未配置时没有任何等待或终端交互。

之后调用 `PlannerRuntime.execute_planned_result()` 执行主轨迹，并以轨迹最后一个七关节角为目标做实际到位确认。只有“真实执行成功且实际关节已到位”后，本次新规划的轨迹才允许写入缓存。

### 阶段 8：松爪位策略

主放置轨迹到达的 `target_tool` 一定是已验证位置。松爪下探只是可选附加动作：

- `release_z_candidates_m=[]`：不发送 LIN 下探，直接在主目标松爪；
- 有下探候选：按配置顺序，对每个候选只改变末端 Z，并用 `planner="lin"` 尝试；第一个成功候选成为松爪位；
- 全部 LIN 下探失败：**不把工件带回去**。模块记录失败并在已验证到达的主目标直接松爪，结果写 `release_fallback_at_target=true`。

`open_left_gripper()` 会先连接 Gripper SDK（左夹爪 hand 固定为 `1`），检查 `module_0_prepare.gripper_state_compat.is_gripper_usable()`，仅在夹爪已初始化可用时执行同步全开：`position=100, force=100, speed=80`。之后记录驱动返回的最终开度。

### 阶段 9：回程

若此前真的执行过 LIN 下探，先 LIN 回主放置位；直接在主目标松爪或下探失败回退时不需要此步。

随后 `return_to_graspup_after_buffer()`（函数名保留了历史名字，实际也服务 Middle）执行：

```text
当前固定目标 → graspup_top → graspup
```

第一段为缓存或规划后的七关节主轨迹，缓存同样要经过首点误差校验；第二段是到固定 `GRASPUP_JOINTS_DEG` 的关节运动：

```python
GRASPUP_JOINTS_DEG = [
  -4.081, -52.722, 3.637, 100.478,
  -10.051, 53.969, 8.843,
]
```

### 阶段 10：写出终态

模块经 `modules.stage_result.write_stage_result()` 写出 JSON。写入先落到同目录临时文件，再用 `os.replace()` 原子替换，因此总控不会读到半个 JSON。

---

## 5. 四条固定路径的轨迹缓存

### 5.1 缓存范围

`trajectory_cache.py` 明确只允许以下四个 route：

```text
graspuptop_to_buffer       buffer_to_graspuptop
graspuptop_to_middle       middle_to_graspuptop
```

对应文件固定在：

```text
trajactory/
├── workpiece_1/
│   ├── graspuptop_to_buffer.json
│   ├── buffer_to_graspuptop.json
│   ├── graspuptop_to_middle.json
│   └── middle_to_graspuptop.json
└── workpiece_2/
    └── 同名四条路径
```

目录名 `trajactory` 是现场已有拼写，代码与文档都保持它，不改为 `trajectory`。

### 5.2 什么时候能复用

缓存 JSON 中保存了：轨迹七关节点、采样频率、所选放置候选（出站路径必需）、创建时间和兼容性签名。兼容性签名是以下字段的规范 JSON 再做 SHA-256：

- 工件编号、route、`placement_kind`；
- 固定目标 XYZ 与所有目标 Z 候选；
- `GRASPUP_ABOVE_JOINTS_DEG`；
- 放置规划器类型；
- `motion_speed` 与回程 `preset_speed`。

因此配置/速度/规划器/起点发生变化、schema 不同、工件或 route 不匹配、JSON 损坏、轨迹点不是有效的 7 关节数值、采样频率非法，都会视为缓存未命中并重新规划。写缓存采用 `.json.tmp` 后 `os.replace()`，避免掉电留下半个文件。

缓存只在真实轨迹已经执行且关节到位验证成功后写入；plan-only 不写缓存。回程缓存若实际关节与缓存第一点误差超过容差，同样会重新规划而不是强行播放。

### 5.3 清空缓存

先只查看：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
./.venv/bin/python modules/module_3_left_place/clear_trajactory.py
```

确认后删除 Module 3 的全部缓存：

```bash
./.venv/bin/python modules/module_3_left_place/clear_trajactory.py --yes
```

该脚本只删除 `trajactory/` 下的直接条目并保留根目录；下一次四条路径真实执行成功时会自动重新生成。它不会删除配置、结果、抓取轨迹或其他模块文件。

---

## 6. 异常与恢复行为

### 6.1 无可行主放置轨迹

在真实模式中，左臂到达 `graspuptop` 后，如果 6 秒内仍未找到可行的主放置轨迹，恢复线程会：

1. 置位 `planning_timed_out`，停止继续投递候选；
2. 将左臂回 `graspup`；
3. 查询 `body_base_T_left_eef`，沿 `body_base` 负 Z 下降 `0.05 m`；
4. 用 `lin` 移到这个安全丢弃位置；
5. 打开左夹爪；
6. 再回 `graspup`；
7. 写出 `outcome="planning_failed_recovered_at_graspup"`。

这里是“规划不可行时不带着工件卡在半空”的受控恢复。若恢复动作本身异常，模块抛出错误，不会伪装为成功。

### 6.2 其他会导致模块失败的情况

| 情况 | 行为 |
| --- | --- |
| handoff 缺失或字段不符合分支契约 | 动作前抛错，无运动 |
| 配置非法 / Z 候选非法 | 动作前抛错，无运动 |
| 到达 graspup_top 失败或到位超时 | 模块失败 |
| 规划器 TF 缺失或格式错误 | 模块失败 |
| 所有候选规划失败（但没有触发 6 秒恢复） | 模块失败 |
| 轨迹首点不匹配且在线重规划为空 | 模块失败 |
| 主轨迹执行后没有有效终点关节或无法到位 | 模块失败 |
| 左夹爪未初始化 / 无状态 / 打开失败 | 模块失败 |
| 回程规划或回程到位失败 | 模块失败 |

“松爪下探失败”是特例：因为主目标已经执行并到位，代码会直接在该主目标松爪，仍写成功结果，同时保留所有下探失败记录，供调整 `release_z_candidates_m`。

### 6.3 Module 3 与掉件判断的关系

Module 3 不再在移动期间检查夹爪是否持件，也不会在放置后读取重量传感器确认工件是否留在 Buffer。对于 Buffer 路径，控制器在 Module 3 成功后把状态推进为“Buffer 预计有件”，随后 Module 4 通过多次观察/ICP 给出实际结论；看不到或始终不可靠的后续处理由总控的 Buffer 观察重试规则决定。

---

## 7. 输出、诊断文件与结果字段

### 7.1 成功 / 纯规划结果

真实成功的关键字段如下（Middle 与 Buffer 的值随分支不同）：

```json
{
  "schema_version": 1,
  "stage": "buffer_place",
  "outcome": "success",
  "module": "module_3_left_place",
  "workpiece": "1",
  "placement_kind": "buffer",
  "place_module": "3-buffer-place",
  "selected_candidate": { "target_tool_pose": { "...": "..." } },
  "planning_mode": "trajectory_cache | offline | online_replan_after_start_mismatch",
  "trajectory_cache": ["..."],
  "release_motion_succeeded": true,
  "release_fallback_at_target": false,
  "release_motion_attempts": ["..."],
  "released": true,
  "released_at_verified_target": true,
  "returned_to_graspup": true
}
```

plan-only 使用 `outcome="planned"`，并将 `released`、`released_at_verified_target`、`returned_to_graspup` 写为 `false`。它的 `planning_mode` 为 `dry_run` 或 `trajectory_cache`，但不会执行也不会把轨迹写入缓存。

对于触发安全丢弃的受控恢复，结果至少包含：

```json
{
  "outcome": "planning_failed_recovered_at_graspup",
  "completed_without_error": true,
  "returned_to_graspup": true,
  "released": true,
  "recovery_action": "discard_below_graspup_then_open_left_gripper"
}
```

总控通过模块退出状态和该结果 JSON 判断后续调度；不要仅依据 shell 返回码把这个恢复结果当成普通放置成功。

### 7.2 规划诊断

每次运行会在 `--result-file` 的同一目录额外追加：

```text
placement_planning_attempts.jsonl
```

每行是一次候选提交、规划成功/失败、缓存命中或松爪下探尝试，包含时间、调度器配置、候选 Z、目标坐标、提交/完成顺序、异常文本等。规划器自身的轨迹记录目录会通过环境变量 `PARM_PLANNING_RECORD_DIR` 指向：

```text
output/left_place_trajectories/<启动时间>/
```

排查问题时优先看最终 result JSON 中的 `planning_attempts`、`trajectory_cache`、`release_motion_attempts`，再结合 JSONL 看所有并发候选的完整过程。

---

## 8. 代码文件职责

| 文件 | 职责 |
| --- | --- |
| `run.sh` | 轻量启动包装；定位项目根目录并强制使用项目 `.venv/bin/python`。 |
| `standalone_left_place.py` | 模块主编排：线程、规划、缓存选择、执行、松爪、回程、终态写出。控制器实际调用此入口。 |
| `placement_support.py` | 命令行解析、hand off 契约、Planner 配置、TF/矩阵转换、关节到位等待、夹爪打开、可选文件门禁。 |
| `placement_config.py` | 严格加载 `workpiece_*_placement_parameters.toml`，只暴露不可变的 `FixedPlacementParameters`。 |
| `placement_parameters.py` | `graspup` / `graspuptop` 七关节常量、6 秒规划恢复阈值、0.05 m 丢弃深度、0.10 m 固定上移常量。 |
| `trajectory_cache.py` | 仅管理四条主路径：校验、签名、读取、原子保存；不连接规划器、不控制机械臂。 |
| `clear_trajactory.py` | 带 `--yes` 二次确认的缓存清理工具。 |
| `workpiece_1_placement_parameters.toml` | 工件 1 的 Buffer/Middle 固定坐标和松爪策略。 |
| `workpiece_2_placement_parameters.toml` | 工件 2 的 Buffer/Middle 固定坐标和松爪策略。 |
| `trajactory/` | 运行生成的、可失效且可重建的四条固定主路径缓存。 |

依赖关系可以概括为：

```text
run.sh
  └── standalone_left_place.py
       ├── placement_support.py
       │    ├── parm_sdk（Arm / Gripper）
       │    └── tools.planner_runtime.PlannerRuntime
       ├── placement_config.py → workpiece_*_placement_parameters.toml
       ├── placement_parameters.py
       ├── trajectory_cache.py → trajactory/
       └── modules.stage_result.py
```

---

## 9. 独立运行

下面的命令仅用于确认模块接口或人工测试；正常生产流程由 `controller/main.py` 调用本模块。

### 9.1 Buffer：纯规划（不发送机械臂、夹爪动作）

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --placement-kind buffer \
  --handoff-file output/module_1_left_grasp/verified_handoff.json \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --result-file output/module_3_left_place/manual_buffer_plan.json
```

### 9.2 Buffer：真实放置

只有机器人周围安全、上游 handoff 确认正确、规划器和驱动均已启动时才可使用：

```bash
modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --placement-kind buffer \
  --handoff-file output/module_1_left_grasp/verified_handoff.json \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --result-file output/module_3_left_place/manual_buffer_execute.json \
  --execute \
  --confirm-real-place
```

### 9.3 Middle：真实放置

```bash
modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --placement-kind middle \
  --handoff-file output/module_5_left_pick/front_regrasp_handoff.json \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --result-file output/module_3_left_place/manual_middle_execute.json \
  --execute \
  --confirm-real-place
```

常用可调参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--post-graspup-planner` | `rrtstar` | 主路径规划器；`rrtconnect` 会映射到 Planner 的 `ompl` |
| `--placement-planning-inflight` | `8` | 同时在飞的 Z 候选规划请求数，范围 1–16 |
| `--offline-start-tolerance-deg` | `1.0` | 缓存/离线轨迹首点与实际关节的最大允许误差 |
| `--preset-speed` | `0.2 * PARM_LEFT_SPEED_SCALE` | 到/离开 `graspuptop`、回程速度 |
| `--motion-speed` | `0.1 * PARM_LEFT_SPEED_SCALE` | 主放置、LIN 下探、异常丢弃的速度 |
| `--pre-place-middle-joints-deg q1..q7` | 未启用 | 显式要求先走的放置前关节中间位 |

---

## 10. 修改与调参原则

1. 改固定放置位置或松爪高度，只改对应 `workpiece_*_placement_parameters.toml` 的对应 section；不要在 `standalone_left_place.py` 写死坐标。
2. 改 `target_*`、速度、规划器或 `graspuptop` 常量后，旧缓存会自动失效；若要保证现场干净，可先运行缓存清理脚本。
3. 新增工件类型时，需要同步扩展 `placement_config.CONFIG_PATHS`、CLI `--workpiece` choices、缓存目录策略和对应 TOML，不能只复制一个配置文件。
4. 新增新的放置区域时，除了配置段，还必须扩展 `placement_kind` choices、`load_handoff()` 的输入契约、允许缓存 route 集合和控制器 registry；不能把新区域伪装为 Buffer 或 Middle。
5. 调整 `release_z_candidates_m` 前，先通过结果 JSON 中的 `release_motion_attempts` 确认主目标是否已稳定可达；下降失败会自动回退直接松爪，但频繁回退说明应重新标定该候选。
6. 不要删除“轨迹首点实际关节校验”和“真实执行后才落缓存”这两个保护，它们是离线轨迹安全复用的前提。

本 README 只描述 Module 3 自身；模块之间的状态机、左右臂并发、Buffer/Middle 状态和人工干预规则见项目根目录 `README.md` 中的总控说明。
