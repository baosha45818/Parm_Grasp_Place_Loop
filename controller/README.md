# Loop 总控

`controller/main.py` 是 Loop 唯一的运行入口。它不实现抓取、ICP、放置或单臂轨迹；它只负责：选择现场配置、调用模块、维护工位状态、保证硬件互斥、安排左右臂并行、处理失败恢复和停止。

具体机器人动作归各模块所有：启动准备在 Module 0，Buffer 观察位在 Module 4，反面反抓/交接准备在 Module 5.2。`controller/` 中不存在单模块的动作运行脚本。

## 运行

先用演练验证总控状态机；不会启动任何模块或连接硬件：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

./.venv/bin/python controller/main.py \
  --mode gzhu \
  --workpiece 1 \
  --dry-run \
  --simulate-face FRONT \
  --left-cycle-limit 1 \
  --prepare-speed 0.3
```

真实执行必须显式带 `--live`。未显式传入的地址由 `--mode` 选择；显式地址优先级更高：

```bash
./.venv/bin/python controller/main.py \
  --mode gzhu \
  --workpiece 1 \
  --live \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --force-host 127.0.0.1 \
  --head-host 127.0.0.1 \
  --max-buffer-observation-attempts 4 \
  --left-cycle-limit 0 \
  --prepare-speed 0.3
```

`--left-cycle-limit 0` 表示持续运行；`--dry-run` 未指定该参数时自动只演练一件。按 `Ctrl+C`、发送 `SIGTERM` 或 `SIGHUP` 时，总控停止发起新任务、向双臂发送 `cancel_task()`，并结束当前模块的整个子进程组。

## 当前现场配置

| `--mode` | arm / gripper | force / head | planner |
| --- | --- | --- | --- |
| `gzhu` | `192.168.88.1` / `127.0.0.1` | `127.0.0.1` / `127.0.0.1` | `127.0.0.1` |
| `xiangxin` | `192.168.3.11` / `192.168.3.11` | `192.168.3.22` / `192.168.3.22` | `127.0.0.1` |

`deployment.py` 还会写入 `PARM_DEPLOYMENT_MODE`、各服务地址以及对应的 `PARM_BODY_HEAD_CALIBRATION`。因此反面交接模块继承的头部标定始终与 `--mode` 一致。

## 调度流程

逻辑模块名 `controller_prepare` 和 `controller_observation_pose` 是总控状态机中的步骤名，不表示其实现位于 controller：前者实际调用 Module 0，后者实际调用 Module 4。

```text
启动
  → controller_prepare
      → Module 0：SAM2/硬件预检、开双夹爪、左臂 graspup、右臂 Idle
  → M1 左臂抓取
  → M3(buffer)
  → controller_observation_pose
      → Module 4：左臂和头部到 Buffer 标定观察位
  → M4.1：Buffer 正反判断与 ICP

FRONT:
  → M5.1
  → M3(middle)
  → 右臂后台：M8.1（内部包含其右侧观察/对齐）→ M9 → M10

BACK:
  → M5.2：反抓 + 左手交接准备
  → M8.2：双臂交接
  → 右臂后台：M9 → M10
```

一旦工件成功发布给右臂，左臂可从 M1 开始下一件。左臂即将占用 `middle` 或交接区时，总控才等待上一件右臂任务结束；因此不会让两件工件争用同一物理工位。

## 状态、结果与恢复

每次启动创建 `controller/runtime/<run_id>/`：

```text
runtime/<run_id>/
├── state.json       # RuntimeState 的最近一次持久化快照
├── artifacts/       # 每次模块的原始结果文件或其指针
├── logs/            # 各模块 stdout/stderr 合并日志
├── results/         # controller 标准化后的 ModuleResult
└── locks/           # LEFT_ARM、RIGHT_ARM、HEAD_D435、LEFT_D405 文件锁
```

`RuntimeState` 的关键字段为：

| 字段 | 含义 | 常见取值 |
| --- | --- | --- |
| `buffer` | Buffer 工位生命周期 | `EMPTY`、`LEFT_PLACING`、`OCCUPIED`、`LEFT_REGRASPING` |
| `middle` | 正面工件的右臂工作区 | `EMPTY`、`OCCUPIED` |
| `handoff` | 反面工件交接区，独立于 Buffer | `EMPTY`、`LEFT_READY`、`TRANSFERRING`、`UNKNOWN` |
| `left_arm` / `right_arm` | 总控掌握的机械臂现场状态 | `IDLE`、`HOLDS_WORKPIECE`、`AT_HANDOFF`、`BUSY`、`FAULT` |
| `orientation` | M4.1 已确认的方向 | `FRONT`、`BACK` 或 `null` |
| `retries` | 当前工件的 M4/M5 重试预算 | 字典 |
| `attempts` | 各逻辑模块的调用编号 | 字典 |
| `artifacts` | 各模块最近一次产物位置 | 模块名到文件路径 |

恢复规则由 `scheduler_rules.py` 固定：

- M4.1 的 `UNCERTAIN`、零件数异常或 ICP 不可靠都消耗同一个 Buffer 观察预算；达到 `--max-buffer-observation-attempts` 后丢弃该 Buffer 生命周期并回 M1。
- M5.1/M5.2 返回退出码 20（未抓稳）时，最多进行 5 次完整“观察位 → M4.1 → M5”闭环；不会复用旧 ICP TF。
- M3 报告 `planning_failed_recovered_at_graspup` 时，说明模块已在安全位置松爪；总控清理本轮工件并回 M1。
- M5.2 已反抓成功但交接准备失败时，结果必须声明 `left_gripper_holds_object=true`。总控把左臂固定为 `HOLDS_WORKPIECE`，人工确认后只运行 `module_5_reverse_handoff`，绝不重新观察或反抓。
- M8.2 失败时交接位置为 `UNKNOWN`；必须人工把现场恢复到“左手交接准备、右手可接取”后才可重试。

## controller 内每个文件

### `main.py` — 主循环、并行与人工恢复

实现内容：

- `parse_args()` 定义 `--mode`、`--live/--dry-run`、服务地址、观察次数和循环上限。
- 启动时调用 `deployment.apply()`，创建 `run_id`、`RuntimeState` 与 `runtime/<run_id>/` 目录结构，然后从 `controller_prepare` 开始。
- `context_for()` 把模式、主机地址、上游结果路径和本次结果路径组装为模块命令上下文。M8.2 使用 `latest_reverse_handoff_result()` 挑选本次最新、有效的 M5.2 交接位姿，避免误用上一件的 ICP TF。
- `run_one()` 为左侧步骤递增 attempt、写入调用前状态、调用 `module_runner.run()`，记录产物和标准结果。
- `RightTask` 与 `run_right_task()` 把右臂消费链放进一个单线程 `ThreadPoolExecutor`。正面后台链是 `M8.1 → M9 → M10`，反面后台链是 `M9 → M10`。
- 主线程通过 `start_right()`、`collect_right()`、`wait_right_clear()` 只允许一个右臂后台任务；在进入 `M3(middle)` 或 M8.2 前等待右臂清空工位。
- `wait_for_operator()` 读取 `y/q`。反抓成功后的交接失败只进入恢复交接入口；Buffer 判断或重抓问题则回观察位；M8.2 失败须先把状态显式改回 `LEFT_READY`。
- 信号处理只设置停止事件；`finally` 负责等待右臂线程收尾。收到停止后调用 `safety.cancel_both_arms()`。

它不包含任何单模块轨迹、抓取、ICP 或相机算法。

### `module_registry.py` — 模块命令与资源声明

`ModuleSpec` 把一个逻辑模块定义为四项：模块名、硬件资源元组、由上下文构建命令的函数、以及该模块是否原生遵守 `--result-file` 协议。

`registry()` 是唯一集中出现模块路径和命令行参数的位置。当前映射如下：

| 逻辑名 | 实际入口 | 资源锁 | 特别处理 |
| --- | --- | --- | --- |
| `controller_prepare` | `modules/module_0_prepare/main.py` | 左臂、右臂 | Module 0 的预检、开夹爪和归位。 |
| `controller_observation_pose` | `modules/module_4_head_judge/observation_pose.py` | 左臂、头部 D435 | Module 4.1 前置观察位。 |
| `module_1_left_grasp` | `modules/module_1_left_grasp/run.sh` | 左臂、左腕 D405 | 真实执行附加 `--execute --confirm-real-grasp`。 |
| `module_3_buffer` / `module_3_middle` | `modules/module_3_left_place/run.sh` | 左臂 | 用 `--placement-kind` 区分目标。 |
| `module_4_buffer_state` | `.../check_buffer_face.py` | 头部 D435 | 强制模块内 `--max-uncertain-retries 0`，观察预算只由总控计数。 |
| `module_5_front` | `.../module_5_1.../main.py` | 左臂 | 真实执行附加 `--execute`。 |
| `module_5_reverse` | `python -m ...module_5_2...pipeline` | 左臂、头部 D435 | 反抓和交接准备的组合入口。 |
| `module_5_reverse_handoff` | `python -m ...module_5_2...resume_handoff` | 左臂、头部 D435 | 仅恢复交接准备。 |
| `module_8_handoff` | `.../module_8_2_left_handoff/main.py` | 左臂、右臂 | 没有原生标准结果文件，由 controller 合成。 |
| `module_8_right_front` | `.../module_8_1_buffer_grasp/workpiece_pipeline.py` | 右臂、头部 D435 | 原生写入 `--result-file`。 |
| `module_9_right_place` | `.../module_9_right_place/place_hole_wiggle.py` | 右臂 | 没有原生标准结果文件，由 controller 合成。 |
| `module_10_right_idle` | `.../module_10_right_idle/main.py` | 右臂 | 没有原生标准结果文件，由 controller 合成。 |

注册表不做状态转换；它只构建命令。是否启动、何时重试由 `main.py` 和 `scheduler_rules.py` 决定。

### `module_runner.py` — 子进程运行器与结果适配器

`run()` 的实现顺序是：从注册表取得命令 → 获取资源锁 → 以项目根目录为 `cwd` 启动独立进程组 → 将 stdout/stderr 写到本轮 `logs/` → 等待退出或停止事件 → 写入标准 `results/`。

停止时 `_terminate_process_group()` 先对整个进程组发 `SIGTERM`，8 秒未结束再发 `SIGKILL`，避免模块启动的子进程遗留。

`_normalize()` 将不同模块的退出码/JSON 转为 `ModuleResult`：

- M4.1 强制校验 `decision` 与退出码对应：`FRONT=0`、`BACK=20`、`UNCERTAIN=21`；前两者是 `SUCCESS`，后者是 `RETRY`。
- M3 校验 `placement_kind` 和 `stage` 与总控分支一致；安全回 `graspuptop` 的规划失败转为 `RETRY`，松爪失败且仍持件转为 `OPERATOR_REQUIRED`。
- M5 的退出码 20 是 `RETRY/NO_GRASP`；通用退出码 21 是 `OPERATOR_REQUIRED`；其余非零退出码为 `FAULT`。
- M8.2、M9、M10 不原生输出统一结果文件，runner 使用退出码生成 `controller_synthesized` 结果；M8.1 原生写入 `--result-file`。

### `scheduler_rules.py` — 纯状态机

`decide(state, result, ...)` 是无硬件副作用的函数：它修改传入的 `RuntimeState`，返回 `Decision(next_module, operator_required, finished)`。

它定义 Buffer/Middle/交接区的全部转换，决定 FRONT 去 M5.1、BACK 去 M5.2，定义 M4/M5 的重试上限，并保护两个危险现场：M5.2 已持件但交接失败，以及 M8.2 后工件归属未知。这里不导入 SDK、不启动进程，也不调用模块。

### `runtime_state.py` — 唯一业务状态源

`RuntimeState` 是 dataclass，保存工位占用、左右臂状态、方向、重试/调用计数和模块产物。`save()` 先写临时文件再原子替换 `state.json`，避免中断时出现半个 JSON；`load()` 用于读取已经保存的快照。只有总控写这个状态，不让模块各自修改 Buffer 或交接区状态。

### `contracts.py` — 统一结果协议

`Status` 枚举定义 `SUCCESS`、`RETRY`、`WAIT`、`OPERATOR_REQUIRED`、`FAULT`、`CANCELLED`。`ModuleResult` 是写入 `runtime/<run_id>/results/` 的标准数据结构，含模块名、结果、产物、错误、来源和协议版本。`write()` 同样采用临时文件原子替换；`now_id()` 生成 UTC 的 `run_id`。

### `locks.py` — 跨进程硬件互斥

`acquire()` 用 `fcntl.flock(... LOCK_EX | LOCK_NB)` 在 `runtime/<run_id>/locks/` 建立进程级锁。已知资源只有 `LEFT_ARM`、`RIGHT_ARM`、`HEAD_D435`、`LEFT_D405`；多个锁按字母顺序申请，避免死锁。等待锁的过程中检查 `stop_event`，收到停止会抛 `LockCancelled`，使 runner 不会在停止后仍开始动作。

Buffer、Middle、handoff 不是文件锁：它们是跨模块的持久业务状态，由 `RuntimeState + scheduler_rules` 约束。

### `deployment.py` — 现场地址与标定绑定

`_PROFILES` 保存 `gzhu`、`xiangxin` 默认地址。`apply(args)` 对每个地址字段采用“显式命令行覆盖，否则按 mode 填默认值”，然后写入 `PARM_PLANNER_HOST`、`PARM_ARM_HOST`、`PARM_GRIPPER_HOST`、`PARM_FORCE_SENSOR_HOST`、`PARM_HEAD_HOST`。

它最后检查并写入该模式专用的头部标定路径。找不到标定文件会在总控尚未启动任何模块前失败，避免在错误现场使用另一套头部标定。

### `safety.py` — SDK 级停止补偿

`cancel_both_arms()` 分别建立左右臂 SDK 客户端并调用 `cancel_task(1/2)`。单侧取消失败只记录错误，仍继续尝试另一侧；随后关闭客户端。它与 runner 的“结束当前模块子进程组”配合：前者尝试中止控制器已接收的任务，后者防止本机模块继续运行。

### `__init__.py` — 包标记

空包标记文件，用于将 `controller` 识别为 Python 包；不参与运行逻辑。

### `CONTROLLER_REFACTOR_PLAN.md` — 历史设计文档

记录总控重构时的目标和阶段性设计，供维护时追溯，不会被 `main.py` 导入或执行。当前运行规则以本 README、源代码和各模块 README 为准。

## 模块外的两项总控依赖

以下文件不在 `controller/`，但由总控明确调度：

| 文件 | 由谁调用 | 做什么 |
| --- | --- | --- |
| [`modules/module_0_prepare/main.py`](../modules/module_0_prepare/main.py) | `controller_prepare` | 只读预检、SAM2 保证可用、双夹爪打开、左臂 `graspup`、右臂 Idle。 |
| [`modules/module_4_head_judge/observation_pose.py`](../modules/module_4_head_judge/observation_pose.py) | `controller_observation_pose` | 左臂 Buffer 观察位、头部到与 D435 标定一致的角度。 |

其余模块的视觉、抓取、轨迹、放置配置和独立运行命令请查看各自 `modules/module_*/README.md`。
