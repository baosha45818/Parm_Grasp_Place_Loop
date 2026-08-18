# 新总控重构方案（讨论确认稿）

> 状态：控制器框架已实现并完成 dry-run 演练：启动预置、统一结果、资源锁、
> 左右并行、观察位复位、停止链与人工恢复均位于 `controller/`。尚未进行真实机械臂
> 运动验收。
>
> 范围：新的总控及其调度、锁、模块结果协议均放在 `controller/`；旧的
> `modules/modular_dual_arm_loop.py -> tests/loop/simple_dual_arm_loop.py` 双层总控
> 在新总控验收前保留，只作为旧实现参考。

## 1. 目标

新总控只做调度，不实现视觉、ICP、抓取、放置等业务细节。每个模块均视为一个
可调用单元：输入可以不同，但必须写出统一格式的结果；总控只根据结果、资源锁和
调度规则决定下一步。单一模块失败不得直接中断整个流程。

对于不能安全自动恢复的异常，总控在终端输出当前状态、故障模块和人工处理建议，
等待操作员输入 `y` 后重新执行**刚刚失败的模块**。

## 2. 已确认的决策

| 项目 | 已确认规则 |
| --- | --- |
| Module 2 | 已删除；新总控不得依赖或等待 Module 2 的任何文件、许可或结果。 |
| Module 5 Buffer 重抓 | 正面（5.1）和反面（5.2）均保留 `5` 次上限，统一含义为最多 **5 次完整“Module 4.1 重新观察并生成新 TF + Module 5 单次抓取”闭环**。 |
| Module 5 达到重试上限 | 不再自动尝试；输出需要人工干预的结果，由总控暂停并等待 `y`；恢复后固定从 Module 4.1 重新观察开始。 |
| Buffer 观察不可靠 | 每次 Module 4.1 调用只采集一次；每个 Buffer 工件的完整观察次数由 `--max-buffer-observation-attempts`（默认 4）统一限制。0/多个工件、ICP 或正反不可靠共用该计数；耗尽后按本次 Buffer 工件已掉/无效处理，清空 Buffer 并回 Module 1。 |
| Module 5 重抓 | 保留 `5` 次上限；耗尽后人工干预。 |
| Module 9 插孔失败 | 允许左臂继续运行；右臂退回 `right place` 起始位置。达到模块内部上限后人工干预，`y` 重启 Module 9。 |
| Buffer 抓取人工恢复 | Module 5 重试耗尽后，人工处理并输入 `y`；不论工件是否被移动，均固定从 Module 4.1 重新观察、生成新 TF 后再进入 Module 5 单次抓取。 |
| Module 1 内部重试 | 视觉稳定、Planner 不可达、重量校验等现有内部重试次数原样保留；本轮重构只补统一结果输出，不改变其策略。 |
| 旧总控 | `modules/modular_dual_arm_loop.py` 与 `tests/loop/simple_dual_arm_loop.py` 保留在原位置作为旧实现参考，但新流程不再调用它们。 |
| 左臂标准交接位 | Module 1 抓取成功和 Module 5.1 正面重抓成功后，均以 `graspuptop` 作为结束位并写出 handoff；不在模块内继续放置。 |
| Module 3 放置类型 | Module 3 统一负责左臂已持件后的放置：Module 1 handoff 使用 `buffer` 目标；Module 5.1 handoff 使用 `middle` 目标。两类目标配置放在 Module 3 的工件 TOML 内，均为固定 `left_eef` 位置并保持进入 `graspuptop` 时的末端旋转。 |

## 3. 新控制器目录

```text
controller/
├── main.py                 # 唯一总控入口：事件循环、启动/停止、人工交互
├── scheduler_rules.py      # 纯调度规则：状态 + 结果 -> 下一模块/动作
├── module_registry.py      # 模块命令、输入映射、结果文件、资源声明
├── module_runner.py        # 启动模块、超时、读取或补齐结果
├── robot_actions.py         # 启动预置、Buffer 观察位（非业务动作）
├── safety.py                # Ctrl+C/停止事件的双臂取消链
├── deployment.py            # xiangxin/gzhu 现场地址和环境统一注入
├── contracts.py            # 统一结果协议及校验
├── runtime_state.py        # Buffer 和双臂的唯一状态源
├── locks.py                # 跨进程资源锁
├── README.md               # 运行和恢复说明
└── runtime/                # 本次运行的结果、状态、锁；不提交运行产物
```

`main.py` 不直接导入或编排各模块的视觉/机器人业务代码；它只调用
`module_runner.py`，再将结果交给 `scheduler_rules.py`。

## 4. 统一模块结果协议

所有模块需要支持 `--input-file`（需要输入时）和 `--result-file`，并保证无论成功、
可重试失败还是异常，尽量都写出一个 JSON。推荐基本结构：

```json
{
  "schema_version": 1,
  "run_id": "20260818T120000_xxxx",
  "module": "module_1_left_grasp",
  "status": "SUCCESS",
  "outcome": "LEFT_HOLDS_SINGLE_WORKPIECE",
  "outputs": {},
  "error": {
    "code": null,
    "message": null,
    "recoverable": false,
    "safe_state": null,
    "operator_hint": null
  },
  "retry": {
    "allowed": false,
    "recommended_action": null
  }
}
```

`status` 只允许：

- `SUCCESS`：模块目标完成；
- `RETRY`：可由调度规则自动重试；
- `WAIT`：等待另一个模块或资源；
- `OPERATOR_REQUIRED`：不能安全自动恢复，等待人工；
- `FAULT`：模块异常、输入非法或未能产出可信结果；
- `CANCELLED`：由总控安全取消。

若子进程崩溃、超时或未写结果，`module_runner.py` 必须合成一个 `FAULT` 结果，保留
退出码、stdout/stderr 摘要和调用参数。M8.2、M9、M10 当前未提供原生结果文件，
由控制器写出标记为 `controller_synthesized` 的相同协议；这样总控永远只处理同一种
结果协议。

## 5. 模块边界和预期接口改造

| 调度单元 | 现有职责 | 新总控看到的输入 | 成功输出 | 失败时应输出 | 需要的改造 |
| --- | --- | --- | --- | --- | --- |
| Module 0 准备 | 服务、相机和环境检查 | 部署模式 | `READY` | `FAULT` + 缺失服务 | 补统一结果文件。 |
| Module 1 左抓 | D405/SAM2/ICP、抓取、重量验证 | mode、任务号 | `LEFT_HOLDS_SINGLE_WORKPIECE` + 在 `graspuptop` 的 verified handoff | `RETRY` 或 `OPERATOR_REQUIRED` | 成功重量验证后移至 `graspuptop` 再写 handoff；保留现有内部重试策略。 |
| Module 3 左放 | 将左臂持件放至 Buffer 或桌中 | Module 1/5.1 handoff、放置类型 | `BUFFER_OCCUPIED` 或 `MIDDLE_PLACED` | `FAULT` + 实际运动结果 | 增加 `--placement-kind buffer|middle`；按 handoff 来源校验允许的放置类型。运输中不以夹爪状态判掉件，Buffer 有效性由 Module 4 统一确认。 |
| Module 4.1 Buffer 状态估计 | Buffer 工件检测、正反判断、ICP TF | mode、Buffer 工件上下文 | `FRONT`/`BACK` + ICP TF | `UNCERTAIN` | 每次只采集一次；控制器对所有 `UNCERTAIN` 共用观察预算。预算耗尽即清空本次 Buffer 并回 Module 1。 |
| Module 5.1 正面抓取 | 左手按 Module 4.1 TF 正抓到 `graspuptop` | Module 4.1 result/TF | `LEFT_HOLDS_FRONT_WORKPIECE` + 在 `graspuptop` 的 handoff | `RETRY` 或 5 次完整闭环耗尽的 `OPERATOR_REQUIRED` | 删除其内部桌中放置；每次闭环均先调用 Module 4.1 重新观察并获取新 TF，再单次重抓。 |
| Module 5.2 反面抓取交接 | 左手按 Module 4.1 TF 反抓并送交接位 | Module 4.1 result/TF | `LEFT_AT_HANDOFF_READY` | `RETRY` 或 5 次完整闭环耗尽的 `OPERATOR_REQUIRED` | 将抓取、交接准备结果统一；每次闭环同样从新的 Module 4.1 观察开始。 |
| 右手正面调整与抓取（复合模块） | Module 4.2 → Module 7 → Module 8.1 的内部串联 | 工件任务和部署模式 | `RIGHT_HOLDS_WORKPIECE` | `OPERATOR_REQUIRED`/`FAULT` + 安全位 | 保留现有小总控实现，但增加稳定 CLI 与统一输入/输出；新总控不拆开其内部细节。 |
| Module 8.2 交接 | 左右手交接和右手放置准备 | 左手 handoff | `RIGHT_HOLDS_WORKPIECE` | `FAULT` + 左右手实际状态 | 标准结果必须声明两臂和交接区的最终状态。 |
| Module 9 右放 | 右手对孔、wiggle、恢复 | 右手持件和目标信息 | `RIGHT_PLACED` | `OPERATOR_REQUIRED`，右手在恢复安全位 | 保留内部插孔重试；结果明确 safe state。 |
| Module 10 右待机 | 放置后回待机 | Module 9 成功结果 | `RIGHT_IDLE` | `FAULT` | 将 `tests/loop` 的实现迁回模块，并提供标准结果。 |
| Module 6 左待机 | 左手回待机 | 当前左臂状态 | `LEFT_IDLE` | `FAULT` | 补清晰 CLI 和标准结果。 |

## 6. 调度主线

```text
Module 0 准备
  ├─ 左线：Module 1 左抓（结束于 graspuptop） → Module 3(buffer) 放入 Buffer → Module 4.1
  │                                            ├─ FRONT → Module 5.1 正抓至 graspuptop
  │                                            │            → Module 3(middle) 放至中间目标
  │                                            └─ BACK  → Module 5.2 反抓至交接
  │                                                         → Module 8.2 交接
  │                                                         → Module 9 右放 → Module 10 右待机
  └─ 右线：右手正面调整与抓取（Module 4.2 → 7 → 8.1 的复合模块）
```

实际下一步不写死在 `main.py`：由 `scheduler_rules.py` 根据模块结果、Buffer 状态、
双臂状态和资源锁返回。这样可以单独审阅、测试和修改规则。

## 7. Buffer 状态与资源锁

Buffer、Middle 与 Handoff 都是左右线共享的工位状态，不能只保留一个“空/非空”布尔量。
`runtime_state.py` 应原子保存：

```text
EMPTY
→ LEFT_PLACING
→ BUFFER_OCCUPIED(workpiece, orientation, ICP TF, observation id)
→ LEFT_REGRASPING 或 LEFT_HANDOFF_READY
→ RIGHT_CONSUMING
→ EMPTY
```

`runtime_state.py` 现记录 `buffer`、`middle` 与 `handoff`。这三个工位都不使用文件锁：
模块进程退出后，工位仍可能被工件或机械臂占用，因此由单一总控的状态机保持其约束。
文件锁只保护机械臂和相机等运行期硬件资源，且只允许控制器持有。正面工件进入 middle
后，右臂后台消费，左臂可从 Module 1 开始下一件；下一件到达 `M3(middle)` 前会等待
上一件右臂完成。

| 锁 | 使用者 | 作用 |
| --- | --- | --- |
| `LEFT_ARM` | 所有左臂运动模块 | 防止左臂同时被两个模块控制。 |
| `RIGHT_ARM` | 所有右臂运动模块 | 防止右臂同时被两个模块控制。 |
| `HEAD_D435` | Module 4.1、4.2、需要头相机的交接流程 | 防止同一相机/标定链并发采集。 |
| `LEFT_D405` | Module 1 | 独占左腕相机。 |

Module 9 执行 Servo 时只持有 `RIGHT_ARM`，**不阻塞左臂普通任务**；失败时右臂退回
`right place` 起始位置。交接动作仅由 `handoff=LEFT_READY` 放行，M8.2 启动前先将其
置为 `TRANSFERRING`；M8.2 成功后才恢复为 `EMPTY`。

旧总控里真正的共享互斥仅是 `buffer_state` 的 `multiprocessing.Value` 锁；
`handoff_left_clear`、`right_idle_event` 等是事件同步，不是资源锁。新方案将这些
隐式同步改为上述显式状态和资源锁。

## 8. 失败、重试与人工干预

1. 模块返回 `RETRY`：规则文件检查该模块的重试计数和安全状态；满足上限前重启该模块。
2. Module 4 观察预算耗尽：自动清空本次 Buffer 状态并回 Module 1；其他模块返回 `OPERATOR_REQUIRED`、`FAULT`，或 Module 5 重试耗尽时，停止发起新的冲突任务，让正在执行的另一臂安全完成当前原子动作。
3. 总控打印：故障模块、错误码、当前 Buffer/双臂状态、建议人工处理动作。
4. 操作员处理完成，输入 `y` 并回车。
5. Module 5 的自动重试和人工恢复均固定从 Module 4.1 开始，重新生成本轮有效观察和 ICP TF；随后只执行一次对应的 Module 5 抓取。
6. 任何重启都使用新的 `run_id` 和新的结果文件，禁止复用历史 TF、handoff 或 `latest` 文件作为当前输入。

## 9. 后续实施顺序

1. 创建控制器框架、结果协议、状态文件和锁文件；不接真实机器人。
2. 先完成左臂边界整理：Module 1 与 5.1 均输出 `graspuptop` handoff；删除 5.1 内部的旧桌中放置调用；Module 3 的配置解析器按 `--placement-kind` 读取现有 `[buffer]` 或 `[middle]` 段，并以进入 `graspuptop` 时的末端旋转移动至固定目标位置。
3. 给 Module 0/1/3/4.1/5/6/8/9/10 加统一 result wrapper，先做 plan-only/模拟结果测试。
4. 将右手 `4.2 → 7 → 8.1` 小总控封装为一个稳定的复合模块接口。
5. 将 Module 10 从 `tests/loop` 迁回模块目录，并删除新总控对 `tests/loop` 的依赖。
6. 编写并单测 `scheduler_rules.py`：正面、反面、UNCERTAIN、重试耗尽、人工恢复、左右并发。
7. 在不执行真实运动的模式下进行端到端状态机演练。
8. 分阶段启用真实运动，并保留每一步的结果 JSON、状态快照和日志以便回放。

## 10. 实施前确认状态

本轮讨论中需要由人工决定的调度与恢复规则均已确认。后续仅需在计划模式和真实
机器人分阶段验证中校验具体点位、标定和安全边界；若现场发现问题，再以实际结果
修订配置，不改变本方案的调度边界。
