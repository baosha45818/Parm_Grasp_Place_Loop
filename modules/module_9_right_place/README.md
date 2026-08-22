# 模块 9：右臂孔位放置

## 1. 模块定位与主要功能

Module 9 负责右臂将已持有的工件放入第一孔位，并在第一孔位确认进入后执行固定的第二孔位路径。它是一个面向真实硬件的放置模块，控制对象包括右臂、右腕力传感器、ServoPose 通道和规划器。

当前实现采用**单段连续直线搜索**：从搜索中心沿配置的世界坐标轴一次性连续推送至最大距离，不在中间按 1 mm 停止、端点采样或重发目标。进洞判定完全依赖搜索过程中的连续力/扭矩监测。

模块完成的工作包括：

1. 回到右臂示教安全关节姿态。
2. 执行世界坐标系下的搜索前接近、下压和力基线采集。
3. 以 50 Hz ServoPose 推送一条单段直线搜索轨迹。
4. 以最高 50 Hz 读取右腕力/扭矩，满足进洞特征或安全接触限时立即停止推送。
5. 第一洞确认后，执行“下探、抬升、Y 偏移、第二洞下降、抬升、X 返回”的固定后续路径。
6. 搜索距离耗尽时，向工件配置入口返回可重试状态；入口按配置执行抬升恢复并开始下一轮。

## 2. 入口、依赖与输入

独立调试入口是 [`place_hole_wiggle.py`](place_hole_wiggle.py)。外部调用必须提供工件编号：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

# 只打印第 1 次尝试将使用的内部参数；不连接硬件
python3 modules/module_9_right_place/place_hole_wiggle.py \
  --workpiece 1 --print-command

# 真实执行：两个安全确认参数缺一不可
python3 modules/module_9_right_place/place_hole_wiggle.py \
  --workpiece 1 --live --confirm-live
```

外层 CLI 参数：

| 参数 | 含义 |
| --- | --- |
| `--workpiece {1,2}` | 选择工件配置。 |
| `--live --confirm-live` | 同时出现才允许真实运动。 |
| `--arm-host HOST` | 覆盖右臂控制器地址。 |
| `--reset-right-task-before-servo` | 开始 ServoPose 前调用 `cancel_task(2)` 清理残留右臂任务。 |
| `--print-command` | 只展开并打印内部执行参数。 |

运行所需输入如下：

| 类别 | 来源 | 作用 |
| --- | --- | --- |
| 工件配置 | `config/workpiece_1_gzhu.toml`、`config/workpiece_2.toml` | 提供工件编号、7 轴示教关节角、搜索轴/方向、搜索前路径及专属进洞阈值。 |
| 规划场景 | `config.hole_a_first_live_fast_return.json` | 用于默认的 `move_j` 回示教起点及避碰规划。 |
| 右臂状态 | 右臂 SDK | 获取实际 TCP、任务状态及控制状态。 |
| 力/扭矩 | 右腕 ForceSensor | 建立基线，提供连续进洞判定与安全限数据。 |
| 网络地址 | 工件 TOML 的 `arm_host`、`force_sensor_host` | 每个工件显式配置；工字工件当前分别为 `192.168.88.1`、`localhost`。 |

工件 TOML 的最小结构为：

```toml
schema_version = 2
workpiece_serial_number = 1
# 右臂控制器 IP 地址。
arm_host = "192.168.88.1"
# 右腕力/扭矩传感器的控制器 IP 地址。
force_sensor_host = "192.168.3.22"
initial_right_joints_deg = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
search_axis = "x"
search_direction = "negative"

[pre_search]
frame = "world"
```

工件配置仅支持 TOML。`initial_right_joints_deg` 必须是 7 个数值；`arm_host` 和 `force_sensor_host` 分别是右臂控制器与右腕力/扭矩传感器 IP，运行时会显式转换为 `--arm-host`、`--force-host`；`search_axis` 为 `x` 或 `y`；`search_direction` 为 `negative` 或 `positive`。工件 1 的 `pre_search.legs` 是从固定起始关节位到力基线点的动态世界坐标运动段；每段都配置 `name`、`axis`、`direction`、`distance_mm` 和 `speed_mm_s`。`press` 配置预压的轴、方向、首轮距离、重试增量和速度，`baseline` 配置稳定时间、采样数量和采样间隔。`stop_conditions` 中的字段会转换为内部搜索参数。

规划器在执行 OMPL 轨迹时会自行创建 SDK 连接，并通过环境变量 `PARM_ARM_HOST` 读取右臂地址。Module 9 会在每次真实放置开始前将该环境变量同步为 TOML 的 `arm_host`，因此规划回起始关节位和后续 ServoPose 使用同一台右臂控制器。

### 交接后到放置起点的轨迹缓存

Module 8 把右臂送到 `place-ready` 后，Module 9 的第一轮会从该位置前往 `initial_right_joints_deg`。此段的缓存路径为：

```text
trajactory/workpiece_<工件号>/place_ready_to_initial.json
```

逻辑与 Module 3 的 `trajactory` 一致：缓存命中时先读取右臂实际关节角，只有与缓存首点最大环绕误差不超过 **1.0°** 才通过 SDK `follow_trajectory` 回放；轨迹参数、目标关节角或规划配置变更时自动失效。首次没有缓存、缓存不匹配，或首点误差过大时，会用 `--initial-joint-planner`（当前工件配置为 `ompl`）在线规划，并在该请求显式带上 `allow_right_hand_tall_contact=true`：规划器会仅从该请求的派生场景移除 `left_table_positive_x_tall_safety_box` 整块碰撞体；全局场景以及其它环境、桌面和双臂碰撞检查保持不变。只有该轨迹真实执行完成且右臂稳定到目标后才原子写入缓存。

搜索失败后的后续重试从恢复抬升位返回起点，已不在 `place-ready`，因此不会错误回放这条缓存，而是继续在线规划。

所有工件相关的运动参数均由 TOML 提供：`search_motion` 配置连续搜索速度和加速度，`pre_search` 配置接近/预压的距离、速度和加速度，`retry` 配置失败后的最大尝试次数及抬升恢复运动，`second_hole` 配置第一洞和第二洞的距离、速度、加速度、停留时间及断连恢复参数；`post_entry.enabled` 控制进洞确认后是否先执行后退和下压。程序不再保留预接近的常量回退路径。

`[second_hole]` 控制第一洞确认后的可选路径。其中 `first_hole_settle_down_mm` 是第一洞确认后的 Z−下压量，`first_hole_lift_mm` 是下压后离开第一洞的 Z+抬升量；`lift_after_second_hole_mm` 则仅用于第二洞完成后的抬升。只有一个洞的工件应设定 `enabled = false`；此时程序记录 `second_hole_skipped` 后结束，不要求填写其余第二洞距离和速度字段。

## 3. 工作原理与操作流程

### 3.1 真实执行流程

1. 建立右臂和力传感器连接；若请求，先取消残留右臂任务。
2. 使用规划 `move_j` 回到示教关节姿态；该姿态刻意与放置点保留距离，因为现场设备区域需要被视为避障屏障，规划器无法可靠规划穿越该区域的路径。
3. 到达该安全关节地标后，使用 ServoPose 按 `pre_search.legs` 的世界坐标分段接近，绕开规划器无法处理的现场屏障区域，再到达力基线采集点。
4. 按 `pre_search.legs` 配置依次完成接近段，并进行本轮预压。
5. 等待机械臂稳定，采集多帧力/扭矩基线；读取实际 TCP 的 `xyz+rpy` 作为搜索中心。
6. 启动右臂 ServoPose，沿 `search_axis` 和 `search_direction` 生成从 `0` 到 `max_back_mm` 的单段连续轨迹。
7. 每个 ServoPose 周期检查连续力帧。有效进洞或安全接触限会停止 ServoPose 并中断搜索。
8. 确认第一洞后，以实际 TCP 为每一段起点执行第二洞固定路径：
   `Z−第一洞下压量 → Z+第一洞抬升量 → Y+偏移 → Z−第二洞下降 → Z+第二洞抬升量 → X−返回量`；所有距离均由工件 TOML 配置。
9. 无论成功、距离耗尽或异常，均确认停止 ServoPose、释放右臂锁、关闭传感器和右臂连接并写入结束日志。

`main()` 是工件级重试入口。若单次 `placement_main()` 返回搜索距离耗尽状态，入口按 TOML 的 `retry.max_attempts` 执行重试，并在下一次尝试前按 `retry` 表的参数调用 `lift_after_exhausted_search()` 做抬升恢复。

### 3.2 单段连续搜索

搜索轨迹由 `motion_primitives.move_continuous_linear_step()` 以 50 Hz 生成。它根据速度和加速度约束计算直线插补点，连续推送至 ServoPose；中间不做端点到位等待。`servo_worker.move_continuous_with_recovery()` 包装该运动：若检测到指定的 ServoPose 通信断连，会读取实际 TCP、重建控制连接、重新启动 ServoPose，并只执行尚未完成的剩余轨迹。

这样做的目标是避免按 1 mm 切段带来的反复加减速和停顿，同时仍保留每个控制周期的力监测。

### 3.3 力/扭矩判定算法

`force_monitoring.monitor_continuous_wrench()` 读取 `filtered_force` 和 `filtered_torque`，减去搜索起点的基线后得到 `dF`、`dT`。它使用以下判据：

| 判据 | 条件 | 结果 |
| --- | --- | --- |
| 安全接触限 | `|dF|` 或 `|dT|` 超过 `max_force_delta_n` / `max_torque_delta_nm` | 立即停止，并按接触进入处理。 |
| Fx 翻转 | 局部正 Fx 峰值在限定距离内转为足够负的 Fx，且连续确认若干帧 | 判定进入。可由工件配置启用。 |
| 合力/合扭矩连续卸载 | 高接触峰值之后，`|dF|`、`|dT|` 连续下降且同时低于绝对阈值，下降量满足阈值 | 判定进入。可由工件配置启用。 |
| Fx/Ty 卸载 | 相邻帧或多帧序列中 Fx/Ty 从接触状态同步下降 | 判定进入。 |
| 重复确认 | `release_events_required` 大于 1 时，需要多次独立有效卸载 | 首次仅记录，达到次数后才停止。 |

`min_entry_back_mm` 用于限制最小搜索距离：在未达到该距离前，不接受释放型进洞判定，降低初始接触或噪声导致误判的风险。

## 4. 参数调整

### 4.1 建议由工件配置调整的参数

优先修改 `config/workpiece_<n>.toml`，并先运行 `--print-command` 检查展开后的参数。

| 参数 | 调整影响 |
| --- | --- |
| `initial_right_joints_deg` | **工件首先需要调整的参数。**它是规划器可安全到达的关节地标，不是放置点；现场设备区域被当作避障屏障，后续由 ServoPose 分段接近放置点。变更前必须完成示教与碰撞验证。 |
| `pre_search.legs` | **工件其次需要调整。**按实际空间路径修改每段 `axis`、`direction`、`distance_mm`，使 ServoPose 从安全地标绕过现场屏障并到达第一洞的力基线点。 |
| `second_hole.first_hole_settle_down_mm` | **工件第三个需要调整。**第一洞确认后沿 Z− 的下压量；该工件使用两个洞，保持 `enabled = true`。 |
| `search_axis`、`search_direction` | 决定单段搜索的世界坐标方向。 |
| `entry_resultant_pre_force_n`、`entry_resultant_pre_torque_nm` | 形成卸载判据前所需的最小接触峰值。 |
| `entry_resultant_force_n`、`entry_resultant_torque_nm` | 卸载后允许的最大残余合力/合扭矩。 |
| `entry_resultant_force_drop_n`、`entry_resultant_torque_drop_nm` | 从接触峰值到卸载状态必须达到的最小降幅。 |
| `entry_resultant_release_frames` | 连续下降确认帧数；50 Hz 下，4 帧约为 80 ms。 |
| `entry_fx_reversal_*` | 启用或调整 Fx 正峰到负载翻转判据；当前工件 2 使用该判据。 |

### 4.2 加速方式

所有运动调速项都在工件 TOML 内。每次只提高一个类别的速度或加速度，先用 `--print-command` 核对，再在低速、空载和受监控条件下验证。接触搜索段优先保证力监测和停止判据，不建议为追求节拍大幅加速。

| 要加速的阶段 | TOML 参数 | 调整方式 |
| --- | --- | --- |
| 规划后到力基线点的接近段 | `pre_search.legs[*].speed_mm_s`、`pre_search.leg_accel_mm_s2` | 可优先提高；仍需保证现场屏障与工件的安全间隙。 |
| 预压 | `pre_search.press.speed_mm_s`、`pre_search.press.accel_mm_s2` | 仅小幅提高，避免预压时冲击工件。 |
| 连续进洞搜索 | `search_motion.speed_mm_s`、`search_motion.accel_mm_s2` | 小步提高并复核进洞判据是否仍能稳定触发。 |
| 第一/第二洞转场 | `second_hole.lateral_speed_mm_s`、`second_hole.descend_speed_mm_s`、`second_hole.linear_accel_mm_s2` | 无接触的上抬和平移可先调；进入第二洞的下降速度应保守。 |
| 搜索失败恢复 | `retry.speed_mm_s`、`retry.accel_mm_s2` | 仅影响失败后的恢复节拍，不影响正常成功路径。 |
| 固定停留 | `second_hole.pause_after_first_down_s`、`second_hole.pause_after_second_down_s` | 确认机构稳定后再缩短；不可直接设为负数。 |

`post_insertion.py` 同时提供了进洞后退让、Z−下降和 raw Fx 上升停止的辅助函数；当前主成功路径实际调用的是第一洞到第二洞路径，未调用退让/下降辅助函数。因此，仅修改 `post_entry_*` 参数不会改变当前主路径。

## 5. 输出、日志与数据分析

### 5.1 文件输出

独立运行 `place_hole_wiggle.py` 时，每次尝试生成一份 UTF-8 JSONL 运行日志：

```text
modules/module_9_right_place/logs/
  hole_a_workpiece_<n>_attempt_<attempt>_<timestamp>.log
```

外层重试还会生成会话日志：

```text
modules/module_9_right_place/logs/
  hole_a_workpiece_<n>_retry_session_<timestamp>.log
```

由新总控 `controller/main.py` 执行时，总控为整轮循环创建唯一运行目录：

```text
controller/runtime/<run_id>/
  run.log                  # 总控、子进程标准输出和错误输出的完整运行记录
  events.jsonl             # 总控结构化事件
  summary.json             # 总控耗时与成功率汇总
  latest_module_status.json # 最新模块状态
```

新总控调用本目录的 `place_hole_wiggle.py`。总控运行时应优先查看
`controller/runtime/<run_id>/logs/`；Module 9 自身仍会把尝试和重试会话日志写入
`modules/module_9_right_place/logs/`。两类日志用途不同：前者用于整轮系统关联，后者用于放置阶段的力数据和进洞判定分析。

常用事件包括：

| 事件 | 用途 |
| --- | --- |
| `run_started`、`run_finished` | 单次尝试边界和耗时。 |
| `force_baseline`、`search_centre_ready` | 力基线及实际搜索中心。 |
| `continuous_force_monitor_tick` | 每个有效连续监测帧的 `dF`、`dT`、范数和搜索距离。 |
| `continuous_force_monitor` | 限频后的控制台同频摘要。 |
| `insertion_release_event`、`insertion_decision` | 释放特征及最终进入/未进入决定。 |
| `servo_disconnect`、`servo_reconnected`、`servo_resume` | 通信恢复过程。 |
| `second_hole_leg_start`、`second_hole_leg_completed` | 第二洞每段路径的目标与实际末端位姿。 |

### 5.2 快速分析

查看进洞决定：

```bash
jq -c 'select(.event == "insertion_decision" or .event == "insertion_release_event")' \
  modules/module_9_right_place/logs/<attempt>.log
```

查看连续力监测帧：

```bash
jq -c 'select(.event == "continuous_force_monitor_tick") | {back_mm, delta_force_n, delta_torque_nm, delta_force_norm_n, delta_torque_norm_nm}' \
  modules/module_9_right_place/logs/<attempt>.log
```

当前目录没有维护中的独立绘图脚本。需要画曲线时，应以 `continuous_force_monitor_tick` 为数据源，横轴使用 `back_mm`，纵轴使用 `delta_force_norm_n`、`delta_torque_norm_nm` 或各分量 `delta_force_n` / `delta_torque_nm`；并在同一图中标注 `insertion_release_event` 与 `insertion_decision` 的搜索距离，以核对阈值是否符合实际接触过程。

## 6. 代码结构

| 文件 | 当前职责 |
| --- | --- |
| `place_hole_wiggle.py` | 新的独立调试入口：生命周期编排、设备连接、单段搜索、成功/失败处理与工件级重试。 |
| `place_hole_a_wiggle_new_config.py` | 旧的整合式入口；**当前总控仍调用此文件**，尚未切换到新的模块化入口。 |
| `cli_and_logging.py` | 内部 CLI、工件 TOML 校验、参数展开、JSONL 日志和控制状态快照。 |
| `placement_session.py` | 一次放置尝试的共享可变运行状态。 |
| `motion_primitives.py` | 关节回位、搜索前接近、ServoPose 直线插补。 |
| `force_monitoring.py` | 基线相对力计算、连续监测、接触/卸载特征维护。 |
| `insertion_detection.py` | 多次卸载事件计数与停止决策。 |
| `servo_worker.py` | 独立 ServoPose 进程、调用超时隔离、断线恢复和末端稳定确认。 |
| `post_insertion.py` | 第一洞到第二洞路径；同时保留进洞后退让/下降辅助函数。 |
| `right_arm_state.py` | 右臂任务与关节目标的状态确认。 |
| `wrench_utils.py`、`pose_utils.py`、`recovery.py` | 力向量、位姿命令和搜索耗尽后的恢复工具。 |

## 7. 现场操作与安全要求

1. 先使用 `--print-command` 核对工件配置和展开的内部参数。
2. 确认右臂、工件、夹具、孔位、坐标系和示教关节姿态均已验证。
3. 确认右腕力传感器零点和网络连接正常，且操作区域无人接触。
4. 真实运动必须同时使用 `--live --confirm-live`；不要将该组合用于未经验证的参数或工件。
5. 若上一次运行异常中断，可在确认机械臂静止后使用 `--reset-right-task-before-servo` 清理右臂残留任务。
6. 每次真实运行后检查 JSONL 中的 `insertion_decision`、`run_finished` 和 ServoPose 停止事件，再决定是否调整阈值或继续下一工件。
