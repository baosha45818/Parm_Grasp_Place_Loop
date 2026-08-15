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

主入口是 [`place_hole_wiggle.py`](place_hole_wiggle.py)。外部调用必须提供工件编号：

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
| 工件配置 | `config/workpiece_1.json`、`config/workpiece_2.json` | 提供工件编号、7 轴示教关节角、搜索轴/方向及专属进洞阈值。 |
| 规划场景 | `config.hole_a_first_live_fast_return.json` | 用于默认的 `move_j` 回示教起点及避碰规划。 |
| 右臂状态 | 右臂 SDK | 获取实际 TCP、任务状态及控制状态。 |
| 力/扭矩 | 右腕 ForceSensor | 建立基线，提供连续进洞判定与安全限数据。 |
| 网络地址 | `PARM_ARM_HOST`、`PARM_FORCE_SENSOR_HOST` 或内部参数 | 默认分别为 `192.168.3.11`、`192.168.3.22`。 |

工件 JSON 的最小结构为：

```json
{
  "schema_version": 1,
  "workpiece_serial_number": 1,
  "initial_right_joints_deg": ["J1", "J2", "J3", "J4", "J5", "J6", "J7"],
  "search_axis": "x",
  "search_direction": "negative",
  "stop_conditions": {}
}
```

其中 `initial_right_joints_deg` 必须是 7 个数值；`search_axis` 为 `x` 或 `y`；`search_direction` 为 `negative` 或 `positive`。`stop_conditions` 中的字段会转换为内部搜索参数。

## 3. 工作原理与操作流程

### 3.1 真实执行流程

1. 建立右臂和力传感器连接；若请求，先取消残留右臂任务。
2. 使用规划 `move_j` 回到示教关节姿态；仅在验证过的近距离场景才应改用 `joint-servo`。
3. 依次完成世界 `Z−` 初始下降、世界 `X+` 前进、世界 `Z−` 接近，并进行本轮预压。
4. 等待机械臂稳定，采集多帧力/扭矩基线；读取实际 TCP 的 `xyz+rpy` 作为搜索中心。
5. 启动右臂 ServoPose，沿 `search_axis` 和 `search_direction` 生成从 `0` 到 `max_back_mm` 的单段连续轨迹。
6. 每个 ServoPose 周期检查连续力帧。有效进洞或安全接触限会停止 ServoPose 并中断搜索。
7. 确认第一洞后，以实际 TCP 为每一段起点执行第二洞固定路径：
   `Z−10 mm → Z+抬升 → Y+偏移 → Z−第二洞下降 → Z+抬升 → X−110 mm`。
8. 无论成功、距离耗尽或异常，均确认停止 ServoPose、释放右臂锁、关闭传感器和右臂连接并写入结束日志。

`main()` 是工件级重试入口。若单次 `placement_main()` 返回搜索距离耗尽状态，入口最多执行 `MAX_SEARCH_ATTEMPTS` 次尝试，并在下一次尝试前调用 `lift_after_exhausted_search()` 做抬升恢复。

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

优先修改 `config/workpiece_<n>.json`，并先运行 `--print-command` 检查展开后的参数。

| 参数 | 调整影响 |
| --- | --- |
| `initial_right_joints_deg` | 搜索前的关节安全地标；变更前必须完成示教与碰撞验证。 |
| `search_axis`、`search_direction` | 决定单段搜索的世界坐标方向。 |
| `entry_resultant_pre_force_n`、`entry_resultant_pre_torque_nm` | 形成卸载判据前所需的最小接触峰值。 |
| `entry_resultant_force_n`、`entry_resultant_torque_nm` | 卸载后允许的最大残余合力/合扭矩。 |
| `entry_resultant_force_drop_n`、`entry_resultant_torque_drop_nm` | 从接触峰值到卸载状态必须达到的最小降幅。 |
| `entry_resultant_release_frames` | 连续下降确认帧数；50 Hz 下，4 帧约为 80 ms。 |
| `entry_fx_reversal_*` | 启用或调整 Fx 正峰到负载翻转判据；当前工件 2 使用该判据。 |

### 4.2 代码级运行参数

以下默认值位于 `cli_and_logging.py` 的内部参数解析及 `profile_argv()`，外层 `--workpiece` 入口不会直接暴露它们；改动前需重新进行现场验证。

| 参数类别 | 典型参数 | 含义 |
| --- | --- | --- |
| 搜索轨迹 | `fast_linear_speed_mm_s`、`fast_linear_accel_mm_s2`、`max_back_mm` | 单段直线的速度、加速度和最大距离。 |
| 力监测 | `continuous_force_monitor_hz`、`continuous_force_output_hz`、`continuous_release_frames` | 监测频率、控制台输出频率和多帧卸载确认长度。 |
| 安全边界 | `max_force_delta_n`、`max_torque_delta_nm`、`min_entry_back_mm` | 接触硬限和最短允许判定距离。 |
| 搜索前接近 | `pre_search_initial_descend_mm`、`pre_search_forward_mm`、`pre_search_descend_mm`、`pre_search_press_mm` | 从示教姿态到搜索起点的世界坐标接近路径。 |
| 第二洞路径 | `second_hole_lift_mm`、`second_hole_y_offset_mm`、`second_hole_descend_mm` 及对应速度 | 第一洞确认后的固定多段轨迹。 |

`post_insertion.py` 同时提供了进洞后退让、Z−下降和 raw Fx 上升停止的辅助函数；当前主成功路径实际调用的是第一洞到第二洞路径，未调用退让/下降辅助函数。因此，仅修改 `post_entry_*` 参数不会改变当前主路径。

## 5. 输出、日志与数据分析

### 5.1 文件输出

每次尝试生成一份 UTF-8 JSONL 运行日志：

```text
modules/module_9_right_place/logs/
  hole_a_workpiece_<n>_attempt_<attempt>_<timestamp>.log
```

外层重试还会生成会话日志：

```text
modules/module_9_right_place/logs/
  hole_a_workpiece_<n>_retry_session_<timestamp>.log
```

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
| `place_hole_wiggle.py` | 生命周期编排、设备连接、单段搜索、成功/失败处理与工件级重试。 |
| `cli_and_logging.py` | 内部 CLI、工件 JSON 校验、参数展开、JSONL 日志和控制状态快照。 |
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
