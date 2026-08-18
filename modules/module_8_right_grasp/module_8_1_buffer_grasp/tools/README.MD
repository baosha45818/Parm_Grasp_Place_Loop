# 工具脚本说明

本目录存放用于单独调试、示教和验证机械臂运动的辅助脚本。它们不是
`buffer_grasp.py` 的正常抓取执行链路；直接带 `--execute` 运行会向真实设备发送命令，必须先确认现场安全、坐标系和目标臂无误。

| 文件 | 用途 | 主要能力 |
| --- | --- | --- |
| `arm_head_teach.py` | 人工示教与云台调试 | 键盘控制手臂 Free 示教、保存关节角和规划器 TCP 位姿、切换夹爪、控制云台俯仰/偏航，以及清除 `ERROR` 后恢复使能。保存结果写入 `tools/saved_poses/`。 |
| `move_by_joints.py` | 按关节角移动单臂 | 读取 7 个关节角，或从保存位姿 JSON 中读取关节角，调用 `move_j` 控制左/右臂。该文件也提供位姿转换、SDK 导入、状态校验和 ServoPose 等共享函数。 |
| `move_by_eef_pose.py` | 按末端笛卡尔位姿规划移动单臂 | 接收 `body_base` 下的 XYZ + RPY，或读取保存位姿/工件位姿 JSON，通过 Planner 生成 `lin` 或 `ptp` 轨迹并执行；可用 `--planar-xy-yaw` 锁定 Z、roll、pitch，仅改变 XY 与 yaw。 |

## 使用关系

```text
arm_head_teach.py
  └─ 独立示教工具；仅依赖 parm_sdk 与 Arm / Planner / Head / Gripper 服务

move_by_joints.py
  └─ 独立关节运动工具；也向 move_by_eef_pose.py 提供共享函数

move_by_eef_pose.py
  ├─ 使用同目录 move_by_joints.py 的位姿、SDK 和状态检查函数
  └─ 使用项目根目录 tool/planner_task_retry.py 的规划任务重试函数
```

三个脚本运行时都需要可用的 `parm_sdk` 和相应的硬件/服务。`arm_head_teach.py` 是交互式脚本，还要求在 Linux/Unix 的真实终端中运行；不适用于无 TTY 的后台任务。

## `arm_head_teach.py` 使用方法

此代码方便手拖动机械臂、开闭夹爪、调整云台角度。

在项目根目录运行。必须指定要示教的手臂：`1` 为左臂、`2` 为右臂、`0` 为双臂。

```bash
python tools/arm_head_teach.py --arm-id 2
```

可按现场服务地址覆盖默认值：

```bash
python tools/arm_head_teach.py --arm-id 2 \
  --arm-host 192.168.88.1 \
  --planner-host localhost \
  --head-host localhost \
  --gripper-host localhost \
  --pitch-step 0.5 --yaw-step 0.5
```

启动后不需要按回车，保持终端焦点并直接按键：

| 按键 | 手臂模式中的动作 |
| --- | --- |
| `D` | 对所选手臂进入 Free 示教；按住踏板约 0.5 秒后可拖动。 |
| `S` | 保存当前关节角、规划器 `body_base → *_hand_transfer_base` TCP 位姿，以及驱动原始末端位姿。 |
| `空格` | 切换所选夹爪开/闭；双臂模式会同时控制两个夹爪。 |
| `X` | 停止所选手臂的示教。 |
| `E` | 仅当系统处于 `ERROR` 时尝试 `clear_error()`，恢复 `RUNNING` 后重新使能。不能解除 `ESTOP`。 |
| `Y` | 进入云台模式；在云台模式中再次按 `Y` 返回手臂模式。 |
| `Q` 或 `Ctrl+C` | 退出；退出时会尝试停止示教并关闭连接。 |

云台模式下使用 `I` / `K` 增加或减小 pitch，`J` / `L` 增加或减小 yaw；默认步长均为 `0.5°`，限制范围为 pitch `[-45°, 45°]`、yaw `[-60°, 60°]`。

按 `S` 后生成的 JSON 位姿文件保存到：

```text
tools/saved_poses/arm_pose_YYYYMMDD_HHMMSS_ffffff.json
```

示教操作会直接改变机器人状态。开始拖动前确认踏板、急停、周围人员和碰撞空间均安全；保存前先松开踏板并等待机械臂稳定。

## 常用预览方式

先不加 `--execute`，只检查参数和目标：

```bash
python tools/move_by_joints.py --arm-id 2 --joint-positions-deg 0 0 0 0 0 0 0
python tools/move_by_eef_pose.py --arm-id 2 --target 0.3 0.0 0.3 0.0 0.0 0.0
python tools/arm_head_teach.py --help
```

确认轨迹、坐标系和碰撞风险后，才为前两个运动命令附加 `--execute`。`arm_head_teach.py` 启动后立即连接设备，并按终端按键执行操作。
