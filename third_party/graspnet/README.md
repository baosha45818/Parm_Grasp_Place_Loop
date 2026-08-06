# grasp-legacy 双相机机器人抓取系统

本项目用于控制 arm1 完成工件观察、双相机点云采集、GraspNet 抓取姿态预测、点云融合、抓取执行和放置。系统使用腕部 RealSense D405 与固定奥比中光 Femto Bolt，机械臂和夹爪通过 `parm_sdk` 控制。

当前可执行主流程位于 `deploy/version5`。`deploy/version6` 是抓取后观察工件姿态并正面朝上放置的后续框架，目前默认不发送真实运动命令。

## 1. 输入与输出

### 1.1 输入

| 输入 | 默认路径或地址 | 用途 |
|---|---|---|
| GraspNet 权重 | `checkpoint/checkpoint-rs.tar` | 单相机与融合点云抓取推理 |
| D405 腕部标定 | `deploy/calibration/output/wrist1/calibration_wrist1.json` | D405 点云转换到 `arm1_base` |
| Femto Bolt 外参 | `deploy/calibration/output/laser/calibration_laser.json` | 固定相机点云转换到 `arm1_base` |
| 夹爪 TCP 标定 | `deploy/calibration/output/tcp_grasp_tcp/calibration_translation2.json` | 抓取中心转换为 EEF 目标位姿 |
| RealSense D405 | 序列号 `260322270242` | 腕部 RGB-D 和点云采集 |
| Femto Bolt | 序列号 `CL8L56301LK` | 外部 RGB-D 和点云采集 |
| arm1 控制器 | `192.168.88.1`，arm1 | 关节运动和末端位姿控制 |
| 夹爪服务 | `localhost`，hand1 | 设置开度、力和速度 |

模型权重和 Python 虚拟环境不在仓库中，需要在部署机器上单独准备。仓库中的标定 JSON 只适用于当前设备；更换相机、安装位置、机械臂或夹爪后必须重新标定。

### 1.2 输出

Version5 默认输出到：

```text
deploy/version5/output/YYYYMMDD_HHMMSS_microseconds/
```

主要文件如下：

| 输出 | 内容 |
|---|---|
| `d405_inference/` | D405 彩色图、深度图、点云和独立抓取结果 |
| `femto_inference/` | Femto Bolt 彩色图、深度图、点云和独立抓取结果 |
| `fused_point_cloud.npy` | 转换到 `arm1_base` 后的双相机融合点云 |
| `fused_point_colors.npy` | 与融合点云逐点对应的 RGB 颜色 |
| `fused_cloud_summary.json` | 两路输入点数及融合点数统计 |
| `fused_inference/` | 融合点云的第三次 GraspNet 推理结果 |
| `all_source_grasp_filter.json` | 三路候选的宽度、姿态、桌面、XY 和来源筛选统计 |
| `grasp_execution_plan.json` | 抓取 TCP、预抓取位姿、下降位姿和执行参数 |

单相机目录中的 `point_cloud.npy` 使用相机坐标系。融合前，程序根据 `meta.json` 中的 `T_base_camera` 将点云转换到 `arm1_base`。

## 2. 系统流程

```text
启动
  │
  ├─ 检查模型、相机和三份标定文件
  ├─ 读取 arm1 状态并确认观察位姿
  │
  ├─ D405 多帧采集
  │    ├─ 深度中值融合与后处理
  │    ├─ 点云转换到 arm1_base
  │    └─ 第一次 GraspNet 推理
  │
  ├─ Femto Bolt 采集
  │    ├─ 点云转换到 arm1_base
  │    └─ 第二次 GraspNet 推理
  │
  ├─ 双相机点云融合与体素降采样
  ├─ 第三次 GraspNet 推理
  ├─ 抓取宽度、姿态、桌面和边界筛选
  ├─ 保存执行计划并显示 Open3D 结果
  │
  └─ 真实执行需再次确认
       ├─ 打开夹爪
       ├─ 移动到预抓取位姿
       ├─ 垂直下降并闭合夹爪
       ├─ 抬升并移动到放置位姿
       └─ 打开夹爪释放工件
```

三次 GraspNet 结果全部转换到 `arm1_base` 并进入统一候选池。每一路先使用自己的点云完成碰撞检测，再统一执行宽度、姿态、融合桌面和 XY 边界筛选；机械臂从三路合格候选中执行最高分结果，执行计划会记录 `selected_grasp_source`。

## 3. 接口调用

### 3.1 相机接口

D405 使用 `utils.robot_camera.D405Camera` 和 `pyrealsense2`，支持多帧深度中值、深度后处理、孔洞填充、ROI 和 XYZ 工作空间裁剪。

Femto Bolt 使用 `utils.robot_camera.FemtoBoltCamera` 和 `pyorbbecsdk`。彩色和深度分辨率不同时，程序按照相机内外参完成对齐和点云着色。

奥比中光 ROI 使用对齐后的彩色图框选：

```bash
.venv/bin/python deploy/tools/roi_graspnet_laser.py --display :1
```

将输出的 `--femto-roi X Y W H` 加入 Version5 命令。D405 使用独立的 `--roi X Y W H`。

### 3.2 机械臂接口

机械臂使用 `/home/dainanjun/Public/data_collector` 提供的 `parm_sdk.Arm`：

- 所有机械臂关节动作均通过 `Planner.plan_joint_trajectory(...)` 规划，再提交到
  `/middle/waypoint/execute` 执行；关节单位为度。
- 预抓取和下降通过 `Planner.plan_cart_trajectory(..., planner="lin")` 规划，再提交同一执行接口；不再调用 SDK servo 接口。
- 末端位姿格式为 `[x, y, z, rx, ry, rz]`；位置单位为米，旋转为轴角弧度。
- 运动前检查系统状态、使能状态、关节范围、最大移动距离和最终到位误差。

### 3.3 夹爪接口

夹爪使用 `parm_sdk.Gripper.move()`。位置、力和速度均使用百分比：

- `--gripper-open-width-m 0.078` 表示 `position=100` 对应约 78 mm 开度。
- `--gripper-width-margin-m` 在预测宽度上增加安全余量。
- `--gripper-speed` 合法范围为 `1~100`；AG 夹爪建议不传，使用 SDK 默认速度。

## 4. 坐标系

D405 点云使用：

```text
T_base_d405 = T_base_eef_capture @ T_eef_wrist1
```

固定 Femto Bolt 使用：

```text
T_base_femto = T_base_laser
```

融合 GraspNet 输出抓取中心后，程序结合 `T_eef_grasp_tcp` 计算机械臂 EEF 目标。所有真实运动目标最终都位于 `arm1_base` 坐标系。

## 5. 安全启动

仅采集、推理并生成计划，不发送观察位姿、夹爪或抓取动作：

```bash
cd /home/dainanjun/qiuguolin/graspnet

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  deploy/version5/run_dual_camera_fused_graspnet.py \
  --skip-observation-move \
  --no-auto-open-gripper \
  --no-vis
```

该命令仍会连接两台相机并执行三次 GraspNet 推理，但不会控制机械臂和夹爪。

完整真实抓取命令见 [deploy/version5/command.txt](deploy/version5/command.txt)。真实执行必须包含 `--execute-grasp --confirm-real-grasp`，关闭 Open3D 窗口后还需要在终端输入 `y` 或 `yes`。执行前必须确认工作空间无人员和障碍物，并核对观察、抬升、放置关节角及 TCP 标定。

## 6. 目录说明

| 目录 | 内容 |
|---|---|
| `deploy/calibration/` | 腕部相机、固定相机、TCP 平移和方向标定 |
| `deploy/tools/` | 关节示教、ROI 和辅助工具 |
| `deploy/version1/` | 双相机独立采集与推理 |
| `deploy/version2/` | 观察运动和多视角融合实验 |
| `deploy/version3/` | D405 单相机抓取 |
| `deploy/version4/` | D405 单相机抓取与固定关节放置 |
| `deploy/version5/` | D405、Femto 和融合点云三次推理及抓取放置 |
| `deploy/version6/` | 抓取后工件姿态估计与正面朝上放置框架 |

各版本均有中文 README。Version5 的完整参数、输入输出和接口说明见 [deploy/README.md](deploy/README.md)。

## 7. 上游依赖

本项目的抓取网络基于 [GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)。训练、数据集评测和原始论文说明请查阅上游仓库；本 README 只描述当前机器人部署流程。
