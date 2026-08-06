# Version5 双相机独立抓取系统

## 1. 输入与输出

### 1.1 输入

程序运行前需要准备以下文件和设备：

| 输入 | 路径或地址 | 用途 |
|---|---|---|
| GraspNet 模型 | `/home/dainanjun/qiuguolin/graspnet/checkpoint/checkpoint-rs.tar` | 默认执行 D405、Femto 两次独立推理 |
| D405 腕部标定 | `deploy/calibration/output/wrist1/calibration_wrist1.json` | 将 D405 点云转换到 `arm1_base` |
| Femto Bolt 激光外参 | `deploy/calibration/output/laser/calibration_laser.json` | 将 Femto 点云转换到 `arm1_base` |
| 夹爪 TCP 标定 | `deploy/calibration/output/tcp_grasp_tcp/calibration_translation2.json` | 将预测抓取中心转换为 EEF 位姿 |
| D405 相机 | 序列号 `260322270242` | 腕部点云采集 |
| Femto Bolt 相机 | 序列号 `CL8L56301LK` | 外部点云采集 |
| arm1 控制器 | `192.168.88.1` | 观察位姿、笛卡尔运动和放置运动 |
| 夹爪服务 | `localhost`，hand1 | 开合夹爪 |

标定文件必须满足以下条件：Femto 标定的 `status` 必须为 `ACCEPTED`；夹爪 TCP 标定的 `point_is_grasp_tcp` 必须为 `true`。

### 1.2 输出

默认输出目录：

```text
/home/dainanjun/qiuguolin/graspnet/deploy/version5/output/YYYYMMDD_HHMMSS_microseconds/
```

主要输出结构：

```text
<run>/
├── d405_inference/<run>/d405/
│   ├── point_cloud.npy
│   ├── color.png
│   ├── depth.png
│   ├── grasps.npy
│   └── meta.json
├── femto_inference/<run>/femto_bolt/
│   ├── point_cloud.npy
│   ├── color.png
│   ├── depth.png
│   ├── grasps.npy
│   └── meta.json
├── fused_point_cloud.npy
├── fused_point_colors.npy
├── fused_cloud_summary.json
├── fused_inference/<run>/merged/
│   ├── point_cloud.npy
│   ├── grasps.npy
│   └── meta.json
├── all_source_grasp_filter.json
└── grasp_execution_plan.json
```

输出含义：

| 输出 | 含义 |
|---|---|
| `d405_inference` | D405 单独采集和一次 GraspNet 推理结果 |
| `femto_inference` | Femto Bolt 单独采集和一次 GraspNet 推理结果 |
| `fused_point_cloud.npy` | 两路点云统一到 `arm1_base` 后融合、清理、降采样的结果 |
| `fused_point_colors.npy` | 与融合点云逐点对应的 RGB 颜色，供 Open3D 彩色预览使用 |
| `fused_inference` | 仅传入 `--enable-fused-inference` 时生成的实验性第三次推理结果 |
| `fused_cloud_summary.json` | D405、Femto 和融合点云的点数统计 |
| `all_source_grasp_filter.json` | 默认两路独立候选的宽度、姿态、桌面、XY 和分来源筛选统计 |
| `ranked_grasp_candidates.json` | 合格候选按分数排序；当前 SDK 无 IK 查询接口，因此明确记录 IK 未检查 |
| `grasp_execution_plan.json` | 最终抓取 TCP、EEF 预抓取位姿、下降位姿和放置参数 |

`point_cloud.npy` 在单相机目录中仍是相机坐标；融合前程序根据对应 `meta.json` 的 `T_base_camera` 将其转换到 `arm1_base`。第三次 GraspNet 推理会临时把融合点云转回 D405 相机坐标系，推理后再把抓取姿态转回 `arm1_base`，避免把任意朝向的基坐标点云直接送入按相机坐标训练的网络。

## 2. 系统流程

```text
启动程序
   │
   ├─ 检查模型、wrist、laser、TCP 标定
   │
   ├─ 读取 arm1 状态
   │    └─ 确认后 Planner PTP 到 D405 观察位姿，再由 waypoint 执行接口执行
   │
   ├─ D405 采集多帧深度
   │    └─ D405 点云转换到 arm1_base
   │    └─ D405 GraspNet 推理
   │
   ├─ Femto Bolt 采集彩色/深度
   │    └─ Femto 点云转换到 arm1_base
   │    └─ Femto GraspNet 推理
   │
   ├─ 默认不执行融合 GraspNet
   │    └─ --enable-fused-inference 才启用实验性第三次推理
   │
   ├─ 汇总 D405、Femto 两路独立抓取候选
   ├─ 两路候选统一筛选并按分数排序
   │    ├─ 抓取宽度
   │    ├─ 向下接近轴与 XOY 平面夹角
   │    ├─ 桌面高度
   │    └─ 可选桌面 XY 边界
   │
   ├─ 保存抓取计划并显示 Open3D
   │
   └─ 输入 y/yes 后执行
        ├─ 设置夹爪开度
        ├─ 移动到预抓取位姿
        ├─ 垂直下降
        ├─ 闭合夹爪
        ├─ Planner PTP 抬升
        ├─ Planner PTP 到放置位姿
        └─ 打开夹爪释放
```

## 3. 接口调用

### 3.1 相机接口

程序通过项目的相机采集与 GraspNet 推理入口调用相机和模型。

D405 使用 `robot_camera.D405Camera` 和 RealSense 接口。参数包括序列号、帧数、深度后处理、孔洞填充、ROI 和 XYZ 工作空间范围。多帧深度使用逐像素中值合成。

Femto Bolt 使用 `robot_camera.FemtoBoltCamera` 和 `pyorbbecsdk` 接口。参数包括序列号、彩色/深度分辨率以及 XYZ 工作空间范围。

Femto ROI 可通过以下工具在对齐后的彩色图上选择：

```bash
.venv/bin/python deploy/tools/roi_graspnet_laser.py --display :1
```

将工具输出的 `--femto-roi X Y W H` 加入 Version5 命令。D405 继续使用独立的 `--roi X Y W H`。

两路 ROI 都是必填项。程序会核对相机输出中的 `roi_xywh`，任何一路没有实际应用 ROI 都会终止，不会退化为全视野融合。

两台相机分别调用一次相机推理命令；融合点云通过 `--merged_point_cloud` 再调用一次推理命令。

### 3.2 机械臂接口

`parm_sdk.Arm` 只保留用于读取状态/EEF 位姿；所有机械臂运动都由本机 Planner 与 plucky 执行接口完成。

观察位姿使用 `Planner.plan_joint_trajectory(group="left_arm", planner="ptp")`：关节角单位为度，目标为 7 个关节角。规划结果统一提交至 `POST /middle/waypoint/execute`，不直接调用 `Arm.move_j`。

抓取和下降调用：

```python
planner.plan_cart_trajectory(
    group="left_arm", planner="lin",
    frame_id="body_base", ik_frame="left_hand_tool_end_link",
    position=[x, y, z], orientation=[qx, qy, qz, qw],
    max_vel=0.1, max_acc=0.1,
)
```

旧流程的 `[x, y, z, rx, ry, rz]` 轴角位姿先从 `arm1_base` 经已验收的
`tool/calibration/left_wrist_d405/planner_tf_bridge.json` 转到 Planner 的
`body_base`，再转换为四元数提交；运行时还会查询 Planner TF 的
`left_arm_base → body_base`。桥接标定未验收或 TF 不存在时，笛卡尔规划会拒绝执行。

### 3.3 夹爪接口

夹爪接口为 `parm_sdk.Gripper`，服务器默认 `localhost`，arm1 默认使用 hand1：

```python
gripper.move(
    hand=1,
    position=position_percent,
    force=force_percent,
    speed=speed_percent_or_none,
    mode="sync",
    preempt=True,
)
```

夹爪位置、力和速度都是百分比。`--gripper-speed` 合法范围是 `1~100`；对于 AG 夹爪建议不传，让 SDK 使用默认速度。`--gripper-open-width-m 0.078` 表示 position=100 对应约 78 mm 物理开度。

## 4. 关键参数

### 4.1 相机和标定

`--d405-serial`、`--femto-serial` 指定相机序列号；`--wrist-calibration`、`--laser-calibration`、`--tcp-translation-calibration` 指定三类标定文件。

`--d405-frame-count` 指定 D405 深度帧数；`--d405-postprocess` 和 `--d405-hole-filling` 控制深度后处理；`--roi X Y W H` 指定 D405 推理区域。

`--d405-z-min/max`、`--d405-x-abs-max`、`--d405-y-abs-max` 控制 D405 工作空间；`--femto-z-min/max`、`--femto-x-abs-max`、`--femto-y-abs-max` 控制 Femto 工作空间。

### 4.2 推理和融合

`--checkpoint-path` 指定模型；`--num-point` 指定输入点数；`--num-view` 指定 GraspNet 视角数量；`--top-k` 指定可视化候选数量；`--fuse-voxel-size-m` 指定融合体素大小。融合前程序分别估计两路桌面高度，并由 `--fusion-object-min-height-m`、`--fusion-object-max-height-m` 提取桌面上方工件点；`--max-grasp-object-distance-m` 防止最终抓取中心落在远离工件点云的桌面或背景上。

`--collision-thresh`、`--collision-voxel-size`、`--collision-approach-dist` 控制抓取碰撞筛选；`--collision-table-thickness`、`--collision-table-plane-dist`、`--collision-table-min-inlier-ratio`、`--collision-table-min-normal-z` 控制桌面碰撞模型。

`--min-grasp-approach-angle-deg` 控制向下接近轴与 XOY 平面的最小夹角；默认 `65` 度，只保留更接近垂直下抓的候选。`--table-clearance-m`、`--table-height-percentile`、`--table-xy-bounds`、`--table-edge-margin-m` 控制融合结果的桌面安全筛选；`--disable-tabletop-filter` 可关闭桌面高度和 XY 边界筛选。

### 4.3 运动和安全

`--observation-joints-deg` 是 D405 观察位姿，单位度；`--movej-speed` 是规划器的 `max_vel/max_acc` 比例。

`--pregrasp-height-m`、`--cartesian-speed-factor` 控制规划的末端抓取轨迹；`--pregrasp-duration-s`、`--descend-duration-s`、`--servo-hz` 仅为旧命令兼容参数，轨迹时间由 Planner 决定；`--max-grasp-distance-m` 限制从当前 EEF 到预抓取位姿的最大距离。

`--postgrasp-lift-joints-deg` 和 `--place-joints-deg` 是抓取后抬升、放置的 7 关节位姿；`--postgrasp-movej-speed` 控制这两次 Planner PTP 的速度比例。

## 5. 完整启动命令

下面命令会真实移动到观察位姿，并在确认后执行抓取、放置和释放。运行前请人工确认机械臂周围无障碍物。

```bash
cd /home/dainanjun/qiuguolin/graspnet

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  deploy/version5/run_dual_camera_fused_graspnet.py \
  --python-bin /home/dainanjun/qiuguolin/graspnet/.venv/bin/python \
  --checkpoint-path /home/dainanjun/qiuguolin/graspnet/checkpoint/checkpoint-rs.tar \
  --wrist-calibration /home/dainanjun/qiuguolin/graspnet/deploy/calibration/output/wrist1/calibration_wrist1.json \
  --laser-calibration /home/dainanjun/qiuguolin/graspnet/deploy/calibration/output/laser/calibration_laser.json \
  --tcp-translation-calibration /home/dainanjun/qiuguolin/graspnet/deploy/calibration/output/tcp_grasp_tcp/calibration_translation2.json \
  --arm-host 192.168.88.1 \
  --arm-id 1 \
  --move-observation \
  --movej-speed 0.1 \
  --observation-joints-deg -6.25 -21.348 12.599 78.268 4.039 56.382 -11.374 \
  --d405-serial 260322270242 \
  --d405-frame-count 5 \
  --d405-postprocess \
  --d405-hole-filling \
  --d405-z-min 0.10 \
  --d405-z-max 1.20 \
  --d405-x-abs-max 0.80 \
  --d405-y-abs-max 0.80 \
  --femto-serial CL8L56301LK \
  --femto-color-width 1280 \
  --femto-color-height 720 \
  --femto-depth-width 640 \
  --femto-depth-height 480 \
  --femto-z-min 0.20 \
  --femto-z-max 2.00 \
  --femto-x-abs-max 1.20 \
  --femto-y-abs-max 1.20 \
  --roi 0 0 512 336 \
  --num-point 50000 \
  --num-view 300 \
  --collision-thresh 0.3 \
  --collision-voxel-size 0.005 \
  --collision-approach-dist 0.01 \
  --collision-table-thickness 0.0 \
  --collision-table-plane-dist 0.005 \
  --collision-table-min-inlier-ratio 0.15 \
  --collision-table-min-normal-z 0.5 \
  --fuse-voxel-size-m 0.004 \
  --table-clearance-m 0.015 \
  --table-height-percentile 5.0 \
  --table-edge-margin-m 0.02 \
  --top-k 50 \
  --min-grasp-approach-angle-deg 65 \
  --pregrasp-height-m 0.02 \
  --gripper-host localhost \
  --gripper-hand 1 \
  --gripper-open-width-m 0.078 \
  --gripper-width-margin-m 0.005 \
  --gripper-force 60 \
  --gripper-close-force 100 \
  --gripper-command-retries 3 \
  --gripper-retry-delay-s 1.0 \
  --postgrasp-lift-joints-deg -6.288 -38.805 12.556 72.427 4.059 67.331 -11.391 \
  --place-joints-deg -62.016 37.312 -23.766 55.214 1.200 -0.677 78.087 \
  --postgrasp-movej-speed 0.1 \
  --pregrasp-duration-s 3 \
  --descend-duration-s 2 \
  --cartesian-speed-factor 1.0 \
  --servo-hz 50 \
  --pose-settle-timeout-s 5 \
  --pose-position-tolerance-m 0.01 \
  --pose-orientation-tolerance-deg 10 \
  --max-grasp-distance-m 0.80 \
  --display :1 \
  --execute-grasp \
  --confirm-real-grasp
```

`--gripper-speed` 没有写入完整命令，因为 AG 夹爪建议使用 SDK 默认速度。如需显式指定，可在命令末尾增加 `--gripper-speed 50`，合法范围为 `1~100`。

## 6. 安全运行方式

仅做采集、推理和计划生成时使用：

```bash
cd /home/dainanjun/qiuguolin/graspnet
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  deploy/version5/run_dual_camera_fused_graspnet.py \
  --skip-observation-move \
  --no-auto-open-gripper \
  --no-vis
```

该模式不发送观察位姿运动、夹爪动作或抓取动作，但会连接相机完成两次独立推理并保存输出。
