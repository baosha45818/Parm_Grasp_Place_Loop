# Version5：D405 + 奥比中光 Femto Bolt 双相机独立抓取

Version5 默认执行两次相互独立的 GraspNet：

1. D405 单独采集、标定、推理。
2. Femto Bolt 单独采集、标定、推理。
3. 两路抓取转换到 `arm1_base`，统一筛选并按分数择优。

默认候选池只包含 D405 与 Femto 两路独立预测。融合和第三次 GraspNet 代码仍保留，但必须显式传入 `--enable-fused-inference` 才会启用，用于相机重新标定后的实验。

## 一、执行流程

程序依次检查标定文件、询问是否移动到 D405 观察位姿、采集 D405、采集 Femto Bolt、合并两路独立候选、统一筛选并生成执行计划。真实执行时还会经过 Open3D 预览和终端 `y/yes` 确认。

当前 `parm_sdk.Arm` 没有纯查询式 IK 接口。程序会输出 `ranked_grasp_candidates.json`，按得分从高到低记录候选，并明确标记 `ik_status: not_checked_sdk_has_no_query_api`；不会通过真实运动试探 IK。

融合前分别从 D405 和 Femto 点云估计桌面高度，只保留各自桌面上方的工件点，避免外参残差形成双层桌面。融合结果筛选包括夹爪宽度、夹爪向下接近轴角度、抓取中心到工件点云的距离、桌面高度，以及可选的桌面 XY 边界。`--min-grasp-approach-angle-deg` 默认是 `65` 度：接近轴必须朝向基坐标系 `-Z`，且与桌面的夹角不小于该值。该参数只筛选候选，不会主动旋转 EEF。

## 二、坐标系和标定

最终抓取结果必须位于 `arm1_base` 坐标系。

### D405 腕部相机

D405 采集时读取 arm1 当前 EEF 位姿，使用 `T_base_d405 = T_base_eef_capture @ T_eef_wrist1`。腕部标定文件是 `deploy/calibration/output/wrist1/calibration_wrist1.json`。

### Femto Bolt

Femto Bolt 使用固定相机外参 `T_base_femto = T_base_laser`。激光外参文件是 `deploy/calibration/output/laser/calibration_laser.json`，并且必须包含 `"status": "ACCEPTED"`，否则程序拒绝融合。

### 夹爪 TCP

TCP 文件是 `deploy/calibration/output/tcp_grasp_tcp/calibration_translation2.json`。文件必须包含 `"point_is_grasp_tcp": true`，表示标定点是两指闭合后的捏取中心。程序根据 `T_eef_grasp_tcp` 将融合抓取中心换算成 EEF 位姿。

## 三、相机接口

D405 和 Femto Bolt 都通过 `deploy/version1/run_dual_camera_inference.py` 调用。D405 使用 `robot_camera.D405Camera`/RealSense 接口，支持多帧深度中值融合、后处理、孔洞填充、ROI 和工作空间裁剪。Femto Bolt 使用 `robot_camera.FemtoBoltCamera`/`pyorbbecsdk` 接口，使用序列号和深度范围采集点云。

version5 对每台相机传入 `--apply_calibration`。相机输出的 `point_cloud.npy` 仍是相机坐标，`meta.json` 中的 `T_base_camera` 用于融合前转换到 `arm1_base`。

Version5 强制要求同时提供 D405 的 `--roi` 和 Femto Bolt 的 `--femto-roi`。每路推理完成后，主程序会核对其 `meta.json` 中实际应用的 `roi_xywh`；缺少 ROI 或 ROI 不一致时立即终止，禁止将任一相机的全视野点云加入融合。

GraspNet 按相机坐标系点云训练。默认模式下两台相机各自在自身相机坐标系独立推理，抓取结果再转换到 `arm1_base`。实验性融合推理启用后，融合目录才会保存 `point_cloud_camera.npy`/`grasps_camera.npy` 等第三次推理中间结果。

## 四、机械臂和夹爪接口

机械臂 SDK 来自 `/home/dainanjun/Public/data_collector` 中的 `parm_sdk`，默认控制服务器是 `192.168.3.11`，使用 arm1。

观察位姿通过 `Planner.plan_joint_trajectory` 规划，关节角单位是度，并将规划结果提交到 `/middle/waypoint/execute`。真实 Version5 必须从 `deploy/tools/graspnet_planner_execute.py` 启动（`command.txt` 已使用该入口）；该入口将 GraspNet 的旧 `arm1_base/eef_pose` 目标按已验收的 `planner_tf_bridge.json` 转成 Planner 的真实 TCP 目标。

末端位姿通过 `Planner.plan_cart_trajectory(..., planner="lin")` 规划。旧的 `[x, y, z, rx, ry, rz]` 轴角输入会转换为四元数；完整变换为 `T_planner_base_base @ T_arm1_base_eef @ T_eef_planner_tool`，再通过 Planner TF 树转换至 `body_base`/`left_hand_tool_end_link` 后提交。

夹爪调用 `parm_sdk.Gripper(host="localhost").move()`，默认 hand1。位置、力和速度均按百分比处理：

| 参数 | 含义 | 默认或范围 |
|---|---|---|
| `--gripper-open-width-m` | 100% 位置对应的物理开度 | `0.078 m` |
| `--gripper-width-margin-m` | 预测宽度安全余量 | `0.005 m` |
| `--gripper-force` | 设置开度时的力 | `60%` |
| `--gripper-close-force` | 闭合抓取时的力 | `100%` |
| `--gripper-speed` | 夹爪速度百分比 | `1~100`；默认不传 |

本地数据采集代码对 AG 夹爪建议不传 `--gripper-speed`，让 SDK 使用默认值。显式设置时，version5 会检查速度必须位于 `1~100`。命令中不需要设置速度时，直接删除 `--gripper-speed` 整行。

## 五、参数说明

相机和标定参数：`--d405-serial`、`--femto-serial`、`--wrist-calibration`、`--laser-calibration`、`--tcp-translation-calibration`、`--d405-frame-count`、`--d405-postprocess`、`--d405-hole-filling`、`--roi`。

D405 范围参数：`--d405-z-min`、`--d405-z-max`、`--d405-x-abs-max`、`--d405-y-abs-max`。

Femto 范围参数：`--femto-color-width`、`--femto-color-height`、`--femto-depth-width`、`--femto-depth-height`、`--femto-z-min`、`--femto-z-max`、`--femto-x-abs-max`、`--femto-y-abs-max`、`--femto-roi X Y W H`。

奥比中光 ROI 使用独立工具在对齐后的彩色图上框选：

```bash
cd /home/dainanjun/qiuguolin/graspnet
.venv/bin/python deploy/tools/roi_graspnet_laser.py --display :1
```

窗口中拖动矩形后按 Enter 或空格确认。工具打印的 `--femto-roi X Y W H` 可直接加入 Version5 命令。该 ROI 与 D405 的 `--roi` 相互独立。

GraspNet 参数：`--checkpoint-path`、`--num-point`、`--num-view`、`--top-k`。

碰撞参数：`--collision-thresh`、`--collision-voxel-size`、`--collision-approach-dist`、`--collision-table-thickness`、`--collision-table-plane-dist`、`--collision-table-min-inlier-ratio`、`--collision-table-min-normal-z`。

融合参数：`--fuse-voxel-size-m` 默认 `0.004 m`；`--fusion-object-min-height-m` 和 `--fusion-object-max-height-m` 默认只保留各自桌面上方 `0.005~0.150 m` 的点；`--max-grasp-object-distance-m` 默认要求抓取中心距离融合工件点云不超过 `0.030 m`。

桌面筛选参数：`--table-clearance-m`、`--table-height-percentile`、`--table-xy-bounds`、`--table-edge-margin-m`、`--disable-tabletop-filter`。

姿态筛选参数：`--min-grasp-approach-angle-deg`。程序取 GraspNet 旋转矩阵第一列作为接近轴，只保留朝向 `arm1_base -Z` 且与 XOY 平面夹角不小于阈值的姿态。第二列是夹爪开口轴，不用于判断机械臂是否垂直下抓。旧的 `--max-grasp-plane-angle-deg` 和 `--gripper-yaw-offset-deg` 不再接受。

运动参数：`--observation-joints-deg`、`--movej-speed`、`--pregrasp-height-m`、`--pregrasp-duration-s`、`--descend-duration-s`、`--cartesian-speed-factor`、`--servo-hz`、`--max-grasp-distance-m`、`--postgrasp-lift-joints-deg`、`--place-joints-deg`、`--postgrasp-movej-speed`。

## 六、安全开关

不传 `--execute-grasp` 和 `--confirm-real-grasp` 时，最终计划不会执行真实抓取；程序仍可采集相机并生成计划。

`--move-observation` 会在启动时询问是否执行观察位姿 `move_j`。使用 `--skip-observation-move` 可使用当前机械臂位姿，不发送观察位姿运动。

`--preview-pregrasp` 是实际运动模式，必须同时传 `--confirm-real-pregrasp`。它会打开夹爪并移动到预抓取位姿，但不会下降和闭合。

真实抓取使用 `--execute-grasp --confirm-real-grasp`。窗口关闭后还需要输入 `y` 或 `yes`。执行顺序是设置夹爪开度、移动到预抓取位姿、垂直下降、闭合、抬升、移动到放置位姿、打开夹爪释放。

即使没有抓取候选通过最终筛选，只要没有传 `--no-vis`，程序仍会打开融合彩色点云。绿色表示最终选中抓取，黄色表示通过筛选的候选，红色表示被筛选掉的原始候选；没有可执行候选时不会发送抓取动作。

## 七、输入和输出

必须存在：`checkpoint/checkpoint-rs.tar`、`deploy/calibration/output/wrist1/calibration_wrist1.json`、`deploy/calibration/output/laser/calibration_laser.json`、`deploy/calibration/output/tcp_grasp_tcp/calibration_translation2.json`。程序还需要 `parm_sdk`、`pyrealsense2`、`pyorbbecsdk`、PyTorch、Open3D 和 `graspnetAPI`。

默认输出目录是 `deploy/version5/output/YYYYMMDD_HHMMSS_microseconds/`。

| 输出 | 内容 |
|---|---|
| `d405_inference/<run>/d405/` | D405 点云、颜色、元数据和独立抓取结果 |
| `femto_inference/<run>/femto_bolt/` | Femto 点云、颜色、元数据和独立抓取结果 |
| `fused_point_cloud.npy` | 转到 `arm1_base` 并降采样后的融合点云 |
| `fused_point_colors.npy` | 与融合点云逐点对应的 RGB 颜色 |
| `fused_cloud_summary.json` | 两路点数和融合点数统计 |
| `fused_inference/<run>/merged/` | 仅启用 `--enable-fused-inference` 时生成的第三次实验结果 |
| `all_source_grasp_filter.json` | 默认两路独立候选的宽度、角度、桌面、XY 和分来源筛选统计 |
| `all_source_grasps.npy` | 默认包含 D405、Femto Bolt 两路独立候选 |
| `ranked_grasp_candidates.json` | 合格候选按分数排序，并明确记录 IK 尚未查询 |
| `all_source_grasp_sources.json` | 候选池中每个抓取的来源 |
| `grasp_execution_plan.json` | 最终 EEF 位姿、TCP 位置和执行参数 |

## 八、命令

安全 dry-run：

```bash
cd /home/dainanjun/qiuguolin/graspnet
CUDA_VISIBLE_DEVICES=0 .venv/bin/python deploy/version5/run_dual_camera_fused_graspnet.py \
  --skip-observation-move \
  --no-auto-open-gripper \
  --no-vis
```

完整真实执行命令保存在 `deploy/version5/command.txt`。其中包含 `--execute-grasp --confirm-real-grasp`，执行前必须确认机械臂、桌面范围、TCP 标定和放置关节位姿均已验证。
