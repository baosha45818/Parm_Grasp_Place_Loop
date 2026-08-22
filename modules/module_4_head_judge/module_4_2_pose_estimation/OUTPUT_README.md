# Module 4.2 输出说明

`tabletop_pointcloud_workpiece_pose.py` 每次成功采集会写入：

```text
output/module_4_2/workpiece_<工件编号>/<递增序号>/
├── mask_pose.json   # 对下游发布的位姿结果
├── metadata.json    # 本次采集、分割和调试元数据
├── rgb.png / rgbd.npz / mask.png
└── seed_*.png / rgb_icp_workpiece_frame.png
```

所有长度默认单位是 **m**；字段名含 `_mm`、`_deg`、`_rad`、`_px` 时，以后缀为准。齐次变换均采用列向量：`p_target = T_target_source @ p_source`。

## 1. `mask_pose.json`：下游位姿接口

这是 Module 8 抓取和 Module 7 翻转应读取的文件。`workpieces` 按 `body_base` 的 Y 从小到大排序；当前相机安装下，`workpieces[0]` 是画面最右、也是 Module 8 默认处理的工件。

### 顶层字段

| 字段 | 含义 |
|---|---|
| `mask` | 同目录 `mask.png` 的绝对路径。 |
| `calibration` | 本次使用的头部相机外参/内参 JSON 路径。 |
| `coordinate_frame` | 主输出坐标系，固定为 `body_base`。 |
| `plane_z_m` | 命令行指定的旧式水平桌面高度参数；实际 ICP 的桌面平面见 `metadata.json.table_plane_head_camera`。 |
| `angle_convention` | 平面朝向角的定义说明。 |
| `workpiece_asset` | 本次使用的工件资产名，例如 `workpiece_1`。 |
| `stl` | 本次参与 ICP 的 STL 路径。 |
| `semantic_calibration` | 工件语义坐标系、正面法线、地标等配置路径。 |
| `workpiece_semantic_frame` | STL 坐标系与工件语义坐标系的关系说明。 |
| `workpieces` | 成功得到 6D 位姿的工件数组。 |
| `rejected_components` | 分割到了、但未通过 ICP/几何约束的连通域；每项含 `label`、`area_px`、`reason`。 |
| `back_face_confirmation` | 仅在 `workpieces[0]` 连续两次检测为 `BACK` 时存在；记录两次确认及按 ICP 质量选择的结果。 |

### `workpieces[i]`：一件工件的结果

#### 下游必须或优先使用的字段

| 字段 | 含义 / 使用方 |
|---|---|
| `T_body_base_workpiece_semantic` | **主位姿。** 4×4 变换，将工件语义坐标的点变到 `body_base`：`p_body = T_body_base_workpiece_semantic @ p_semantic`。Module 8 抓取用它计算语义抓取点；Module 7 翻转也用它取完整方向。 |
| `T_head_camera_workpiece` | 同一工件语义坐标系在头部相机坐标系下的 4×4 变换。用于相机画面叠加或相机系诊断。 |
| `pose_body_base` | 主位姿的 xyz、ZYX RPY 和 `rotation_matrix` 展开形式。Module 7 用其中 xyz 作为工件中心；与主变换表达的是同一语义位姿。 |
| `orientation` | 正反判断：`FRONT`（语义正面朝上）、`BACK`（语义反面朝上）、`OTHER`（两侧无法可靠区分）。Module 8 仅对 `FRONT` 继续自动抓取。 |
| `head_direction_base_angle_deg` | 语义 `tail → head`（即语义 +X）在 `body_base` XY 平面的朝向，范围 `[0, 360)`。Module 8 用于判断是否处于可直接抓取的 Q1 角度窗。 |
| `head_direction_base_quadrant` | 上述朝向所在象限 `1`–`4`。 |

`pose_body_base.yaw_deg` 是完整旋转矩阵的 ZYX Euler yaw；当工件有 roll/pitch 时，它不必严格等于 `head_direction_base_angle_deg`。桌面抓取朝向应使用后者。

#### 位姿、投影和掩码诊断

| 字段 | 含义 |
|---|---|
| `T_head_camera_stl` | 原始 STL 坐标系到头部相机的 ICP 变换。若语义坐标系相对 STL 有旋转/平移，它会与 `T_head_camera_workpiece` 不同。 |
| `center_pixel_xy` | 语义中心投影到 RGB 图像的像素 `[u, v]`。 |
| `head_pixel_xy`、`tail_pixel_xy` | 语义 head/tail 地标投影像素。 |
| `grasp_center_pixel_xy` | 基于分割形状得到的粗略抓取中心像素，仅作诊断；最终位姿以 ICP/CAD 为准。 |
| `head_base_xy_m`、`tail_base_xy_m` | 语义 head/tail 地标在 `body_base` 的 XY 坐标。 |
| `head_direction_base_vector` | `tail → head` 的三维 `body_base` 向量。 |
| `head_direction_base_xy_unit` | 上述向量的单位 XY 投影。 |
| `area_px` | 连通域面积（像素数）。 |
| `rotated_box_pixel_corners_xy` | 分割连通域最小旋转外接框的四个像素角点。 |
| `head_width_px`、`tail_width_px`、`head_to_tail_width_ratio` | 掩码端部宽度的粗略测量，仅供检查分割形状，不参与最终 6D 位姿。 |

#### ICP 与正反判断

| 字段 | 含义 |
|---|---|
| `icp.rmse_mm` | 最终 ICP 的点到平面 RMSE，越小越好。 |
| `icp.fitness` | ICP 内点匹配比例，越大越好。 |
| `icp.depth_score_mm` | CAD 按候选位姿投影到深度图后，与实测深度的评分，越小越好。 |
| `icp.observed_points` | 本次 mask 内用于 ICP 的有效观测点数。 |
| `icp.initial_hypotheses`、`icp.initialization` | FPFH/RANSAC 粗匹配与最终候选的来源，供复现诊断。 |
| `icp.positive_face_axis_semantic` | 语义坐标中定义为“正面”的法线轴。 |
| `icp.positive_face_up_dot_table_normal` | 正面法线变到相机后与桌面法线的点积；接近 `+1` 表示正面朝上，接近 `-1` 表示反面朝上。 |
| `icp.origin_pixel_xy`、`icp.origin_constraint` | 语义原点必须投影在该工件 SAM2 mask 内的约束及其像素。 |
| `icp.table_penetration_mm` | CAD 候选落到拟合桌面以下的最大深度；过大候选会被拒绝。 |
| `face_hypotheses.FRONT/BACK` | 正、反面各自最优候选的 `depth_score_mm`、`rmse_mm`、`fitness`、法线点积和初始化来源。缺失一侧表示该侧没有合格候选。 |
| `orientation_score_margin_mm` | FRONT 与 BACK 最优深度评分之差；越大，正反区分越明确。只有一侧候选时为 `null`。 |
| `candidate_confidence` | 每个面内的候选排行榜与近似最优候选的位姿离散程度。用于排查“评分接近但位姿差很大”的不稳定情况。 |

#### 质量结论

| 字段 | 含义 |
|---|---|
| `pose_confidence.level` | `HIGH`、`MEDIUM`、`LOW`。 |
| `pose_confidence.recommended_for_grasp` | 是否通过自动抓取建议门；`LOW` 时为 `false`。 |
| `pose_confidence.low_reasons` | 不通过的具体原因，如点数不足、fitness 低、RMSE/深度评分高、正反不清楚等。 |
| `pose_confidence.advisories` | 不阻止输出但值得注意的提示，例如近似最优候选不稳定。 |
| `pose_confidence.thresholds` | 本次判定实际使用的阈值，便于离线复盘。 |
| `pose_confidence.report_only` | 恒为 `true`，表示此对象描述的是诊断结果；重拍/回退的运行级决策见 `metadata.json`。 |

## 2. `metadata.json`：采集与调试记录

此文件不作为抓取的位姿输入。它记录这一次输出是如何采集、分割和挑选出来的，适合复盘质量问题。

| 字段 | 含义 |
|---|---|
| `capture_index` | 对应输出目录的递增序号。 |
| `captured_at`、`serial`、`resolution` | 采集时间、相机序列号和 RGB-D 分辨率。 |
| `calibration`、`intrinsics`、`depth_scale_m` | 使用的标定路径、相机内参和深度单位比例。 |
| `depth_fusion` | 多帧深度融合方式。 |
| `head_move` | 头部 pitch/yaw 请求值、实际读回值、稳定等待等记录。 |
| `table_plane_head_camera` | 相机坐标系中的桌面平面 `[nx, ny, nz, d]`，满足 `n·p + d = 0`。 |
| `roi_normalized_rect_override`、`roi_polygon_px` | 桌面 ROI 覆盖参数及实际使用的像素多边形。 |
| `seed_height_mm`、`seed_max_gray` | 生成初始前景种子的高度和灰度过滤阈值。 |
| `seed_groups` | 初始种子分组，每组的 `seed_label` 与正提示像素 `positives_xy`。 |
| `sam2_groups`、`sam2_device` | SAM2 每组掩码的像素数/接收状态，以及推理设备。 |
| `mask_foreground_pixels`、`pose_components` | 最终 mask 前景像素总数、成功输出的工件数量。 |
| `pose_source` | 位姿估计方法文字说明。 |
| `confidence_attempt`、`low_confidence_retries_allowed` | 本次是第几次质量尝试，以及允许的 LOW 重拍次数。 |
| `confidence_fallback_used`、`confidence_fallback_selection_key` | 若所有尝试均为 LOW，是否启用了“从低质量尝试中择优发布”，以及择优键。 |
| `back_face_confirmation` | 仅 BACK 连续两次确认时存在；记录两次采集及最终选择。 |
| `files` | 本目录内图像、点云、mask 和 JSON 文件名映射。 |
