# Module 4.1：Buffer 状态估计

## 1. 模块主要功能

本模块在抓取前从头部 Intel RealSense D435i 获取一帧 RGB-D 观测，估计 Buffer 中工件的以下状态：

- 工件数量与桌面安全工作区内的目标 mask；
- 工件正反：`FRONT`（正面朝上）、`BACK`（反面朝上）或 `UNCERTAIN`（不允许猜测）；
- 工件相对于头部相机和机器人 body-base 的 6D 位姿；
- 配准、深度匹配和置信度诊断，用于决定是否允许后续抓取。

模块当前支持工件 `1` 和 `2`，支持 `xiangxin`、`gzhu` 两个现场模式。模式是相机序列号、手眼标定、采集分辨率和桌面 ROI 的原子组合，不能交叉使用。模块只负责感知和放行判断，不计算抓取 TCP，也不直接控制机械臂或夹爪。

## 2. 工作原理

整体链路为：

```text
头部 D435i RGB-D
  → 对齐深度中值融合、桌面平面拟合、现场 ROI 约束
  → 高出桌面的深度种子 + SAM2 提示分割
  → CAD 与 mask 深度点云配准
  → 正反竞争、质量/置信度检查
  → FRONT / BACK / UNCERTAIN 与 6D 位姿、诊断文件
```

1. **采集与现场绑定**：`deployment_modes.py` 读取现场标定文件，将 D435 序列号、图像尺寸、fps、body-base 到头部相机的外参和工件专属 ROI 一起传给子流程。运行时会核对采集 `metadata.json` 中的序列号，防止将错误现场的标定用于当前相机。
2. **桌面几何种子**：深度图先转换为相机坐标点云；通过 RANSAC 拟合桌面平面，选取 ROI 内、高于桌面的像素作为候选种子，再经过形态学去噪和连通域筛选。
3. **SAM2 精分割**：从种子区域生成分散的正提示点（工件 1 还带负提示），调用 SAM2 得到工件 mask。最终 mask 仍受现场 ROI、最小面积和配置的连通域策略约束。
4. **位姿与正反判定**：对每个 mask 深度点云执行 CAD 配准。CAD 的物理正面由工件配置/语义文件中的 `+Z` 定义；配准结果同时用于输出正反和位姿，避免两个结果来自不同算法而相互矛盾。
5. **安全放行**：只有唯一工件、`FRONT` 且质量门通过时才返回成功；`BACK` 必须由上层进入反抓恢复或安全停止，`UNCERTAIN` 必须停止，不能按经验放行。

## 3. 操作流程

### 3.1 复用已有 RGB-D 采集，执行状态估计

采集目录必须至少包含 `metadata.json`、`rgb_000.png` 和 `depth_aligned_000.png`，且位于项目的 `output/module_4_1/` 下（Buffer Check 的安全约束）。

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

.venv/bin/python \
  modules/module_4_head_judge/module_4_1_buffer_state_estimation/run_orientation.py \
  --mode xiangxin --workpiece 1 \
  --capture-dir output/module_4_1/captures/head_rgbd_<timestamp>
```

将 `--workpiece` 改为 `2`，或将 `--mode` 改为 `gzhu`，即可切换到相应的工件和现场。若 SAM2 服务不在默认 Unix socket，可用 `--socket <url>` 指定服务地址。

### 3.2 执行 Buffer 严格放行检查

以下命令在未指定 `--capture-dir` 时会自动采集。深度默认以 60 帧中值融合；当正反假设均因观测质量被拒绝时，最多额外采集 3 次，因此总采集次数最多为 4 次。

```bash
.venv/bin/python \
  modules/module_4_head_judge/module_4_1_buffer_state_estimation/check_buffer_face.py \
  --mode xiangxin --workpiece 1
```

可选地复用一份已有采集，或将最终报告复制给调用方：

```bash
.venv/bin/python \
  modules/module_4_head_judge/module_4_1_buffer_state_estimation/check_buffer_face.py \
  --mode gzhu --workpiece 2 \
  --capture-dir output/module_4_1/captures/head_rgbd_<timestamp> \
  --result-file /path/to/buffer_face_check.json
```

退出码和上层动作约定如下：

| 退出码 | 判定 | 上层处理 |
| --- | --- | --- |
| `0` | `FRONT` | 可以继续后续流程。 |
| `20` | `BACK` | 显式进入已验证的反抓恢复，或安全停止。 |
| `21` | `UNCERTAIN` | 报错并停止，不能猜测正反。 |
| `22` | 运行/配置错误 | 检查相机、标定、输入文件和服务状态。 |

### 3.3 推荐运行前检查

- 使用项目的 `.venv`，其中应具备 RealSense、OpenCV、Open3D、SAM2 客户端等依赖。
- 确认目标现场对应的 D435 已连接；`xiangxin` 与 `gzhu` 不能替换序列号或套用彼此标定。
- 确认 SAM2 服务可连接。默认 socket 为 `tools/sam2-service/runtime/sam2-service.sock`；本地 socket 不存在时，mask 流程会尝试启动服务。
- 仅修改对应现场、对应工件的配置文件；修改后先使用历史采集验证，再接入真实流程。

## 4. 算法应用

| 阶段 | 共用算法 | 工件 1 | 工件 2 |
| --- | --- | --- | --- |
| 分割 | 桌面 RANSAC、ROI、离桌高度阈值、连通域、SAM2 point/box prompt | 仅保留最大候选，要求最终 mask 深度有效 | 保留全部候选；同一物体的重复 SAM2 mask 按 IoU 去重 |
| 初始位姿 | CAD 表面采样与深度点云 | 正面/反面各 4 个桌面 yaw 初值 | FPFH/RANSAC 全局粗配准，扩展为 4 个语义半转候选 |
| 精配准 | ICP 与 CAD 深度验证 | 受桌面约束的 point-to-plane ICP：只优化绕桌面法向的 yaw 与 XYZ 平移，不能把一面翻到另一面 | 无约束全 6DoF point-to-plane ICP |
| 正反选择 | CAD 可见表面与观测深度的一致性评分 | 两面候选比较 ICP fitness、深度匹配率和 RMSE | 以渲染 CAD 深度分数、桌面法向夹角、桌面穿透约束及候选间分差综合判定 |

工件 1 的正反候选保持各自固定的朝向约束，因此不会出现 ICP 通过翻转旋转“伪造”另一面的情况。工件 2 的几何更适合全局粗配准和完整 6D 优化；其候选必须满足物理朝向及桌面约束，若一面没有物理可行候选，另一面仍需通过绝对质量门才可被接受。

## 5. 输入输出

### 输入

| 类别 | 文件/来源 | 关键内容 |
| --- | --- | --- |
| RGB-D 采集 | `<capture-dir>/rgb_000.png`、`depth_aligned_000.png`、`metadata.json` | 对齐 RGB/深度、深度尺度、相机内参、相机序列号。 |
| 现场标定 | `tools/calibration/head_d435/<mode>_reverse_observation.json` | 相机序列号、采集规格、`T_body_base_head_camera` 与现场信息。 |
| 工件配置 | `configs/workpiece_1.json`、`configs/workpiece_2.json` | STL、正面定义、mask 与 ICP 阈值。 |
| 现场工作区 | `configs/<mode>_workpiece_<id>_workspace.json` | 参考分辨率和桌面安全 ROI 多边形。 |
| CAD/语义 | `assets/workpiece_*` 及配置引用的 STL | CAD 形状、语义坐标系和正面法向定义。 |

### 输出

默认采集目录为 `output/module_4_1/captures/head_rgbd_<timestamp>/`。核心产物位于其下的 `sam2_masks/`：

| 文件 | 用途 |
| --- | --- |
| `masks.npz` | 通过面积与 ROI 检查的最终二值 mask。 |
| `meta.json` | 使用的配置、ROI、桌面平面、seed 和 mask 统计。 |
| `orientation_diagnostics.json` | 每个目标的分类、候选质量、置信度和 6D 位姿；这是直接状态估计的主输出。 |
| `orientation_overlay.png` | 正反/不确定分类及质量指标叠加到 RGB。 |
| `icp_alignment_overlay.png`（工件 1） | mask 轮廓、CAD 投影和物体 XYZ 坐标轴。 |
| `icp_alignment_3d*.png`（工件 1） | 观测深度点云与配准 CAD 的三维对照。 |
| `buffer_face_check.json` | Buffer Check 的最终报告：`decision`、`continue_allowed`、位姿、诊断路径、相机/标定和重试信息。 |

`orientation_diagnostics.json` 中的 `pose` 至少包含：

- `valid`、`orientation_valid`：位姿及正反是否可用；
- `method`：实际使用的配准方法；
- `T_head_camera_workpiece`：工件在头部相机坐标系下的齐次变换；
- `T_body_base_workpiece`：转换到机器人 body-base 后的齐次变换；
- 工件 2 还提供 `body_base_xyz_rpy`、`pose_confidence`、`candidate_confidence` 和 `icp` 指标。

## 6. 数据可视化分析

建议按以下顺序排查一次结果：

1. 查看 `depth_preview_000.png`：黑色区域代表无效深度或预览范围外的数据，确认工件区域的深度连续。
2. 查看 `workspace_roi_overlay.png`：ROI 外被压暗、黄色边界为实际加载的工作区。工件必须完整位于该区域，且文件名应与当前现场和工件一致。
3. 查看 `seed_raw.png`、`seed_cleaned.png`、`seed_selected.png`：确认离桌高度种子没有漏掉工件，也没有把桌面杂物并入候选。
4. 查看 `sam2_prompt_overlay.png` 和 `sam2_masks_overlay.png`：检查正/负提示点、提示框和最终 mask 是否覆盖真实工件而未包含邻近背景。
5. 查看 `orientation_overlay.png`：绿色为 `positive`、红色为 `reverse`、黄色为 `unknown`；标签中的 `f` 为配准 fitness，`d` 为深度分数或匹配相关指标（依工件而定）。
6. 对工件 1，继续查看 `icp_alignment_overlay.png` 与 `icp_alignment_3d*.png`：CAD 投影应贴合 mask 和深度点云，XYZ 轴方向应与语义坐标系一致。
7. 最后读取 JSON：重点检查目标数、`classification`、`pose.valid`、`pose.orientation_valid`、两面候选指标与 `buffer_face_check.json` 的 `decision`。视觉图看似正确但质量门未过时，仍必须按 `UNCERTAIN` 处理。

## 7. 参数调整

优先通过 JSON 配置调整，避免在算法代码中写死现场差异。修改任一阈值后，应在已标注的正面、反面、低质量和无目标采集上回归验证。

| 配置位置 | 参数 | 作用与调整建议 |
| --- | --- | --- |
| `configs/<mode>_workpiece_<id>_workspace.json` | `safe_roi_polygon_px`、`reference_color_size_wh` | 限定桌面安全区。只在重新标定现场后调整；坐标会按实际分辨率缩放。ROI 过大易引入背景，过小会截断工件。 |
| `workpiece_*.json` → `mask` | `seed_height_mm` | 离桌种子高度。漏检时可小幅降低；桌面噪声进入时提高。 |
| 同上 | `seed_max_gray` | 深色种子筛选上限。受光照变化影响，应结合 seed 图调整；设为 `null` 可关闭。 |
| 同上 | `minimum_depth_m`、`maximum_depth_m` | 有效深度范围。用于排除过近/过远背景；工件 1 可不设上、下限。 |
| 同上 | `minimum_seed_area_px`、`minimum_mask_area_px` | 过滤小噪点和过小 mask。误检多时提高，远距离工件漏检时谨慎降低。 |
| 同上 | `positive_points`、`negative_points`、`sam_box_mode`、`sam_box_padding_px` | SAM2 提示策略。mask 外溢时优先增加负提示或缩小 box；漏分割时增加正提示或 padding。 |
| `workpiece_1.json` | `minimum_reverse_icp_fitness`、`minimum_reverse_depth_match_fraction` | 反面几何放行阈值；提高更保守，降低会增加误判风险。 |
| `workpiece_2.json` | `reverse_full_6d_*` | FPFH/RANSAC、ICP 的采样、迭代、对应距离和并发参数；先保证候选正确，再考虑增加迭代或收紧对应距离。 |
| `workpiece_2.json` | `minimum_face_alignment`、`max_table_penetration_m`、`minimum_face_margin_mm` | 控制候选的朝向、与桌面关系及正反区分度。误放行风险上升时应收紧。 |
| `quality_gate.json` | `minimum_positive_icp_fitness`、`minimum_positive_depth_match_fraction` | Buffer `FRONT` 的最终安全门。它独立于优化过程；不应只为通过某一帧而降低。 |
| 命令行 | `--depth-fuse-frames`、`--max-uncertain-retries` | 前者提高深度稳定性但增加采集时间；后者只针对自动采集时可重新观测的低质量不确定结果。 |

## 8. 历史问题及解决方案

| 历史问题 | 解决方案 |
| --- | --- |
| 两个现场的相机、标定和 ROI 有被混用的风险。 | 引入 `deployment_modes.py`，以 `xiangxin`/`gzhu` 原子绑定序列号、手眼标定、采集尺寸和工件 ROI；采集 metadata 与当前模式序列号不一致时立即拒绝。 |
| 直接在单个工件 mask 上拟合桌面，可能把工件表面当作桌面并翻转法向。 | 工件 1 改为从完整场景深度点云拟合桌面，再将该桌面法向用于配准约束与正反判定。 |
| 工件 1 的全自由度配准可能把正面假设旋转成反面，导致“配准好但面别错”。 | 对两面分别建立 4 个 yaw 初值，并采用受桌面约束的 point-to-plane ICP，只允许绕桌面法向旋转和 XYZ 平移；两面不再能在优化中互相翻转。 |
| SAM2 的 mask 可能跨出工作区、与邻近区域粘连，或因提示点落在边界而不稳定。 | mask 在输出前再次与 ROI 相交；按工件配置使用最大连通域/提示连通域保留策略；正提示改为从距离边界足够远的内部区域采样，并保留 seed 与 prompt 可视化用于复核。 |
| 工件 2 仅靠 yaw 编号初始化，无法覆盖真实的 6D 偏差。 | 改用 FPFH/RANSAC 全局粗配准、语义半转候选和全 6DoF point-to-plane ICP，并加入物理朝向、桌面穿透、候选分差和绝对质量检查。 |
| 单帧质量不佳、双面评分接近或检测到多个目标时，流程可能给出不可靠放行。 | 输出 `UNCERTAIN` 而非猜测；Buffer Check 要求唯一目标和严格质量门。自动采集时，仅对可重新观测的几何不确定情况有限重试，其他情况安全停止。 |
| 下游重新从 mask 中推算中心，会丢失已验证的 CAD/ICP 位姿。 | 将胜出候选的 `T_head_camera_workpiece` 与 `T_body_base_workpiece` 写入统一诊断和 `buffer_face_check.json`，由下游直接消费该位姿。 |
