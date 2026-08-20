# 左腕 D405 手眼标定与代码使用

本文说明 Module 1 当前使用的两套左腕 Intel RealSense D405 手眼标定、坐标链、
重新标定方法，以及如何把标定结果传给代码。

> 本目录中的 `sdk_base_alignment/` 是其他基座对齐实验数据。Module 1 当前实际读取
> 的腕部标定结果仍位于 `tools/calibration/`，不要仅因为本文放在 `calibration/`
> 就移动这些 JSON。

## 当前相机与标定绑定

`--mode` 会把相机序列号、手眼链和现场服务地址作为一整套参数绑定。两套腕部
标定不能交叉使用。

| Module 1 模式 | D405 序列号 | 手眼链 | 必需标定文件 |
| --- | --- | --- | --- |
| `gzhu` | `260322270242` | SDK 手眼 + Planner bridge | `tools/calibration/left_wrist_d405/calibration_wrist1.json` 和 `tools/calibration/graspnet_planner_bridge_5604_20260803/planner_tf_bridge_5604_candidate.json` |
| `xiangxin` | `260322270600` | Planner-direct | `tools/calibration/wrist1_planner_final_v3/calibration_wrist1_planner.json` |

当前 Module 1 直接通过 `pyrealsense2` 打开 D405，不使用 Camera Broker。
Planner-direct 标定文件中的 `camera_profile.module_1_profile` 仍写着旧版 Camera
Broker，这只是标定文件的历史来源说明，不代表当前运行方式。

## 坐标变换约定

本文和代码统一使用列向量。`T_A_B` 表示“把 B 坐标系中的点变换到 A 坐标系”：

```text
p_A = T_A_B @ p_B
```

多个变换按从右到左的顺序复合。所有外参均为 4×4 齐次矩阵，最后一行必须为
`[0, 0, 0, 1]`。

## 方法一：SDK 手眼 + Planner bridge

这套方法绑定 D405 `260322270242`，分为两个彼此独立的标定步骤。

### 1. 标定 SDK 末端到 D405

D405 刚性安装在左臂末端，标定板固定不动，属于 eye-in-hand 标定。当前结果使用：

- 640×480 D405 图像；
- 11×8 内角点的普通棋盘格；
- 方格边长 30 mm；
- SDK `state.left.eef_pose` 作为机器人位姿来源；
- 15 个输入姿态，剔除 2 个离群姿态后保留 13 个。

每个静止姿态需要同时得到：

```text
T_sdk_base_eef       SDK 基座到左臂末端
T_d405_board         D405 彩色光学坐标系到固定标定板
```

固定标定板约束为：

```text
T_sdk_base_board
    = T_sdk_base_eef
    @ T_eef_d405
    @ T_d405_board
```

对多个具有明显三维旋转差异的姿态联合求解固定的 `T_eef_d405` 和
`T_sdk_base_board`。当前结果将 `T_eef_d405` 保存为：

```text
tools/calibration/left_wrist_d405/calibration_wrist1.json
└── T_eef_wrist1
```

该结果采用 2 mm 平移或 0.5° 旋转的离群阈值。最终保留样本中的最大平移残差约
0.865 mm，最大旋转残差约 0.463°。

### 2. 标定 SDK 坐标链到 Planner 坐标链

这一步不使用相机，只在多个手臂静止姿态下同步读取：

```text
T_sdk_base_eef
T_left_arm_base_left_arm_end_link
```

求解两端固定变换：

```text
T_left_arm_base_left_arm_end_link
    = T_left_arm_base_sdk_base
    @ T_sdk_base_eef
    @ T_eef_left_arm_end_link
```

结果文件使用以下字段名：

```text
tools/calibration/graspnet_planner_bridge_5604_20260803/
  planner_tf_bridge_5604_candidate.json

T_planner_base_base   == T_left_arm_base_sdk_base
T_eef_planner_tool    == T_eef_left_arm_end_link
```

当前 bridge 为 `ACCEPTED`，使用 9 个姿态；最大平移残差约 0.032 mm，最大旋转
残差约 0.006°。文件声明的 Planner 坐标系必须保持为：

```text
planner_base_frame = left_arm_base
planner_tool_frame = left_arm_end_link
source_base_frame  = base
source_eef_frame   = eef_pose
```

采样时至少应满足以下原则：

1. 机械臂由操作员移动到多个差异明显的姿态，每次采样时完全静止。
2. 至少采集 5 个姿态，末端姿态的旋转跨度至少达到 20°；实际建议 8～12 个。
3. 每个样本都要用相同关节角的 FK 复核 Planner TF，不能用 bridge 掩盖过期关节
   状态或错误 URDF。
4. 求解结果只有在 frame 名称正确且质量门限通过后才能标记为 `ACCEPTED`。

### 3. Module 1 运行时复合

采图前后，Module 1 只读获取 SDK 末端位姿并检查机械臂静止；同时从 Planner
读取 `body_base` 到 `left_arm_base`。最终相机位姿为：

```text
T_body_base_d405
    = T_body_base_left_arm_base             # 运行时 Planner TF
    @ T_left_arm_base_sdk_base              # bridge: T_planner_base_base
    @ T_sdk_base_eef                        # 运行时 SDK eef_pose
    @ T_eef_d405                            # wrist: T_eef_wrist1
```

实现位置：

- `modules/module_1_left_grasp/main.py`：采图前后读取 SDK/Planner 位姿并写入元数据；
- `modules/module_1_left_grasp/icp_pipeline.py`：`base_camera_transform()` 复合矩阵。

## 方法二：Planner-direct 手眼标定

这套方法绑定 D405 `260322270600`。D405 同样刚性安装在左臂末端，但机器人
位姿直接来自 Planner，因此不需要 SDK/Planner bridge。

当前结果使用：

- 640×480、30 FPS，深度对齐到彩色图；
- `DICT_4X4_100` ChArUco 板；
- 13×5 方格，方格边长 15 mm，marker 边长 11 mm；
- 20 个输入姿态，剔除 2 个离群姿态后保留 18 个；
- Planner `left_arm_base` 到 `left_arm_end_link` 作为机器人位姿来源。

每个静止姿态需要同步得到：

```text
T_left_arm_base_left_arm_end_link
T_d405_color_optical_board
```

固定标定板约束为：

```text
T_left_arm_base_board
    = T_left_arm_base_left_arm_end_link
    @ T_left_arm_end_link_d405_color_optical
    @ T_d405_color_optical_board
```

当前结果保存为：

```text
tools/calibration/wrist1_planner_final_v3/calibration_wrist1_planner.json
└── T_left_arm_end_link_d405_color_optical
```

文件的 `parent_frame` 必须为 `left_arm_end_link`，`base_frame` 必须为
`left_arm_base`。该结果采用 2 mm/0.5° 离群阈值；最终保留样本中的最大平移
残差约 1.442 mm，最大旋转残差约 0.307°。

现场 Planner TF 网关返回的矩阵方向与请求标签相反。采集和运行代码都必须：

1. 对返回矩阵做 inverse-normalize；
2. 同时查询正反两个方向并检查互逆闭环；
3. 检查动态 TF 时间戳确实前进；
4. 在相机曝光前后检查末端变化不超过 0.5 mm/0.25°。

Module 1 运行时直接复合：

```text
T_body_base_d405
    = T_body_base_left_arm_end_link          # 运行时 Planner TF
    @ T_left_arm_end_link_d405_color_optical # planner-direct 标定
```

## 重新标定的采集要求

以下任一情况发生后必须重新标定对应链路：相机或支架拆装、相机序列号更换、
末端坐标系定义变化、SDK 位姿定义变化、Planner URDF/基座/末端 frame 变化。

重新采集时应遵守：

1. 固定标定板，确认实际板参数与 JSON 完全一致。
2. 确认采集的是目标 D405 序列号，并从 active profile 读取实际内参。
3. 让标定板覆盖画面中央、边缘、近处和远处，并提供绕多个轴的明显旋转。
4. 每个姿态先等待机械臂和标定板稳定，再成对记录图像观测与机器人位姿。
5. 使用正反向 Planner TF 做方向和时间戳检查，拒绝运动中或状态陈旧的样本。
6. 首次求解后按 2 mm/0.5° 检查离群样本，再用保留样本重算。
7. 独立拿若干未参与求解的姿态做闭环验证，确认标定板在基座坐标系中的估计
   不随机械臂姿态漂移。
8. 先在 plan-only 模式复核点云、桌面高度和抓取中心，确认后才允许真实抓取。

仓库当前状态需要特别注意：

- `tools/calibration/calibration_head_chest/calibration_common.py` 中保留了
  `solve_wrist()` 和 `solve_wrist_with_outlier_rejection()` 的通用 eye-in-hand
  数值求解逻辑；
- 当前工作树没有用于重新采集这两套腕部标定数据的正式 CLI，因此不能把头部/
  胸部相机的 `calibration_head.py --collect` 命令直接用于腕部 D405；
- SDK/Planner bridge 的旧采集求解器可在 Git 提交 `27254cb` 的
  `third_party/graspnet/deploy/tools/calibrate_planner_tf_bridge.py` 中查阅，但它的
  默认端口和 frame 是旧部署值，不能原样执行。当前环境使用 Planner 端口
  5604/5609/5610 和 frame `left_arm_base`/`left_arm_end_link`。

因此，现阶段支持的是“使用和验证现有结果”。如需重新生成结果，应先把腕部采集
入口按上述两套数据源恢复为当前部署版本，再进行现场采样，不应手工编辑矩阵。

## Module 1 使用方法

在项目根目录运行。下面是不控制机械臂和夹爪的 plan-only 示例；程序仍会打开
D405、只读获取当前机器人位姿、运行 SAM/ICP 并请求 Planner 做轨迹预检。

### D405 `260322270242`：SDK + bridge

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_1_left_grasp/run.sh \
  --mode gzhu \
  --skip-observation-move \
  --frames 1
```

### D405 `260322270600`：Planner-direct

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_1_left_grasp/run.sh \
  --mode xiangxin \
  --skip-observation-move \
  --frames 1
```

不带 `--execute --confirm-real-grasp` 时不会发送机械臂或夹爪动作。只传其中一个
参数也会在连接机器人之前报错。真实动作的安全条件和完整命令见
`modules/module_1_left_grasp/README.md`。

### 覆盖标定文件路径

`gzhu` 模式可以替换 SDK 手眼结果和 bridge 结果：

```bash
modules/module_1_left_grasp/run.sh \
  --mode gzhu \
  --skip-observation-move \
  --frames 1 \
  --bridge-wrist-calibration /absolute/path/to/calibration_wrist1.json \
  --planner-bridge-calibration /absolute/path/to/planner_tf_bridge.json
```

`xiangxin` 模式可以替换 Planner-direct 结果：

```bash
modules/module_1_left_grasp/run.sh \
  --mode xiangxin \
  --skip-observation-move \
  --frames 1 \
  --direct-wrist-calibration /absolute/path/to/calibration_wrist1_planner.json
```

`--handeye-mode` 不能用来打破模式绑定：`gzhu` 只允许 `bridge`，`xiangxin` 只
允许 `planner-direct`。自定义路径目前只会被检查是否存在以及运行时所需字段/
frame 是否正确，标定 JSON 内没有可靠的相机序列号字段；操作者仍必须确认文件
确实属于所选 D405，不能只靠文件名判断。

### 运行记录

每次采集的目录中，`raw/d405/meta.json` 会记录：

- 实际 D405 序列号、FPS 和采集来源；
- 最终解析出的 `handeye_mode`；
- 三个标定参数的绝对路径；
- 采图时的 SDK 或 Planner 末端位姿；
- 相机曝光前后的末端稳定性检查；
- Planner TF 时间戳和正反方向闭环诊断（适用时）。

排查抓取坐标问题时，应把该文件与本轮 `traditional/report.json`、点云可视化和
实际加载的标定 JSON 一起保存，避免只根据启动命令猜测所用坐标链。

## 标定与抓取偏置必须分开

手眼矩阵只描述刚体坐标关系。Module 1 的以下抓取补偿由业务代码另行应用，
不得写入手眼矩阵：

| 手眼模式 | 模式专属补偿 |
| --- | --- |
| SDK + bridge | 正面沿 `body_base +Z` 固定抬高 21 mm；反面固定抬高 27 mm |
| Planner-direct | 正反面固定 Z 抬高为 0；按夹爪水平轴投影补偿抓取中心，最大 7 mm |

两种模式还共同保留反面工件局部 `-X` 方向 1.5 mm 偏置，并允许在固定 Z 偏置
基础上再自动向上搜索最多 30 mm 以避开低处堆叠点。若把这些数值烘焙进标定
矩阵，运行时会发生重复补偿。

## 常见报错

| 报错或现象 | 优先检查 |
| --- | --- |
| D405 serial 与预期不一致 | `--mode` 是否选错、另一台相机是否占用设备 |
| selected hand-eye calibration file is missing | 三个标定参数的路径是否存在 |
| unexpected direct Planner D405 calibration frames | direct JSON 的 `parent_frame`/`base_frame` |
| unexpected Planner bridge calibration | bridge 是否为 `ACCEPTED`，四个 frame 字段是否匹配当前部署 |
| Planner TF timestamp did not advance | Planner 关节状态发布是否停止或陈旧 |
| eef/end-link remained in motion across capture | 等待机械臂完全静止、排查反馈抖动后重新采图 |
| 点云整体方向正确但抓取中心固定偏移 | 先核对所选标定链，再区分手眼误差和模式专属抓取偏置 |

