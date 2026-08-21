# 头部与胸部相机外参标定

本目录使用同一个入口脚本，为固定安装的两台相机执行 eye-to-hand（相机相对机器人基座）标定：

- 头部 Intel RealSense D435i；
- 胸部 Orbbec Femto Bolt。

采集时将 ChArUco 标定板固定在机械臂右爪末端（理论上左爪也行，但左爪不好抓标定板）。脚本读取末端在 SDK 基座坐标系中的位姿和相机观测，求解相机到基座的变换。相机在一次采集期间必须保持刚性固定；头部相机的结果还只对采集时记录的 pitch/yaw 有效。

## 目录结构

    calibration_head_chest/
    ├── calibration_head.py                 两台相机共用的命令行入口
    ├── calibration_common.py               采集、检测、求解、质量检查公共逻辑
    ├── head_d435i_camera.py                RealSense D435i 彩色相机适配器
    ├── chest_orbbec_camera.py              Orbbec Femto Bolt 彩色相机适配器
    ├── configs/
    │   ├── calibration_head_config.json    头部 D435i 的 ChArUco 配置
    │   ├── calibration_head_chessboard_11x8_config.json
    │   │                                   头部 D435i 的 11×8 棋盘格配置（30 mm 方格）
    │   └── calibration_chest_config.json   胸部 Orbbec 的配置
    ├── outputs/                            采集样本和标定结果；首次采集时自动创建
    │   ├── head_d435i/
    │   └── chest_orbbec/
    └── __pycache__/                        Python 自动生成的字节码缓存，可忽略

## 快速使用

请使用项目虚拟环境：

    cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tools/calibration/calibration_head_chest
    PY=/home/dainanjun/Public/Parm_Grasp_Place_Loop/.venv/bin/python

### 头部 D435i

    $PY calibration_head.py \
      --config configs/calibration_head_config.json \
      --collect --solve \
      --samples 18 \
      --head-pitch-deg 0 \
      --head-yaw-deg 0

### 头部 D435i（11×8 棋盘格）

    $PY calibration_head.py \
      --config configs/calibration_head_chessboard_11x8_config.json \
      --collect --solve \
      --samples 18 \
      --head-pitch-deg 38.5 \
      --head-yaw-deg 0

默认输入/输出目录为 outputs/head_d435i/，成功结果为：

    outputs/head_d435i/calibration_head.json

棋盘格配置使用独立目录和文件名，避免与 ChArUco 样本混用：

    outputs/head_d435i_chessboard_11x8/calibration_head_chessboard_11x8.json

注意！！！云台的 pitch/yaw 不由本脚本控制；这两个参数仅写进结果 JSON。重新调整云台姿态后，必须重新采集和标定。

### 胸部 Orbbec

    $PY calibration_head.py \
      --config configs/calibration_chest_config.json \
      --collect --solve \
      --samples 18

默认输入/输出目录为 outputs/chest_orbbec/，成功结果为：

    outputs/chest_orbbec/calibration_chest.json

胸部相机使用 Orbbec SDK；不要把它的序列号填入头部配置，或将头部 D435i 的序列号填入胸部配置。

### 只重新求解已有样本

已有采样图像和 sample_*.json 时，不连接相机或机械臂也可以重跑求解：

    $PY calibration_head.py \
      --config configs/calibration_head_config.json \
      --solve

可用 --data /path/to/samples 指定非默认样本目录。默认目录按相机分开；如手动覆盖 --data，请避免让两台相机共用同一个目录。

## 选择与修改配置

--config 选择 configs/ 下的 JSON 文件。每份 JSON 包含：

| 字段 | 用途 |
| --- | --- |
| camera_name | 样本元数据、求解筛选与默认输出目录的相机名称 |
| camera_type | realsense_d435i 选择 RealSense；orbbec_femto_bolt 选择 Orbbec |
| camera_model | 写入最终标定 JSON 的设备型号 |
| default_serial | 默认设备序列号；可由命令行 --serial 临时覆盖 |
| default_board | 标定板定义；必须与实际物理标定板一致 |
| calibration_type | 写入结果的标定类型描述 |
| transform_key | 最终结果中相机外参矩阵的字段名 |
| intrinsics_source | 默认内参来源说明 |
| output_subdir | outputs/ 下的默认相机子目录 |
| result_filename | 质量通过时的标定结果文件名 |

`calibration_head_config.json` 和胸部配置使用 DICT_4X4_100、13×5 方格、方格边长 15 mm、marker 边长 11 mm 的 ChArUco 板。`calibration_head_chessboard_11x8_config.json` 使用 11×8 内角点、方格边长 30 mm 的普通棋盘格；左上、左下为白色，右上为黑色。若更换了物理板，必须同步修改对应配置的 default_board 后重新采集。

胸部配置当前记录的默认序列号为 CL8654103PJ。若现场设备不同，推荐先临时运行：

    $PY calibration_head.py \
      --config configs/calibration_chest_config.json \
      --collect --serial 实际序列号

确认后再更新 configs/calibration_chest_config.json 的 default_serial。

## 各代码文件的作用

### calibration_head.py

唯一的入口脚本。它读取 --config 指定的 JSON，根据 camera_type 选择相机适配器，发起采样或离线求解，并按配置决定结果字段和输出位置。

常用参数：

- --collect：连接相机、机械臂和可选夹爪，打开预览窗口并采集。
- --solve：读取样本并计算外参；可与 --collect 同时使用。
- --samples N：目标采样数量，默认 18；至少需要 8 个样本，默认质量门限要求至少保留 10 个。
- --serial：临时覆盖 JSON 的默认序列号。
- --arm-host、--arm-id：机械臂 SDK 服务地址及手臂编号。
- --disable-gripper-toggle：禁用预览窗口的空格键夹爪开闭。
- --intrinsics FILE.json：用指定内参覆盖样本中的相机内参。
- --exclude-index N：求解时排除某一个采样编号，可重复传入。

采集窗口中：c 保存当前样本，d 删除本次会话最后一张，空格切换夹爪，q 或 Esc 退出。ChArUco 板的角点由 ID 唯一确定，因此不需要人工翻转角点方向。

### calibration_common.py

两种相机共用的核心逻辑：

- 检测棋盘、ArUco 或 ChArUco 标定板，并利用当帧内参估计板位姿；
- 控制 SDK 的自由示教，读取末端 eef_pose；
- 采样时检查前后两次机械臂读数，运动超过 0.5 mm 或 0.25° 的帧会被拒绝；
- 保存图像、相机内参、畸变、末端位姿和板位姿；
- 用稳健优化求解外参并剔除离群样本；
- 计算平移/旋转残差，并执行质量门限检查。

### head_d435i_camera.py

头部 RealSense D435i 的彩色流适配器。它通过 pyrealsense2 打开指定序列号的 1280×720@30 彩色流（可通过参数修改），读取 active profile 的内参与畸变参数，并将图像转为 OpenCV 使用的 BGR 数组。

### chest_orbbec_camera.py

胸部 Orbbec Femto Bolt 的彩色流适配器。它使用 pyorbbecsdk 按序列号选择设备，优先尝试请求的分辨率/帧率，若设备不提供精确 profile 则从可用彩色 profile 中选择最合适的一个。它会把 RGB、BGR、MJPG、YUYV 或 NV12 图像统一转成 BGR，并保存实际 profile 的内参。

## outputs/ 中的文件

一次采集会写入对应相机的子目录：

| 文件 | 内容 |
| --- | --- |
| sample_000.png 等 | 每个被接受样本的彩色图像 |
| sample_000.json 等 | 同编号图像的末端位姿、板位姿、内参、畸变、重投影误差和稳定性数据 |
| session.json | 本次会话的样本清单、采样数、示教模式和夹爪设置 |
| calibration_head.json / calibration_chest.json | 质量门限通过后的最终外参 |
| *_rejected.json | 质量不通过时仍保存的诊断结果；不要将其用于机器人任务 |

头部成功结果包含 T_base_head_camera 和 head_pose_deg；胸部成功结果包含 T_base_chest_camera。两者的 T_base_... 都以 SDK 报告的机器人基座为参考坐标系。若下游程序使用 Planner 的 body_base，应先确认或额外标定这两个基座坐标系之间的变换。

## 如何检查标定结果

主要检查最终结果 JSON 的 residual 中这两个平移指标：

    "residual": {
      "translation_rms_mm": ...,
      "translation_max_mm": ...
    }

- translation_rms_mm：所有最终保留样本的平移残差均方根，单位毫米。它反映整组样本的典型总体误差；数值越小，说明不同机械臂姿态推算出的固定相机位置越一致。
- translation_max_mm：所有最终保留样本中最大的单帧平移残差，单位毫米。它用于发现仍未被剔除的最差样本；数值越小，说明没有某一帧与整体解明显不一致。

这里的平移残差是同一块板在“机器人末端位姿 + 末端到标定板变换”和“相机观测到的标定板位姿”两条链路之间的闭环位置误差，并不是图像像素误差。

当前默认质量门限为：

    translation_rms_mm <= 5.0
    translation_max_mm <= 15.0

若不满足，程序会：
    
    1. 仍保存诊断结果，但状态写为 REJECTED_QUALITY_GATE；
    2. 不生成/不覆盖正常的标定结果文件；
    3. 改为输出拒绝文件：
        头部：calibration_head_rejected.json
        胸部：calibration_chest_rejected.json
    4. 最后抛出报错，明确提示标定被拒绝以及 RMS/最大误差。

**但是，实际最好满足：**
    
    translation_rms_mm <= 2.0
    translation_max_mm <= 3.0

二者都满足才建议使用标定结果，否则误差较大。

程序仍会同时保留旋转残差、样本数和离群样本信息，并将它们作为自动质量门限的一部分；如果最终 status 不是 ACCEPTED，不应使用该结果。

## 用于 TF 的外参

用于相机与机器人基座之间 TF 变换的是结果 JSON 中的 4×4 齐次矩阵：

- 头部 D435i：T_base_head_camera；
- 胸部 Orbbec：T_base_chest_camera。

这两个矩阵的方向均为“相机坐标系到 SDK 机器人基座坐标系”，即可将相机坐标系中的点变换到 base 坐标系：

    p_base = T_base_camera @ p_camera

不要把 T_eef_board 当作相机 TF；它是本次标定中求出的“末端坐标系到标定板坐标系”关系，只用于求解过程和结果诊断。若下游 TF 树以 Planner 的 body_base 为根，需要先完成 SDK base 与 body_base 的坐标系对齐，再复合该变换。

## 与 Module 4-2 位姿估计的兼容性

当前的 Module 4-2 入口为：

    modules/module_4_head_judge/module_4_2_pose_estimation/tabletop_pointcloud_workpiece_pose.py

标定完成后将标定文件拷贝一份到/home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_2_pose_estimation/calibration里供 Module 4-2 消费（需要修改4-2的引用）。

将头部标定结果通过该脚本的 --calibration 参数传入时，它主要使用：

- T_base_head_camera：最主要的外参。它把相机坐标系中的工件位姿转换到机器人基座坐标系，关系为：

      T_base_workpiece = T_base_head_camera @ T_head_camera_workpiece

- camera_matrix 和 distortion_coeffs：用于像素去畸变、图像投影及相机射线计算。
- image_size：脚本会拒绝实际采集分辨率与标定分辨率不一致的组合。
- status：必须为 ACCEPTED；如果标定文件是 *_rejected.json 或 status 不是 ACCEPTED，Module 4-2 会直接报错停止。

residual.translation_rms_mm 和 residual.translation_max_mm 不会被 Module 4-2 直接参与位姿计算；它们通过上游标定的质量门限决定 status 是否为 ACCEPTED。

> 重要限制：tabletop_pointcloud_workpiece_pose.py 当前只接受头部字段 T_base_head_camera（以及旧格式 T_body_base_head_camera）。它不识别胸部文件中的 T_base_chest_camera。因此，胸部标定结果当前不能直接作为该脚本的 --calibration 参数。
## 采集前检查

1. 使用项目虚拟环境，确认其中有 numpy、scipy、OpenCV ArUco/ChArUco、parm_sdk、pyrealsense2 和 pyorbbecsdk。
2. 确认目标相机已连接，序列号与配置一致，且图形桌面可显示 OpenCV 预览窗口。
3. 确认机械臂服务、夹爪服务和地址参数正确。
4. 将 ChArUco 板稳固固定在指定夹爪，采集至少 15 张（建议 18 张）具有明显三维旋转变化的手臂姿态。
5. 每次按 c 前松开踏板并等待手臂静止；避免只在同一平面或姿态附近采样。

> 注意：采集会启动机械臂自由示教，并可能通过空格键操作夹爪。确认现场安全后再加 --collect；若不需要夹爪开闭，请添加 --disable-gripper-toggle。
