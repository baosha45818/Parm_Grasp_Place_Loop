# Module 4.2：头部 RGB-D 工件位姿估计

这是从 `push_T` 的相机/视觉检测链路重构出的独立副本。它采集头部 D435i 的对齐 RGB-D、用桌面几何产生 SAM2 提示、分割工件，并用 STL 的多起点全 6DoF ICP 输出工件在 `body_base` 坐标系中的语义位姿。

本目录自带运行必需的标定和 CAD/语义配置：

- `calibration/sdk_base_to_planner_alignment.json`：头相机到 `body_base` 的标定；
- `assets/workpiece_1/`、`assets/workpiece_2/`：两个工件的 STL 和语义坐标定义；
- `tabletop_pointcloud_workpiece_pose.py`：完整采集、分割和位姿估计入口。

运行时依赖是硬件与共享服务，而非源项目文件：RealSense/`parm_sdk`、视觉虚拟环境，以及 `tool/sam2-service`。脚本不会读取源项目中的代码、标定或 CAD 文件。复制的标定 JSON 中保留了一条原始标定的来源字段，仅作历史追溯，不参与路径解析。

先只检查参数和依赖解释（不会连接硬件）：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_2_pose_estimation/tabletop_pointcloud_workpiece_pose.py --help
```

执行工件 1 或工件 2 的完整采集与估计：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_2_pose_estimation/tabletop_pointcloud_workpiece_pose.py --workpiece 1


python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_2_pose_estimation/tabletop_pointcloud_workpiece_pose.py --workpiece 2
```

默认使用 `/home/dainanjun/Public/Parm_Grasp_Place_Loop/.venv/bin/python`，会将头部移至已标定的 `pitch=38.5°`、`yaw=0°`。每次运行的 RGB-D、SAM2 掩码、调试图、`mask_pose.json` 和元数据会按工件写入 `/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_4_2/workpiece_1/<序号>/` 或 `workpiece_2/<序号>/`。离线调试已在观察位的相机时可增加 `--skip-head-move`。

`mask_pose.json` 中每个 `workpieces` 条目还会输出 `orientation`：`FRONT` 表示语义配置 `positive_face.upward_axis` 指向桌面上方，`BACK` 表示其反向朝上，`OTHER` 表示两种假设的深度评分差距不足或其中一面没有通过几何约束。`face_hypotheses` 保留正、反两套候选的评分，阈值可用 `--minimum-face-margin-mm` 调整。


可用环境变量覆盖部署位置：`POSE_ESTIMATION_PYTHON`、`POSE_ESTIMATION_SAM2_SOCKET`、`POSE_ESTIMATION_SAM2_BIN`；所有业务参数也可由脚本的 CLI 覆盖，例如 `--calibration`、`--stl`、`--semantic-calibration`、`--capture-root`。
