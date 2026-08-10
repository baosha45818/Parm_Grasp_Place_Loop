# Module 8.1：右臂缓冲区直接抓取

本目录是从原直接抓取链路重构出的独立模块。入口 `buffer_grasp.py` 读取相机/视觉模块生成的 `mask_pose.json`，根据选定工件的 TOML 配置，规划后方对齐点、沿工件方向接近、闭合夹爪并执行确认后的抬升。

`workpiece_pipeline.py` 是三个模块的闭环总控。它依次协调头部视觉、右臂姿态调整和右臂直接抓取：视觉结果在 Q1 的 15°--75° yaw 区间时直接抓取；否则执行调整，随后重新观察并继续判断。总控只调用以下新模块：

- `module_4_head_judge_new/module_4_2_pose_estimation/tabletop_pointcloud_workpiece_pose.py`；
- `module_7_right_align_new/right_align.py`；
- 本目录的 `buffer_grasp.py`。

目录内已包含完整的本地依赖：

- `buffer_grasp.py`：直接抓取入口；
- `configs/workpiece_1_grasp.toml`、`configs/workpiece_2_grasp.toml`：两种工件的抓取点、TCP 补偿、接近、夹爪确认与抬升参数；每项均带中文说明；
- 右臂目标直接使用规划器原生 `right_eef`；
- `move_by_joints.py`、`move_by_eef_pose.py`、`planning_log.py`：抓取执行需要的本地运动和规划公共代码。

先进行预演（不加 `--execute` 不会发送机械臂或夹爪动作）：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp_new/module_8_1_buffer_grasp/buffer_grasp.py \
  --workpiece 1 \
  --arm-id 2
```

`--workpiece` 会自动选择同编号抓取配置，并读取 `output/module_4_2/workpiece_<编号>/` 中最新的 `mask_pose.json`；可用 `--workpiece-pose-file` 显式覆盖，但文件的工件标记必须一致。确认预演与现场状态后，追加 `--execute` 才会实际打开夹爪、规划并执行抓取、闭合夹爪和抬升。模块需要可访问的 `parm_sdk`、右臂/夹爪服务和规划器。抓取入口不会调用调整模块；应仅在视觉结果已经满足直接抓取条件时使用。

总控先预演：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp_new/module_8_1_buffer_grasp/workpiece_pipeline.py --workpiece 1 \
--execute
```

确认预演和现场状态后再追加 `--execute`。总控使用项目 `.venv`，本目录 `saved_poses/` 中的 `3.json`、`6.json`、`10.json` 作为观察和调整过程中的右臂关节姿态；运行日志写入 `/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_8_1/`。
