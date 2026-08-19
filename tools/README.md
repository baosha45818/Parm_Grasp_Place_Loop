# 项目公共运行工具

This directory contains the project-local runtime dependencies used by the
正式模块与新总控使用的公共运行依赖。

## Layout

- `sam2-service/`: SAM2/EfficientViT 服务、第三方后端、检查点及本地运行环境。
- `graspnet_runtime/`: the minimal GraspNet-derived Planner and D405 Camera
  Broker source closure. GraspNet inference models, checkpoints, jobs, output
  and its 8.9 GB virtual environment are intentionally excluded.
- `calibration/`: 左腕、头部 D435、Planner bridge 和 TCP 标定。
- `workpiece_1/`、`workpiece_2/`: ICP 所需的 CAD 模型。
- `inspection_support/`: M4 与反面交接观察共用的相机、分割和点云辅助代码。
- `handoff_observation/`: 右手交接观察、位姿估计、交接规划和执行的通用工具；由 M5.2、M8.2 调用，不是独立业务模块。
- `grasp_weight_check/`: 左右夹爪抓取后的力传感器重量判定工具；由 M1、M5.2 和 GraspNet 运行时调用。
- `planner_runtime.py`、`planner_task_retry.py`、
  `left_motion_right_servo_lock.py`: 规划与双臂动作支持。
- `icp.py`: 共享 ICP 求解器。

旧 `tool/` 目录已经删除；任何正式模块都不得再引用它。`graspnet_runtime/LICENSE`
保留上游 GraspNet 许可证，SAM 后端许可证与源码放在一起。

## SAM2 environment

For a fully independent SAM2 environment, run:

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tools/sam2-service
uv sync --python 3.10
```

启动脚本只使用本目录的 `.venv`，不再回退到旧目录。
