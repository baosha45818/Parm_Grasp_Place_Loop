# 模块 4：Head Judge

## 职责

该模块使用头部 D435i 判断桌面工件朝向，并统一管理所有云台动作：

1. 采集桌面 RGB-D，或者读取已有采集目录。
2. 使用 SAM2 分割桌面工件。
3. 工件1执行正反多yaw受约束ICP；工件2执行FPFH/RANSAC粗配准和全6DoF ICP。
4. 从同一个最佳ICP变换同时输出正反和工件位姿。

`FRONT` 表示 CAD `+Z` 正面朝上，`BACK` 表示 CAD `-Z` 反面朝上。
`OTHER` 表示结果不明确、几何质量不合格或目标不在安全工作区，不能直接
解释为工件一定侧放。

该模块不导入规划器，不控制云台、左右机械臂或夹爪。调用方必须在进入 Module 4
前将云台移动到对应标定观察位。

反抓后双臂交接使用的固定观察角 `pitch=15°、yaw=35°` 不具备通用性，已归回
Module 5-2 反手交接流程，不属于 Module 4。

`module_4_1_buffer_state_estimation/` 采用一个 `run_orientation.py` 入口和一套
`vision/` 公共实现；工件差异保存在 `configs/workpiece_1.json`、
`configs/workpiece_2.json` 及各自 ICP 策略中。抓取点/TCP 预测归 Module 5-1。

## 运行

对已有 RGB-D 执行朝向判断：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

/home/dainanjun/Public/data_collector/.venv/bin/python \
  modules/module_4_head_judge/run.py \
  --mode xiangxin \
  --capture-dir /path/to/head_rgbd_capture
```

结果写入：

```text
<capture-dir>/sam2_masks/tabletop_orientation_result.json

桌面候选区域由 `module_4_1_buffer_state_estimation/configs/workpiece_<编号>.json`
中 `mask.safe_roi_polygon_px` 定义。它按采集分辨率缩放并约束深度种子和
SAM2输出。
```

## Python 接口

```python
from modules.module_4_head_judge import inspect_tabletop_state

result = inspect_tabletop_state(capture_dir, workpiece_id="2", mode="xiangxin")
print(result.orientation)
```

当结果包含 `OTHER` 且 `tilt_recovery_candidate=True` 时，调用方可以执行未来
的摆正策略，动作结束后必须重新采集并再次调用本模块确认。Module 4 本身
不会执行恢复动作。

## 当前视觉依赖

工件1、2视觉实现直属 Module 4，位于：

```text
modules/module_4_head_judge/module_4_1_buffer_state_estimation/
  run_orientation.py
  vision/capture_rgbd.py
  vision/sam2_client.py
  vision/mask.py
  vision/workpiece_1_icp.py
  vision/workpiece_2_icp.py
```

本目录保存分割、正反判断和工件位姿；抓取框架、TCP 计算和抓取可视化位于 Module 5-1。
本模块不再依赖 `/home/dainanjun/zhangfei/tools/tabletop_orientation`。

## Buffer 正反面检查

`module_4_1_buffer_state_estimation/check_buffer_face.py` 在 Module 4 的公开正反判断结果之上增加
Buffer 专用的严格质量门，输出 `FRONT`、`BACK` 或 `UNCERTAIN`。它可以现场采集
头部 D435i，也可以复用其 `captures/` 下的历史采集；不控制云台、机械臂或夹爪。
