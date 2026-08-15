# 模块 4：Head Judge

## 职责

该模块使用头部 D435i 判断桌面工件朝向，并统一管理所有云台动作：

1. 采集桌面 RGB-D，或者读取已有采集目录。
2. 使用 SAM2 分割桌面工件。
3. 对 CAD 正面和反面执行受桌面约束的 ICP。
4. 输出 `FRONT`、`BACK` 或 `OTHER`。

`FRONT` 表示 CAD `+Z` 正面朝上，`BACK` 表示 CAD `-Z` 反面朝上。
`OTHER` 表示结果不明确、几何质量不合格或目标不在安全工作区，不能直接
解释为工件一定侧放。

该模块不导入规划器，不控制云台、左右机械臂或夹爪。调用方必须在进入 Module 4
前将云台移动到对应标定观察位。

反抓后双臂交接使用的固定观察角 `pitch=15°、yaw=35°` 不具备通用性，已归回
Module 5-2 反手交接流程，不属于 Module 4。

Module 4-1 `module_4_1_reverse_grasp_prediction/` 负责左臂从 Buffer 反抓前的
云台观察、RGB-D、SAM2/ICP 和唯一抓取点预测。Module 5-1 只消费 4-1 的预测并
执行左臂动作，绝不再控制云台。

## 运行

对已有 RGB-D 执行朝向判断：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

/home/dainanjun/Public/data_collector/.venv/bin/python \
  modules/module_4_head_judge/run.py \
  --capture-dir /path/to/head_rgbd_capture
```

结果写入：

```text
<capture-dir>/sam2_masks/tabletop_orientation_result.json

桌面候选区域由 `module_4_1_reverse_grasp_prediction/workpiece_2_pipeline/config/tabletop_workspace.json`
中的 `safe_roi_polygon_px` 定义为随分辨率缩放的桌面多边形。它同时约束深度种子、
SAM2 输出和负提示，以排除桌外地面；只有离线调试时才使用
`sam2_tabletop_masks.py --roi X Y W H` 的矩形覆盖。
```

## Python 接口

```python
from modules.module_4_head_judge import inspect_tabletop_orientation

result = inspect_tabletop_orientation(capture_dir)
print(result.orientation)
```

当结果包含 `OTHER` 且 `tilt_recovery_candidate=True` 时，调用方可以执行未来
的摆正策略，动作结束后必须重新采集并再次调用本模块确认。Module 4 本身
不会执行恢复动作。

## 当前视觉依赖

为了保持与已验证反抓视觉链路一致，本模块调用 Loop 内以下只读视觉脚本：

```text
modules/module_5_left_pick/module_5_1_reverse_grasp/implementation/
  capture_head_rgbd.py
  sam2_tabletop_masks.py
  sam2_icp_grasp.py
```

它只复用采集、SAM2、CAD、标定和 ICP，不调用 Module 5 的机械臂抓取代码。
本模块不再依赖 `/home/dainanjun/zhangfei/tools/tabletop_orientation`。
