# Workpiece 2 reverse-grasp prediction

此目录是独立、只预测的左臂反抓工具。它不修改原有模块、采集目录或任何机器人
服务，只会在本目录写入预测 JSON、覆盖图和 `reverse_grasp_prediction.log.jsonl`。
即使传入 `--output`、`--overlay` 或 `--log`，路径也必须位于本目录。

工件模型和方向来自 `tool/workpiece_2`：`X_object=-X_stl`、
`Y_object=-Y_stl`、`Z_object=Z_stl`。反抓点沿用已标定的 STL
`[0.0405, 0, 0] m`，即工件坐标的 `[-0.0405, 0, 0] m`（左侧 -X 端向内 15 mm）。

先以当前 `body_base` ICP 结果生成预测：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
.venv/bin/python reverse_grasp/predict_and_confirm.py \
  --result output/reversed_head_pose_d435_workpiece_2/20260801_220424/result.json
```

检查 `reverse_grasp/latest_grasp_overlay.png` 与 `latest_prediction.json` 后，记录
人工确认（这依然不会执行运动）：

```bash
.venv/bin/python reverse_grasp/predict_and_confirm.py \
  --result output/reversed_head_pose_d435_workpiece_2/20260801_220424/result.json \
  --confirm-grasp-point
```

`latest_prediction.json` 仅是规划/人工复核输入，绝不能直接作为实机执行授权。

## 完整视觉链路（等价于 Module 4-1 的视觉部分）

对由上游在反抓观察位采集的 RGB-D 运行：

```bash
.venv/bin/python reverse_grasp/run_full_visual_prediction.py \
  --capture-dir /path/to/head_rgbd_capture
```

输入目录必须有 `metadata.json`、`rgb_000.png` 和 `depth_aligned_000.png`。
程序不会修改该输入目录：它会将所需三文件复制到
`reverse_grasp/runs/<时间>/capture/`，在该沙盒中依次运行 SAM2 桌面分割、
双面受限 CAD/ICP、反面几何质量门、候选去重/唯一性判定，以及 2D/3D 抓取可视化。
仅唯一反面候选会生成 `prediction.json`；否则报错并保留本目录内的诊断工件。

## 直接采集并预测

上游已将云台与工件放在反抓观察位后，可直接运行：

```bash
.venv/bin/python reverse_grasp/capture_and_predict.py
```

它仅打开头部 D435i 采集一帧中值融合 RGB-D，随后自动进入上述完整视觉链路；
不会发送云台、机械臂或夹爪命令。采集写入 `reverse_grasp/captures/`。

## 固定桌面反面位姿复核

当操作者已确认“工件在桌面上的反面”时，可固定采用该反面假设，不让正/反
评分差拒绝结果，并保存正式工件坐标轴：

```bash
.venv/bin/python reverse_grasp/estimate_fixed_reverse_pose.py \
  --capture-dir reverse_grasp/runs/<运行ID>/capture
```

输出 `fixed_reverse_axes_overlay.png`（RGB 上 X 红、Y 绿、Z 蓝）与
`fixed_reverse_pose_3d.png`（点云、CAD、三维轴），用于确认位姿。 
