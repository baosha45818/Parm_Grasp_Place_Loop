# Module 4-1 Workpiece 2 Pipeline

这是 Module 4-1 的新工件 2 隔离实现，不会覆盖同级旧 `main.py` 或旧
`implementation/`。它负责：头部 RGB-D 采集、以桌面几何自动生成正点的 SAM2 分割、固定
桌面反面假设下的全 6DoF CAD/ICP，以及凸起抓取点预测。

目录内的 `vision_support/` 是新流程专用的相机/SAM2 分割/ICP 支持代码；`config/` 是该流程
自己的头部标定、几何工作区和 ICP 执行质量门限副本。新工件 STL 仍使用全项目唯一的
`tool/workpiece_2/workpiece_2.stl`。

入口由 Module 5-1 的 `workpiece_2_pipeline/run_full_reverse_grasp.sh` 编排；也可以单独运行：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
.venv/bin/python modules/module_4_head_judge/module_4_1_reverse_grasp_prediction/workpiece_2_pipeline/capture_sam_fixed_icp.py \
  --run-dir modules/module_4_head_judge/module_4_1_reverse_grasp_prediction/workpiece_2_pipeline/runs/test
```

该入口会移动云台并采集图像，但不会移动机械臂或夹爪。
## 每次运行的输出

每次运行目录固定分层，避免把分割、ICP 和抓取点日志混在一起：

```text
<run>/capture/             原始 RGB-D 与多帧融合结果
<run>/01_sam2_mask/        正点图、SAM2 完整工件 mask 和 mask overlay
<run>/02_icp/              反面全 6DoF ICP、CAD 坐标轴和候选日志
<run>/03_grasp/            抓取点预测图、抓取点点云和 JSON
```
