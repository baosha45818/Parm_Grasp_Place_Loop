# workpiece_2 左臂凸起抓取

本目录只消费 `reverse_grasp` 已确认的反面位姿与凸起抓取点；不采集图像、不运行
SAM/ICP，也不修改 Module 4-1 或 Module 5 的文件。

夹爪采用已部署的 Version7/GraspNet TCP 约定：`+X` 是接近方向、`+Y` 是张开方向。
本任务固定为：

- TCP `+X = workpiece +Z`：反面工件的 `+Z` 指向桌面，故为竖直向下接近；
- TCP `+Y = workpiece +X`：左指在抓取点的工件 `+X` 一侧，右指在 `-X` 一侧；
- TCP `+Z = +X × +Y`。

先做无硬件 dry-run：

```bash
.venv/bin/python reverse_grasp/grasp/execute_convex_tab_grasp.py \
  --prediction-json reverse_grasp/runs/20260801_232315/capture/table_reverse_verified_final/convex_tab_grasp_xy_refined/convex_tab_grasp_prediction.json \
  --output-dir reverse_grasp/grasp/runs/check_20260801_232315
```

确认输出 JSON 后才用 `--execute`。执行时会移动左臂、开闭夹爪并抬升工件。
可先在同一命令末尾增加 `--preflight-only`，只调用 Planner 验证 PTP 预抓取和
LIN 下探是否可达，不会发送运动或夹爪命令。
