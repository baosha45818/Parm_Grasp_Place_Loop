# Module 4.2：头部 RGB 工件位姿估计

本目录是独立副本：云台设定、头部 D435i RGB 采集、SAM2 分割、掩码位姿计算、
SAM2 源码、模型权重和标定 JSON 都位于此处。运行时不会调用
`/home/dainanjun/zhuyiming` 下的代码或文件。

首次在本目录安装依赖：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_2_pose_estimation
uv run --no-sync python --version
```

执行完整链路（会将云台移动至 pitch 38.5°、yaw 0°）：

```bash
uv run --no-sync python run_pose_estimation.py
```

uv run python run_pose_estimation.py

每次运行都会在 `/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/4_2/`
新建 RGB、`*_mask.png`、`*_mask_pose.json` 和标注图。
若云台已经位于观察位，可加 `--skip-head-move`。可用 `--head-pitch`、`--head-yaw`、
`--roi` 和 `--outlier-mad-scale` 覆盖默认参数。
