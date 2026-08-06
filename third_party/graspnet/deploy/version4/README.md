# Version4：arm1 腕部 D405 抓取与放置

Version4 只使用完成手眼标定的 arm1 腕部 D405，不会打开 Femto Bolt，也不会加载 `T_base_laser`。

## 安全推理

从机械臂当前位置采集并推理：

```bash
cd /home/dainanjun/qiuguolin/graspnet

CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  deploy/version4/run_d405_graspnet.py
```

## 观察位姿

先让 arm1 移动到固定 D405 观察位姿，再执行推理：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  deploy/version4/run_d405_graspnet.py \
  --move-observation \
  --confirm-real-movej
```

默认观察关节角为：

```text
[-1.559, -8.031, -153.808, -57.798, -162.852, 59.599, -42.667] deg
```

该目标通过 `Planner.plan_joint_trajectory(group="left_arm", planner="ptp")` 规划，再由 `POST /middle/waypoint/execute` 执行；不直接调用 `Arm.move_j`。

最高分抓取以 `arm1_base` 坐标保存到 `top_grasp.json`。腕部标定文件必须包含 `T_eef_wrist1`，不需要激光相机标定。

## 真实抓取与放置

检查 Open3D 预测结果后，如需执行抓取，必须同时加入两个执行开关：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  deploy/version4/run_d405_graspnet.py \
  --roi X Y W H \
  --execute-grasp \
  --confirm-real-grasp
```

程序会在打开 Open3D 前写入 `grasp_execution_plan.json`。抓取计划中的笛卡尔目标在 `arm1_base`；执行前程序会用已验收 TF bridge 和 Planner 的实时 TF 转到 `body_base`。关闭窗口后，终端还会要求输入 `y` 确认。确认后程序执行以下动作：

1. 将夹爪打开到 `预测宽度 + 0.005 m`。
2. 用 Planner Cartesian LIN 到预测 XY、旋转姿态和 `Z + 0.05 m` 的预抓取位姿。
3. 用 Planner Cartesian LIN 垂直下降并闭合夹爪。
4. 保持夹爪闭合，依次用 Planner PTP 到抬升关节位姿和放置关节位姿。
5. 打开夹爪并释放工件。

默认抓取后抬升关节角为：

```text
[-6.288, -38.805, 12.556, 72.427, 4.059, 67.331, -11.391]
```

默认放置和释放关节角为：

```text
[-62.016, 37.312, -23.766, 55.214, 1.200, -0.677, 78.087]
```

只有需要修改这些已经验证的位姿时，才使用 `--postgrasp-lift-joints-deg` 或 `--place-joints-deg`。`--postgrasp-movej-speed` 同时控制抬升和放置两次运动，默认值为 `0.1`。

夹爪默认映射为 `100% = 0.078 m` 开口宽度，旋转偏置默认为 `0` 度。只有确认实际夹爪标定后才能覆盖这些参数。
