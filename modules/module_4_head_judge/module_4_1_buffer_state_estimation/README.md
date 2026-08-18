# Module 4.1：Buffer 正反与工件位姿

Module 4.1 使用一次头部 RGB-D 采集，同时发布工件正反和 ICP 位姿。它不计算
抓取 TCP，也不控制机械臂或夹爪。

- 工件1：正反两面各4个 yaw 初值，执行桌面约束 ICP。
- 工件2：FPFH/RANSAC 粗配准、语义半转候选和全6DoF点到平面 ICP。
- 两种工件共用采集、SAM2客户端、桌面几何和一份配置驱动的 mask 实现。
- 一次调用首次不确定时最多额外采集3次，即最多4次采集。

## 目录

```text
module_4_1_buffer_state_estimation/
├── check_buffer_face.py           # Buffer严格放行入口
├── deployment_modes.py            # xiangxin/gzhu 相机与标定原子绑定
├── run_orientation.py             # 工件1/2统一视觉入口
├── full6d_pose_solver.py
├── vision/
│   ├── capture_rgbd.py
│   ├── sam2_client.py
│   ├── geometry.py
│   ├── mask.py                    # 两工件唯一mask实现
│   ├── workpiece_1_icp.py
│   └── workpiece_2_icp.py
├── configs/
│   ├── workpiece_1.json
│   ├── workpiece_2.json
│   └── <mode>_workpiece_1_workspace.json / workpiece_2_workspace.json
├── calibration/
├── assets/
└── tests/
```

原来的 `workpiece_1_orientation_pipeline/` 和
`workpiece_2_orientation_pipeline/` 已删除。工件差异由两份 JSON 配置和两个
ICP策略表达，不再用两套目录复制公共代码。

## 统一入口

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

.venv/bin/python \
  modules/module_4_head_judge/module_4_1_buffer_state_estimation/run_orientation.py \
  --mode xiangxin --workpiece 1 --capture-dir /path/to/capture

.venv/bin/python \
  modules/module_4_head_judge/module_4_1_buffer_state_estimation/run_orientation.py \
  --mode gzhu --workpiece 2 --capture-dir /path/to/capture
```

结果统一写入：

```text
<capture-dir>/sam2_masks/orientation_diagnostics.json
```

`module.py` 只调用上述统一入口，不再维护 `VISION_ROOTS`。

## Buffer Check

```bash
.venv/bin/python \
  modules/module_4_head_judge/module_4_1_buffer_state_estimation/check_buffer_face.py \
  --mode xiangxin --workpiece 1
```

退出码：`0=FRONT`、`20=BACK`、`21=UNCERTAIN`、`22=运行错误`。

`--mode` 必须显式为 `xiangxin` 或 `gzhu`。两套 D435 序列号、外参和采集尺寸
只从 `tools/calibration/head_d435/<mode>_reverse_observation.json` 读取；代码会核对
采集 metadata 中的序列号，发现相机与标定串用时立即拒绝继续。

报告 `buffer_face_check.json` 同时包含 `decision`、唯一物体的 ICP `pose`、
`orientation_diagnostics`、`body_head_calibration`、`camera_serial` 和 `mode`。
Module 5-1 直接消费该报告生成抓取 TCP，不再采集或重复 ICP。

历史工件2采集和运行结果已从源码目录迁移到：

```text
output/module_4_1/archive/workpiece_2_orientation_pipeline/
```
