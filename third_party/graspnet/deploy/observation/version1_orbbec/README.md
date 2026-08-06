# Version 1：Orbbec 直接观察与路由

该版本对应方法 1：工件抓住后，先由既有 `go_graspup.py` 抬到安全位，再由既有 `go_pc.py` 到固定 Orbbec 观察位。这里的代码只采集、估计和输出路由，**不会发送机械臂或夹爪命令**。

```text
go_graspup（外部执行） → go_pc（外部执行）
→ capture_orbbec.py
→ 深度坐标 ROI + depth-aligned SAM2 mask.npy
→ classify_orbbec.py
→ front_grip / back_grip / side_grip
```

分类来自 STL 点云配准后在 `go_pc` 时刻向下的 CAD 面：

| 向下面 | 输出类别 | 下一步 |
| --- | --- | --- |
| front | `front_grip` | `front_flow` |
| back | `back_grip` | `manual_review` |
| left / right | `side_grip` | `go_pc2` |

该映射定义在 `config_template.json`，在实机样本验证后可改；`back_grip` 当前刻意不自动继续。

## 采集

确认 arm1 已停在 `go_pc` 后运行：

```bash
cd /home/dainanjun/qiuguolin/graspnet
.venv/bin/python /home/dainanjun/qiuguolin/graspnet/deploy/observation/version1_orbbec/capture_orbbec.py
```

采集程序只读取 arm1 EEF 位姿并保存 Femto RGB-D。输出目录会包含 `depth.npy`、`depth_intrinsics.npy`、`color.png` 和 `meta.json`。

然后用以下工具在深度图坐标中选择 ROI：

```bash
.venv/bin/python /home/dainanjun/qiuguolin/graspnet/deploy/observation/version1_orbbec/select_depth_roi.py \
  --capture-dir /home/dainanjun/qiuguolin/graspnet/deploy/observation/version1_orbbec/output/captures/<timestamp>/go_pc
```

再保存一个与 `depth.npy` 同分辨率、仅含工件的 `mask.npy`。若 `meta.json` 中 `depth_in_color_pixels` 为 `false`，不得直接使用彩色图产生的 SAM2 mask。

## 离线分类

```bash
.venv/bin/python /home/dainanjun/qiuguolin/graspnet/deploy/observation/version1_orbbec/classify_orbbec.py \
  --capture-dir /home/dainanjun/qiuguolin/graspnet/deploy/observation/version1_orbbec/output/captures/<timestamp>/go_pc
```

结果保存为 `output/classification/.../result.json`。它包含 `T_eef_object`、ICP 质量指标、当前向下面和下一步路由。
