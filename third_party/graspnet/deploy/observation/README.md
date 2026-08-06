# 工件观察与位姿估计（第一版）

本目录只做离线观察和位姿估计，**没有机械臂、夹爪或相机控制代码**。

实现严格采用 `graspnet/example/place/pc_pose` 已验证的流程：

```text
SAM2 mask + 对齐深度 + Femto 内参
  -> 工件点云（laser）
  -> T_base_laser 转到 arm1 base
  -> STL 表面采样
  -> PCA 多初值
  -> point-to-point ICP
  -> T_base_object
```

`T_base_object` 将 STL 对象坐标转换为 arm1 `base` 坐标。它是后续计算“屁股端朝向 arm2”所需的唯一位姿输入；本第一版不做该规划或任何真实运动。

## 输入要求

- `mask`：SAM2 输出的二值掩膜；必须与深度图像像素对齐。
- `depth`：Femto 的对齐深度，支持 `.npy` 或单通道 16 位图像。
- `depth-scale`：一个深度单位对应的米数。
- `laser-calibration`：现有的 `calibration_laser.json`，必须具有 `status: ACCEPTED`、`camera_matrix` 和 `T_base_laser`。
- `stl`：工件 CAD。本工件当前 STL 已使用米单位；程序也兼容常见的毫米 STL。

## 运行

```bash
cd /home/dainanjun/qiuguolin/graspnet
.venv/bin/python /home/dainanjun/qiuguolin/graspnet/deploy/observation/pose_estimator.py \
  --mask /path/to/sam2_mask.png \
  --depth /path/to/aligned_depth.npy \
  --depth-scale 0.001 \
  --stl /home/dainanjun/qiuguolin/graspnet/deploy/tools/welding_up_side_1118_only_2_small_screws_removed.stl \
  --laser-calibration /home/dainanjun/qiuguolin/graspnet/deploy/calibration/output/laser/calibration_laser.json \
  --output /home/dainanjun/qiuguolin/graspnet/deploy/observation/output/pose.json
```

输出包含最佳及全部 PCA/ICP 候选、`fitness`、`rmse_m`、内点数和 `T_base_object`。当前阶段应人工检查结果和质量指标；在接入任何放置动作前，再根据实测数据确定可接受阈值。
