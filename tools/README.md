# 项目公共运行工具

This directory contains the project-local runtime dependencies used by the
正式模块与新总控使用的公共运行依赖。

## Layout

- `sam2-service/`: SAM2/EfficientViT 服务、第三方后端、检查点及本地运行环境。
- `graspnet_runtime/`: the minimal GraspNet-derived Planner and D405 Camera
  Broker source closure. GraspNet inference models, checkpoints, jobs, output
  and its 8.9 GB virtual environment are intentionally excluded.
- `calibration/`: 左腕、头部 D435、Planner bridge 和 TCP 标定。
- `workpiece_1/`、`workpiece_2/`: ICP 所需的 CAD 模型。
- `inspection_support/`: M4 与反面交接观察共用的相机、分割和点云辅助代码。
- `handoff_observation/`: 右手交接观察、位姿估计、交接规划和执行的通用工具；由 M5.2、M8.2 调用，不是独立业务模块。
- `grasp_weight_check/`: 左右夹爪抓取后的力传感器重量判定工具；由 M1、M5.2 和 GraspNet 运行时调用。
- `planner_runtime.py`、`planner_task_retry.py`、
  `left_motion_right_servo_lock.py`: 规划与双臂动作支持。
- `icp.py`: 共享 ICP 求解器。

旧 `tool/` 目录已经删除；任何正式模块都不得再引用它。`graspnet_runtime/LICENSE`
保留上游 GraspNet 许可证，SAM 后端许可证与源码放在一起。

## ICP 求解器（`icp.py`）

`icp.py` 只提供点云配准的数值原语，不处理 SAM mask、CAD 正反面、
抓取点或机器人动作。调用方负责生成点云、提供初始位姿，并根据返回的
`fitness` 和 `rmse` 做业务层决策。

### 输入、输出与约定

- `source`: `N x 3` 浮点数组，通常是 CAD 表面采样点。
- `target`: `M x 3` 浮点数组，通常是 RGB-D mask 反投影得到的观测点云。
- 所有点、搜索半径、对应距离和 RMSE 都以米为单位。
- `initial` 和返回值都是 `4 x 4` 齐次变换，方向为
  `target_frame <- source_frame`，即 `p_target = T @ p_source`。
- ICP 是局部优化器；初始位姿差得过大时，可能收敛到错误的局部极值。

### 公开接口

| 接口 | 自由度 | 用途 |
| --- | --- | --- |
| `pca_initial_guesses` | 初值生成 | 用 PCA 主轴对齐生成 4 个合法旋转假设，再加一个仅对齐质心的初值。 |
| `solve_multi_hypothesis_icp` | 6DoF | 对每个初值分别运行 Open3D 点面和/或点点 ICP，返回全部候选。 |
| `full_6d_point_to_plane_icp` | 6DoF | 使用 Open3D 点面 ICP 精配准，可同时优化 XYZ 平移和 roll/pitch/yaw。 |
| `tabletop_constrained_icp` | 4DoF | 点点距离版本；只优化 XYZ 平移和绕桌面法向的 yaw。 |
| `tabletop_constrained_point_to_plane_icp` | 4DoF | 项目定制的点面版本；用于固定正反面假设，防止 ICP 通过 roll/pitch 把工件翻面。 |

`solve_multi_hypothesis_icp` 不会自动挑选“最佳”结果，因为不同业务可能还需要检查
正反面、桌面穿透、安全 ROI 或 CAD 深度分数。

### 点面 ICP 怎样建立匹配

设当前位姿变换后的一个源点为 `p_i`，它在目标点云中的最近邻为
`q_i`，`q_i` 处的单位法向为 `n_i`。点面 ICP 不直接最小化
`||p_i - q_i||`，而是最小化连线在目标法向上的投影：

```text
r_i = n_i^T (p_i - q_i)
minimize  sum_i w_i r_i^2
```

因此，沿表面切向的小误差影响较小，沿表面法向的误差会被直接纠正。
在初始位姿足够接近真值、目标法向可靠时，它通常比点点 ICP 收敛更快。

`full_6d_point_to_plane_icp` 将这个过程交给 Open3D：

1. 在目标点云上估计法向；法向搜索半径为
   `max(2 * max_corr_m, 0.008)`，最多使用 40 个邻点。
2. 用当前位姿变换 `source`，并在 `max_corr_m` 内建立最近邻对应。
3. 使用 Tukey loss 抑制边缘、遮挡和 mask 溢出带来的外点；
   Tukey 尺度为 `0.5 * max_corr_m`。
4. 求解全 6DoF 增量，更新位姿，再重新建立对应，直到收敛或达到
   `max_iterations`。

### 桌面约束点面 ICP

`tabletop_constrained_point_to_plane_icp` 自行实现了一个 Gauss-Newton 迭代。
它不求解 6DoF，而只求解四个增量：

```text
x = [delta_yaw, delta_tx, delta_ty, delta_tz]
```

其中 `delta_yaw` 的旋转轴是单位桌面法向 `u`。小角度线性化后，
每组对应的 Jacobian 为：

```text
J_i = [n_i^T (u x p_i), n_ix, n_iy, n_iz]
```

每次迭代的实际流程是：

1. 用当前 `T` 变换 CAD 源点。
2. 用 `cKDTree` 为每个源点寻找最近目标点。
3. 先保留距离小于 `max_corr_m` 且目标法向有效的对应；如果少于
   30 组，尝试补入最近的最多 250 组。
4. 按欧氏对应距离删掉最差的 18%，降低边缘和错配影响。
5. 计算点面残差 `r_i`，再使用 Tukey biweight：

   ```text
   s = max(0.5 * max_corr_m, 1e-6)
   w_i = (1 - (r_i / s)^2)^2,  if |r_i| < s
   w_i = 0,                    otherwise
   ```

6. 求解带轻微阻尼的加权最小二乘：

   ```text
   (J^T W J + D) x = -J^T W r
   ```

   如果正规方程不可解，则回退到 `numpy.linalg.lstsq`。
7. 每次迭代将 yaw 增量限制在 `+/-5 deg`，每个平移分量限制在
   `+/-5 mm`，然后用 `T <- delta_T @ T` 更新。
8. 当 `|delta_yaw| < 2e-5 rad` 且平移增量范数小于 `2e-6 m`时提前结束。

这个 4DoF 限制是 Module 4 工件 1 正反面判定的关键：调用方分别给出
正面朝上和反面朝上的初始姿态，ICP 只能绕桌面旋转和平移，不能用
roll/pitch 把一个面假设翻成另一个。

### 质量指标

`full_6d_point_to_plane_icp`、`tabletop_constrained_icp` 和
`tabletop_constrained_point_to_plane_icp` 返回：

```python
transform, rmse_m, fitness = ...
```

- `transform`: 最终 `target <- source` 齐次变换。
- `rmse_m`: 最终内点对应的欧氏距离 RMSE，单位是米。
- `fitness`: 源点中最终最近邻距离小于 `max_corr_m` 的比例。

`solve_multi_hypothesis_icp` 则返回候选列表；每个候选包含变换矩阵及
`method`、`initialization`、`fitness`、`rmse_m` 诊断字段。

注意：定制 4DoF 点面 ICP 的优化残差是点面距离，但为了与旧质量门槛
保持量纲一致，返回的 `rmse_m` 仍然是最终对应的欧氏距离 RMSE，
不是点面残差 RMSE。

### 最小使用示例

```python
import numpy as np

from tools.icp import tabletop_constrained_point_to_plane_icp

# CAD 点云和观测点云均以米为单位。
source = np.asarray(cad_points_m, dtype=np.float64)
target = np.asarray(observed_points_m, dtype=np.float64)

initial = np.eye(4)
initial[:3, 3] = target.mean(axis=0) - source.mean(axis=0)

transform, rmse_m, fitness = tabletop_constrained_point_to_plane_icp(
    source,
    target,
    initial,
    table_normal=np.array([0.0, 0.0, 1.0]),
    max_corr_m=0.009,
    max_iterations=45,
)
```

调用方不应只根据单一 `fitness` 放行结果。实际工件检查还应结合
RMSE、CAD 深度一致性、正反候选分差、桌面穿透和安全工作区。

### 持久 CAD 点云模板

Module 4 直接加载下列固定模板，不再为每次状态估计重新从 STL 采样：

- `tools/workpiece_1/cad_pointcloud_template.npz`
- `tools/workpiece_2/cad_pointcloud_template.npz`

每个档案包含单位为米的 `icp_points_m`（14,000 点）和
`render_points_m`（90,000 点），以及源 STL 路径、SHA-256 和采样种子。
修改 CAD 后必须重新生成模板：

```bash
.venv/bin/python tools/generate_cad_pointcloud_templates.py --workpiece all
```

## SAM2 environment

For a fully independent SAM2 environment, run:

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tools/sam2-service
uv sync --python 3.10
```

启动脚本只使用本目录的 `.venv`，不再回退到旧目录。
