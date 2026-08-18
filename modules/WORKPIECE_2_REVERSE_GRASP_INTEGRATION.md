# 工件2反面抓取接入说明

本说明记录工件2如何接入左臂的“桌面反面物体位姿 -> 反面抓取”流程。

## 模块边界

| 阶段 | 目录 | 责任 | 输出 |
| --- | --- | --- | --- |
| Module 4 | `module_4_head_judge/module_4_1_buffer_state_estimation/` | 头部 RGB-D、SAM2、正反面与物体位姿 | `buffer_face_check.json` |
| Module 5 反面分支 | `module_5_left_pick/reverse_pick_to_handoff/grasp/workpiece_2/` | 凸耳抓取位姿、Planner 预检、左臂抓取与抬升 | `convex_tab_grasp_prediction.json`、`grasp_plan.json` |

唯一 STL 来源：`tool/workpiece_2/workpiece_2.stl`。

## Module 4：方向与物体位姿

入口：`capture_workpiece_2_reverse_pose.py`。

1. 云台移动到头部观察位，采集一帧新的头部 RGB-D；不移动机械臂或夹爪。
2. 使用该目录专用的 SAM2/深度/工作区配置分割桌面上的工件。
3. 以“桌面反面”作为已确认假设进行 ICP：工件正式坐标系为 `workpiece2_top_view`，工件 `+Z` 指向桌面、`-Z` 面向相机和外侧。
4. `estimate_workpiece_2_reverse_pose.py` 对 STL 采样点与分割点云进行动态 yaw ICP，写出 `fixed_reverse_pose.json`、RGB 坐标轴图和 3D 可视化。
5. Module 5-1 的 `grasp/predict_convex_tab_grasp.py` 消费该物体位姿，取得凸耳抓取点并计算相机坐标下的接触点、外法向和接近方向。
6. 自动反抓不叠加 `manual_grasp_offset.json`；抓取点仅使用 CAD 名义点及沿桌面法向的接触下移。

预测输出事件固定为：

```json
"event": "workpiece_2_convex_tab_grasp_prediction"
```

关键字段：

- `frame_id` / `object_frame`：预测所用坐标系，要求为 `workpiece2_top_view`。
- `grasp_contact_point_object_m`：工件坐标系下的凸耳接触点。
- `grasp_contact_point_camera_m`：头部相机坐标系下的同一接触点。
- `outward_direction_camera`、`approach_direction_camera`：抓取方向。
- `source_pose`：对应的 `fixed_reverse_pose.json`，供 Module 5-1 复核 ICP 质量。
- `artifacts`：二维叠加图、点云和 3D 可视化路径。

## Module 5-1：执行逻辑

入口：`workpiece_2_pipeline/grasp/execute_convex_tab_grasp.py`。

1. 只接受事件为 `workpiece_2_convex_tab_grasp_prediction` 的 JSON，拒绝旧工件预测或其他格式。
2. 重新读取 `source_pose` 并检查 ICP 质量门限；不通过时禁止任何运动。
3. 由 `body_head_calibration.json`、`grasp_tcp_calibration.json`、`planner_tool_bridge.json` 将头部相机下的抓取 TCP 转为 Planner 的 `body_base` 下左臂末端目标。
4. 构造抓取坐标系：TCP `+X` 为向桌面的接近方向，TCP `+Y` 与工件 `+X` 在桌面平面的投影对齐；因此左指对应工件 `+X` 侧，右指对应工件 `-X` 侧。
5. 首先到 `reverse_grasp_handoff_positions.toml` 中工件2的固定反抓预备关节位。
6. 对多个接近角做离线预检：PTP 到预抓取位，再 LIN 下探到抓取位。只有预抓取和下探都可规划时才接受该候选。
7. `--preflight-only` 只做规划预检，不发送机械臂或夹爪命令。
8. `--execute` 才执行：PTP 预抓取 -> LIN 下探 -> 夹爪闭合 -> LIN 抬升。每一段都会检查 Planner 返回轨迹和机械臂到位状态。

## 总入口与产物交接

完整编排脚本：

```bash
bash modules/module_5_left_pick/reverse_pick_to_handoff/grasp/workpiece_2/run_full_reverse_grasp.sh
```

默认仅执行：头部采集、SAM2、ICP、凸耳预测、抓取可达性预检。只有增加 `--execute` 才会真正移动左臂、闭合夹爪并抬升。

同一次运行中的路径关系：

```text
Module 4 output/module_4_1/workpiece_2_reverse/runs/<run-name>/02_icp/fixed_reverse_pose.json
    -> Module 5-1 grasp/runs/<run-name>_prediction/convex_tab_grasp_prediction.json
    -> Module 5-1 grasp/runs/<run-name>_preflight/grasp_plan.json
    -> Module 5-1 grasp/runs/<run-name>_execute/grasp_plan.json   (仅 --execute)
```

## 总控选择

模块化总控通过可选参数选择工件类型：

```bash
# 工件1：不传参数时保持默认工件1配置。
uv run modules/modular_dual_arm_loop.py ...

# 工件2：仅反面分支进入本说明的固定反面预测与凸耳反抓流水线。
uv run modules/modular_dual_arm_loop.py --workpiece 2 ...
```

总控将选择值写入 `PARM_LOOP_WORKPIECE`。在左臂反抓阶段，
`run_head_judge_then_pick.py` 仅当该值为 `2` 时调用
`workpiece_2_pipeline/run_full_reverse_grasp.sh --execute`；值为 `1` 时继续
执行原有头部采集、旧预测、方向判断和旧反抓代码，不存在隐式替换。

## 工件2交接接入

工件2反抓成功后，总控沿原有反面分支继续调度，但 Module 5-2/8-2 会因
`PARM_LOOP_WORKPIECE=2` 选择各自模块目录内的工件2实现：

```text
Module 5-2：D435观察和ICP -> 左手送 body_base [0.42, 0.10, 0.25]
Module 8-2：右手从左手接取 -> 双臂撤离到后续放置准备状态
Module 9：按 place_hole_a_wiggle_workpiece_2.json 执行右手孔位放置
Module 10：仍由原总控调度，负责右手 Idle
```

Module 5-2、Module 8-2 与模块化总控共同读取
`modules/reverse_grasp_handoff_positions.toml`。该文件按工件分区统一保存左手反抓
预备位、左手准备/观察/回程位、云台观察角、交接中心和右手交接位；其中右手
预交接/预抓取位与释放后撤离量仅属于工件2。握手中的 `profile_id` 和工件
类型必须与本次总控选择一致，否则在右臂运动前拒绝继续。

工件1保留自己的 STL、SAM2/ICP 参数和右手抓取相对几何，但观察与交接也使用
上述高位区域。第一类当前使用在新中心附近重新记录的高位右手示教关节位，旧低位
数组只保留为代码注释且不会再次叠加中心 TF 变化；预交接入口和放置准备位保持
工件1独立配置，不复用工件2不同尺寸模型的右手关节目标或工件—夹爪关系。
第一类最终接取前的预交接点由实时工件 TF 沿工件 `-X` 偏移 `50 mm` 生成，先到
该点，再直线进入当前示教交接位。

当工件2以正面/侧面路径进入缓冲区时，总控的 Module 7 与 Module 8-1 不使用
modules 内原工件链，而是直接运行外部已验证脚本：

```bash
python3 /home/dainanjun/zhuyiming/push_T_ALL/workpiece_2/q1_closed_loop_pipeline.py \
  --arm-id 2 --execute
```

该文件保留在 `zhuyiming/push_T_ALL/workpiece_2/`，总控只引用它，不复制、
不修改其实现。工件1改由模块内
`module_8_right_grasp/module_8_1_buffer_grasp/workpiece_pipeline.py` 运行。

`tests/reversed_head_observation_2/full-execute` 仍是独立联调工具，不会被总控
调用，因为它还包含右手最终放置，会与 Module 9 的职责冲突。

## 当前限制

- 固定反面假设仅适用于已经确认“工件 +Z 朝桌面”的桌面摆放状态。
- 自动执行前仍应先查看 SAM2、ICP 坐标轴和凸耳抓取点可视化，再使用 `--execute`。
