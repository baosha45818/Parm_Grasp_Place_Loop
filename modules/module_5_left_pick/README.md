# 模块 5：左手 Buffer 重抓

Module 5 只消费 Module 4.1 发布的正反状态与 ICP TF；不采集图像、不运行 ICP。
总控根据 Module 4.1 的标准结果选择分支：

- `module_5_1_front_pick_to_graspuptop/`：`FRONT`。左手按 ICP TF 重抓后保持持件并回到 `graspuptop`；总控再调用 Module 3 的 `middle` 放置。
- `module_5_2_reverse_pick_to_handoff/`：`BACK`。左手按 ICP TF 反抓并进入双臂交接准备，随后由 Module 8.2 接取。

`UNCERTAIN`、观察次数、重抓次数和人工恢复全部由 `controller/scheduler_rules.py` 处理；Module 5 不猜测工件状态。

Module 5.1 的输入契约、独立启动命令、安全边界和输出字段见
[`module_5_1_front_pick_to_graspuptop/README.md`](module_5_1_front_pick_to_graspuptop/README.md)。

## Module 5.2 反抓配置

工件 1 的反抓配置按现场原子绑定，均为 TOML：

- `module_5_2_reverse_pick_to_handoff/grasp/config/gzhu_workpiece_1_grasp.toml`
- `module_5_2_reverse_pick_to_handoff/grasp/config/xiangxin_workpiece_1_grasp.toml`

调用时由 Module 4.1 结果中的 `mode` 选择对应文件，不能将两个现场的配置交叉使用。
一份文件同时保存工件语义、抓取点补偿与执行接近角；不再为 gzhu/xiangxin 的工件 1 反抓维护 JSON 配置。

| 参数 | 含义 |
| --- | --- |
| `wide_head_outward_grasp_offset_m` | 宽头背离相机一侧时，抓取中心沿工件长轴向外的补偿，单位 m。 |
| `wide_head_inward_grasp_offset_m` | 宽头朝向相机一侧时，抓取中心沿工件长轴向内的补偿，单位 m。 |
| `wide_head_image_left_grasp_lateral_offset_m` / `wide_head_image_right_grasp_lateral_offset_m` | 宽头在图像左/右时，沿 `grasp_frame_+Y` 的横向补偿，单位 m。当前两个现场均为 0。 |
| `grasp_width_axis_stl` | CAD 坐标系中的夹爪开合轴；未设置时由 CAD 几何自动推导。 |
| `execution.approach_angles_deg` | 相对桌面的接近角候选。`90` 为垂直下抓，`±75` 为相对垂直侧倾 `±15` 的候选。Planner 只会执行通过可达性与碰撞预检的候选。 |

当前 gzhu 保留宽头内外补偿（向外 5 mm、向内 10 mm）；xiangxin 的
`wide_head_outward_grasp_offset_m` 与 `wide_head_inward_grasp_offset_m` 都为 `0`，因此不施加宽头内外偏置。
