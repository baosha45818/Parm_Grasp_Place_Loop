# 模块 5：左手 Buffer 重抓

Module 5 只消费 Module 4.1 发布的正反状态与 ICP TF；不采集图像、不运行 ICP。
总控根据 Module 4.1 的标准结果选择分支：

- `module_5_1_front_pick_to_graspuptop/`：`FRONT`。左手按 ICP TF 重抓后保持持件并回到 `graspuptop`；总控再调用 Module 3 的 `middle` 放置。
- `module_5_2_reverse_pick_to_handoff/`：`BACK`。左手按 ICP TF 反抓、观察并得到本轮交接 ICP；随后先让左臂到高位交接中心，确认成功后再让右臂从 Idle 到安全预抓取位。两臂均确认到位后才交给 Module 8.2 做最终 LIN 接取。

`UNCERTAIN`、观察次数、重抓次数和人工恢复全部由 `controller/scheduler_rules.py` 处理；Module 5 不猜测工件状态。

Module 5.1 的输入契约、独立启动命令、安全边界和输出字段见
[`module_5_1_front_pick_to_graspuptop/README.md`](module_5_1_front_pick_to_graspuptop/README.md)。
需要测试“自动重抓”或“人工已持件后的预放置”时，使用统一入口
[`TESTING.md`](TESTING.md)；不要把测试命令接入总控运行。

## Module 5.2 配置边界

所有会驱动机械臂的反抓和交接位置统一来自
[`reverse_grasp_handoff_positions.toml`](../reverse_grasp_handoff_positions.toml)：

```text
deployment.gzhu.workpiece_1
deployment.gzhu.workpiece_2
deployment.xiangxin.workpiece_1
deployment.xiangxin.workpiece_2
```

每套都包含 `left_reverse_grasp`、`left_handoff`、`right_handoff`。总控默认选择
`gzhu/workpiece_1`；显式传入 `--mode`、`--workpiece` 时，只读取对应的一个分区。
旧 `grasp/config/reverse_ready.json` 已删除。

工件 1 的视觉语义、CAD 抓取点补偿和接近角仍是算法配置，按现场保存在下列文件；它们不保存
机械臂反抓预备位或交接关节位：

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

## Module 5.2 串行交接准备

工件1的 M5.2 在 ICP 成功后严格串行执行：左臂先按拍摄时的 TCP—工件刚性关系把工件送到
固定高位交接中心并确认到位；随后右臂确保夹爪张开，再从 Idle 到安全预抓取位。右臂优先回放
`module_8_right_grasp/module_8_2_left_handoff/trajactory/workpiece_1/idle_to_pregrasp.json`；缓存不可用时，
根据名义高位交接姿态请求 RRTstar，并确认七轴到位后更新该缓存。

该预抓取只是安全接近，不使用本轮 ICP 终点作为缓存复用条件。两个串行动作完成构成屏障：只有左臂
在高位、右臂在预抓取位时，M5.2 才发布成功结果。后续 M8.2 重新读取实时左 TCP，以本轮 ICP
关系计算真正抓取 TCP，并用 LIN 完成最后一段；因此预抓取缓存不会替代最终抓取精度。
