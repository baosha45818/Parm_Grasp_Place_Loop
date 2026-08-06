# 模块 3：左手放

模块 3 将左夹爪持有的工件放到目标位置。放置阶段不使用 Inspect 的分类或
夹爪开度来选择放置动作：两个工件都会生成正面与反面候选，并以 `T_body_base_object` 中工件 `+Z` 相对
`body_base +Z/-Z` 的夹角决定候选顺序。

- 夹角小于配置门限且接近 `+Z`：先完整尝试正面，再尝试反面。
- 夹角小于配置门限且接近 `-Z`：先完整尝试反面，再尝试正面。
- 其余姿态：正反候选交替提交。
- 候选经滑动窗口并发规划；任一候选首先返回可行解后，不再提交新的候选。

共同执行骨架为：回 `graspup`、LIN 上移至 `graspup_above`、离线规划目标、
下降松夹爪、返回 `graspup`。最终目标末端必须满足
`left_hand_transfer_end +Z` 朝 `body_base -Z`。

## 参数文件

`placement_parameters.py` 只保留共享的 `graspup`、上方关节起点、超时和丢弃
恢复参数。每个工件的正反面放置配置独立存放：

- `workpiece_1_placement_parameters.toml`：保留工件1既有正反面中心、Z、yaw 和
  局部 RPY 候选；当前仅使用中心 XY。
- `workpiece_2_placement_parameters.toml`：保留工件2现有正反面局部 RPY 候选；每面
  仅使用中心 XY。

两个工件的 Roll/Pitch/Yaw 都定义在工件局部坐标系；正面零候选为
`[0, 0, 0]`，反面零候选为局部 `roll=180°`。TOML 由
`placement_config.py` 读取并在规划前校验。

可用 `--post-graspup-planner ptp|rrtconnect|rrtstar` 切换离线笛卡尔规划器，
默认 `rrtstar`；`graspup` 上移仍为 LIN。
