# 模块 3：左手放

模块 3 将左夹爪持有的工件放到目标位置。放置阶段不使用 Inspect 的分类或
夹爪开度来选择放置动作，而是以 Inspect 同步写入的 `T_left_eef_object` 中工件
`+Z` 相对 `left_eef -X/+X` 的夹角决定候选顺序。

- 工件 `+Z` 与 `left_eef -X` 夹角小于配置门限：工件1先完整尝试正面、再完整尝试反面；
  两者均无解后才尝试直接放置。工件2仍保持正面、反面的原有顺序。
- 工件 `+Z` 与 `left_eef +X` 夹角小于配置门限：工件1只完整尝试反面；反面无解后才尝试
  直接放置，不会提交正面。工件2仍先反面、再正面。
- 两边都不小于门限：工件1走 `[direct place]`，在 `graspup_top` 读取实际
  `left_hand_transfer_end` RPY 并保持不变，只将工件中心移到该分支的目标 XYZ；工件2仍正反
  候选交替提交。
- 候选经滑动窗口并发规划；任一候选首先返回可行解后，不再提交新的候选。

到达 `graspup_top` 后的候选等待上限为 6 秒。对工件1的轴向正反候选，超过该时限不会丢弃：
已在飞的正反请求结果不再选用，并转入 `DIRECT`；`DIRECT` 本身无解或不满足末端朝下硬约束时，
该轮会安全失败，绝不强行执行。工件2仍按其原有超时恢复处理。

共同执行骨架为：回 `graspup`、LIN 上移至 `graspup_above`、离线规划目标、
下降松夹爪、返回 `graspup`。最终目标末端必须满足
`left_hand_transfer_end +Z` 朝 `body_base -Z`。

## 参数文件

`placement_parameters.py` 只保留共享的 `graspup`、上方关节起点、超时和丢弃
恢复参数。每个工件的正反面放置配置独立存放：

- `workpiece_1_placement_parameters.toml`：保留工件1既有正反面中心、Z、yaw 和
  局部 RPY 候选；并定义 `direct place` 的固定工件中心和 Z 候选；当前仅使用中心 XY。
- `workpiece_2_placement_parameters.toml`：保留工件2现有正反面局部 RPY 候选；每面
  仅使用中心 XY。

正反面候选的 Roll/Pitch/Yaw 都定义在工件局部坐标系；正面零候选为
`[0, 0, 0]`，反面零候选为局部 `roll=180°`。TOML 由
`placement_config.py` 读取并在规划前校验。

可用 `--post-graspup-planner ptp|rrtconnect|rrtstar` 切换离线笛卡尔规划器，
默认 `rrtstar`；`graspup` 上移仍为 LIN。
