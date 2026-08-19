# 模块 5：左手 Buffer 重抓

Module 5 只消费 Module 4.1 发布的正反状态与 ICP TF；不采集图像、不运行 ICP。
总控根据 Module 4.1 的标准结果选择分支：

- `module_5_1_front_pick_to_graspuptop/`：`FRONT`。左手按 ICP TF 重抓后保持持件并回到 `graspuptop`；总控再调用 Module 3 的 `middle` 放置。
- `module_5_2_reverse_pick_to_handoff/`：`BACK`。左手按 ICP TF 反抓并进入双臂交接准备，随后由 Module 8.2 接取。

`UNCERTAIN`、观察次数、重抓次数和人工恢复全部由 `controller/scheduler_rules.py` 处理；Module 5 不猜测工件状态。
