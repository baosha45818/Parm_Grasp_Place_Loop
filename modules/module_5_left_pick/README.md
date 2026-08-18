# 模块 5：左手 Buffer 抓取

Module 5 只消费 Module 4-1 已发布的正反状态和 ICP 工件 TF，不采集图像、不运行 ICP。

## 两个业务分支

- `front_pick_to_graspuptop/`：唯一 `FRONT` 工件。左手按 ICP TF 正面抓取并抬到
  `graspuptop`；总控随后调用 Module 3，将已验证持件放到中间目标位置。
- `reverse_pick_to_handoff/`：唯一 `BACK` 工件。左手按同一份 ICP TF 反面抓取，
  再由本分支的 handoff 模块送到双臂交接位置。

`buffer_pick_router.py` 是唯一的 Module 4-1 → Module 5 路由契约：
`FRONT_TO_INTERMEDIATE`、`BACK_TO_HANDOFF` 或安全拒绝的 `OTHER`。
`OTHER` 由总控处理重试与人工暂停，Module 5 不得猜测抓取。

## 职责

根据 Module 4 Head Judge 给出的桌面工件状态，控制左手抓起指定工件。

模块化总控通过 Module 4-1 的 JSON 报告和 `buffer_pick_router.py` 串联 Module 4
与 Module 5。Module 4 独占云台 RGB-D/SAM2/ICP，并一次发布正反与工件 TF；
Module 5 绝不再次采集或 ICP：

1. `FRONT_TO_INTERMEDIATE`：`front_pick_to_graspuptop/main.py` 消费正面 ICP TF，左手
   正面抓取后由 Module 3 送到中间 Buffer。
2. `BACK_TO_HANDOFF`：工件 1 由 `reverse_pick_to_handoff/main.py`、工件 2 由凸耳
   适配器消费同一份 Module 4-1 TF，随后 Module 5-2 送到双臂交接位。
3. `OTHER`：不允许抓取；总控重新调用 Module 4-1。默认额外重试 2 次，仍无法
   判断时等待双臂完成当前任务并由人工处理后重新检测。

总控参数：`--buffer-check-mode xiangxin|gzhu` 选择对应相机标定，
`--module4-uncertain-retries N` 设置完整 Module 4-1 调用的额外重试次数。
Module 4-1 自己每次调用最多四次采集的策略不变。

## 子模块

- `buffer_pick_router.py`：Module 4-1 → Module 5 的唯一状态/TF 路由契约。
- `reverse_pick_to_handoff/handoff/`：左手持件到观察位、复检并移动到双臂交接位。

Module 5-1 完成桌面反抓；Module 5-2 保持左夹爪夹紧，将工件送到右手
可接取的位置并输出本轮 ICP `result.json`。右臂接取属于 Module 8-2。
