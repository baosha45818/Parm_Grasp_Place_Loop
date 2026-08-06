# 当前双臂总控调度规则

## 左臂

1. Module 1 完成左手抓取。
2. Module 2 判断夹爪内工件为 `FRONT`、`SIDE` 或 `REVERSED`。
3. Module 3 严格按分类选择 3-1、3-2 或 3-3 完成放置。
4. 桌面状态判断属于新增 Module 4 Head Judge。
5. 桌面工件重新抓取属于 Module 5 Left Pick。
6. 无任务时进入 Module 6 Left Idle。

模块化总控在 `REVERSED` 分支中先调用 Module 4 Head Judge，确认桌面上为
单个 `BACK` 工件后再进入 Module 5-1；随后 Module 5-2 把左手持件送到
交接准备位。

## 右臂

1. 需要摆正缓冲区工件时进入 Module 7。
2. 从缓冲区抓取进入 Module 8-1。
3. 从左手接取进入 Module 8-2。
4. 抓取完成后进入 Module 9 Right Place。
5. 放置完成后进入 Module 10 Right Idle。

模块化总控直接调用 Module 8-2 的模块入口。旧
`simple_dual_arm_loop.py` 仍保持原来的兼容入口和行为。

## Module 3 映射

| 模块 | 分类 | 兼容分支名 |
|---|---|---|
| 3-1 正面放 | `FRONT` | `BRANCH_1_FRONT` |
| 3-2 侧面放 | `SIDE` | `BRANCH_2_SIDE` |
| 3-3 反面放 | `REVERSED` | `BRANCH_1_REVERSED` |

分类与分支名不一致时必须停止，不能猜测或跨分支执行。

## 共享状态

新模块化总控与旧 `simple_dual_arm_loop.py` 使用相同状态：

```text
0 = EMPTY
1 = TABLE_READY
2 = HANDOFF_READY
```

新入口复用已验证的调度实现，只替换业务模块入口，不改变这套共享状态、
ServoPose 互斥或 `Ctrl+C -> cancel_task()` 停止链路。
