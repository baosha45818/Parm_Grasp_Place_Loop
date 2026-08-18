# 当前调度边界

旧总控等待重写。新的左臂调度器应遵守以下最小顺序：

1. 启动并检查 Planner、SAM2、Camera Broker和模式对应的外设服务。
2. 调用 Module1完成左臂抓取。
3. 只在 Module1成功写出有效 `SINGLE` handoff后调用 Module3。
4. Module3把持件放到固定 Buffer并返回 `graspup`。
5. 后续从 Module4开始处理 Buffer中的工件。

Module2 已删除，新总控不得再等待 Module2 ready/permit文件、Inspect分类或
`T_body_base_object/T_tool_object` 结果。

## 授权与自动执行

- Module1真实运动必须同时传入 `--execute --confirm-real-grasp`。
- Module3真实运动必须同时传入 `--execute --confirm-real-place`。
- 两个模块内部均没有人工回车或 `y/n` 提示。
- 可选的文件式 ready/permit门禁只用于外部调度器同步；未配置时模块连续执行。

## 失败传播

- Module1退出码非零或未写出有效 handoff：不得启动 Module3。
- Module3检测到掉件、Planner失败或恢复失败：停止本轮并保存结果，不得假定
  Buffer已有工件。
- 新总控必须以本轮 handoff和结果文件为准，不能复用上一轮文件。
