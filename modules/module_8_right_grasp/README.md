# 模块 8：Right Grasp

本目录是右手抓取模块组：

- `8-1`：从中间缓冲区抓取。
- `8-2`：读取 Module 5-2 的本轮 ICP 结果，直接从左手接取并撤离到
  Module 9 上方准备位。

Module 8-2 的独立入口位于 `module_8_2_left_handoff/main.py`。旧兼容总控已删除，
Module 5.2 与 Module 8.2 仅由 `controller/main.py` 调度。
