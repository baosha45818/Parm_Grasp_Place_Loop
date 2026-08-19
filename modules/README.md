# 模块目录

当前模块均由新总控 `controller/main.py` 编排：

```text
Module 1 → Module 3(buffer) → Module 4.1 → Module 5
  ├─ FRONT → Module 3(middle) → Module 4.2/7/8.1 → Module 9 → Module 10
  └─ BACK  → Module 8.2 → Module 9 → Module 10
```

Module2 已删除；Buffer 正反和 ICP 位姿由 Module 4.1 统一发布。

## 当前模块

- `module_0_prepare/`：公共服务与机器人基础设施预检。
- `module_1_left_grasp/`：D405 + SAM2 + STL/ICP左臂抓取。
- `module_3_left_place/`：Buffer 与 Middle 固定放置。
- `module_4_head_judge/`：Buffer 状态估计和右手抓取前位姿估计。
- `stage_result.py`：Module3及其他模块仍使用的通用原子结果文件协议，必须保留。

旧总控、旧实验链和根目录 `tests/` 已删除；不要再从旧脚本启动实机流程。
