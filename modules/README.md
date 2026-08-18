# 模块目录

当前已整理并可独立运行的左臂链路为：

```text
module_1_left_grasp
  → output/module_1_left_grasp/verified_handoff.json
module_3_left_place
```

Module2 已删除。Module1抓住并验证单个工件后，Module3直接将它放到固定
Buffer；不再经过胸前相机 Inspect、正反面分类或 Module2结果文件。

## 当前模块

- `module_0_prepare/`：部分公共服务预检与 Camera Broker管理。
- `module_1_left_grasp/`：D405 + SAM2 + STL/ICP左臂抓取。
- `module_3_left_place/`：固定 Buffer放置。
- `module_4_head_judge/` 及后续目录：保留的后续工序，等待新总控重新编排。
- `stage_result.py`：Module3及其他模块仍使用的通用原子结果文件协议，必须保留。

旧 `modular_dual_arm_loop*.py` 未适配当前 Module1/Module3接口，仅作为待重写
代码保留，不应作为当前实机入口。
