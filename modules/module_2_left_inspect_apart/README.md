# Module 2 Left Inspect Apart

Module 2 只负责在胸前 Femto Bolt 视角估计工件位姿，输出
`T_body_base_object` 和 `T_tool_object`，然后左臂回到 `graspup`。它不生成
放置候选、不占用 Buffer，也不执行放置。

## 统一流程

所有工件共用一份运行代码、SAM2 提示/掩码后处理和 ICP 质量门限：

1. 先移动到 `PC1`，采集 RGB-D、调用 SAM2 并执行 Point-to-Plane/Point-to-Point
   ICP；两种 ICP 都要通过三维和二维投影门限。
2. `PC1` 的点云不足、SAM2/深度处理失败或 ICP 未通过时，才移动到 `PC2` 重试。
3. `PC2` 仍失败则本次 Inspect 返回错误；不再存在 `PC3`，夹爪开度也不再决定
   观察位。
4. 成功时保存 mask、有效深度、ICP 指标、相机/`body_base` 下三维可视化和结果 JSON。

`--prompt-mode` 对所有工件的含义一致：默认 `auto`；`manual` 使用 ROI 和正负点；
`auto-then-manual` 在自动提示不可用时再进入手动提示。人工接受的 SAM2 原始 mask
不会被二维规则修改，ICP 点云仅使用该 mask 最近有效深度带。

工件编号只用于选择 STL 及 STL 到工件坐标系的固定变换。工件2使用
`tool/workpiece_2/workpiece_2.stl`，并应用 `X=-X_stl, Y=-Y_stl, Z=+Z_stl`；
工件1直接使用 `inspect_config.json` 中的 STL 坐标系。

`standalone_inspect.py` 是总控保持不变的兼容入口；实际通用实现位于
`standalone_inspect_global.py`，不要再向两者分别添加逻辑。
