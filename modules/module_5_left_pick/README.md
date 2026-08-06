# 模块 5：Left Pick

## 职责

根据 Module 4 Head Judge 给出的桌面工件状态，控制左手抓起指定工件。

模块化总控通过 `run_head_judge_then_pick.py` 串联 Module 4 和 Module 5：

1. Module 4-1 控制云台、采集头部 RGB-D，并预测唯一反抓点。
2. Module 4 复用同一次采集确认桌面工件为 `BACK`。
3. Module 5-1 只消费 prediction JSON，进入左臂预备位并执行抓取。
Module 4 独占云台移动与 RGB-D 采集，并将 `capture_dir` 传给 Module 5-1。
因此该路径中的 Module 5-1 只复用已有图像做反抓定位，不会再次移动云台或
重新拍摄。直接运行 5-1 的兼容入口时，仍可自行采集，供旧总控保持原行为。

## 子模块

- `module_5_1_reverse_grasp`：已迁入并验证的桌面工件反面抓取流程。
- `module_5_2_handoff_prepare`：左手持件到观察位、读取 Module 4-3 的
  D435/SAM2/ICP 结果，并移动到双臂交接位。

Module 5-1 完成桌面反抓；Module 5-2 保持左夹爪夹紧，将工件送到右手
可接取的位置并输出本轮 ICP `result.json`。右臂接取属于 Module 8-2。
