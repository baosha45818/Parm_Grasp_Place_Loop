# 模块 1-2：DP-Grasp 抓取

该模块当前留空，不参与第一版 Loop。`placeholder.py` 只保留明确抛出 `NotImplementedError` 的接入口，避免被误认为已经实现。

## 职责

使用 DP 策略完成左手抓取。它与 GraspNet 的内部实现不同，但必须向调度器提供相同的结果格式。

## 输入

- DP 模型和 CKPT。
- DP 所需的相机观测和机器人状态历史。
- 工件与场景配置。
- 左臂、规划器和夹爪控制入口。
- 抓取成功、空抓和抓多的判定参数。

## LeftArmState 更新

- `left.object_pose`：最终抓取位姿。
- `left.holding=EMPTY/SINGLE/MULTIPLE/UNKNOWN`。
- `left.grasp_stable`。
- `left.pose_confidence` 和 `left.last_error`。

## 完成条件

只抓到一个工件并确认抓稳时返回统一 `ModuleResult(status=SUCCESS, previous_module="1-2", ...)`。返回结构与模块 1-1 完全一致，业务差异只体现在 `LeftArmState`。
