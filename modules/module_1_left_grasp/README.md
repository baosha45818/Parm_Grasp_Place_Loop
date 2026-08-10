# 模块 1：左手抓

## 职责

最终版本调度 GraspNet 或 DP-Grasp，从左侧工件堆抓取一个工件，并判断抓取为空、单个还是多个。

当前第一版只调用 1-1 占位函数，并固定写入“抓到单个工件且抓稳”。真正 GraspNet 由其他同学实现后替换该占位函数。DP-Grasp 当前不参与流程。

## 子模块

- `module_1_1_graspnet`：历史兼容入口；当前会转发至 `module_1_3_icp_grasp`。
- `module_1_3_icp_grasp`：当前总控使用的传统 SAM/STL/ICP 抓取实现。
- `module_1_2_dp_grasp`：仅保留空模块位置，当前调用会抛 `NotImplementedError`。

## 输入

- 当前 `LeftArmState`，要求左手为空。
- 抓取策略配置。
- 胸部相机和腕部相机数据入口。
- 工件模型、相机外参、TCP 和规划场景。
- 左臂、规划器和夹爪控制入口。

## LeftArmState 更新

```text
left.holding: EMPTY | SINGLE | MULTIPLE | UNKNOWN
left.grasp_stable: true | false | unknown
left.grasp_strategy: GRASPNET | DP
left.object_pose
left.pose_confidence
```

## 统一返回

模块只返回标准 `ModuleResult`。只有 State 中 `holding=SINGLE` 且 `grasp_stable=true` 时返回 `SUCCESS`；调度器读取 State 后决定是否进入模块 2。
