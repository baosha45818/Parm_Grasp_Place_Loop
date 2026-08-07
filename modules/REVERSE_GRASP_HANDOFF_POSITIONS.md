# 反抓与交接预设位置

所有第一、二类工件在左臂反抓和右臂交接阶段使用的预设关节位，统一配置在
[`reverse_grasp_handoff_positions.toml`](reverse_grasp_handoff_positions.toml)。

- `workpiece_*/left_reverse_grasp`：桌面反抓前左臂预备位、到位门限、闭夹爪后
  `body_base +Z` 抬升距离和最大重抓次数。
- `workpiece_*/left_handoff`：左臂 D435 观察位、云台角、交接中心与左臂回程位。
- `workpiece_*/right_handoff`：右臂交接抓取位和 `place_ready` 位；工件2另有预抓取位与交接后撤离距离。

Module 5-1、5-2 与 Module 8-2 均在每次启动时读取 TOML。修改任一工件的
位置不会改变另一工件；修改后重新启动总控或对应独立模块即可生效。旧的
`left_reverse_regrasp_ready_joints.json` 和 `workpiece_2_handoff_profile.json`
只保留用于历史追溯，不再是这些执行链的配置来源。

反抓重量分段、采样稳定性和采样窗口参数则统一在
[`../config/grasp_weight_thresholds.conf`](../config/grasp_weight_thresholds.conf)。
两类工件的 Module 5-1 均使用该文件；闭夹爪、上抬后只有 `SINGLE` 才会继续。
