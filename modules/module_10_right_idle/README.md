# 模块 10：Right Idle

将右臂移动到专用安全休息位姿，并持续等待缓冲区或交接任务。

统一返回中的 `previous_module` 为 `"10"`。

模块化总控入口为 `main.py`。它兼容已经验证的放后松爪、沿
`body_base +Z` 抬升、关节规划回 Idle，以及 `right_gripper_opened`
握手文件协议。默认 Idle 关节位姿保存在 `right_idle_pose.json`。
