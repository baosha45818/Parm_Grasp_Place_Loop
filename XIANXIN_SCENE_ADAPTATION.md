# 祥鑫现场适配

本项目的运行时通信统一使用以下地址：

| 服务 | 地址 |
| --- | --- |
| 左右臂、左右夹爪 | `192.168.3.11` |
| 云台、左右力传感器 | `192.168.3.22` |
| 本机轨迹规划器 / TF | `127.0.0.1` |
| 本机规划执行后端 | `http://127.0.0.1:5667/middle/waypoint/execute` |

入口 `main.py` 与两个模块化总控会先加载 `xianxin_runtime.py`，为子进程设置
`PARM_ARM_HOST`、`PARM_GRIPPER_HOST`、`PARM_HEAD_HOST`、`PARM_FORCE_SENSOR_HOST`
和 `PARM_PLANNER_HOST`。直接运行某个模块时也可显式传入同名命令行参数。

项目虚拟环境已使用交付 SDK：
`260812/祥鑫现场交付_260812/parm_sdk-1.0.0+20260811.182400-py3-none-any.whl`。
本机不再启动旧的夹爪、力传感器、云台或踏板 driver。

## 规划器坐标系

当前规划器可用的统一基坐标系是 `body_base`。左右臂的可用末端包括
`left_hand_transfer_end`、`right_hand_transfer_end`、
`left_hand_transfer_base` 和 `right_hand_transfer_base`；旧的
`left_hand_tool_end_link`、`right_hand_tool_end_link` 不存在，不能作为 IK frame。

## 必须现场复核的内容

此次改动完成的是网络、SDK、力传感器地址和规划器 TF 名称的适配。旧场地保存的
关节示教位姿、桌面/缓冲区位置、相机外参、SAM2/ICP 阈值和工件放置参数不是网络配置，
无法自动转换为祥鑫现场的真实物理位置。首次实机运行前必须逐项核验，并在低速、空载、
人工监护下重新标定或确认。
