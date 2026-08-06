# 模块 7：Right Align（只调整，不最终抓取）

## 职责与边界

Module 7 的实机入口是 `push_to_q1_pipeline.py`。它调用 Module 4-2 观察工件，
并在需要时把缓冲区中的工件调整至右臂可直接抓取的 Q1 朝向；它不执行交给
Module 8-1 的最终 Q1 抓取。

可直接抓取的判定为：视觉结果在 Q1，且红线朝向 yaw 位于 **15°--75°**。
Q2--Q4 以及 Q1 的边缘朝向都需要调整。

## 输入与依赖

- 右臂使用 `--arm-id 2`。
- 每次观察均从示教关节位 `3.json` 进行，并调用 Module 4-2；该模块将云台设为
  pitch=`38.5°`、yaw=`0°`，采集 head RGB、分割 mask、计算工件的 `Q1--Q4` 与 yaw。
- Module 7 从 Module 4-2 的 `output/4_2` 中读取本次新生成的 pose JSON。
- 调整动作使用本模块 `vendor/push_t/reorient_q3_to_q1_old.py`；
  该动作会抓住工件、抬起、转向，并放回可抓取的 Q1 区域。
- 调整前右臂经 `10.json` 过渡到 `6.json`；调整后经 `10.json` 回 `3.json`。
- 本模块的 `vendor/push_t/` 包含调整脚本、TCP 配置与示教关节位；
  SDK 优先使用工程统一的 `runtime/parm_sdk/`，`vendor/parm_sdk/` 仅作为
  随模块携带的离线备用副本。不会读取 `/home/dainanjun/zhuyiming`。

## 流程

```text
启动
  → 到 3.json
  → 拍照 / mask / 位姿计算
  → 是否已处于可直接抓取 Q1（Q1 且 yaw 15°--75°）？
       ├─ 是，且从未调整
       │    → 输出“该阶段不需要右臂调整，可直接抓”
       │    → 结束（不抓取）
       └─ 否
            → 到 6.json
            → 抓住工件并旋转、放回 Q1
            → 回 3.json
            → 重新拍照 / mask / 位姿计算
            → 是否已可直接抓取？
                 ├─ 是 → 输出“右臂调整完成，可进行下一步抓取” → 结束（不抓取）
                 └─ 否 → 继续下一次调整
```

## 重试与异常处理

- `--max-attempts` 控制最多调整次数，默认 `10`，必须至少为 `1`。
- 调整命令失败时，先打开右夹爪，确认已松开后回 `3.json` 再观察并重试；
  这样不会带着状态不明的工件回观察位。
- 达到最大次数仍无法转至可抓 Q1 时，流程以错误结束。
- 若 Module 4-2 检测到缓冲区/桌面没有工件，Module 7 输出
  `桌面没有工件，右臂无法抓取。` 并以专用退出码 **24** 结束。该码表示没有
  可供右臂处理的工件，调用方不得进入 Module 8-1 或将其当成摆正重试。
- 无论初始可抓还是调整成功，本模块都不调用最终 Q1 抓取脚本，也不更新为
  “已抓取”状态；下一步应由 Module 8-1 执行。
- 每次运行会在 `output/right_trajectories/<时间戳>/` 写入右臂规划 JSONL。
  每条规划 RPC 都保存起点、目标、规划器/速度参数、规划轨迹、成功或失败状态：
  关节 `PTP` 保存实际起点与目标/轨迹终点的 7 轴关节角；笛卡尔 `LIN` 保存规划前
  IK 末端位姿与目标末端位姿。包括经 `10.json` 的关节 PTP，以及调整过程中的所有
  笛卡尔候选预规划和执行规划。

## 运行

默认 `--motion-speed-scale=2.5`；可显式传入 `--motion-speed-scale 1` 恢复原始
速度。该倍率会同时作用于 Module 7 的初始关节运动、关节过渡和抓住工件后的
调整轨迹，Module 4-2 相机流程不受影响。

```bash
/home/dainanjun/Public/data_collector/.venv/bin/python \
  modules/module_7_right_align/push_to_q1_pipeline.py --execute
```

不带 `--execute` 时仅预演首个到 `3.json` 和观察步骤，不会下发机器人、云台或
夹爪运动。

`implementation.py` 仍只供早期、未接入实机流程的 scheduler 骨架使用。
