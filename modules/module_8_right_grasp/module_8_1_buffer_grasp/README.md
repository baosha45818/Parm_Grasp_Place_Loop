# 模块 8-1：Buffer Grasp（直接抓取）

## 职责与边界

Module 8-1 的直接抓取入口是 `push_to_q1_grasp_pipeline.py`。它消费 Module 4-2
判定为可直接抓取、或已经由 Module 7 调整好的工件，并执行最终 Q1 抓取。它不旋转、
不重定向工件。

每次观察都要求工件处于可直接抓取的 Q1：视觉结果为 Q1，且 yaw 位于
**15°--75°**。如果不满足该条件，Module 8-1 输出“工件位姿已不满足直接抓取
情况，请回到module_7重新调整”后停止，不会尝试用抓取代替调整。

## 输入与依赖

- 右臂使用 `--arm-id 2`，初始先到示教关节位 `3.json`。
- 每次尝试都调用 Module 4-2：云台设为 pitch=`38.5°`、yaw=`0°`，随后采集
  head RGB、生成 mask 并计算工件 Q 区和 yaw。Module 8-1 从
  `output/4_2` 读取该次观测生成的 pose JSON。
- 确认可抓取后，调用已验证的
  本模块 `vendor/push_t/push_workpiece_to_q1.py`，以视觉结果中的
  工件中心和朝向执行 Q1 直接抓取。
- 本模块 `vendor/push_t/` 包含 Q1 抓取代码、TCP 配置和示教关节位；SDK 优先使用
  工程统一的 `runtime/parm_sdk/`，`vendor/parm_sdk/` 是随模块携带的离线备用副本。
  不会读取 `/home/dainanjun/zhuyiming`。

## 流程

```text
启动
  → 到 3.json
  → 拍照 / mask / 位姿计算
  → 是否为可直接抓取 Q1（Q1 且 yaw 15°--75°）？
       ├─ 否 → 输出“工件位姿已不满足直接抓取情况，请回到module_7重新调整”
       │       → 结束，不再抓取
       └─ 是 → 执行 Q1 直接抓取
                 → 是否抓取成功？
                      ├─ 是 → 输出“Q1 抓取成功。” → 结束
                      ├─ 路径不可规划 → 输出“无法抓取，需module_7再次调整位姿”并结束
                      ├─ 空抓 / 窄抓恢复失败且未满 3 次
                      │    → 松爪 → 经 10.json 回 3.json → 重新观察并再抓
                      └─ 其他失败，或空抓 / 窄抓已满 3 次
                           → 输出“无法抓取，需module_7再次调整位姿”并结束
```

## 重试与结束条件

- `--max-attempts` 控制空抓或窄抓恢复失败后的最大总抓取次数，范围为 `1--3`，
  默认 `3`。
- Q1 直接路径不可规划时立即输出“无法抓取，需module_7再次调整位姿”并结束。
- 空抓、或抓取脚本已耗尽其内部“松爪 → Z 向抬高 → 复抓”窄抓恢复后，若仍未抓住：
  先确认右夹爪已打开，再经 `10.json` 回 `3.json` 重新观察并重抓；最多总计 3 次。
- 除上述三类外的脚本失败（例如执行、通信或抓后上抬失败）立即输出
  “无法抓取，需module_7再次调整位姿”并结束。
- 抓取成功即结束。
- 本入口不会改动总控，不会调用 Module 7 的调整动作；总控或上层流程应在
  Module 7 成功后再调用该入口。
- 每次运行会在 `output/right_trajectories/<时间戳>/` 写入右臂规划 JSONL。
  每条抓取候选可达性检查、预规划和执行规划都会保存起点、目标、规划器/速度
  参数、轨迹及成功/失败状态：笛卡尔 `LIN` 含规划前/目标末端位姿，关节 `PTP`
  含实际起点与目标/轨迹终点的 7 轴关节角。

## 运行

默认 `--motion-speed-scale=2.5`；可显式传入 `--motion-speed-scale 1` 恢复原始
速度。该倍率会作用于 Module 8-1 的初始关节运动、失败后的回观察位和 Q1
直接抓取轨迹，Module 4-2 相机流程不受影响。

```bash
/home/dainanjun/Public/data_collector/.venv/bin/python \
  modules/module_8_right_grasp/module_8_1_buffer_grasp/push_to_q1_grasp_pipeline.py \
  --execute
```

不带 `--execute` 时仅预演首个到 `3.json` 和观察步骤，不会下发机器人、云台或
夹爪运动。

早期 scheduler 骨架仍使用同目录的 `implementation.py` 占位函数；它不等同于
上述实机直接抓取入口。
