# Module 7：右臂工件姿态调整

 `right_align.py` 接收相机/视觉模块（module 4_2）产生的 `mask_pose.json`，根据对应工件的单一 TOML 配置，使用右臂抓取、提起、调整姿态和放置，即把 Q2--Q4 朝向的工件调整到 Q1 的可抓取朝向后释放。Q1-4是以body_base坐标系为基准的第x象限简称。

目录内已包含完整的本地依赖：

- `right_align.py`：调整姿态入口；
- `configs/workpiece_1_turn_candidates.toml`、`configs/workpiece_2_turn_candidates.toml`：两种工件唯一的调整参数源；参数按预抓取、下探/闭抓、抬起、摆正、安全下放顺序编排并附中文注释；
- 右臂目标使用规划器原生 `right_eef`，即右臂夹爪指尖闭合中心。
- `planning_log.py`：规划请求和闭爪刚性关系的结构化 JSONL 日志工具；其余运动、几何、SDK 与夹爪控制代码已内置于 `right_align.py`。

先进行预演（不加 `--execute` 不会发送机械臂或夹爪动作）：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_7_right_align/right_align.py \
  --workpiece 1 \
  --arm-id 2
```

`--workpiece` 会自动选择同编号调整配置，并读取 `output/module_4_2/workpiece_<编号>/` 中最新的 `mask_pose.json`；可用 `--workpiece-pose-file` 显式覆盖，但文件的工件标记必须一致。确认预演结果与现场状态后，追加 `--execute` 才会实际打开夹爪、抓取、翻转、放置并释放。该模块需要可访问的 `parm_sdk`、右臂/夹爪服务和规划器。

这个模块的log日志，只累积统计选择了哪些翻转候选点、以及这个候选点实际规划时的各类信息，便于多次运行得到一些统计数据后的调参。关于这个模块的全部终端打印信息，会被Module 8 的 [workpiece_pipeline.py (line 100)](/home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp/workpiece_pipeline.py:100)记录。

| 文件 | 写入方式 | 内容 |
|---|---|---|
| `workpiece_1_turn_candidate_statistics.json` / `workpiece_2_turn_candidate_statistics.json` | 覆盖更新（原子替换） | 翻转候选的累计统计：总运行次数、每个“中心偏移 + 局部 RPY”候选的 `attempts`、`plannable`、`selected` 计数、最后更新时间。 |
| `workpiece_1_turn_planning_records.jsonl` / `workpiece_2_turn_planning_records.jsonl` | 追加，一行一个 JSON | 本次实际提交给翻转规划器的每个候选：运行 ID、视觉 pose 文件、候选优先级、中心偏移、局部 RPY、是否可规划、是否被选中、失败原因。 |
| `reorient_q3_records/grasp_xy_<时间戳>.json` | 新建 | 仅传 `--calibrate-grasp-xy` 时写入。记录抓取点、命令 TCP、规划器实际 TCP、XY 误差和建议补偿量；该模式不会闭爪或翻转。可用 `--calibration-record-file` 改路径。 |
