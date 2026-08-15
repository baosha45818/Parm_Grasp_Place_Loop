# Module 7：右臂工件姿态调整

本目录是从原调整链路重构出的独立模块。入口 `right_align.py` 接收相机/视觉模块产生的 `mask_pose.json`，根据对应工件的单一 TOML 配置，使用右臂抓取、抬升、转向和放置，把 Q2--Q4 朝向的工件调整到 Q1 的可抓取朝向后释放。

目录内已包含完整的本地依赖：

- `right_align.py`：调整姿态入口；
- `configs/workpiece_1_turn_candidates.toml`、`configs/workpiece_2_turn_candidates.toml`：两种工件唯一的调整参数源；按预抓取、下探/闭抓、抬起、摆正、安全下放顺序编排并附中文注释；
- 右臂目标直接使用规划器原生 `right_eef`；
- `planning_log.py`：规划请求和闭爪刚性关系的结构化 JSONL 日志工具；其余运动、几何、SDK 与夹爪控制代码已内置于 `right_align.py`。

先进行预演（不加 `--execute` 不会发送机械臂或夹爪动作）：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_7_right_align/right_align.py \
  --workpiece 1 \
  --arm-id 2
```

`--workpiece` 会自动选择同编号调整配置，并读取 `output/module_4_2/workpiece_<编号>/` 中最新的 `mask_pose.json`；可用 `--workpiece-pose-file` 显式覆盖，但文件的工件标记必须一致。确认预演结果与现场状态后，追加 `--execute` 才会实际打开夹爪、抓取、翻转、放置并释放。该模块需要可访问的 `parm_sdk`、右臂/夹爪服务和规划器。

校准记录默认写到 `/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_7/reorient_q3_records/`。可通过 `--calibration-record-file` 指定其他记录文件。
