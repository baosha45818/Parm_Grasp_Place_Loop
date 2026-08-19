# 反面工件交接观察支持

这里是 Module 5.2 与 Module 8.2 共用的头部 D435 观察、位姿估计、交接预检和交接执行实现。
它原先误放在根目录 `tests/`，现已迁入正式模块目录。

正常运行不直接调用本目录，而由新总控的 Module 5.2 / Module 8.2 调用：

```bash
python3 controller/main.py --mode gzhu --workpiece 1 --live ...
```

主要入口：

- `estimate_pose_head_d435.py`：D435 采集与工件位姿估计。
- `preflight_handoff.py`：交接轨迹与几何预检。
- `execute_taught_tail_handoff.py`：工件 1 的已示教反面交接。
- `execute_handoff.py`：交接动作执行辅助。

本目录的 JSON 是交接标定与运行状态数据；运行时可更新相应状态文件。
