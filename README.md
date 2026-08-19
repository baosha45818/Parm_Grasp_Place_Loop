# PARM Grasp-and-Place Loop

实机流程唯一入口为新总控：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
python3 controller/main.py --mode gzhu --workpiece 1 --live \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --force-host 127.0.0.1 \
  --head-host 127.0.0.1
```

调度逻辑位于 `controller/`，模块实现位于 `modules/`。总控主线为：

```text
M1 → M3(Buffer) → M4.1
  ├─ FRONT → M5.1 → M3(Middle) → M4.2/M7/M8.1 → M9 → M10
  └─ BACK  → M5.2 → M8.2 → M9 → M10
```

- Module 1 直接管理左腕 D405；不使用 Camera Broker。
- Module 4.1 直接管理头部 D435，发布 Buffer 工件的正反结果与 ICP TF。
- 总控统一处理观察重试、重抓重试、左右臂并行、人工恢复和停止信号。
- 运行产物位于 `controller/runtime/<run_id>/`。

旧双层总控、旧反抓实验链与根目录 `tests/` 均已删除，不能作为实机入口。
详细运行与恢复规则见 [controller/README.md](controller/README.md)。
