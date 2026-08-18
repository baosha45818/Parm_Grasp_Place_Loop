# 新总控

入口是 `main.py`。它只根据 `scheduler_rules.py` 调度模块；模块命令集中在
`module_registry.py`，所有跨进程锁在 `locks.py`，运行状态和结果写在
`runtime/<run_id>/`。

每次启动都先执行 `controller_prepare`：检查视觉服务与机器人基础设施，打开
双夹爪，左臂回 `graspup`、右臂回 `Idle`。它不是业务模块，不做视觉或抓取判断。
相机不在总控预检阶段打开：Module 1 直接管理左腕 D405，Module 4 直接管理头部
D435i。

`--mode xiangxin|gzhu` 不只选择 Module 4 的相机标定；它还统一为所有模块子进程
设置该现场的机械臂、夹爪和力传感器地址。只有显式传入 `--arm-host`、
`--gripper-host`、`--force-host`、`--planner-host` 或 `--head-host` 时才覆盖现场默认值。

先演练正面流程：

```bash
python3 controller/main.py --mode xiangxin --workpiece 1 --dry-run --simulate-face FRONT
```

正面主线是 `M1 → M3(buffer) → 观察位 → M4.1 → M5.1 → M3(middle)`；之后右臂
后台运行复合模块 `M4.2 → M7 → M8.1`、再运行 `M9 → M10`。反面则经 `M5.2 →
M8.2` 完成双臂交接，再由右臂后台运行 `M9 → M10`。工件交给右臂后，左臂可立即
开始下一件；下一件将要使用 middle 或交接区时才等待右臂完成，所以不会争用工位。

M1 成功时左手已在 `graspuptop` 持件。Buffer 是单容量：总控只会在其状态为
`EMPTY` 时接受这次 M1 handoff，并将状态改为 `LEFT_PLACING` 后调用 M3(buffer)。
若状态仍表示上一件正在 Buffer 生命周期中，总控拒绝放置并保持左手在
`graspuptop`，不会用新件覆盖旧件。

真实执行必须显式提供 `--live`，不会调用旧
`modules/modular_dual_arm_loop.py` 或 `tests/loop/simple_dual_arm_loop.py`。

```bash
python3 controller/main.py --mode xiangxin --workpiece 1 --live
```

默认持续运行；用 `--left-cycle-limit N` 可限制成功送给右臂的工件数，便于现场分段
验收。`--dry-run` 未给该参数时只演练一件。

Module 4.1 每次调用只采集一次。`--max-buffer-observation-attempts`（默认 4）统一
限制每个 Buffer 工件的完整观察次数：0/多个工件、ICP 不可靠、正反不可靠均共用该
计数；耗尽后认定本次 Buffer 工件无效，清空状态并回 Module 1 抓下一件。Module 5
每次重抓都会先让左臂撤回 Buffer 观察位、头部回固定观察角，再调用 Module 4.1
重新生成 ICP TF；达到 5 次完整闭环后等待人工输入 `y`，恢复同样从观察位开始。

按 `Ctrl+C` 或收到停止信号时，总控会停止发起新任务、发送两臂 `cancel_task()`，并
终止正在运行模块的子进程组；不会继续使用旧总控。

每次模块调用都有独立的 attempt 编号：原始结果在 `artifacts/`，日志在 `logs/`，
控制器标准结果在 `results/`。尚未提供原生 `--result-file` 的 M8.2、M9、M10 由
控制器写出同一份标准结果协议，并标为 `controller_synthesized`。
