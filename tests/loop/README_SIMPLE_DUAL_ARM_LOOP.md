# 简单双臂串联

这个入口只负责串联现有项目，不替代任何原有单独入口：

- 左臂：Version7 GraspNet -> Module 2 inspect -> 放置 -> graspup。
- 正面或侧面：共享状态写为 `1`；右臂完成 pushT 和抓取后，总控以关节 RRTConnect 到放置准备位。工件1保持原 `right_place_ready` `[30.4353, 23.8637, 0.5905, 66.3409, -62.0439, -64.3304, 25.7149]`（deg）；工件2先到交接流程同一准备位 `[102.2483, 21.8477, -87.2013, 83.4439, -147.2882, -32.2943, -66.0196]`（deg），再进入各自 Module 9 放置起点。
- 反面交接：左臂完成 Module 5-1 桌面反抓后，由左 worker 运行
  `module_5_2_handoff_prepare/main.py`，调用 `auto_head_d435_mask.py` 完成固定
  观察位自动 SAM2 mask、ICP 并移动到交接位，
  再将共享状态写为 `2`。右 worker 先用规划器关节 PTP 到预交接位
  `[161.36, 19.07, -107.979, 71.451, -55.833, -14.896, 88.2]`（deg），
  然后将 5-2 发布的本轮 `result.json` 显式传给
  `module_8_2_left_handoff/main.py`。总控不覆盖该流程的交接速度、抓取位或
  Module 9 上方准备位参数。
- inspect 判定为反面后、左臂启动桌面反抓前，会把云台校正并确认到 pushT 相同的命令视角：`pitch=38.5°`、`yaw=0°`。
- 反面：左臂先执行 desktop_grasp 的重新抓取；完成后共享状态写为 `2`，右臂执行既有交接流程。
- 右臂：两个分支都先运行 `modules/module_9_right_place/place_hole_a_wiggle.py --live --confirm-live` 完成正式孔位摆动放置；随后松开夹爪、沿 `body_base +Z` 上升 5 cm，再用规划器 PTP 到 `saved_poses/3.json` 的 Module 10 Idle。

启动时如果右臂已经在 `saved_poses/3.json` 的 Idle，就跳过右臂运动；否则先尝试直接规划 PTP 到 Idle，只有直接规划失败才经过固定启动过渡位。随后左臂到 graspup，两者到位后才启动第一轮 loop。除反向交接外，左右臂使用两个独立 worker，并通过状态 `0/1/2` 和观察位握手协调。

每次启动总控的准备阶段，都会先同步打开左、右夹爪，并确认两侧实际开度至少为 95% 后，才允许任一机械臂开始规划运动。

状态 `1/2` 不会阻止左臂回到 graspup 并移动到 GraspNet 观察位。左臂实际到达观察位后会通知右臂，右臂随即启动 ServoPose 孔位放置；左臂保持不动并完成图像预测和候选可达性规划。GraspNet 在真正下发预测出的预抓取轨迹前等待状态 `0`，所以 ServoPose 会话期间左臂不会执行新的轨迹。Module 2 最终放置前仍会再次检查状态，作为安全兜底。

反向交接的时序是：右手实际到 `right_place_ready` 并接稳工件后，交接脚本让左臂回到中间初始位。左臂随后回 graspup 并到达下一轮 GraspNet 观察位；到位信号触发右臂开始 ServoPose，左臂只预测和规划。右臂完成放置并确认松开夹爪后，状态才写为 `0`，左臂才会执行预测出的抓取轨迹；右臂之后继续上升并回 Idle。

GraspNet 最终抓取保留人工 `y/yes` 确认；总控自动确认 Module 2、反抓和
交接中的既有运动确认。反面交接的 Module 5-2 默认自动生成 ROI、SAM2
正负点和 mask；其他观察流程是否人工标注仍由各自模块决定。

## 前提

规划器、机械臂驱动、夹爪服务必须已启动。SAM2 需要单独常驻：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service
bash start_uds_server.sh
```

它使用 `/tmp/sam2-service.sock`，供反抓和右手交接使用。Module 2 仍会按原有实现临时启动自己的 tracking-only 服务 `/tmp/parm-loop-module2-sam2.sock`；两个 socket 不冲突。

## 启动总控

```bash
/home/dainanjun/Public/data_collector/.venv/bin/python \
  /home/dainanjun/Public/Parm_Grasp_Place_Loop/tests/loop/simple_dual_arm_loop.py \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.3.11
```

右臂状态为 `1` 时，总控直接运行既有 pushT+抓取流程；不会再等待人工输入 `y`。抓取完成后，pushT 自行上抬，总控直接关节 PTP 到 `right_place_ready`，右臂执行持续 ServoPose 正式放置。只有工件放入孔位并确认松开夹爪后，状态才回到 `0`，左臂才会执行已经预测和规划好的抓取。

若使用 `--left-cycle-limit` 做有限轮测试，总控仍会等右臂完成最后一次正式放置并实际回到 Idle 后才退出，不会因为状态已提前释放为 `0` 而中断右臂。

`Ctrl+C` 会停止两个 worker，并分别对 arm 1、arm 2 使用独立 SDK 客户端调用 `cancel_task()`；随后终止它们启动的子流程。

## 兼容性

原来的命令仍可独立使用。新增的 GraspNet 抓取执行许可只在总控设置
`LOOP_GRASPNET_EXECUTION_PERMIT_FILE` 时启用；Module 2 放置许可只在总控设置
`MODULE2_PLACEMENT_READY_FILE` 与 `MODULE2_PLACEMENT_PERMIT_FILE` 时启用。直接运行
`run_grasp_to_module2.sh` 或 `standalone_inspect.py` 不会等待总控。
