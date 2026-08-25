# Module 5 独立测试

统一入口是 `test_handoff.py`。默认现场和工件是 `gzhu/workpiece_1`，测试结果自动写入
`output/module_5_test/<时间戳>/`，不需要指定结果文件。该入口直接控制机器人，测试前必须确认 Buffer、Middle 与交接区没有其他工件或运动任务。

无论自动还是人工模式，程序启动后都会先执行公共准备流程：检查服务与设备，左臂回到 `graspup` 初始位、右臂回到 `Idle` 初始位；只有双臂都确认到位后，才打开左右夹爪。任何一侧归位失败时不会继续执行打开夹爪及后续测试。

自动抓取：
```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
./.venv/bin/python modules/module_5_left_pick/test_handoff.py

```

人工抓取：
```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
./.venv/bin/python modules/module_5_left_pick/test_handoff.py --face front

```

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
./.venv/bin/python modules/module_5_left_pick/test_handoff.py --face back

```


## 从 Buffer Check 自动抓取

不带参数即自动模式：程序从 Module 4 的本轮 ICP 自动重抓，并按实际正反面继续执行，直到工件由右臂持有且右臂到达放置准备位。

```bash
./.venv/bin/python modules/module_5_left_pick/test_handoff.py
```

自动模式先运行 Module 4 观察位和 Buffer Check，再按实际正反判定自动进入相应抓取分支；不需要也不能提供旧的 Buffer Check 结果文件。

## 人工已抓稳后测试抓后预放置

传入 `--face front` 或 `--face back` 即人工模式，不使用 Buffer Check。公共准备完成、双夹爪打开后，左臂再运动到统一的 `ready_joints_deg`。到位后空格可反复切换左夹爪开/闭（无需回车，使用 `async + preempt=True`）；按 `y` 时只有反馈为 `gripping/arrived` 且开度大于 20% 才进入交接流程，否则继续停留在按键控制界面。按 `q` 退出。

正面人工持件后依次执行 `graspuptop → M3 Middle 放置 → M8.1 右手抓取 → 右臂放置准备位`：

```bash
./.venv/bin/python modules/module_5_left_pick/test_handoff.py \
  --face front
```

反面人工持件后执行 M5.2 观察，先让左臂到高位交接中心，再让右臂到安全预抓取位；两臂到位后继续执行 M8.2 最终 LIN 接取、左手释放和右臂撤离，终点为右臂放置准备位：

```bash
./.venv/bin/python modules/module_5_left_pick/test_handoff.py \
  --face back
```

切换现场或工件时加 `--mode xiangxin --workpiece 2`，并按现场需要覆盖主机地址，例如 `--arm-host`、`--gripper-host`、`--head-host`。
