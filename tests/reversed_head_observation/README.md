# 反向抓取到头部观察位：碰撞规划测试

本目录检查左夹爪抓住反向工件后，从记录的中间位移动到头部观察位是否能由
当前 plucky/MoveIt 模型规划成功。脚本固定使用以下目标：

- 中间位（度）：`[-112.208, 51.35, -3.083, 71.536, -0.401, 30.53, 84.765]`
- 头部观察位（度）：`[-64.123, 3.918, -13.619, 101.684, -103.678, 83.784, 75.869]`
- 云台目标（度）：`pitch=2, yaw=26.5`

运行：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
python3 -m tests.reversed_head_observation.check_plan
python3 -m unittest tests.reversed_head_observation.test_check_plan -v
```

关节限制修正并重新通过只规划检查后，执行脚本为：

```bash
python3 -m tests.reversed_head_observation.execute_plan
```

执行脚本分成两段，每段分别规划、确认和执行：

1. 从规划器实时状态规划到记录的中间起始点；
2. 第一段执行完成后，重新读取实时状态，再规划到目标观察位。

脚本默认速度比例为 `0.05`。每段规划和轨迹内容校验成功后，必须在终端输入
`y` 才会直接通过 SDK 向左臂发送该段轨迹；第二段完成后才调用
云台角以 `check_plan.py` 中当前 `HEAD_TARGET_DEG` 为准。使用 `--skip-head`
可以不转云台。

## 使用头部 D435i 估计工件位姿

手臂和云台到达观察位后运行：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
python3 -m tests.reversed_head_observation.estimate_pose_head_d435 \
  --sam2-url unix:///tmp/sam2-service.sock \
  --output-dir /home/dainanjun/Public/Parm_Grasp_Place_Loop/output/reversed_head_pose_d435
```

该入口连接头部 Intel RealSense D435i（序列号 `261222076212`），不会连接胸前
Femto Bolt。默认校验云台命令角 `pitch=2°, yaw=26.5°`，并输出
`head_d435i_color_optical_frame` 下的工件位姿。

### 自动 SAM2 mask

默认 `interactive` 模式保留原有的人工 ROI/正负点标注。对于已验证的固定观察位，
可改用非交互式的 3 个工件主轴正点和 2 个夹爪负点：

```bash
python3 -m tests.reversed_head_observation.estimate_pose_head_d435 \
  --mask-mode auto \
  --sam2-url unix:///tmp/sam2-service.sock
```

提示坐标定义在 `head_d435_auto_mask_config.json`，以归一化坐标随 RGB 分辨率缩放。
自动流程会保存 `head_d435_mask.png`、`head_d435_mask_overlay.png` 与提示记录到本轮
输出目录；若夹具或工件观察位改变，须先核验叠加图再用于 ICP。

当前相机 profile 为彩色 `1280x720@30`、深度 `848x480@30`。深度会通过
RealSense 对齐到彩色坐标后再应用 SAM2 mask；该 profile 在当前近距离反光金属
场景的有效深度明显优于 `1280x720` 深度。

当前头部 D435 测试配置关闭 ICP 硬质量门限。结果仍记录
`quality_gate_passed`、fitness 和 RMSE；未达到原门限时只报警，不中止位姿输出。

### 将工件中心移动到固定交接点

最新检测结果定义的交接点是 `body_base` 中的工件中心
`[0.517668, 0.018779, -0.200000] m`。脚本只改变工件中心位置，保持 ICP
得到的工件姿态以及左夹爪与工件的刚性关系不变。

先只规划：

```bash
python3 -m tests.reversed_head_observation.move_object_center \
  --result output/module_2_inspect/20260722_165216/result.json
```

确认规划输出后执行：

```bash
python3 -m tests.reversed_head_observation.move_object_center \
  --result output/module_2_inspect/20260722_165216/result.json \
  --execute
```

执行前仍需在终端输入 `y`。该脚本不移动右手、不松开左夹爪。

现有 `sdk_base_to_planner_alignment.json` 的头部 D435 外参只在标定云台角
`pitch=38.5°, yaw=0°` 有效，因此当前 `2°/26.5°` 观察角不能直接得到可信的
`body_base` 位姿。当前角重新标定并生成新的 alignment 文件后，才可增加：

```bash
  --output-frame body_base --alignment /path/to/alignment_at_pitch2_yaw26p5.json
```

当前部署的 PARM Studio URDF 包含 `head_joint_link1`（yaw）、
`head_joint_link2`（pitch）及 `head_joint_cam` 的完整轴心和安装偏移，因此也可将
`38.5°/0°` 实测外参通过名义运动学传播到 `2°/26.5°`：

```bash
python3 -m tests.reversed_head_observation.derive_head_alignment
```

输出为：

```text
/home/dainanjun/Public/Parm_Grasp_Place_Loop/calibration/sdk_base_alignment/
sdk_base_to_planner_alignment_pitch2_yaw26p5_kinematic.json
```

这是运动学推导值，并非在 `2°/26.5°` 重新完成的独立相机标定；精度受云台零偏、
轴心尺寸和装配误差影响。用于运动前必须重新拍摄生成同步左 TCP 快照，并先只规划。
云台命令下发后会等待最多 15 秒并校验实际状态；可用 `--head-timeout` 调整。
已使能时不会重复调用 `Head.init()`。由于当前 SDK 对 pitch 下发取反但状态读取未恢复
符号，目标 `pitch=2°` 对应的底层状态反馈为约 `-2°`。

手臂已经到位时，只测试云台：

```bash
python3 -m tests.reversed_head_observation.execute_plan --head-only
```

若左臂 SDK 状态显示已有任务运行，脚本会询问是否取消该任务。
输入 `y` 后直接调用 SDK `cancel_task(1)`，随后从新的实时状态重新规划和校验；不会直接
复用旧轨迹。若其他程序持续占用左臂，应先停止该程序，不能反复抢占控制权。

`check_plan.py` 通过关节限制预检后才访问 5604 规划服务，既不执行轨迹，
也不实例化 `Arm`、`Head` 或 `Gripper`。结果写入 `last_plan_report.json`。成功只表示路径在
当前 MoveIt 机器人模型和规划场景中通过；规划失败或碰撞均返回退出码 1。

当前两个记录位均通过模型关节限制预检。执行脚本先从实时状态规划到记录中间位，
执行完成后再从实时中间状态规划到观察位；第二段起点默认每轴误差不超过 `0.5°`。

## 当前模型限制

`/home/dainanjun/Public/plucky_hjt_parm/welding_config/config/joints.yaml` 中云台
`gimbal` 仍配置为 `simulation: true`、默认角度 0°，所以现有规划器不能按目标
`pitch=2°, yaw=26.5°` 联合检查头部姿态。当前场景也没有附着到夹爪上的工件
碰撞几何体。因此，在把真实云台状态和工件几何体接入 PlanningScene 前，测试结果
不能证明“夹持工件的实体机器人一定不会碰撞”。实体运行前仍需低速、急停就绪并由
现场人员观察确认。

云台驱动调用为 `Head.move_to(pitch=2.0, yaw=26.5)`；`check_plan.py` 不执行它。

## 左手到胸前、右手固定接取预检

交接采用统一的齐次变换约定 `p_A = T_A_B @ p_B`。程序根据观察位的实时左 TCP
和 `body_base` 系 ICP 工件位姿计算 `T_left_tcp_object`，然后反算左手把工件送到
固定胸前位姿时的 TCP 目标；右手目标由标定好的固定 `T_right_tcp_object` 反算。

配置模板：

```text
tests/reversed_head_observation/handoff_config.json
```

以下三项必须通过示教/标定填写，模板故意保留为 `null`，程序不会猜测安全交接位：

- `T_base_object_handoff`：胸前工件中心的完整 6D 目标位姿；
- `T_right_tcp_object`：右手夹稳时，工件坐标系在右 TCP 中的固定变换；
- `right_pregrasp_offset_tcp_m`：从抓取位退到预抓取位的右 TCP 局部坐标偏移。

不要手工计算矩阵。取得新的 `body_base` ICP 结果后，用明确指定的胸前中心坐标生成
独立配置文件；工件目标姿态沿用这次 ICP 的姿态：

```bash
python3 -m tests.reversed_head_observation.configure_handoff set-target \
  --icp-result output/reversed_head_pose_d435/<时间>/result.json \
  --center-m <body_x> <body_y> <body_z> \
  --output-config tests/reversed_head_observation/handoff_config.calibrated.json
```

随后仅在工件已经真实位于该交接目标、右 TCP 已由人工对准期望夹持位置时，记录固定
右手抓持关系。下面偏移只是参数格式示例，不应照抄，方向必须按实际右 TCP 轴确认：

```bash
python3 -m tests.reversed_head_observation.configure_handoff record-right \
  --input-config tests/reversed_head_observation/handoff_config.calibrated.json \
  --output-config tests/reversed_head_observation/handoff_config.ready.json \
  --pregrasp-offset-tcp-m <dx> <dy> <dz> \
  --confirm-object-at-handoff
```

两个子命令都只读取文件/Planner TF并写新配置，不发送机械臂、云台或夹爪命令。
`set-target` 同时根据同步 TCP 快照生成保守工件包围盒和
`T_left_tcp_object`，供 MoveIt 附着碰撞体使用。

生成 `handoff_config.ready.json` 后，用桌面+交接碰撞场景重启规划器：

```bash
cd /home/dainanjun/Public/plucky_hjt_parm
docker compose \
  -f docker-compose.yml \
  -f docker-compose.tables.yml \
  -f docker-compose.handoff.yml \
  up --build --force-recreate
```

另开终端确认 MoveIt 实际收到附着物体：

```bash
cd /home/dainanjun/Public/plucky_hjt_parm
docker compose \
  -f docker-compose.yml \
  -f docker-compose.tables.yml \
  -f docker-compose.handoff.yml \
  exec plucky_hjt_parm ros2 topic echo --once /attached_collision_object
```

输出必须包含 `id: held_reverse_workpiece`、`link_name: left_hand_transfer_end` 和
配置文件中的 box 尺寸。未看到该消息时不能认为规划包含了工件碰撞。

完成当前 `pitch=2°, yaw=26.5°` 的 D435 外参后，先产生 `body_base` 结果：

```bash
python3 -m tests.reversed_head_observation.estimate_pose_head_d435 \
  --output-frame body_base \
  --alignment /path/to/alignment_at_pitch2_yaw26p5.json \
  --sam2-url unix:///tmp/sam2-service.sock \
  --output-dir output/reversed_head_pose_d435
```

`body_base` 模式会在采集前后读取 `left_hand_transfer_end`，默认要求左 TCP 变化不超过
`2 mm / 1°`，并把拍摄时的 `T_base_left_tcp` 写入 `result.json`。交接预检只使用该
同步快照计算 `T_left_tcp_object`，且要求运行预检时左 TCP 仍在拍摄位附近；旧结果
因没有同步 TCP 快照会被拒绝。

填好交接配置后执行只规划预检：

```bash
python3 -m tests.reversed_head_observation.preflight_handoff \
  --icp-result output/reversed_head_pose_d435/<时间>/result.json \
  --config tests/reversed_head_observation/handoff_config.ready.json \
  --confirm-attached-collision-active
```

该入口只查询 TF并规划 `left_handoff`。只有检测到左臂已经真实位于交接目标时，才把
`right_pregrasp → right_grasp` 作为 PTP+LIN 两路点连续轨迹预检；否则右臂阶段标为
`DEFERRED_UNTIL_LEFT_IS_PHYSICALLY_AT_HANDOFF`。没有轨迹执行或夹爪控制代码。报告写入
`last_handoff_preflight.json`。当前 MoveIt 没有
附着工件，而且单臂规划不能验证右臂路径与“未来左臂交接姿态”的全过程碰撞，因此
三段均规划成功仍只是第一层预检，不能直接启用自动交接。

## 分阶段交接执行

先运行默认只规划模式。该命令还会通过状态文件 request/ack 握手确认附着碰撞发布器
确实在线；发布器未应答会在规划前停止：

```bash
python3 -m tests.reversed_head_observation.execute_handoff \
  --icp-result output/reversed_head_pose_d435/<时间>/result.json \
  --config tests/reversed_head_observation/handoff_config.ready.json
```

只规划和现场检查通过后，才增加 `--execute`：

```bash
python3 -m tests.reversed_head_observation.execute_handoff \
  --icp-result output/reversed_head_pose_d435/<时间>/result.json \
  --config tests/reversed_head_observation/handoff_config.ready.json \
  --execute
```

执行顺序固定为：

1. 规划并经 `y` 确认后执行左手到胸前；读取实时 TCP 验证到位。
2. 检查右夹爪开度，必要时经 `y` 确认张开；随后在左手真实交接姿态下重新规划
   右手 `PTP预抓取 → LIN抓取`。
3. 经 `y` 确认后执行右手接近，再经 `y` 确认后闭合右夹爪。
4. 必须人工确认右手夹稳；碰撞发布器确认所有权切换到右 TCP 后才打开左夹爪。
5. 重新规划并经 `y` 确认后，右手携件 LIN 退到预抓取位。

任何阶段取消或失败都不会自动软停止/抢占其他任务，也不会越过当前阶段继续释放。
初版完成后左臂仍张开停在交接位，尚未自动撤回安全位。
