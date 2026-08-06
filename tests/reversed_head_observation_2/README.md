# 第二类工件：左手反抓观察与右手交接

本目录复用已经现场验证过的双臂运动和状态校验实现，只替换第二类工件相关的模型、
坐标、碰撞盒和右手抓取几何。模型为
`tool/workpiece_2/workpiece_2.stl`。

## 已适配内容

- ICP 采样前把 STL 坐标转换到约定工件坐标：
  `X_object=-X_stl, Y_object=-Y_stl, Z_object=Z_stl`。
- 工件原点采用 STL 包围盒中心，尺寸校验为
  `111.0 x 87.52 x 62.13 mm`。
- attached collision 使用每个方向总计约 4 mm 余量的保守盒：
  `115.0 x 91.52 x 66.14 mm`。
- 交接保持工件 `+Z` 正面朝上、`+X` 朝机器人右臂。右手几何规划种子位于
  `+X` 边界向内 15 mm，即 `[0.0405, 0, 0] m`；左手位于 `-X` 端。

右手接触点只能由 STL 给出规划初值，不能证明真实夹持面、指尖净空或夹持力合适。
因此只规划不要求额外确认，实体执行必须显式加入
`--confirm-workpiece-2-grasp-geometry`。自动 SAM2 提示也默认禁用，首次必须使用交互式
mask；取得固定观察位实拍后再标定自动提示点。

## 执行脚本

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
tests/reversed_head_observation_2/run_workpiece_2.sh test
tests/reversed_head_observation_2/run_workpiece_2.sh check-observation
tests/reversed_head_observation_2/run_workpiece_2.sh move-observation
```

到达观察位后，采集可用于运动的 `body_base` ICP 结果（外参必须与当前
`pitch=2°, yaw=26.5°` 一致）：

```bash
tests/reversed_head_observation_2/run_workpiece_2.sh estimate \
  --output-frame body_base
```

设结果路径为 `RESULT=output/reversed_head_pose_d435_workpiece_2/<时间>/result.json`。
先根据本轮同步 TCP 快照生成附着碰撞配置（中心坐标按现场交接点填写，下面仅展示
命令格式）：

```bash
tests/reversed_head_observation_2/run_workpiece_2.sh prepare-config \
  --icp-result "$RESULT" \
  --center-m X Y Z \
  --overwrite
```

随后用桌面场景、通用 handoff 场景和工件2覆盖共同重启规划器：

```bash
cd /home/dainanjun/Public/plucky_hjt_parm
docker compose \
  -f docker-compose.yml \
  -f docker-compose.tables.yml \
  -f docker-compose.handoff.yml \
  -f /home/dainanjun/Public/Parm_Grasp_Place_Loop/tests/reversed_head_observation_2/docker-compose.handoff.yml \
  up --build --force-recreate
```

必须确认 `/attached_collision_object` 中盒尺寸为
`[0.115, 0.09152, 0.06614]`、所有者为左 TCP 后，再严格按先规划、后执行的顺序：

```bash
tests/reversed_head_observation_2/run_workpiece_2.sh left-plan \
  --result "$RESULT" --config tests/reversed_head_observation_2/handoff_config.ready.json
tests/reversed_head_observation_2/run_workpiece_2.sh left-execute \
  --result "$RESULT" --config tests/reversed_head_observation_2/handoff_config.ready.json

tests/reversed_head_observation_2/run_workpiece_2.sh right-plan \
  --result "$RESULT" --config tests/reversed_head_observation_2/handoff_config.ready.json
tests/reversed_head_observation_2/run_workpiece_2.sh right-execute \
  --result "$RESULT" \
  --config tests/reversed_head_observation_2/handoff_config.ready.json \
  --switch-attachment \
  --confirm-workpiece-2-grasp-geometry
```

`left-execute` 和 `right-execute` 都保留实时 TCP、规划响应、夹爪状态和到位
检查。右手的预抓取、尾部抓取和放置准备位均以七关节反馈作为到位门禁，
不再用未标定本地 FK 生成的 TCP 二次否决正确示教位；实时 TCP 仅用于后续
笛卡尔推进和持件撤离。右手固定按两组现场示教关节角执行 PTP：先到预抓取位
`[173.9519, 50.8063, -79.5218, 85.0397, -25.3112, 4.5552, 79.6344]`，
再到工件2尾部抓取位
`[162.5467, 59.2095, -70.8701, 76.9661, -21.2146, 0.4807, 80.0175]`。
ICP与实时左TCP只用于确认工件中心和轴向符合交接条件，不再补偿这两个右手目标。
到位后右夹爪保持张开，操作者可输入 `a`
沿实时工件 `+X` 方向每次 LIN 推进 5 mm；直到尾部进入夹爪后输入 `y`
闭合，或输入 `q` 停止并由左手继续持件。未加 `--execute` 时不会运动机械臂或夹爪。

## 完整反抓、交接与放置

完整入口为 `full-execute`。它依次执行：第二类工件反抓、头部观察、交互式分割与
ICP、左手送交接位、右手示教位 `-X` 预抓取—TF补偿抓取，以及右手
放置。右手夹稳且左夹爪释放后，右手先保持当前姿态，沿实时工件坐标系
`-X 30 mm、+Z 30 mm` 做一次 LIN 安全撤离，避免到放置准备位的运动碰到
左手夹爪；撤离到位后再运动到
`[102.2483, 21.8477, -87.2013, 83.4439, -147.2882, -32.2943, -66.0196]`，
确认右手已离开交接区后，再按新的实时双臂状态规划左手恢复到中间初始位。
不在双手靠近的交接状态规划左手回位。释放前会记录右手 TCP—工件刚性关系和
撤离目标；恢复入口会识别释放位、撤离位及放置准备位，避免重复执行 3 cm 位移。
第二类工件的左手交接使用原来的笛卡尔 `planner="ptp"`：工件 `+Z` 严格朝上、
`+X` 严格朝 `body_base +Y`，禁止 yaw/tilt 冗余；若名义中心目标不可用，
只在已定义的安全交接区域内搜索中心位置。右手接取阶段的轴向门限为 `2°`，
仅用于到位和 TF 误差，`yaw_p15/yaw_m15` 等偏移姿态会被拒绝。

若中断时右手已夹稳、左夹爪已张开，使用恢复入口，不得重复抓取：

```bash
tests/reversed_head_observation_2/run_workpiece_2.sh right-execute \
  --result output/reversed_head_pose_d435_workpiece_2/<时间>/result.json \
  --resume-after-left-release \
  --confirm-workpiece-2-grasp-geometry \
  --require-confirmation
```

按当前现场决定，第二类工件最终放置沿用第一类工件的 Module 9/10 配置。映射文件为
`first_workpiece_place_config.json`，流程依次使用第一类 `right_place_ready`、Module 9
固定关节起点与力控 ServoPose 参数，成功后按 Module 10 松夹、上抬并回 Idle。执行前
仍会再次要求操作者确认第二类工件可沿用相同孔位、抓持姿态和力阈值。

完成反抓覆盖图、右手交接点和最终放置三项现场核验后执行：

```bash
tests/reversed_head_observation_2/run_workpiece_2.sh full-execute \
  --handoff-center-m X Y Z \
  --confirm-workpiece-2-reverse-grasp-geometry \
  --confirm-workpiece-2-handoff-geometry \
  --confirm-workpiece-2-place-calibration
```

如果左夹爪已经在本轮抓稳工件，不得再次运行桌面反抓识别，应从观察位阶段续跑：

```bash
tests/reversed_head_observation_2/run_workpiece_2.sh full-execute \
  --already-left-held \
  --handoff-center-m 0.42 0.20 0.12 \
  --confirm-workpiece-2-handoff-geometry \
  --confirm-workpiece-2-place-calibration
```

该模式从左臂实时状态规划到记录中间位，再到头部观察位，并将云台转到
`pitch=2°, yaw=26.5°`；不会读取桌面反抓候选，也不会再次闭合左夹爪。
头部观察采集固定使用完全手动 SAM2 标注：在 OpenCV 窗口左键拖 ROI，ROI 内左键
添加工件正点、右键添加背景/夹爪负点；终端输入 `s` 生成预览，确认 mask 后输入
`y`，输入 `r` 可清空重标。该分支显式关闭自动夹爪提示，不会加载固定自动点。

完整入口默认沿用第一类工件的预设点执行方式，不重启 Docker、不启用 MoveIt
attached collision，也不等待 attachment owner 握手。只有明确需要启用第二类工件
附着碰撞场景时，才额外传入 `--restart-planner-for-workpiece-2-collision`。

可使用已有桌面 RGB-D 采集复核反抓时增加 `--capture-dir DIR`。完整实体流程保留各阶段
终端 `y` 确认；任一子阶段失败会终止后续阶段，主入口会尝试向左右臂发送
`cancel_task`。
