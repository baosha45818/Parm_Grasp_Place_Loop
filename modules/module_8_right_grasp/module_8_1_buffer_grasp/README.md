# Module 8.1：右臂直接抓取与闭环总控

本目录提供两种入口：

- `buffer_grasp.py`：已获得视觉位姿后，单独执行一次右臂直接抓取（直接抓，不会先观测，也不会先移动到右臂的待机位姿4.json，因此如果单独运行这个代码，需要先把右臂调整到4.json）；
- `workpiece_pipeline.py`：负责右手的位姿调整、抓取。流程是先观察、判断是否可以直接抓、直接抓取或调整的闭环总控，机械臂在任意姿态都可直接运行这个程序，会自动调整到待机位姿，但有些位姿会由于关节超限而无法调整，因此有必要手动拖拽到差不多位姿再运行。

所有会下发实际运动的命令都必须显式附加 `--execute`。先运行预演确认参数、工件编号、坐标系和现场安全，再执行真实动作。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `buffer_grasp.py` | 右臂（仅 `arm_id=2`）的直接抓取入口。读取 Module 4.2 的 `mask_pose.json` 与工件 TOML 标定，计算语义抓取点；RRTstar 到 TCP −X 后方对齐点，再沿 TCP +X 进行 ServoPose（默认）或 RRT 接近；闭爪后通过夹爪最终开度确认是否抓住，确认后原路撤离并规划 +Z 抬升。 |
| `planning_log.py` | 进程安全的结构化日志公共模块。它通过环境变量 `WORKPIECE_PIPELINE_EVENT_LOG` 写入 JSONL；日志失败不会中断机器人控制。`buffer_grasp.py` 用它记录每次规划的起点、目标、耗时和结果；`workpiece_pipeline.py` 用它记录子进程、观察和决策事件。 |
| `workpiece_pipeline.py` | 右臂工件闭环总控。它移动到保存姿态、调用视觉、判断工件朝向和 Q1/yaw 条件，选择直接抓取或 Module 7 调整；抓取/调整后继续观察直到成功或达到周期上限。 |

工件专属标定在 `configs/workpiece_1_grasp.toml`、`configs/workpiece_2_grasp.toml` 中，包含语义抓取点、TCP 补偿、接近窗口、Servo 参数、夹爪确认开度和抓后抬升候选高度。

## 单独使用 `buffer_grasp.py`

适用场景：已经拥有正确的 Module 4.2 视觉结果，且已确认工件是正面并满足直接抓取条件。该脚本不调用 Module 7 调整模块。

进入本目录：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp
```

### 使用最新视觉结果

不传 `--workpiece-pose-file` 时，脚本会自动选择：

```text
/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_4_2/
  workpiece_<工件编号>/<最新数字目录>/mask_pose.json
```

先预演：

```bash
python3 buffer_grasp.py --workpiece 1 --arm-id 2
```

确认后实际执行：

```bash
python3 buffer_grasp.py --workpiece 1 --arm-id 2 --execute
```

### 指定某次视觉结果

当要复用某个固定的视觉输出时，显式指定文件。文件中的 `workpiece_asset` 必须与 `--workpiece` 一致。

```bash
python3 buffer_grasp.py \
  --workpiece 1 \
  --arm-id 2 \
  --workpiece-pose-file \
    /home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_4_2/workpiece_1/379/mask_pose.json \
  --workpiece-index 0
```

在以上命令末尾增加 `--execute` 才会执行真实动作。`--workpiece-index` 选择 JSON 中 `workpieces[]` 的目标，默认是 `0`。

### 执行流程与结果

```text
mask_pose.json + 工件 TOML
  → 算出 body_base 坐标系下的抓取点
  → RRTstar 搜索 TCP −X 后方安全对齐点
  → 最终 TCP +X 直线推进至抓取点
  → 闭爪，以标定开度区间确认抓取
  → 未确认：张爪后沿 body_base +Z 小幅复抓一次（因为之前会出现实际抓取点偏低的情况，所以加了这块）
  → 已确认：Servo 路径反向撤离，再 RRTstar +Z 抬升
```

成功返回 `0`。若整个后方接近窗口都不可规划，返回 `20`；该状态表示尚未闭爪，调用方可安全地转入调整策略。其他失败返回 `1`。

需要可访问的项目 `.venv`/`parm_sdk`、右臂服务、右夹爪服务和 Planner 服务。默认地址可通过 `PARM_ARM_HOST`、`PARM_GRIPPER_HOST`、`PARM_PLANNER_HOST` 等环境变量，或对应命令行选项覆盖。

## `workpiece_pipeline.py` 关联模块

这是右臂的小总控，负责调用module 4_2观察姿态、调用module 7调整工件姿态、调用module 8_1抓取工件。建议调试右臂时直接启动这个代码，会自动拍照位姿估计、判断直接抓或调整并运行对应模块代码。注意这个代码不负责左右手交接、右臂放置到焊接点。

总控会协调以下模块与数据：

```text
workpiece_pipeline.py
├─ saved_poses/{4,6,10}.json
│  └─ 用关节 RRTstar 在观察位、调整位之间移动右臂
├─ Module 4.2：modules/module_4_head_judge/module_4_2_pose_estimation/
│  └─ tabletop_pointcloud_workpiece_pose.py
│     这是头部相机观测模块，生成 output/module_4_2/workpiece_*/<编号>/mask_pose.json
├─ 本目录 buffer_grasp.py
│  └─ 工件姿态处在 Q1 且 yaw 位于 15°--75° 时执行直接抓取
├─ Module 7：modules/module_7_right_align/right_align.py
│  └─ 若不满足直接抓取条件、路径不可达或连续抓取失败后执行调整；
│     right_align.py 根据 --workpiece 自行选择并读取其调整配置
├─ configs/workpiece_{1,2}_grasp.toml
│  └─ buffer_grasp.py 根据 --workpiece 自行选择并读取
├─ Module 7 的 configs/workpiece_{1,2}_turn_candidates.toml
│  └─ module 7 的 right_align.py 会自行调用此路径
├─ planning_log.py 与 tool/planner_task_retry.py
└─ parm_sdk、Arm / Gripper / Planner 及视觉设备服务
```

决策逻辑如下：

1. 初始移动到 `4.json`，在观察位运行 Module 4.2。
2. 若视觉判定 `orientation` 不是 `FRONT`，即工件非正面摆放，总控暂停等待人工处理；处理好后输入 `y` 后重新观察。
3. 若工件在 Q1 且 yaw 在 `15°--75°`，调用 `buffer_grasp.py`，直接抓取。
4. 若 `buffer_grasp.py` 返回 `20`（后方直接抓取路径不可达），则直接转 Module 7 调整。
5. 其他抓取失败会释放夹爪并回到观察位，连续失败达到阈值（ 3 次）后强制进入 Module 7 调整。
6. 不满足 Q1/yaw 条件时，右臂按 `4 → 10 → 6` 前往调整位姿；调整完后，按 `6 → 10 → 4` 返回待机位姿。返回期间到达 `10.json` 后拍照，视觉计算与 `10 → 4` 回程并行，可以省点时间。
7. 直至抓取成功或达到 `--max-cycles`（默认 20）为止。



## 使用 `workpiece_pipeline.py`

先进行不接触硬件的流程预演：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp

python3 workpiece_pipeline.py --workpiece 1
```

预演会检查模块、配置和保存姿态是否存在，并打印流程；不会拍照、移动机械臂或操作夹爪。

确认现场安全后执行完整闭环：

```bash
python3 workpiece_pipeline.py --workpiece 1 --execute
```

常用控制参数：

```bash
python3 workpiece_pipeline.py \
  --workpiece 2 \
  --workpiece-index 0 \ （表示桌面上第几个工件，但一般桌面上只会有一个工件，所以默认 0）
  --max-grasp-failures 3 \ 
  --max-cycles 20 \
  --observation-attempts 3 \
  --joint-speed 0.1 \
  --joint-speed-scale 1.0 \
  --planner-host 127.0.0.1 \
  --execute
```

其中`--joint-speed × --joint-speed-scale` 必须在 `(0, 1]` 内。

默认日志目录为：

```text
/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_8_1/workpiece_<工件编号>/
```

每次运行生成一份终端文本日志 `.log` 和一份结构化事件日志 `.jsonl`。总控会通过项目 `.venv` 重新启动自身，并把同一个事件日志路径传递给 `buffer_grasp.py` 等子进程。

## 运行前检查

- 已确认工件编号、`workpiece-index`、视觉模型和对应 TOML 标定匹配；
- 右臂及夹爪已可通信，Planner 的 `body_base`、`right_eef` TF 可用；
- Module 4.2、Module 7、保存姿态和两类配置文件均存在；
- 确认观察位、调整位和抓取路径周围没有人或障碍物；
- 对真实动作，务必先不加 `--execute` 完成预演，再显式添加它。
