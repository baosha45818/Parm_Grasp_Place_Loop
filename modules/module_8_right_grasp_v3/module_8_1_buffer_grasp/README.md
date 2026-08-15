# Module 8.1：右臂缓冲区直接抓取

本目录是从原直接抓取链路重构出的独立模块。入口 `buffer_grasp.py` 读取相机/视觉模块生成的 `mask_pose.json`，根据选定工件的 TOML 配置，规划后方对齐点、沿工件方向接近、闭合夹爪并执行确认后的抬升。

`workpiece_pipeline.py` 是三个模块的闭环总控。它依次协调头部视觉、右臂姿态调整和右臂直接抓取：视觉结果在 Q1 的 15°--75° yaw 区间时直接抓取；否则执行调整，随后重新观察并继续判断。总控只调用以下新模块：

- `module_4_head_judge/module_4_2_pose_estimation/tabletop_pointcloud_workpiece_pose.py`；
- `module_7_right_align/right_align.py`；
- 本目录的 `buffer_grasp.py`。

目录内已包含完整的本地依赖：

- `buffer_grasp.py`：直接抓取入口；
- `configs/workpiece_1_grasp.toml`、`configs/workpiece_2_grasp.toml`：两种工件的抓取点、TCP 补偿、接近、夹爪确认与抬升参数；每项均带中文说明；
- 右臂目标直接使用规划器原生 `right_eef`；
- `move_by_joints.py`、`move_by_eef_pose.py`、`planning_log.py`：抓取执行需要的本地运动和规划公共代码。

先进行预演（不加 `--execute` 不会发送机械臂或夹爪动作）：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp/buffer_grasp.py \
  --workpiece 1 \
  --arm-id 2
```

`--workpiece` 会自动选择同编号抓取配置，并读取 `output/module_4_2/workpiece_<编号>/` 中最新的 `mask_pose.json`；可用 `--workpiece-pose-file` 显式覆盖，但文件的工件标记必须一致。确认预演与现场状态后，追加 `--execute` 才会实际打开夹爪、规划并执行抓取、闭合夹爪和抬升。模块需要可访问的 `parm_sdk`、右臂/夹爪服务和规划器。抓取入口不会调用调整模块；应仅在视觉结果已经满足直接抓取条件时使用。

## 手动移动到保存位姿

`saved_poses/3.json`、`4.json`、`6.json`、`10.json` 都保存的是右臂（`arm_id=2`）位姿。先不加 `--execute` 预览目标；确认机器人周围无碰撞风险、右臂处于可运动状态后，再在同一命令末尾加 `--execute`。默认机械臂地址为现场的 `192.168.3.11`，可用 `--arm-host` 覆盖。

按保存的七轴关节角移动（控制器 `move_j`，适合回到已示教的固定姿态）：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp

python3 move_by_joints.py \
  --arm-id 2 \
  --pose-file saved_poses/4.json \
  --execute
```

实际执行时：

```bash
python3 move_by_joints.py \
  --arm-id 2 \
  --pose-file saved_poses/4.json \
  --execute
```

将 `4.json` 替换为 `3.json`、`6.json` 或 `10.json` 即可使用相应保存姿态；如需降低关节运动速度，可添加例如 `--joint-speed 0.1`。

按保存的末端位姿移动（规划器规划，默认笛卡尔直线 `lin`；使用保存文件中的 `body_base` 下 `eef_pose`）：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp

python3 move_by_eef_pose.py \
  --arm-id 2 \
  --pose-file saved_poses/4.json
```

实际执行时：

```bash
python3 move_by_eef_pose.py \
  --arm-id 2 \
  --pose-file saved_poses/4.json \
  --execute
```

末端位姿移动默认连接本机规划器 `127.0.0.1`；如规划器不在本机，可用 `--planner-host <地址>` 指定。可用 `--planner ptp` 改为关节空间规划，或用 `--max-vel 0.05 --max-acc 0.05` 降低规划速度和加速度比例。







8-1模块单独跑，不加 --workpiece-pose-file 默认就是output/module_4_2/workpiece_<工件编号>/<最新编号>/mask_pose.json，加了参数可以指定--workpiece-pose-file \
  /home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_4_2/workpiece_1/379/mask_pose.json


cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp

python3 buffer_grasp.py \
  --workpiece 1 \
  --arm-id 2 \
  --execute





总控执行：

```bash
python3 /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_8_right_grasp/module_8_1_buffer_grasp/workpiece_pipeline.py --workpiece 1 \
--execute
```

确认预演和现场状态后再追加 `--execute`。总控使用项目 `.venv`，本目录 `saved_poses/` 中的 `3.json`、`6.json`、`10.json` 作为观察和调整过程中的右臂关节姿态；运行日志按工件分别写入 `/home/dainanjun/Public/Parm_Grasp_Place_Loop/output/module_8_1/workpiece_1/` 或 `workpiece_2/`。
