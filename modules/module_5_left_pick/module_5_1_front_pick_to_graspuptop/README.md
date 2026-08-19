# Module 5.1：正面工件 Buffer 重抓与 graspuptop 交接

Module 5.1 只处理 Module 4.1 已确认的唯一 `FRONT` 工件。它不采集图像、
不运行 SAM2/ICP，也不负责桌中放置。

```text
Module 4.1 FRONT + ICP TF
→ 校验唯一工件和位姿
→ 生成两个夹爪姿态候选
→ 预规划 RRTStar 预抓、LIN 下探和 LIN 上提
→ 打开左夹爪
→ 到预抓位
→ LIN 下探并闭爪
→ 沿 body_base +Z 上提 3 cm
→ 确认持件
→ RRTStar 到 graspuptop
→ 再次确认持件
→ 写出 front_regrasp_handoff
```

handoff 成功后夹爪保持闭合，`released=false`。桌中放置由 Module 3 使用
`--placement-kind middle` 独立完成。

## 输入契约

必须提供 Module 4.1 的结果文件。代码要求：

- `decision="FRONT"`；
- `continue_allowed=true`；
- `objects` 恰好包含一个工件，且 `orientation="FRONT"`；
- `workpiece_id` 为 `"1"` 或 `"2"`；
- `capture_dir` 存在，用于归档 Module 4.1 可视化；
- 位姿至少提供以下一种：
  - `objects[0].diagnostics.pose.T_body_base_workpiece`；
  - `objects[0].diagnostics.positive.T_camera_object`，并同时提供有效的
    `body_head_calibration` 文件路径。

输入不满足契约时，在连接 Planner、机械臂和夹爪之前失败。

## 抓取几何

默认参数：

- 预抓位：ICP原点沿 `body_base +Z` 抬高5 cm；
- LIN下探：2 cm；
- 默认闭爪点：ICP原点上方3 cm；
- 闭爪后LIN上提：3 cm。

程序根据ICP工件轴生成两个右手夹爪姿态候选，先为每个候选完整规划
“预抓→下探→上提”，再选择预抓关节位移较小的可达候选。上提轨迹必须包含
有效的七关节终点，否则不会闭爪执行该候选。

## 运行

查看参数不会连接或控制机器人：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

.venv/bin/python \
  modules/module_5_left_pick/module_5_1_front_pick_to_graspuptop/main.py \
  --help
```

Module 5.1 当前没有 plan-only 模式。真实重抓必须显式传入 `--execute`：

```bash
.venv/bin/python \
  modules/module_5_left_pick/module_5_1_front_pick_to_graspuptop/main.py \
  --buffer-check-result output/module_4_head_judge/buffer_face_check.json \
  --result-file output/module_5_left_pick/front_regrasp_handoff.json \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.3.11 \
  --gripper-host 192.168.3.11 \
  --execute
```

`gzhu` 环境应由控制器或命令行传入 `--arm-host 192.168.88.1` 和
`--gripper-host 127.0.0.1`。相机标定已经由 Module 4.1 固化在其输出位姿中，
Module 5.1 不按D405序列号重新选择标定。

## 返回与输出

没有可达抓取候选，或上提后未确认持件时，进程返回退出码 `20`，不发布成功
handoff。未持件分支会先打开左夹爪。

成功结果的关键字段为：

```text
stage = "front_regrasp"
module = "module_5_1_front_pick_to_graspuptop"
outcome = "front_regrasp_handoff"
classification = "FRONT"
handoff_pose = "graspuptop"
released = false
```

Module 4.1 的可视化证据会复制到
`output/module_5_left_pick/front_buffer_regrasp/<timestamp>/`，handoff副本为
`front_regrasp_handoff.json`。

## 安全边界

- 缺少 `--execute` 时立即退出，不读取结果文件，也不连接硬件。
- 三段抓取轨迹必须全部规划成功后才允许打开夹爪并开始真实重抓。
- 闭爪后必须完成3 cm LIN上提并确认 `motion_state="gripping"`。
- 到达 `graspuptop` 后再次确认持件，确认失败不发布handoff。
- 本模块不会松爪、不会移动到Middle，也不会调用Module 3。
