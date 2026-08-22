# Module 1：左臂 ICP 抓取

Module 1 是独立的左臂传统视觉抓取模块，只保留 D405 + SAM2 + STL/ICP
路线，不依赖 Module 2、Module 3 或旧总控。

```text
D405 RGB-D
→ SAM2 实例分割
→ 深度反投影
→ STL 多初值 ICP
→ 工件正反面与夹爪安全检查
→ Planner 预抓取 + 10 cm LIN 下探
→ 左夹爪抓取
→ 重量和夹持状态验证
→ 持件返回 `graspuptop` 标准交接位
→ verified_handoff.json
```

## 部署模式

必须通过 `--mode` 选择整套部署参数，不能混用相机、标定和服务地址。

| 模式 | D405 | 手眼标定 | FPS | Arm | Gripper | Force sensor |
| --- | --- | --- | ---: | --- | --- | --- |
| `gzhu` | `260322270242` | SDK + bridge | 15 | `192.168.88.1` | `127.0.0.1` | `127.0.0.1` |
| `xiangxin` | `260322270600` | planner-direct | 30 | `192.168.3.11` | `192.168.3.11` | `192.168.3.22` |

模式会同时约束 `--d405-serial`、`--handeye-mode`、`--arm-host`、
`--gripper-host` 和 `--force-sensor-host`。显式参数与模式不一致时，程序在
采集或运动前直接报错。

### 标定与偏置

`gzhu` 使用：

- `tools/calibration/left_wrist_d405/calibration_wrist1.json`
- `tools/calibration/graspnet_planner_bridge_5604_20260803/planner_tf_bridge_5604_candidate.json`
- 正面沿 `body_base +Z` 抬高 21 mm
- 反面沿 `body_base +Z` 抬高 27 mm
- 不使用 planner-direct 的 7 mm 水平夹爪轴补偿

`xiangxin` 使用：

- `tools/calibration/wrist1_planner_final_v3/calibration_wrist1_planner.json`
- 正反面固定 Z 抬高均为 0
- 按水平夹爪轴投影应用最大 7 mm 抓取中心补偿

两种模式都保留反面工件局部 `-X` 方向 1.5 mm 公共偏置。点云避碰允许在
各模式的固定 Z 偏置基础上再自动抬高，默认上限 30 mm：`gzhu` 从正面
21 mm/反面 27 mm 开始搜索，`xiangxin` 从 0 mm 开始搜索。

## 运行

### 无动作规划

不提供真实执行参数时，只采集当前位置图像并生成抓取与 Planner 预检结果，
不会控制机械臂或夹爪：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_1_left_grasp/run.sh \
  --mode gzhu \
  --skip-observation-move \
  --frames 1
```

### 全自动真实抓取

真实动作必须同时提供 `--execute` 和 `--confirm-real-grasp`。授权后各运动阶段
连续自动执行，不进行终端交互：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_1_left_grasp/run.sh \
  --mode gzhu \
  --execute \
  --confirm-real-grasp \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json
```

香信环境只需将模式改为：

```bash
--mode xiangxin
```

`run.sh` 会检查或启动 `tools/sam2-service`。D405 由 `main.py` 在每次观察时直接
通过 `pyrealsense2` 打开、采集并关闭；不使用 Camera Broker。`gzhu` 的本机夹爪和
力传感器服务需要在运行前已经监听。

如需外部控制器在最终运动前放行，可设置 `MODULE1_EXECUTION_PERMIT_FILE`；
文件内容必须为 `allow`。未设置时不会等待外部门禁。

## 退出条件
Module 1 只有在以下条件全部通过后才写满足退出条件：

- Planner 预抓取和 10 cm LIN 下探均可达
- 抓取后反向 LIN 抬升完成
- 左夹爪稳定报告持件
- 力传感器判定为单个工件 `SINGLE`
- 返回抓取后安全抬升位仍保持 `SINGLE`
- 到达最终 `graspuptop` 交接位后仍确认左夹爪持件

推荐固定输出路径：

```text
output/module_1_left_grasp/verified_handoff.json
```

该路径需要通过 `--weight-handoff-file` 显式指定；未指定时，handoff 写入本轮
运行目录下的 `weight_handoff.json`。

Module 3 只接受其中 `weight_check_enabled=true` 且
`weight_check.classification="SINGLE"` 的结果。

## 代码结构

- `run.sh`：模式绑定、SAM2 启动检查。
- `main.py`：采集、Planner 预检、真实抓取、重试和 handoff 编排。
- `icp_pipeline.py`：SAM 深度点云、STL 姿态、工件语义、抓取与碰撞筛选。
- `weight_verification.py`：抓取前后力传感器采样与单工件判定。
- `tools/icp.py`：公共纯数值配准，包括 PCA 多初值和 Open3D ICP。
- `tools/graspnet_runtime/utils/robot_camera.py`：D405 的直接驱动适配器。

运行产物写入 `output/module_1_left_grasp/`，包括 SAM/ICP 可视化、候选报告、
Planner 预检、最终抓取计划和 handoff。

## 重构后的函数边界

Module 1 的业务步骤均通过命名函数表达；`main()` 只负责参数声明、模式绑定、
安全校验和顶层编排。所有函数及类都带中文用途说明。

`main.py` 的主要函数：

| 函数 | 职责 |
| --- | --- |
| `capture_log()` | 直接通过 pyrealsense2 采集 D405 RGB-D，并固化采集时的手眼链元数据 |
| `traditional_perception()` | 调用 `icp_pipeline.py` 完成一次纯感知规划 |
| `ranked_planner_records()` | 从感知报告提取有序 Planner 候选 |
| `paired_pregrasp_and_descent()` | 将预抓取 PTP 与 10 cm LIN 下探作为同一候选进行规划 |
| `paired_preflight()` | 在动作前验证成对轨迹及起点一致性 |
| `select_reachable_candidate()` | 按视觉排序选择首个 Planner 可达候选 |
| `execute_paired_plan()` | 执行已验证的预抓取、下探、夹取和反向抬升 |
| `prepare_traditional_candidate()` | 完成一次采集、ICP、候选筛选和 Planner 预检 |
| `run_traditional_cycles()` | 管理无抓取、不可达和重量门控失败后的有限重试 |
| `move_to_initial_observation_if_authorized()` | 仅在真实执行授权后移动到初始观察位 |

`icp_pipeline.py` 按以下阶段组织：

1. 校验参数、相机序列号、手眼模式和模式专属偏置。
2. 读取 RGB-D、内参及采集时机械臂位姿。
3. 调用 SAM2 并反投影各实例点云。
4. 调用 `tools/icp.py` 执行 PCA 多初值 STL-ICP。
5. 验证单 STL 完整性、正反面特征、夹持材料和 SAM 轮廓。
6. 检查张开夹爪与 10 cm 下探扫掠体的点云碰撞。
7. 生成排序后的 Planner 候选、复核图和诊断报告。

纯数值 ICP、SAM2、标定文件、PlannerRuntime 和观察位工具均从项目级
`tools/` 调用，Module 1 文件夹内只保留本模块的业务编排和规则。

## 安全边界

- `--execute` 与 `--confirm-real-grasp` 必须同时出现或同时不出现。
- plan-only 路径不会发送机械臂运动或夹爪命令，也不会移动到观察位；
  `bridge` 手眼链仍会只读获取采集时的机械臂位姿。
- ICP 流程只生成候选文件，不直接发送机器人命令。
- 真实执行只使用通过成对 Planner 预检的预抓取和 LIN 下探轨迹。
- 未通过夹爪持件与 `SINGLE` 重量门控时不会生成有效 handoff。
- 模式绑定冲突、标定链缺失或相机序列号不一致都会在运动前失败。
