# Version7：常驻 GraspNet 抓取流程

Version7 按 Version5 的双相机候选、外参转换和候选筛选逻辑运行；模型、D405 和 Femto 在后台常驻。前台只负责到 `go_graspnet`、双相机采集、CPU 点云预处理、batch=2 推理、Version5 筛选和 Planner 抓取。

## 运行前

- arm1、D405、Femto、夹爪和 Planner 均应已上电且可用；
- GraspNet 权重位于 `checkpoint/checkpoint-rs.tar`；
- 使用 `.venv`；
- 当前固定使用 arm1、D405 `260322270242`、Femto `CL8L56301LK`；
- 相机服务与 GraspNet 服务必须使用同一个任务目录。

以下命令使用当前默认配置：D405 后处理开启、孔洞填充开启、任务目录为 `deploy/version7/jobs`。

```bash
cd /home/dainanjun/qiuguolin/graspnet

export GRASPNET_V7_JOBS_ROOT=/home/dainanjun/qiuguolin/graspnet/deploy/version7/jobs
export GRASPNET_V7_NUM_VIEW=300
export GRASPNET_V7_ARM_HOST=192.168.88.1
export GRASPNET_V7_ARM_ID=1
export GRASPNET_V7_D405_POSTPROCESS=1
export GRASPNET_V7_D405_HOLE_FILLING=1
```

## 必须提前启动的后台服务

### 1. GraspNet GPU 服务

```bash
cd /home/dainanjun/qiuguolin/graspnet
bash deploy/version7/start_graspnet_service.sh
tail -f deploy/version7/runtime/graspnet-service.log
```

服务 PID 写入 `deploy/version7/runtime/graspnet-service.pid`；GCS 与模块化总流程共用该运行目录。

继续之前，日志必须依次出现：

```text
GraspNet service ready
GraspNet warmup complete
listening on /tmp/graspnet-v7.sock
```

### 2. D405 + Femto 常驻相机服务

```bash
cd /home/dainanjun/qiuguolin/graspnet
bash deploy/version7/start_camera_service.sh
tail -f deploy/version7/logs/camera_service.log
```

继续之前，日志必须出现：

```text
Version7 camera service ready: D405 and Femto are open
camera service listening on /tmp/graspnet-v7-camera.sock
```

前台启动时会自动核对任务目录、`num_view`、arm host/id、相机序列号、D405 后处理和孔洞填充配置。任何不一致都会在机械臂运动前退出。

## 前台抓取命令

推荐使用已固化配置的脚本，不要手动粘贴超长命令：

```bash
cd /home/dainanjun/qiuguolin/graspnet
bash deploy/version7/run_grasp_v7.sh
```

该命令会：

1. Planner 移动 arm1 到 `go_graspnet` / D405 观察位；此步骤不再等待终端确认；
2. 到位后自动以 100% 打开夹爪；
3. D405 与 Femto 并行采集，CPU 并行预处理；
4. 使用后台 GraspNet batch=2 推理；
5. 按 Version5 逻辑汇总、过滤和显示可执行抓取；
6. 用户确认真实抓取后，使用 Planner 执行抓取，并移动到 `post-grasp lift` / `graspup` 位；夹爪保持闭合，流程结束，交给模块 2 做工件观察与位姿估计。

Version7 默认启用 `--stop-after-graspup`，因此不会再前往 buffer 或打开夹爪。若只为旧的独立放置调试而需要恢复旧行为，可显式传入 `--no-stop-after-graspup`。

在任何真实运动之前，Version7 会按 GraspNet 分数惰性检查前 3 个候选。每个候选只使用与当前末端姿态距离更近的平行夹爪等价姿态，并记录最终选用的 `0°` 或 `180°`；不会额外尝试另一个等价姿态。若实际 pregrasp 或夹爪闭合前的 LIN 下抓明确无轨迹，机械臂会安全回观察位并继续下一个分数候选。全部不可达时，默认重新采集并推理 3 次。规划器连接、TF、网络或轨迹格式故障不会伪装成“候选不可达”，而是立即停止。诊断记录写入：

```text
deploy/version7/output/<run>/weight_cycle_XX/reachability_cycle_XX/planner_candidate_reachability.json
```

可调整这两个数量，例如只检查前 2 个候选、允许一次重新推理：

```bash
bash deploy/version7/run_grasp_v7.sh \
  --planner-candidate-top-k 2 \
  --max-planner-unreachable-retries 1
```

脚本默认 `--max-no-grasp-retries 10`，即无有效候选时最多总计 11 次视觉尝试。可在脚本命令后覆盖，例如只允许两次视觉尝试：

```bash
bash deploy/version7/run_grasp_v7.sh --max-no-grasp-retries 1
```

每轮会输出服务预检、观察位、夹爪、相机采集、CPU 预处理、GraspNet 请求、Version5 筛选和总耗时。总计时在预测结果即将显示前，或视觉重试全部失败后停止。

## 力传感器单件验证（可选）

力传感器没有单独的 Version7 启动服务；由前台经 SDK 连接 `--force-sensor-host`。启用真实抓取后的单件验证：

```bash
cd /home/dainanjun/qiuguolin/graspnet

bash deploy/version7/run_grasp_v7.sh \
  --grasp-weight-check \
  --max-weight-retries 3 \
  --force-sensor-host 127.0.0.1 \
  --force-sensor-hand 1
```

流程在预抓取位采集空载基线；闭爪后不做成功判定，始终先执行完整 `post-grasp lift` 到 `graspup`。到达 graspup 后才检查夹爪保持状态并采集载荷、判定 `SINGLE`。`EMPTY`、`MULTIPLE` 或 `UNKNOWN` 会在 graspup 释放，再回 `go_graspnet` 开始新一轮 Version7 观察、Version5 筛选与抓取。每次重量判定保存到：

```text
deploy/version7/output/<run>/weight_cycle_XX/weight_check_result.json
```

当前默认阈值：

```text
EMPTY:    <= 30 g
SINGLE:   >30 g 且 <130 g
MULTIPLE: >= 130 g
UNKNOWN:  仅采样不稳定或力数据不可用
```

边界按安全侧处理：正好 `30 g` 算 `EMPTY`，正好 `130 g` 算
`MULTIPLE`。`EMPTY`、`MULTIPLE` 和 `UNKNOWN` 都会在 graspup 打开夹爪，
然后重新采集点云并推理；只有 `SINGLE` 保持夹持并交给模块 2。

这些阈值须以最终检查位姿、当前夹具和本工件进行实机标定后再用于生产。

## 可选服务：SAM2 与 Version6 常驻 ICP

它们不参与 Version7 的初始 GraspNet 抓取；仅在后续 Version6 工件观察/位姿估计流程使用。

### SAM2 UDS 服务

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service
./start_uds_server.sh
```

默认 socket：`/tmp/sam2-service.sock`。

### Version6 常驻 STL/CAD/ICP 服务

```bash
cd /home/dainanjun/qiuguolin/graspnet
bash deploy/version7/start_pose_service.sh
tail -f deploy/version7/logs/pose_service.log
```

运行 Version6 时加入：

```bash
--pose-service-socket /tmp/graspnet-v7-pose.sock
```

## Socket 遗留处理

所有后台启动脚本都拒绝覆盖现有 Unix socket。确认旧服务已停止后，才可删除对应遗留 socket 并重新启动，例如：

```bash
rm -f /tmp/graspnet-v7.sock /tmp/graspnet-v7-camera.sock
```
