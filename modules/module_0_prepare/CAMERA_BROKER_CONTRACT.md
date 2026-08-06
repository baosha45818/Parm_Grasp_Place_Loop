# Camera Broker 稳定帧数据契约（CBFC）

状态：**设计基线**。首次实现使用 `schema_version: 1`；任何不兼容的
字段语义、单位或坐标系变化必须升级主版本，不得静默复用同一版本号。

## 1. 边界与所有权

Camera Broker 是 D405 和胸前 Femto 的唯一 SDK 所有者。业务模块、GraspNet
和调试脚本只能通过 Unix socket 请求采集，不能直接打开这两个设备序列号。

Broker 由 Module 0 启动或复用，并在整个 Loop 中持有设备。它提供：

- 单相机采集：`capture`；
- D405/Femto 同步采集：`capture_pair`；
- 健康与配置查询：`status`；
- 不提供连续视频流、任意 SDK 参数透传或删除采集结果。

头部 D435 是独立设备，后续使用独立 Broker；不得混入本契约的相机列表。

## 2. 固定 profile

客户端只传 `profile_id`，不传分辨率、序列号或 SDK 开关。profile 由 Module 0
配置，并在 `status` 返回其不可变摘要；这避免 Module 1、Module 2 用同一相机却
期待不同的内参或深度处理。

初始 profile：

| `profile_id` | 相机 | 用途 |
|---|---|---|
| `graspnet_d405_v1` | D405 | Module 1 抓取点云 |
| `graspnet_femto_v1` | Femto | Module 1 与 D405 同步点云 |
| `inspect_femto_v1` | Femto | Module 2 PC1/PC2 检查 |

每个 profile 固定设备序列号、实际分辨率/FPS、深度后处理、预热帧数及采集帧数。
Broker 启动时记录实际获得的 profile；若它与配置不一致则健康检查失败。

## 3. 请求与响应

传输为一行一个 UTF-8 JSON 的 Unix-domain socket RPC。所有请求必须带
`request_id`（UUID）和 `schema_version: 1`。

```json
{
  "schema_version": 1,
  "request_id": "b7da6d1f-d03c-4e2c-b5e6-091a52fdb421",
  "op": "capture",
  "camera": "femto_bolt",
  "profile_id": "inspect_femto_v1",
  "consumer": "module_2_left_inspect_apart",
  "purpose": "pc1"
}
```

`capture_pair` 使用同一 `profile_id` 列表（D405/Femto 各一个）并可声明
`max_start_skew_ms`；Broker 返回实际时间差，超过阈值时整次请求失败，不产出
`complete` 工件。

成功响应只返回 manifest 路径与摘要，图像和数组绝不嵌入 JSON：

```json
{
  "ok": true,
  "schema_version": 1,
  "request_id": "b7da6d1f-d03c-4e2c-b5e6-091a52fdb421",
  "capture_id": "20260802T131500.123Z_b7da6d1f",
  "manifest_path": "/.../captures/20260802T131500.123Z_b7da6d1f/manifest.json",
  "sha256": "<manifest sha256>"
}
```

失败响应含稳定的 `error_code`、人可读 `message`、`request_id`。首版错误码为
`INVALID_REQUEST`、`UNKNOWN_PROFILE`、`PROFILE_MISMATCH`、`DEVICE_UNAVAILABLE`、
`WARMUP_FAILED`、`CAPTURE_TIMEOUT`、`QUALITY_REJECTED`、`SYNC_SKEW_EXCEEDED`、
`STORAGE_FAILED` 与 `INTERNAL`。客户端只能据 `error_code` 决策重试；不得解析
`message`。

## 4. 不可变 Capture Bundle

成功采集产生一个只写一次的目录：

```text
captures/<capture_id>/
├── color.npy                 # uint8, Hc×Wc×3，BGR
├── depth.npy                 # uint16, Hd×Wd，原始整数深度
├── color_intrinsics.npy      # float64, 3×3
├── depth_intrinsics.npy      # float64, 3×3
├── color_distortion.npy      # 可选，float64, N
├── depth_distortion.npy      # 可选，float64, N
├── T_depth_color.npy         # 可选，float64, 4×4
├── preview_color.png         # 非契约诊断文件，不可作为计算输入
└── manifest.json             # 唯一权威元数据
```

Broker 先写入同一文件系统上的临时目录，完成数组、manifest 与 hash 校验后再原子
`rename` 为最终目录。`manifest.json` 最后写入，且含 `state: "complete"`。客户端
只接受完整目录、`state=complete` 和 SHA-256 校验通过的 bundle；失败或取消的临时
目录不得被客户端读取。

`capture_id` 全局唯一，Broker 永不覆写既有 capture。保留策略由独立清理任务执行，
不得在采集路径中删除历史工件。

## 5. manifest：稳定字段及其语义

```json
{
  "schema_version": 1,
  "state": "complete",
  "capture_id": "...",
  "request": {
    "request_id": "...",
    "consumer": "module_2_left_inspect_apart",
    "purpose": "pc1",
    "profile_id": "inspect_femto_v1"
  },
  "camera": {
    "name": "femto_bolt",
    "serial": "CL8L56301LK",
    "profile_fingerprint_sha256": "..."
  },
  "frames": {
    "requested_count": 5,
    "accepted_count": 5,
    "depth_aggregation": "median_uint16",
    "color_selection": "last_accepted"
  },
  "geometry": {
    "color_order": "BGR",
    "depth_dtype": "uint16",
    "depth_scale_m_per_unit": 0.001,
    "color_shape_hw": [720, 1280],
    "depth_shape_hw": [720, 1280],
    "registration": "depth_in_color_geometry"
  },
  "timing": {
    "capture_started_unix_ns": 0,
    "capture_finished_unix_ns": 0,
    "capture_started_monotonic_ns": 0,
    "capture_finished_monotonic_ns": 0
  },
  "quality": {
    "valid_depth_ratio": 0.0,
    "dropped_frames": 0,
    "checks": {"intrinsics_finite": true, "depth_nonempty": true}
  },
  "files": {"depth.npy": "sha256:...", "color.npy": "sha256:..."}
}
```

强制语义：

- `color.npy` 一律为 BGR、`uint8`；禁止根据 SDK 或消费者改成 RGB。
- `depth.npy` 一律为未经单位换算的 `uint16` 整数；米值恒为
  `depth * depth_scale_m_per_unit`。`depth_scale_m_per_unit` 必须大于零。
- 内参均为像素坐标中的 `float64` 3×3 矩阵。它们分别对应保存的 color/depth 数组，
  不得假定二者相同。
- `registration=depth_in_color_geometry` 时，depth 的像素坐标与 color 相同，
  `depth_intrinsics.npy` 必须等于 `color_intrinsics.npy`，`T_depth_color.npy` 不出现。
- `registration=separate_optical_frames` 时，必须提供 `T_depth_color.npy`，定义为
  `p_color = T_depth_color @ p_depth`；消费者不得以数组形状相同推断已对齐。
- `frames.depth_aggregation` 与 color 选择策略必须显式记录。Module 2 的初始 profile
  固定为 `median_uint16` + `last_accepted`，匹配当前实现。

## 6. Broker 行为与并发

- 同一物理相机的 SDK 调用严格串行；不同相机可并行。
- `capture_pair` 以两个相机的独立锁并行采集，并以一次逻辑事务提交两个 bundle 和
  pair manifest；任一失败则 pair manifest 不标记 complete。
- 请求进入有界 FIFO 队列；超过队列上限返回 `DEVICE_UNAVAILABLE`，不无限阻塞。
- 客户端断开不取消已开始的 SDK 采集；Broker 完成并保留结果，以防设备停在半状态。
- 任何 profile、设备序列号或 SDK 流参数变化均要求 Broker 重启；运行中拒绝热切换。

## 7. 健康检查与准入

`status` 返回 socket ABI 版本、每台设备状态、实际 profile fingerprint、队列长度、
最近成功采集时间及每个 profile 最近失败错误码。Module 0 在机械臂运动前必须验证：

1. Broker ABI 为 1，D405 与 Femto 的序列号符合配置；
2. 所有本轮启用的 profile 存在且 fingerprint 一致；
3. 执行一次不落盘的 warmup/readiness probe，检查 SDK 可读、内参有限、深度尺度
   合法、有效深度比例达到 profile 阈值；
4. 每个 consumer 仅获准调用其声明的 profile；例如 Module 2 只能请求
   `inspect_femto_v1`。

Module 1、Module 2 仍须在真正采集后校验 manifest；健康检查不能替代单帧质量判断。

## 8. Module 2 适配边界

Module 2 的 `BrokerFemtoCamera.read()` 读取和校验一个完整 bundle，并构造现有
`RGBDFrame`：`color`、`depth`、两套内参、`depth_scale`、相机名、序列号、畸变和
外参。`_capture_and_register()` 之后的 SAM2、点云与 ICP 逻辑不变。

适配器必须拒绝未知 schema、未完成工件、hash 不匹配、BGR/uint8/uint16 不符、非有限
内参、非法尺度和注册语义缺失；这些都以本地 `FrameContractError` 明确失败，禁止
猜测或自动修正。

## 9. 验收门槛

在替换 Module 2 前，必须通过以下兼容性测试：

1. 用同一原始 Femto 采集，直连路径和 Broker 路径反序列化后的 frame 字段逐项一致；
2. 固定 PC1/PC2 回放输入下，mask、有效点数、`T_camera_object` 与分类结果在既定
   数值容差内一致；
3. Module 1 `capture_pair` 与 Module 2 单 Femto 请求交错执行，验证没有 SDK
   打开冲突、无并发读帧及无未完成目录被消费；
4. 断开客户端、写盘失败、设备拔插、坏 socket 和 profile 不匹配均得到稳定错误码，
   且不会删除已完成工件或关闭其他基础硬件服务。
