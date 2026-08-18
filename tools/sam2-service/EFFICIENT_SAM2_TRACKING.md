# Efficient-SAM2 在线 Tracking 使用说明

本文档面向接入方。正式 runtime 位于 `src/efficient_sam2_tracking.py`，模型、predictor、
CUDA context 和 inference state 由一个专用工作线程持有。调用者提交的图像会被复制；
`append()` 只入队，只有显式调用 `propagate()` 才执行传播。

## 1. 环境和资源

项目根目录：

```text
/mnt/data2/shurenbin/sam2_service
```

默认资源已经按相对路径配置：

```text
checkpoints/sam2.1_hiera_base_plus.pt
3rdparty/efficient-sam2/bypass/ckpt/bypass_bottleneck_base.pth
3rdparty/efficient-sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
```

Efficient-SAM2 必须使用 NVIDIA CUDA。当前验证环境的 Python 是：

```text
/mnt/data2/shurenbin/.conda/envs/dp/bin/python
```

如果服务目录位于 `/mnt/data2/shurenbin/sam2_service`，应把它的父目录加入
`PYTHONPATH`，或者在程序启动时加入 `sys.path`：

```python
import sys
sys.path.insert(0, "/mnt/data2/shurenbin")
```

## 2. 推荐接入方式

推荐每路视频创建一个 `EfficientSam2Tracker`。不要让多路无关视频共享同一个 tracker，
因为每个实例只维护一条视频的 inference state。

```python
import cv2
from sam2_service import EfficientSam2Tracker

first_bgr = cv2.imread("first.png", cv2.IMREAD_COLOR)
first_mask = cv2.imread("first_mask.png", cv2.IMREAD_GRAYSCALE)
frame_1 = cv2.imread("frame_1.png", cv2.IMREAD_COLOR)
frame_2 = cv2.imread("frame_2.png", cv2.IMREAD_COLOR)

with EfficientSam2Tracker(
    device="cuda:0",
    queue_capacity=16,
    overflow_policy="raise",
    mixed_precision="bf16",
    compile_model=True,
    compile_mode="default",
) as tracker:
    initial = tracker.initialize_from_mask(first_bgr, first_mask)

    tracker.append(frame_1, frame_id=1)  # 只复制并入队，不推理
    tracker.append(frame_2, frame_id=2)
    batch = tracker.propagate()          # 传播当前队列的两帧

    latest_mask = batch.latest.mask       # uint8 HxW，取值 0/1
    latest_overlay = batch.latest.overlay # uint8 BGR HxWx3
```

`start()` 会同步完成模型加载；启用编译时还会完成 compile warmup，全部成功后才返回。
退出时必须调用 `close()`，使用 `with` 可以保证自动关闭。

## 3. 初始化首帧

每次 tracking 必须先选择一种提示方式初始化：

```python
# 已有二值 mask；非零值均视为前景
result = tracker.initialize_from_mask(image_bgr, mask, frame_id=0)

# 正负点；坐标是原图像素坐标 (x, y)，label 1=正点、0=负点
result = tracker.initialize_from_points(
    image_bgr,
    points=[[320, 240], [100, 100]],
    labels=[1, 0],
    frame_id=0,
)

# xyxy 框；坐标是原图像素坐标
result = tracker.initialize_from_box(
    image_bgr,
    box_xyxy=[100, 80, 500, 600],
    frame_id=0,
)
```

再次调用任意初始化函数会清空旧队列、结果历史和旧 inference state，并开始一条新视频。
后续帧的高宽和通道数必须与首帧一致，输入格式必须为 OpenCV BGR `HxWx3 uint8`。

## 4. 入队与传播

```python
frame_id = tracker.append(frame)               # 自动分配 ID
frame_ids = tracker.append_many([frame1, frame2])

result = tracker.propagate()                   # 只传播已有队列
result = tracker.propagate(new_frame)          # 已有队列 + 单个新帧
result = tracker.propagate([new1, new2])       # 已有队列 + 一组新帧
```

两种 `propagate` 都返回 mask。队列为空时，`propagate()` 不执行推理，返回当前 anchor：

```python
result.propagated_count == 0
result.latest.mask
```

传播成功后队列会被清空，最新帧成为当前 anchor。模型只处理新增帧；历史视觉结果和
mask memory 保留在持久 inference state 中。

队列满时：

- `overflow_policy="raise"`：抛出 `BufferError`，适合不允许丢帧的任务。
- `overflow_policy="drop_oldest"`：丢弃最早的未传播帧，适合实时任务。

## 5. 返回值

`TrackingFrameResult`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | `int` | 调用者指定或 runtime 自动生成的帧 ID |
| `mask` | `np.ndarray` | `uint8 HxW` 二值 mask，值为 0/1 |
| `overlay` | `np.ndarray` | 带红色 mask 和绿色 bbox 的 BGR 图像 |
| `bbox` | `list[float] \| None` | `[x0, y0, x1, y1]`，空 mask 时为 `None` |
| `mask_pixels` | `int` | 前景像素数量 |

`TrackingBatchResult`：

| 字段/属性 | 含义 |
|---|---|
| `frames` | 本次传播产生的逐帧结果 tuple |
| `masks` | 本次结果的 mask tuple |
| `latest` | 本次最后一帧结果 |
| `propagated_count` | 本次实际传播帧数 |
| `queued_before/queued_after` | 调用前后队列长度 |
| `latency_ms` | 本次传播墙钟耗时，不包含此前模型启动时间 |

返回数组是副本，调用方可以安全修改，不会破坏 tracker 内部状态。

## 6. 混合精度和编译

```python
EfficientSam2Tracker(
    mixed_precision="bf16",  # 推荐；也支持 fp16、fp32
    compile_model=True,
    compile_mode="default",
)
```

- `bf16` 已完成精度和速度回归，是默认值。
- `fp16` 可用，但数值范围小于 bf16，需要业务数据回归后再使用。
- `fp32` 关闭 autocast，显存和耗时更高。
- `compile_model=False` 默认不编译，启动更快。
- 启用编译后，worker 在返回 ready 前用两帧虚拟数据完成 warmup。

只编译形状稳定的 memory encoder、prompt encoder 和 mask decoder。bypass image encoder
含动态 Python 控制流，memory attention 的输入长度随历史变化，这两个模块保持 eager，
避免在线运行期间重复编译卡顿。

当前三帧 bf16 编译测试中，warmup 约 10.3 秒（不计入在线推理），最新传播约
62.5 ms；三帧 IoU 为 0.9921、0.9885、0.9931。具体速度随 GPU、驱动和输入变化。

## 7. 状态与生命周期

```python
status = tracker.start()
status = tracker.status()
latest = tracker.latest()
tracker.reset()  # 清空当前视频，但不卸载模型
tracker.close()  # 停止线程并释放 runtime 引用
```

`status()` 的常用字段包括：

- `initialized`、`device`
- `queued_frames`、`queue_capacity`、`dropped_frames`
- `anchor_frame_id`、`state_frames`
- `propagation_count`、`last_latency_ms`
- `mixed_precision`、`compile_model`、`compile_warmup_ms`

多个业务线程可以调用同一个 tracker；所有状态修改和 CUDA 推理在内部 worker 中串行。
但关闭 tracker 时应先停止上游生产者，避免在 `close()` 后继续提交帧。

持久 state 会随视频增长。超长连续视频需要监控显存，可以在业务边界调用 `reset()`
并重新初始化；目前 runtime 没有自动截断已经进入模型 state 的历史帧。

## 8. 模块级单例接口

简单的单路视频程序也可以使用根包导出的全局接口：

```python
from sam2_service import (
    append_tracking_frame,
    close_tracking,
    initialize_tracking_from_mask,
    propagate_tracking,
    start_tracking,
)

start_tracking(device="cuda:0", queue_capacity=16, mixed_precision="bf16")
try:
    initialize_tracking_from_mask(first_bgr, first_mask, frame_id=0)
    append_tracking_frame(next_bgr, frame_id=1)
    result = propagate_tracking()
finally:
    close_tracking()
```

已启动全局 tracker 后，不能通过再次 `start_tracking(**new_options)` 修改配置；必须先
`close_tracking()`。复杂程序和多路视频仍建议直接实例化 tracker。

## 9. 验收测试

```bash
cd /mnt/data2/shurenbin/sam2_service

/mnt/data2/shurenbin/.conda/envs/dp/bin/python \
  test/test_efficient_sam2_tracking.py \
  --frames-dir /tmp/efficient_sam2_first3_frames \
  --seed-mask test/test_tracking_queue/frame_0000_mask.png \
  --output test/test_tracking_queue/efficient_sam2_result \
  --device cuda:0 \
  --queue-capacity 3 \
  --propagate-every 1 \
  --max-frames 3 \
  --mixed-precision bf16 \
  --compile \
  --compile-mode default
```

输入也可以换成 `--video`、`--camera`；初始化可以换成 `--point` 或 `--box`。
输出目录包含 `masks.npz`、`tracking_overlay.mp4`、预览图和 `summary.json`。

## 10. 异常处理

- `RuntimeError: Efficient-SAM2 requires CUDA`：当前进程看不到 CUDA。
- `FileNotFoundError: checkpoint not found`：权重路径不完整或服务目录迁移不完整。
- `BufferError`：队列已满；提高容量、及时传播或使用 `drop_oldest`。
- `ValueError: frame shape changed`：视频中途改变了分辨率，应统一尺寸或重新初始化。
- `cannot import name '_C' from sam2`：当前环境未构建 SAM2 CUDA 后处理扩展；基础推理
  仍可运行，但小孔填充后处理会跳过。
