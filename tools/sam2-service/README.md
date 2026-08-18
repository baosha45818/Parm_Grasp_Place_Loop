# sam2-service

面向机器人 RGB-D 场景的常驻首帧分割与视频目标跟踪服务。仓库包含运行代码、
EfficientViT-SAM、Efficient-SAM2、SAM2 依赖源码，以及可直接使用的模型权重。

首帧接口接收原始 BGR、与其对齐的深度图、相机内参和深度尺度，在内部完成：

1. 深度反投影和背景平面去除；
2. 非平面点重新投影到 RGB；
3. 仅在非平面区域生成稀疏 AMG 提示点；
4. EfficientViT-SAM-L0 单次图像编码和批量候选解码；
5. 深度支撑过滤，并选择有效 bbox 面积最大的候选；
6. 返回普通 mask、轴对齐 bbox mask 和旋转 bbox mask。

首帧 mask 可以直接交给 `EfficientSam2Tracker`，完成后续视频帧跟踪。

## 仓库结构

```text
sam2_service/
├── __init__.py                         # 对外 Python API
├── src/
│   ├── sam2_first_frame_selector.py    # RGB-D 首帧选择器
│   └── efficient_sam2_tracking.py      # 常驻视频跟踪器
├── checkpoints/                        # Git LFS 模型权重
│   ├── efficientvit_sam_l0.pt          # 默认首帧模型
│   ├── efficientvit_sam_xl0.pt         # 高质量备用模型
│   └── sam2.1_hiera_base_plus.pt       # SAM2 / tracking 权重
├── 3rdparty/                            # 随仓库交付的第三方源码
├── test/                                # 离线、相机和机械臂测试
├── checksums.sha256                     # 权重校验值
└── requirements.txt
```

## 获取仓库和权重

权重使用 Git LFS 保存。克隆方必须先安装 Git LFS，否则只会得到很小的指针文件。

```bash
git lfs install
git clone git@github.com:srbLinux/sam2-service.git
cd sam2-service
git lfs pull
sha256sum -c checksums.sha256
```

## 环境

当前验证环境：Python 3.10、CUDA、PyTorch、OpenCV、Open3D 和 RealSense。

```bash
pip install -r requirements.txt
```

EfficientViT 和 Efficient-SAM2 的源码已经放在 `3rdparty/`。默认路径和 checkpoint
均由包内部解析，正常调用不需要传仓库路径或权重路径。

### 独立 uv 环境与本地服务

服务进程建议拥有独立环境，避免业务模块中的 ROS、Open3D 或其他 PyTorch 版本与 CUDA
模型环境冲突。首次使用在 `sam2-service` 目录执行：

```bash
uv sync --python 3.10
```

`torch` 必须安装与本机 CUDA 驱动匹配的构建；如已有验证过的机器人 Conda 环境，也可以
使用该解释器创建/运行 uv 环境。权重仍需先完成 Git LFS 拉取。

启动本机 HTTP-over-UDS 服务。一个服务进程会常驻加载 EfficientViT 分割模型和
Efficient-SAM2 跟踪模型：

```bash
./start_uds_server.sh
```

默认使用 EfficientViT、`/tmp/sam2-service.sock`、队列容量 4、请求超时 60 秒。通过环境
变量覆盖，例如：

```bash
SAM2_SERVICE_SOCKET=/run/user/$UID/sam2-service.sock \
SAM2_SERVICE_QUEUE_CAPACITY=8 \
./start_uds_server.sh
```

服务仅监听本机 Unix Socket，不提供 TCP/HTTP 网络监听。

它在 `/tmp/sam2-service.sock` 上提供：

- `GET /healthz`：分割队列与模型状态；
- `POST /v1/segment`：全图分割；
- `POST /v1/first-frame/mask`：RGB-D 首帧选择（`select_first_frame_mask`）；
- `POST /v1/first-frame/candidates`：RGB-D 首帧候选（`select_first_frame_candidates`）；
- `POST /v1/tracking/initialize`：用首帧 BGR + mask 初始化跟踪；
- `POST /v1/tracking/append`：追加原始 BGR 帧；
- `POST /v1/tracking/propagate`：传播已追加的帧；
- `POST /v1/tracking/close`：关闭当前跟踪流并释放跟踪模型。

分割和追加帧请求体均为原始 `uint8 HxWx3` BGR bytes（`application/octet-stream`），不是
Base64/JSON；带 mask 的跟踪初始化和所有 mask 响应均使用压缩 NumPy `.npz` 二进制容器。
服务端分割队列满时返回 HTTP 429，超时返回 HTTP 504。一个服务当前只允许一个活动跟踪流，
避免多个 module 相互覆盖 SAM2 视频状态。生产系统建议将 socket 放在受限权限目录并设置适合
机器人用户组的 `--socket-mode`。

业务模块无需导入模型或 GPU 依赖，只需安装 `aiohttp` 和 `numpy`。仓库内的模块统一从
`tool.sam2_uds_client` 导入；它直接复用本目录的客户端实现，不需要复制文件：

```python
import asyncio
from tool.sam2_uds_client import segment_full_image_top_n

result = asyncio.run(segment_full_image_top_n(
    image_bgr,
    socket_path="/tmp/sam2-service.sock",
    num_masks=5,
    roi=(100, 80, 540, 400),
))
print(result.masks.shape, result.bboxes, result.metadata)
```

首帧分割后初始化并传播跟踪的客户端流程：

```python
from tool.sam2_uds_client import (
    append_tracking_frame,
    initialize_tracking_from_mask,
    propagate_tracking,
)

initial = await initialize_tracking_from_mask(first_bgr, first_mask)
stream_id = initial.metadata["stream_id"]
await append_tracking_frame(next_bgr, stream_id=stream_id)
tracked = await propagate_tracking(stream_id=stream_id)
latest_mask = tracked.masks[-1]
```

RGB-D 首帧 API 同样通过客户端对外暴露。它使用 NPZ 二进制体传输 BGR、对齐深度图、
3x3 内参与深度尺度；除图像外的参数通过 `**options` 传递，且必须 JSON 可序列化：

```python
from tool.sam2_uds_client import (
    select_first_frame_candidates,
    select_first_frame_mask,
)

first = await select_first_frame_mask(
    image_bgr, aligned_depth, intrinsics, depth_scale,
    target_center=(230, 228),
    target_window=None,
)
mask = first.mask

candidates = await select_first_frame_candidates(
    image_bgr, aligned_depth, intrinsics, depth_scale,
    num_candidates=5,
)
# 固定 5 个槽位；candidate_present=False 的槽位无有效候选。
print(candidates.candidate_present, candidates.candidate_bboxes)
```

若业务模块不安装完整 `sam2-service` 包，可以只复制 `uds_http_client.py` 到其公共客户端包；
该文件不导入 SAM2、Torch 或模型权重。

## 首帧分割：调用方最小示例

确保包目录的父目录位于 `PYTHONPATH`：

```python
import sys
sys.path.insert(0, "/path/to/sam2-service-parent")

from sam2_service import (
    close,
    segment_full_image_top_n,
    select_first_frame_candidates,
    select_first_frame_mask,
    start,
)
```

模型可以显式常驻：

```python
start(backend="efficientvit")

try:
    result = select_first_frame_mask(
        image_bgr,       # uint8, HxWx3, OpenCV BGR
        aligned_depth,   # HxW，必须与 BGR 像素对齐
        intrinsics,      # float 3x3 相机内参矩阵
        depth_scale,     # 原始深度单位转换为米的比例
        return_formats=("mask", "bboxmask", "rotated_bboxmask"),
    )
finally:
    close()
```

不提前调用 `start()` 也可以；首次调用 `select_first_frame_mask()` 会自动启动并常驻
EfficientViT-SAM。持续处理多帧/多任务时建议显式 `start()`，程序退出时调用
`close()`。

### 输入约束

- `image_bgr`：`numpy.ndarray`，`uint8`，形状 `H x W x 3`；
- `aligned_depth`：形状 `H x W`，必须已经对齐到彩色图；
- `intrinsics`：形状 `3 x 3`，包含 `fx/fy/cx/cy`；
- `depth_scale`：将深度原始值转换为米，例如 RealSense 常见值约 `0.001`；
- 相机、箱体或工作平面发生明显移动后，需要重新确认 `target_center` 和 ROI。

### 常用参数

```python
result = select_first_frame_mask(
    image_bgr,
    aligned_depth,
    intrinsics,
    depth_scale,
    roi=(0.20, 0.10, 0.88, 0.72),
    min_depth=0.15,
    max_depth=1.0,
    plane_distance=0.006,
    min_object_height_above_plane_m=0.006,
    target_center=(230, 228),
    # 默认 None：保留 ROI 内的全部非平面区域；只想搜索中心局部时再设 (150, 150)
    target_window=None,
    max_amg_points=256,
    amg_grid_points_per_side=12,
    amg_point_separation=16.0,
    # 固定观测视角下主连通主体的 SAM mask 面积；只保留 0.7～1.2 倍
    # 参数名保留兼容性，但数值现在表示 mask 像素面积
    normal_object_bbox_area_px=10000,
    bbox_area_min_factor=0.7,
    bbox_area_max_factor=1.2,
    # 最大面积组内相差不超过 10% 时，优先选择离桌面更高的工件
    similar_bbox_area_ratio=0.10,
    return_formats=("mask", "bboxmask", "rotated_bboxmask"),
)
```

### 返回值

```python
mask = result["formats"]["mask"]
bbox_mask = result["formats"]["bboxmask"]
rotated_bbox_mask = result["formats"]["rotated_bboxmask"]
bbox_xyxy = result["bbox"]

print(result["selection_rule"])
print(result["candidate_scores"])        # 所有有效候选框及评分
print(result["preprocess_timing_ms"])    # 点云和非平面提示生成
print(result["timing_ms"])               # EfficientViT AMG 耗时
```

mask 均为 `uint8`，背景值为 `0`，前景值为 `255`。设置 `return_debug=True` 时还会
返回 `debug.allowed_mask`、内部 AMG 点和点云诊断；生产调用一般不需要开启。

需要取得前 N 个通过面积过滤的候选时，可使用：

```python
batch = select_first_frame_candidates(
    image_bgr, aligned_depth, intrinsics, depth_scale,
    num_candidates=5,
)
print(batch["actual_count"])
for candidate in batch["candidates"]:
    if candidate is None:
        continue
    mask = candidate["mask"]
    bbox = candidate["bbox"]
    source_index = candidate["source_index"]
```

该函数只进行一次批量推理，`candidates` 始终包含请求的 N 个槽位。有效候选不足
时后面的槽位为 `None`；`actual_count` 是非空槽位数，`available_count` 是通过过滤
的候选总数。排序与 `select_first_frame_mask` 一致：面积明显不同时优先最大面积，
面积相近时优先离桌面更高的候选。

## RGB 全图分割：按面积返回前 N 个 mask

不需要深度图、内参或点云去平面时，使用 `segment_full_image_top_n()`。该接口只对
BGR 图像运行一次 AMG，并按 mask 面积从大到小返回前 N 个结果：

```python
from sam2_service import close, segment_full_image_top_n, start

start(backend="efficientvit")
try:
    result = segment_full_image_top_n(
        image_bgr,
        num_masks=5,
        # 可选，像素坐标 (x0, y0, x1, y1)，右/下边界不包含。
        # 不传或传 None 时，默认使用整张图。
        roi=(100, 80, 540, 400),
        min_mask_area_px=100,
        return_formats=("mask",),
    )
finally:
    close()

for item in result["masks"]:
    mask = item["mask"]             # 原图大小的 uint8 mask，ROI 外恒为 0
    bbox = item["bbox"]              # ROI 裁剪后 mask 的 xyxy bbox
    area = item["area_px"]           # ROI 内的前景像素面积
    raw_area = item["raw_area_px"]   # AMG 原始全图 mask 面积
```

`roi` 不会触发点云处理：它仅过滤 AMG 的像素结果。指定 ROI 后，每个候选 mask 都会被
裁到该区域，按裁剪后的 `area_px` 排序；完全不与 ROI 相交或面积小于
`min_mask_area_px` 的候选不会返回。返回数量可能少于 `num_masks`，可通过
`actual_count`、`available_count` 和 `roi` 字段判断实际结果。该接口不会自动排除桌面、
背景或大物体；这些结果同样会按面积参与排序，需要时由调用方裁图或后处理。

## 多调用者与并发

首帧分割模型由一个常驻工作线程独占，多个业务线程可以安全地同时调用任意首帧 API，
但 AMG 推理会严格串行执行。这样能避免同一个 predictor/model 被并发访问，也避免多份
模型占用 GPU 显存；代价是总吞吐量约等于单次推理吞吐量，后续调用会等待前一调用完成。

该模式适合机器人控制等低并发场景。若接入 HTTP 服务或存在大量调用者，建议在服务层
设置有界请求队列、超时和丢帧/最新帧优先策略；不要把每个请求都无限等待。需要提高
吞吐量时，应部署多个独立进程（每个进程一个模型实例），按 GPU 显存与设备数量分流，
而不要在同一模型实例上并发调用。

## 首帧接入跟踪

```python
from sam2_service import EfficientSam2Tracker

first = select_first_frame_mask(
    first_bgr,
    first_depth,
    intrinsics,
    depth_scale,
    return_formats=("mask",),
)

with EfficientSam2Tracker(queue_capacity=16) as tracker:
    tracker.initialize_from_mask(first_bgr, first["mask"])
    tracker.append(frame_1)
    tracker.append(frame_2)
    batch = tracker.propagate()
    latest_mask = batch.latest.mask
    latest_bbox = batch.latest.bbox
```

跟踪器的完整接口、队列行为、box/point/mask 初始化方式和测试命令见
[EFFICIENT_SAM2_TRACKING.md](EFFICIENT_SAM2_TRACKING.md)。

## RGB-D 机械臂测试

```bash
/mnt/data2/shurenbin/.conda/envs/dp/bin/python \
  test/test_efficient_rgbd_first_frame_selector_3_73.py \
  --move-robot \
  --output ./efficientvit_rgbd_live \
  --max-amg-points 12 \
  --amg-point-separation 16 \
  --min-allowed-ratio 0.30 \
  --target-center 230 228 \
  --target-window 150 150
```

典型输出：

- `00_wrist_rgb.png`：原始彩色图；
- `01_depth.png`：深度可视化；
- `02_pointcloud_prompts.png`：非平面区域和 AMG 点；
- `03_all_candidate_boxes.png`：全部有效候选框，绿色为最大框；
- `04_efficientvit_restricted_amg_mask.png`：最终 mask；
- `summary.json`：候选、bbox、选择规则和耗时；
- `rgbd_prompts_masks.npz`：用于离线复现的 RGB-D 数据。

## 权重

| 文件 | 用途 | 大小（约） |
|---|---|---:|
| `efficientvit_sam_l0.pt` | 默认 RGB-D 首帧分割 | 139 MB |
| `efficientvit_sam_xl0.pt` | 高质量对比/回退 | 468 MB |
| `sam2.1_hiera_base_plus.pt` | SAM2 与 tracking | 324 MB |

三份权重均通过 Git LFS 上传，并使用 `checksums.sha256` 校验完整性。
