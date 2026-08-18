# SAM2 / EfficientViT-SAM 首帧常驻接口

常驻逻辑位于 `src/sam2_first_frame_selector.py`，不再需要独立 service 对象。模块内部
维护一个专用工作线程和一个 AMG generator；模型只在线程中创建与推理，直到调用
`close()` 才释放模块持有的引用。

## 推荐用法：显式预热

在机械臂开始运动前调用 `start()`，可以提前发现 checkpoint、依赖或 CUDA 问题：

```python
from eval_scripts_package.sam2_first_frame_selector import (
    close,
    select_largest_sam2_mask,
    start,
    status,
)

start(
    ckpt="/path/to/sam2.1_hiera_base_plus.pt",
    cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
)

try:
    result = select_largest_sam2_mask(
        image_bgr,
        ckpt="/path/to/sam2.1_hiera_base_plus.pt",
        cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
        candidate_roi_norm=(0.2, 0.2, 0.8, 0.8),
        return_formats=("mask", "bboxmask"),
    )
finally:
    close()
```

重复调用 `start()` 且参数完全相同时是幂等的，不会重新加载模型。运行期间若要更换
checkpoint、config、device 或 AMG 参数，必须先 `close()` 再重新 `start()`。

## 懒启动

不调用 `start()` 也可以直接使用原 API：

```python
result = select_largest_sam2_mask(image_bgr, ckpt=ckpt, cfg=cfg)
```

第一次调用会自动启动线程并加载模型，后续调用复用常驻模型。程序退出前仍建议显式
调用 `close()`。如果没有传 `ckpt/cfg`，会继续使用原来 `my_robot` 中的默认值。

## 自定义预热参数

```python
start(
    ckpt=ckpt,
    cfg=cfg,
    device="cuda",
    amg_options={"points_per_side": 24, "points_per_batch": 64},
)
```

`amg_options` 会覆盖默认 AMG 构造参数。推理 API 不需要再次传递 device 或 AMG
参数，只要 checkpoint/config 与当前常驻模型一致即可。

## 生命周期与并发

- `start()`：创建线程、同步完成模型加载，然后返回状态字典。
- `select_largest_sam2_mask()`：未启动时自动启动；已启动时通过队列执行推理。
- `status()`：返回是否启动、checkpoint、config、device 和排队请求数。
- `close()`：等待当前推理结束、停止线程；未启动时调用也是安全的。
- 多个业务线程可以调用选择 API；SAM2 推理在内部线程上严格串行。
- 输入图像进入队列前会复制，因此相机可以立即复用原始帧缓冲区。

该模块只负责 AMG 首帧选择。已有视频跟踪逻辑仍由 `sam2_mask_runtime.py` 负责，
避免把两套不同 predictor 的状态和生命周期塞进首帧选择 API。

## EfficientViT-SAM 后端

当前使用随服务迁移的 `3rdparty/efficientvit`。默认使用512分辨率 L0 和对应权重，
以降低首帧延迟；XL0 权重仍保留用于质量对比：

```python
start(backend="efficientvit")
try:
    result = select_largest_sam2_mask(image_bgr)
finally:
    close()
```

也支持不预热直接懒启动：

```python
result = select_largest_sam2_mask(image_bgr, backend="efficientvit")
```

EfficientViT L0 默认使用 `points_per_side=16`、`points_per_batch=128`、
`crop_n_layers=0`、`pred_iou_thresh=0.75`、`stability_score_thresh=0.90` 和
`min_mask_region_area=100`，用于提高小目标候选召回。可在 `start()` 的
`amg_options` 中覆盖。返回字典及 ROI、面积、连通域、bbox span 过滤逻辑与 SAM2
后端一致，`result["backend"]` 会标记实际后端。EfficientViT 只替换首帧 AMG，后续
视频跟踪仍使用原来的 SAM2 runtime。
