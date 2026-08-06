# 模块 1-1：左臂 GraspNet 抓取

本目录是模块化 Loop 的 **生产入口层**。真实 GraspNet 实现、常驻服务和
模型资源统一位于 `third_party/graspnet`；本目录不保存另一份生产实现。

## 生产调用链

```text
modules/modular_dual_arm_loop.py
  -> tests/loop/simple_dual_arm_loop.py（并发调度器）
  -> modules/module_1_left_grasp/module_1_1_graspnet/
     run_grasp_to_inspect_place.sh
  -> third_party/graspnet/deploy/version7/run_grasp_to_module2.sh
  -> third_party/graspnet/deploy/version7/run_grasp_v7.sh
  -> third_party/graspnet/deploy/version7/run_persistent_graspnet.py
  -> modules/module_2_left_inspect_apart/run_inspect_then_place.py
```

`run_grasp_to_inspect_place.sh` 是唯一的模块化生产入口。它只转发到经过实机
验证的 Version7，不会调用本目录下的 Version9。

## 当前生产配置

生产参数以
`third_party/graspnet/deploy/version7/run_grasp_v7.sh` 为唯一准源：

- D405 连续 5 帧深度中值融合、后处理和孔洞填充；
- 当前关闭 Femto，GraspNet 只使用 D405；
- Planner 使用 `5604/5609/5610`；
- 规划目标为 `left_eef` 的直接抓取中心；
- 左夹爪为 `hand=1`，物理最大开口 `0.085 m`；
- 成功条件是重量检查通过并到达 `graspup`；
- 只有生成有效 handoff 文件后才允许进入 Module 2。

### 工件2的内置点云加厚

当总控传入 `PARM_LOOP_WORKPIECE=2` 时，Module 1-1 自动使用
vendored `third_party/graspnet/deploy/version7` 内置的推理点云加厚。它不依赖
`/home/dainanjun/qiuguolin/graspnet/deploy/version11/`，也不改变相机、手眼、
Planner、夹爪或候选筛选逻辑：V7 在 D405 预处理完成后保留原始
`point_cloud.npy`，另写一份仅供 GraspNet 推理的加厚点云。

该过程先用 RANSAC 找桌面，再对离桌面 4–120 mm 的点估计局部法向，并且**仅**
为“局部平面与桌面夹角不小于 65°”的点生成法向两侧虚拟层；偏移为
±1.5 mm、±3.0 mm，总厚度 6 mm。后续工件参考、边缘筛选和执行仍使用未修改的
物理 D405 点云。工件1固定保持原 Version7 点云，不加厚。

运行时终端会打印解析后的生产入口、真实抓取脚本和 Module 2 入口。若这些
路径不是 `third_party/graspnet/deploy/version7`，应立即停止并检查环境变量。

## 环境覆盖

正常 Loop 不需要设置以下变量：

```text
GRASPNET_GRASP_SCRIPT
GRASPNET_FLOW_RUNNER
```

它们是调试覆盖口，会改变实际执行脚本。生产运行前若怀疑环境污染，可执行：

```bash
env | grep -E '^(GRASPNET_GRASP_SCRIPT|GRASPNET_FLOW_RUNNER)='
```

没有输出表示未启用调试覆盖。

## Version9 实验目录

[`version9/`](version9/README.md) 是保留的 URDF 名义 TCP 实验，不属于生产
Loop 调用链。它使用另一组 Planner 端口和旧实验参数，默认仅规划。不要通过
Version9 修改生产 Version7 参数。

## 模块边界

Module 1-1 负责相机采集、GraspNet 推理、候选筛选、Planner 可达性检查、
真实抓取、重量验证以及到达 `graspup`。工件正反面/侧面检测和最终左臂放置
分别由 Module 2 和 Module 3 负责。
