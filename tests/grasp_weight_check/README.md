# 力传感器抓取重量判定方案

## 判定流程

在机械臂静止且不接触环境时，使用力传感器区分抓空、单件和多件：

1. 每次抓取前，空夹爪先经过固定的“抬升检查位姿”，采集 31 个不同时间戳的
   `filtered_force`，得到逐轴中位数作为本次抓取的空载基线。
2. 下移到抓取位并闭合夹爪，然后垂直上移，使工件完全离开桌面；上移终点回到
   **相同的抬升检查姿态**。
3. 规划器确认上移动作完成后静置约 0.3 秒，调用 `verify_grasp_after_lift()`
   采集 31 帧载荷数据并立即分类。
4. 计算三轴中位力的差向量，重量为
   `mass_g = norm(F_loaded - F_baseline) / 9.80665 * 1000`。
5. 先检查两个窗口的逐轴 MAD；振动过大时返回 `UNKNOWN`，不执行放置。
6. 根据现场标定阈值返回 `EMPTY`、`SINGLE`、`MULTIPLE` 或 `UNKNOWN`。

使用三轴差向量的模长，而不是固定使用 `Fz`，可避免传感器安装轴向不同造成的
误判。抬升高度必须足以让工件和夹爪都脱离桌面及周边物体。空载和载荷必须保持
相同腕部姿态，且推荐使用相同检查位姿；否则工具重力分量、线缆牵引力或环境接触力
会进入差值，使其不再只代表工件重量。检测不能在上移运动过程中执行。

Loop 建议调用顺序：

```text
空夹爪到抬升检查位 -> collect_force_window() 保存 baseline
    -> 下移抓取 -> 闭合夹爪 -> 垂直上移回检查位
    -> 等轨迹执行完成 -> verify_grasp_after_lift(sensor, baseline, thresholds)
    -> 根据 EMPTY / SINGLE / MULTIPLE / UNKNOWN 分支
```

## 异常处理建议

| 判定 | 建议主流程动作 |
| --- | --- |
| `SINGLE` | 更新 `holding=SINGLE`、`grasp_stable=true`，进入观察/放置 |
| `EMPTY` | 更新 `holding=EMPTY`，退回安全位，有限次数重新抓取 |
| `MULTIPLE` | 更新 `holding=MULTIPLE`，回料区安全释放，抖散后重新抓取 |
| `UNKNOWN` | 保持当前位置，不进入放置；静置后重采一次，仍未知则进入人工/故障恢复 |
| 采样超时 | 视为传感器故障，不允许仅凭夹爪闭合状态继续流程 |

建议空抓和多抓分别限制重试次数，并把估算重量、三轴差值、MAD、阈值和尝试编号
写入日志。多抓时不得直接在 Buffer 或成品位释放。

## 阈值标定

不要直接使用示例数值。先在最终验证位姿分别采集至少 30 组空载、单件和已知双件
数据，再留出不分类的迟滞区：

- `empty_max_g`：略高于空载差值分布的高分位；
- `single_min_g` / `single_max_g`：覆盖单件重量及测量误差；
- `multiple_min_g`：低于双件数据低分位，但必须高于 `single_max_g`；
- `max_axis_mad_n`：略高于静止窗口 MAD 的高分位。

若工件本身重量差异导致单件与双件重量区间重叠，仅靠力传感器无法可靠区分，
必须联合视觉或夹爪开度判定。

## 运行

纯软件单元测试（不需要 SDK 和硬件）：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
python3 -m unittest discover -s tests/grasp_weight_check -v
```

真实左手传感器交互测试示例（下列阈值仅演示命令格式）：

```bash
python3 -m tests.grasp_weight_check.manual_force_sensor_check \
  --host 127.0.0.1 --hand 1 \
  --empty-max-g 30 --single-min-g 80 \
  --single-max-g 140 --multiple-min-g 180
```

退出码为：`0=SINGLE`、`2=EMPTY`、`3=MULTIPLE`、`4=UNKNOWN`、`5=采样/SDK故障`。

后续合入模块 1-1 或 6-1 时，抓取前使用 `collect_force_window()` 保存空载基线，
抓取上移完成后调用 `verify_grasp_after_lift()`。真实运行模块负责检查位姿、夹爪动作、
重试计数和把 `GraspClass` 映射为 `Holding` / `ModuleResult`，本测试代码不控制机械臂。
