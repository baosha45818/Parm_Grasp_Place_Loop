# Module 5-1: Reverse Grasp

`main.py` 是保留的旧一体化入口，供旧流程单独运行，行为不变。

模块化总控使用 `execute_prediction.py`：它只接受 Module 4-1 的 prediction JSON，
执行左臂反抓预备位和真实抓取；不控制云台、不采集图像、不调用 SAM2/ICP。

## 耗时记录

每次通过 `execute_prediction.py` 执行工件 1 后，会在该次 RGB-D `capture_dir`
写入 `reverse_grasp_timing.json`，记录总耗时以及“到反抓预备位”和“抓取规划、
执行、持件确认”两个阶段。即使抓取失败或中断，也会写入已完成阶段与状态。

同一目录还会生成 `reverse_grasp_perception_timing.json`，其中拆分
`sam2_mask_generation`、`icp_grasp_pose`、`grasp_visualization` 和结果写入耗时。
抓取执行的细分记录位于 `sam2_masks/reverse_grasp_execution_timing.json`，其中包括
并发 Planner 预检、预抓动作、下探规划与动作、闭夹爪、上提及持件确认。每个接近角
候选的规划秒数同时保存在 `sam2_masks/planner_pose_debug.json` 的
`approach_preflight[*].planning_duration_seconds`。

工件 2 的凸耳反抓将同类耗时写入其本次
`workpiece_2_pipeline/grasp/runs/<run>/grasp_plan.json` 的 `timing` 字段；细分为
预备位、并发候选预检、预抓、LIN 下探、闭夹爪和上提/力确认。

`implementation/` 和 `main.py` 的一体化感知入口保留，供旧总控和独立调试使用；
模块化调用链不会经过它们。

```bash
/home/dainanjun/Public/data_collector/.venv/bin/python execute_prediction.py \
  --execute --prediction-file /tmp/reverse_grasp_prediction.json
```
