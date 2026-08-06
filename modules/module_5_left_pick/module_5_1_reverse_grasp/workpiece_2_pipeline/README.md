# Module 5-1 Workpiece 2 Pipeline

这是 Module 5-1 的新工件 2 隔离实现，不会覆盖同级旧 `main.py`、
`execute_prediction.py` 或 `implementation/`。它只消费 Module 4-1 新工件流程输出的
`workpiece_2_convex_tab_grasp_prediction`，并使用 Planner 执行 PTP 预抓取、LIN 下探、夹爪闭合和抬升。

自动反抓流程不叠加工件坐标系手动偏置；抓取目标只由 CAD 名义抓取点和沿桌面法向的接触下移构成。

完整自动入口：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
bash modules/module_5_left_pick/module_5_1_reverse_grasp/workpiece_2_pipeline/run_full_reverse_grasp.sh --execute
```

该入口尚未接入总控；总控后续可以在旧工件链路和本目录的新工件链路之间显式选择。
