# Module 1-3：传统 SAM/STL-ICP 抓取

该模块不启动或调用 GraspNet 推理服务。流程为：D405 多帧采集、SAM 实例分割、
STL-ICP 姿态估计、Planner 可达性预检、预抓取与 10 cm LIN 下探、闭夹爪、原 LIN
抬升、同姿态力传感器质量检测，最后回到 `graspup`。质量检测只接受稳定的 `SINGLE`：
空抓、多件、未知或不稳定采样都会在预抓取高度开夹爪并拒绝进入 Module 2。

独立执行真实抓取：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_1_left_grasp/module_1_3_icp_grasp
./run_live_traditional_pick.sh --execute --confirm-real-grasp
```

大 Loop 使用 `run_grasp_to_inspect_place.sh`。它在抓取成功后调用由总控传入的
Module 2 入口并传递放置许可和 `icp_weight_handoff.json`。该文件由本模块实际测得，
记录重量、力窗口与 `SINGLE` 判定；不会伪造 GraspNet 交接文件。
