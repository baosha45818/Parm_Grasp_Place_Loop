# Version9：URDF 名义 TCP 实验（非生产入口）

> 模块化 Loop 不调用本目录。生产入口固定转发到
> `third_party/graspnet/deploy/version7/run_grasp_v7.sh`。

Version9 复用 Version7 的常驻相机、GraspNet、候选筛选和 Planner 路径，只将抓取规划中的 `T_eef_grasp_tcp` 替换为左夹爪 URDF 的闭合两指中心：`[0.245062, 0, 0.000285] m`。

规划必须使用 GraspNet 专用 Planner：轨迹 `5605`、TF `5614`、waypoint `5615`；脚本已显式传入这三项。轨迹执行仍通过后端 `5667`，它不是 Planner 的规划/TF 服务端口。

该值与当前实测 TCP 相差约 25.13 mm，因此此版本默认 `--planner-plan-only`，不会执行任何真实运动。

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
bash /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_1_left_grasp/module_1_1_graspnet/version9/run_grasp_v9.sh
```

启动前仍需按 Version7 文档启动 Version7 的常驻相机与 GraspNet 服务；Version9 复用其 socket 和任务目录。运行记录写入本模块的 `version9/output/`。

先审阅生成的执行计划中的 `tcp_calibration` 字段和 Planner 轨迹。要运行完整的 Version7 实物流程（观察位移动、开夹爪、双相机采集、常驻 GraspNet 推理、候选预检、终端确认后的抓取、力传感器单件校验和 graspup 抬升），使用 shell 级显式开关：

```bash
V9_ALLOW_EXECUTE=1 bash /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_1_left_grasp/module_1_1_graspnet/version9/run_grasp_v9.sh
```

该开关会自动传入 `--execute-grasp`、`--confirm-real-grasp` 和 `--allow-urdf-tcp-execution`；仍会在抓取候选出现后要求操作者确认。真实执行时仍会读取实测 TCP 仅用于比较；默认偏差上限为 30 mm。通过 `--urdf-tcp-max-disagreement-mm` 可收紧该上限，设为 `0` 可关闭该比较阈值。
