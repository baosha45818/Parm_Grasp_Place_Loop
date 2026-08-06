# Tool 目录

本目录只保存多个模块共用、但不属于 Loop 调度本身的工具与资源：

- `planner_runtime.py`：统一使用规划器查询 TF、生成笛卡尔轨迹，并通过 SDK 直接执行关节轨迹。
- `welding_up_side_1118_only_2_small_screws_removed.stl`：当前工件 STL。
- `planner_joint_motion.py`：五个固定左臂关节位的共用 PTP 规划和执行实现。
- `go_graspnet.py`、`go_buffer.py`、`go_graspup.py`、`go_pc.py`、`go_pc2.py`：原 graspnet 工具目录下五组关节角的规划器版本。
- `toggle_left_gripper.py`：读取左夹爪开度并在完全打开、完全闭合之间切换一次。

运行前需先启动 plucky 规划器：

```bash
cd /home/dainanjun/Public/plucky_hjt_parm
docker compose up
```

另开一个终端，在已有 `parm_sdk` 的 data_collector 环境中运行。脚本先做关节空间 `ptp` 规划、打印规划轨迹，只有输入 `y` 才直接调用 SDK 的 `Arm.follow_trajectory()`：

```bash
cd /home/dainanjun/Public/data_collector

uv run /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/go_graspnet.py --speed 0.1
uv run /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/go_buffer.py --speed 0.1
uv run /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/go_graspup.py --speed 0.1
uv run /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/go_pc.py --speed 0.1
uv run /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/go_pc2.py --speed 0.1
```

先验证规划但不执行时，在任一命令末尾添加 `--plan-only`。

夹爪切换不依赖规划器，但要求本机 `parm_gripper_server` 已启动且左夹爪已经完成初始化：

```bash
cd /home/dainanjun/Public/data_collector
uv run /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/toggle_left_gripper.py
```

当前开度大于等于 `50%` 时脚本闭合到 `0%`；低于 `50%` 时打开到 `100%`。不自动执行 `init()`。

业务模块实现应放在对应的 `modules/module_*` 目录，不应继续添加到本目录。




能不能得到工件坐标系相对新的base坐标系的笛卡尔坐标（/home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/welding_up_side_1118_only_2_small_screws_removed.stl这是工件的stl，应该可以根据他构建工件坐标系），得到之后先让机械臂回到graspup向上20cm的地方，然后开始规划，规划到工件正面朝上，并且xyz是[-0.519526,  0.231085, -0.16202 ]分别相对新base坐标系
