# PARM Grasp-and-Place Loop

项目当前正在重构。左臂已经形成不经过 Module2 的独立闭环：

```text
Module1：D405 + SAM2 + STL/ICP 抓取
  → verified_handoff.json
Module3：读取 SINGLE handoff，固定放置到 Buffer
```

Module2 已删除，不再参与感知、分类、交接或放置。旧模块化总控尚未适配这条
新链路，不应继续作为实机入口；后续会单独编写新的总控。

## 当前可独立运行模块

- `modules/module_1_left_grasp/`：左臂传统视觉抓取，支持 `gzhu` 和
  `xiangxin` 两套相机、标定、偏置及服务地址绑定。
- `modules/module_3_left_place/`：读取 Module1 handoff，把左手工件放到固定
  Buffer，并返回 `graspup`。
- `tools/`：SAM2、标定、纯数值 ICP、PlannerRuntime、观察位以及精简后的
  GraspNet运行支持。

详细参数和安全边界见：

- [Module1 README](modules/module_1_left_grasp/README.md)
- [Module3 README](modules/module_3_left_place/README.md)

## Module1 + Module3 全自动运行

以下命令会真实控制机械臂和夹爪：

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

modules/module_1_left_grasp/run.sh \
  --mode gzhu \
  --execute \
  --confirm-real-grasp \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json \
&& \
modules/module_3_left_place/run.sh \
  --workpiece 1 \
  --arm-host 192.168.88.1 \
  --gripper-host 127.0.0.1 \
  --weight-handoff-file output/module_1_left_grasp/verified_handoff.json \
  --execute \
  --confirm-real-place
```

整个链路没有回车、`y/n` 或其他终端交互确认。真实动作分别通过成对参数
`--execute --confirm-real-grasp` 和 `--execute --confirm-real-place` 授权。

## 外设和本机服务

- Planner运行在本机，默认监听 `127.0.0.1:5604/5609/5610`。
- SAM2由 `tools/sam2-service` 提供。
- D405通过 Camera Broker采集。
- `gzhu` 的机械臂服务位于 `192.168.88.1`，夹爪和左力传感器服务位于
  `127.0.0.1`。
- `xiangxin` 的机械臂和夹爪服务位于 `192.168.3.11`，力传感器服务位于
  `192.168.3.22`。
