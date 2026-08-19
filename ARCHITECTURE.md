# 当前架构

## 左臂有效链路

```text
D405 RGB-D
  → SAM2实例分割
  → STL/ICP位姿与抓取候选
  → Planner预抓取和LIN下探预检
  → Module1抓取及SINGLE重量验证
  → verified_handoff.json
  → Module3固定Buffer放置
  → 返回graspup
```

Module2 已删除。Module3不再接收工件姿态分类，只读取 Module1已验证的
`SINGLE` handoff；工件1和工件2分别使用各自的固定 Buffer TCP配置。

## 模块边界

### Module1

负责 D405采集、SAM2分割、STL/ICP、抓取语义与点云安全筛选、Planner成对
轨迹预检、真实抓取、持件复检和 handoff输出。

相机序列号同时绑定标定链、抓取偏置和服务地址：

- `gzhu`：`260322270242`、SDK + bridge、`192.168.88.1`。
- `xiangxin`：`260322270600`、planner-direct、`192.168.3.11`。

### Module3

负责 handoff校验、持件复检、到达 `graspup/graspup_top`、并发规划固定
Buffer Z候选、执行放置、松爪、失败恢复和回程。

### tools

项目公共实现统一放在 `tools/`：SAM2、标定、ICP、PlannerRuntime、观察位和
精简后的 GraspNet运行支持。Module1和Module3只保留各自业务规则。

## 总控状态

旧双层总控已删除。当前唯一有效入口为 `controller/main.py`；它直接编排正式模块，
不再声明 Module2阶段或其结果协议。

## 安全原则

- plan-only 不发送机械臂运动或夹爪命令。
- 真实抓取和真实放置分别要求两项显式命令行授权参数。
- Module1只有通过夹爪状态及 `SINGLE` 重量门控后才写 handoff。
- Module3在关键持件运动前复检夹爪状态。
- Planner轨迹起点不匹配时不执行旧轨迹。
- 当前两个模块均不包含终端交互确认。
