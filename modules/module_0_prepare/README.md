# 模块 0：系统准备

## 职责

在任何机械臂调度开始前，一次性准备整个 Loop 使用的共享资源。

## 输入

- 系统配置：机器人地址、端口、速度、超时和重试参数。
- GraspNet、DP、正反识别模型及 CKPT 路径。
- 工件 STL、规划场景和碰撞模型路径。
- 相机与机器人标定文件。
- 左右臂安全初始位姿、观察位姿、缓冲区位姿、交接位姿和最终放置位姿。

## 内部工作

- 加载模型和共享资源，避免每轮重复加载。
- 创建左右机械臂、夹爪、相机和规划器连接。
- 验证规划器 TF 中的 `body_base`、工具坐标系和相机坐标系。所有笛卡尔规划必须使用 `frame_id="body_base"`。
- 创建左右手 State、BufferState 和 SystemState。
- 创建唯一的 Buffer 锁、容量为 1 的左返回值队列、容量为 2 的右返回值队列和全局停止标志。
- 将 `buffer.capacity=1`、`buffer.full=false`、`buffer.needs_alignment=false`、`buffer.operator=NONE`。
- 检查设备健康，打开双夹爪，并使用规划器规划、执行双臂到安全初始位姿。不得调用 SDK servo 或旧 `move_to_pose`。

## Context 与 State 更新

```text
RuntimeContext
  models
  robot_clients
  gripper_clients
  camera_clients
  planner_client
  transforms
  shared_states
  synchronization_objects
  configuration
```

模块 0 创建上述共享 Context，并初始化各 State。它不把这些对象塞进 `ModuleResult`。

## 统一返回

准备成功时返回标准 `ModuleResult(status=SUCCESS, previous_module="0", ...)`，随后由主线程设置 `system.ready=true`。任一关键资源失败时返回同结构的 `FAULT`，不得启动左右调度器。
