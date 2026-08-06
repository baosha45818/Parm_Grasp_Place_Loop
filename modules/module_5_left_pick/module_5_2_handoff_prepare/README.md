# Module 5-2：Left Handoff Prepare

## 职责

Module 5-1 完成左手反抓后，本模块依次完成：

```text
左手持件到头部 D435 观察位
→ Module 4-3 控制云台到 pitch=2°、yaw=26.5°
→ Module 4-3 运行 auto_head_d435_mask、D435 + SAM2 + ICP
→ 读取 Module 4-3 生成的本轮 result.json
→ 复用左 TCP 到工件的刚性关系
→ 左手将工件送到双臂交接位
```

结束时左夹爪仍保持夹紧，右臂尚未运动。

## 独立执行

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

python3 modules/module_5_left_pick/module_5_2_handoff_prepare/main.py \
  --handoff-result-file /tmp/module_5_2_handoff.json
```

默认使用四倍速度并自动确认机械臂动作。两类工件均使用统一的较高准备位、
观察位和交接中心。两类工件的 SAM2 默认均使用手动 ROI 和正负点标注，避免
固定观察位自动提示因工件交接姿态或现场背景变化而分割到错误区域。

工件1此前的固定观察位自动提示配置仍保留在：

```text
tests/reversed_head_observation/auto_head_d435_mask.py
tests/reversed_head_observation/head_d435_auto_mask_config.json
```

如需显式恢复工件1自动提示，可加 `--mask-mode auto`。自动流程不打开标注窗口，
会保存 `head_d435_mask.png`、`head_d435_mask_overlay.png` 及提示记录。默认手动
模式会打开标注窗口；拖 ROI 后添加正负点，再在终端确认 SAM2 mask。

```bash
python3 modules/module_5_left_pick/module_5_2_handoff_prepare/main.py \
  --workpiece 2
```

两类工件未传 `--mask-mode` 时默认 `interactive`，需要手动完成 ROI 和 SAM2
正负点。工件2尚未标定自动提示，显式传 `--mask-mode auto` 仍会被底层安全检查拒绝。

需要恢复运动确认时加 `--require-confirmation`。

成功后终端会输出：

```text
MODULE_5_2_RESULT=/.../output/reversed_head_pose_d435/<timestamp>/result.json
```

`--handoff-result-file` 的 JSON 包含：

```json
{
  "event": "module_5_2_left_handoff_ready",
  "result": "/absolute/path/to/result.json",
  "output_frame": "body_base",
  "left_gripper_holds_object": true
}
```

将其中的 `result` 传给 Module 8-2。观察完成后到 Module 8-2 释放左夹爪
之前，工件不能相对左夹爪滑动。

## 工件2

总控以 `--workpiece 2` 启动时，本模块选择本目录 `workpiece_2_config/` 中独立
保存的 D435、SAM2 和 ICP 配置；运动位置与工件1共用统一高位配置：

```text
左手到工件2较高准备位 [14.281, -29.886, -85.633, 102.504, 37.378, 56.016, 9.241]
→ 左手到工件2头部观察位
→ 云台 pitch=2°、yaw=26.5°
→ 工件2 STL 的 D435 + SAM2 + ICP
→ 左手将工件2送到 body_base [0.42, 0.10, 0.25]
```

准备位、观察位、云台角、外参和交接中心统一由
`modules/workpiece_2_handoff_profile.json` 提供。Module 5-2 的握手 JSON 会写入
`workpiece`、`profile_id` 和 `handoff_center_body_m`，供总控和 Module 8-2
拒绝跨工件或跨版本结果。

此分支只控制左臂和云台；不会闭合右夹爪、移动右臂或执行最终放置。模块运行时
不引用 `tests/reversed_head_observation_2/`，因此独立测试代码保持隔离。

## 工件1统一高位

工件1保留自己的 STL、自动 SAM2 提示和 ICP 参数，但复用上述较高准备位、观察位、
云台角、对应外参以及 `body_base [0.42, 0.10, 0.25]` 交接中心。这样两种工件
进入交接模块时使用同一规划区域。
