# 模块 9：Right Place

右手持有稳定单工件后执行孔位放置，并在成功后松开夹爪。

本目录保留现有 `place_hole_a_wiggle.py` 等右臂放置代码。模块编号由原来的
7 更新为 9，运动逻辑不因重编号而改变。

统一返回中的 `previous_module` 为 `"9"`，完成后进入 Module 10。

## 工件2总控接入

总控使用 `--workpiece 2` 时，Module 9 会选择
`run_workpiece_2_from_loop.py`。该入口仅兼容旧调度器的公共参数，随后严格执行：

```bash
python3 modules/module_9_right_place/run_place_hole_a_wiggle_config.py \
  --config modules/module_9_right_place/place_hole_a_wiggle_workpiece_2.json \
  --live --confirm-live
```

工件1不传 `--workpiece`，仍直接调用原来的 `place_hole_a_wiggle.py`，不会经过
该入口或工件2 JSON 配置。
