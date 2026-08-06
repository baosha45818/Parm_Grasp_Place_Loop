# Module 4-1: Reverse Grasp Prediction

该模块是“左臂从 Buffer 反抓工件”前唯一允许控制云台的模块：

1. 读取调用方在标定观察位采集的头部 RGB-D。
2. 调用 SAM2 和 ICP 预测桌面反面工件的抓取点。
3. 仅在得到**唯一**候选时写出 prediction JSON，供 Module 5-1 消费。

它不移动云台或机械臂、不控制夹爪、不执行抓取。

```bash
/home/dainanjun/Public/data_collector/.venv/bin/python \
  /home/dainanjun/Public/Parm_Grasp_Place_Loop/modules/module_4_head_judge/module_4_1_reverse_grasp_prediction/main.py \
  --execute --capture-dir /path/to/head_rgbd_capture \
  --prediction-file /tmp/reverse_grasp_prediction.json
```

输出 JSON 中包含本次 `capture_dir`、唯一的 `grasp_poses.json` 和候选数量。预测失败或候选数量不为 1 时退出非零，Module 5-1 不会执行。
