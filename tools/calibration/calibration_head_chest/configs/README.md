# 配置

每个 `calibration_*_config.json` 都可直接通过 `--config` 启动，只维护四项：

```json
{
  "camera_model": "任何便于人理解的名称",
  "camera_serial": "设备序列号",
  "position": "安装位置说明",
  "board_config": "boards/实际标定板.json"
}
```

- `camera_serial`：用于连接对应设备，并自动判断使用 RealSense 还是 Orbbec SDK。
- `board_config`：用于标定板检测和位姿求解，必须与实物一致。
- 其中`camera_model` 与 `position`：只写入日志和结果，便于人识别；可以改成任意清楚的名称，不参与相机驱动选择、样本字段或数学求解。

换相机只改 `camera_serial`（可选改 `camera_model`）；换安装位只改 `position`。默认样本目录取自配置文件名，而不是位置字段。更换相机后，请配合 `--data` 指向全新的样本目录，避免混入旧相机的采样。



配置json文件命名为：相机类型_位置_标定板类型.json

输出json文件命名为：calibration_相机类型_位置_标定板类型.json
