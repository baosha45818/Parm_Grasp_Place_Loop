# 新电脑部署：本次问题、根因与最终恢复步骤

适用现场：当前 `gzhu` 主线，机械臂控制器为 `192.168.88.1`。本文记录 2026-08-19
将项目迁至新电脑时实际遇到的问题，后续更换电脑可按此文恢复，不需要重新传整个项目或
重新制作虚拟环境。

> 本文中的“本机外设”指两路力传感器、双夹爪、头部云台和 free 示教踏板。它们由
> `/home/dainanjun/Public/parm_sdk_1.0.0` 启动；机械臂本体 driver 在
> `192.168.88.1` 的控制器上。

## 1. 项目代码同步

新电脑可以保留已传好的 `.venv`，仅用 Git 更新代码。该仓库在新电脑上需要禁用旧代理并
强制使用 HTTP/1.1（默认 Git HTTPS 请求曾卡住）。只需配置一次：

```bash
cd ~/Public/Parm_Grasp_Place_Loop
git config --local http.proxy ""
git config --local https.proxy ""
git config --local http.version HTTP/1.1
```

以后要以 GitHub 最新 `main` **覆盖本机已跟踪代码**时：

```bash
git fetch origin
git reset --hard origin/main
git status -sb
git log -1 --oneline
```

不要把 `git clean -fd` 当成日常更新命令。它会删掉未跟踪的模型、标定输出等资源；它不会
删 `.venv`，但仍可能删掉后续需要的其他本地文件。

## 2. 两只踏板：识别、确认与写入映射

两只 Key-1 `FD-KYMS` 踏板的 VID/PID 和序列号相同，不能用 `by-id` 区分。每一只踏板在
Linux 中还会显示为两个 `event` 节点（keyboard + mouse/consumer-control），所以两只
踏板显示为四个 `event` 节点是正常现象。

使用项目脚本确认左右关系；它会要求依次踩左、右踏板：

```bash
cd ~/Public/Parm_Grasp_Place_Loop
./.venv/bin/python tools/confirm_foot_pedals.py --timeout 40 --apply
```

本次新电脑实测得到：

```text
左踏板 / arm 1: /dev/input/by-path/pci-0000:00:14.0-usb-0:5.4:1.0-event-kbd
右踏板 / arm 2: /dev/input/by-path/pci-0000:00:14.0-usb-0:7:1.0-event-kbd
```

脚本会更新下面这个外设 driver 配置，并生成时间戳备份：

```text
~/Public/parm_sdk_1.0.0/driver_servers/run_activate_teach.sh
```

配置中应当是：

```bash
--pedal arm=1:device=/dev/input/by-path/pci-0000:00:14.0-usb-0:5.4:1.0-event-kbd \
--pedal arm=2:device=/dev/input/by-path/pci-0000:00:14.0-usb-0:7:1.0-event-kbd
```

### 已修复的脚本问题

早期版 `confirm_foot_pedals.py --apply` 有路径替换正则错误，可能把新路径与旧路径尾部拼接，
例如出现 `...event-kbdsb-0:5...`。此时不要启动 driver；手工把上述两行恢复正确，或使用
修复后的脚本重新确认。脚本现已按两只物理踏板分组，而不是误把四个 event 节点当四只踏板。

## 3. 踏板权限

踏板 driver 启动报错：

```text
ERROR: pedal #0: open ... event-kbd: Permission denied
```

说明踏板路径正确，但当前用户没有读取 `/dev/input/event*` 的权限。安装专用 udev 规则：

```bash
cd ~/Public/parm_sdk_1.0.0/driver_servers/udev
sudo cp 99-parm-footpedal.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=input
sudo udevadm settle
```

把运行用户加入 `input` 组（只需一次）：

```bash
sudo usermod -aG input dainanjun
```

真正注销并重新登录后，该权限永久生效。无需注销时，可用下列方式让本次启动进程继承
`input` 组：

```bash
sg input -c 'cd /home/dainanjun/Public/parm_sdk_1.0.0 && ./start.sh --no-web-demo'
```

验证踏板事件设备应显示 `root input`：

```bash
ls -l /dev/input/event24 /dev/input/event26
```

## 4. FT4232H 串口外设别名

夹爪、云台、左右力传感器共用一颗 FT4232H。虽然内核会产生变化的
`/dev/ttyUSB0` 至 `/dev/ttyUSB3`，项目 driver **只允许使用**下列稳定别名：

```text
/dev/parm-force-sensor-left  -> ttyUSB0
/dev/parm-force-sensor-right -> ttyUSB1
/dev/parm-gripper            -> ttyUSB2
/dev/parm-head               -> ttyUSB3
```

首次迁移时若 `ls -l /dev/parm-*` 报“没有那个文件”，安装 FT4232H 规则：

```bash
cd ~/Public/parm_sdk_1.0.0/driver_servers/udev
sudo ./install.sh
```

该安装脚本第一次使用的触发过滤可能不会匹配已经存在的 tty 设备；因此无论输出如何，再
强制触发一次所有 tty：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=tty
sudo udevadm settle
ls -l /dev/parm-*
```

若仍无 `ttyUSB*`，先确认硬件/线缆和供电：

```bash
lsusb
ls -l /dev/ttyUSB* 2>&1
```

应能看到 `0403:6011 Future Technology Devices International FT4232H`。

## 5. 本机外设服务的正确启动与停止

自动总控不需要 `web_demo`，所以不应因为 web_demo 缺 Python 依赖而运行笼统的根目录
`./install.sh`。推荐启动命令：

```bash
sg input -c 'cd /home/dainanjun/Public/parm_sdk_1.0.0 && ./start.sh --no-web-demo'
```

正常状态必须包含：

```text
✓ 5 个 server 启动
driver: 5 个
```

五个服务为：左右力传感器、双夹爪、云台、踏板 free-teach 服务。启动云台 driver 时会执行
初始化/归中，启动前确认云台周围安全。保持该终端打开；按 `Ctrl+C` 会优雅停止这次启动的
所有外设服务。

若只显示 `driver: 4 个`，优先查看踏板日志：

```bash
tail -n 60 ~/Public/parm_sdk_1.0.0/logs/activate_teach.log
```

## 6. 启动前最小检查清单

```bash
# FT4232H 的 4 个稳定串口别名
ls -l /dev/parm-*

# 两个踏板键盘事件设备（本次实际映射）
ls -l /dev/input/by-path/*5.4*event-kbd /dev/input/by-path/*7*event-kbd

# 本机外设服务数量（启动后）
pgrep -af 'parm_(force_sensor_server|gripper_server|head_app|activate_teach)'
```

通过后再启动 Planner 和 `controller/main.py`。本次解决的是新电脑的 Git、USB/udev、踏板
映射与外设服务权限问题；不涉及抓取、ICP、总控状态机或机器人动作逻辑的改动。

## 7. Planner Docker 镜像：使用离线 tar，不登录 Registry

Planner 由 `~/Public/plucky_hjt_parm/start_planner.sh xuexiao` 通过 Docker Compose 启动。
若出现以下错误：

```text
error from registry: Invalid credential
```

不要尝试登录 `papjia-docker.pkg.coding.net`。项目交付目录
`~/Public/parm_sdk_1.0.0/docker_images/` 应包含三份离线镜像；在新电脑 Docker 安装完成后
直接加载：

```bash
cd ~/Public/parm_sdk_1.0.0
docker load -i docker_images/parm_studio_backend_fixed-sub.tar
docker load -i docker_images/parm_studio_frontend_2.0.0.tar
docker load -i docker_images/parm-planner_1.0.4.tar
```

然后启动 Planner：

```bash
cd ~/Public/plucky_hjt_parm
./start_planner.sh xuexiao
```

若 `docker_images/*.tar` 缺失，只需从旧电脑补传这三份镜像文件，不需要重传整个项目或虚拟环境。

## 8. SAM2 服务缺 bypass tracking 权重

启动 `tools/sam2-service/start_uds_server.sh` 时，若 EfficientViT 分割模型已经加载成功，但
在 Efficient-SAM2 tracking 阶段报错：

```text
checkpoint not found: .../3rdparty/efficient-sam2/bypass/ckpt/bypass_bottleneck_base.pth
```

说明 `.venv` 和大模型并非问题，只缺少一个未被 Git 跟踪的 bypass 权重（约 0.8 MB）。这类
文件可能被 `git clean -fd` 删除。只需从旧电脑补传：

```text
tools/sam2-service/3rdparty/efficient-sam2/bypass/ckpt/bypass_bottleneck_base.pth
```

新电脑的目标目录为：

```text
~/Public/Parm_Grasp_Place_Loop/tools/sam2-service/3rdparty/efficient-sam2/bypass/ckpt/
```

传入后重新执行：

```bash
cd ~/Public/Parm_Grasp_Place_Loop/tools/sam2-service
./start_uds_server.sh
```
