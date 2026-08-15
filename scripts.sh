cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

# 第一类工件
.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 1 \
  --planner-host 127.0.0.1 \
  --arm-host 192.168.3.11

.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 1 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.3.11 \
  --left-cycle-limit 1 \
  --require-confirmation

cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

# 第二类工件（使用较高的准备位、观察位和交接位）
.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 2 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.3.11

.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 2 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.3.11 \
  --left-cycle-limit 1 \
  --require-confirmation

更新相机序列号
头部D435 261722072445
左手D405 260322270600
右手D405 260322273553
