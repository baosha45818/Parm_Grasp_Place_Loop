cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

# 第一类工件
.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 1 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1

.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 1 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --left-cycle-limit 1 \
  --require-confirmation

cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

# 第二类工件（使用较高的准备位、观察位和交接位）
.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 2 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1

.venv/bin/python modules/modular_dual_arm_loop.py \
  --workpiece 2 \
  --planner-host 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --arm-host 192.168.88.1 \
  --left-cycle-limit 1 \
  --require-confirmation
