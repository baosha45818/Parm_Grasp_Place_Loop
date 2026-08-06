cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tool/sam2-service
./.venv/bin/sam2-service \
  --socket /tmp/sam2-service.sock \
  --backend efficientvit \
  --queue-capacity 4 \
  --request-timeout-s 60

cd /home/dainanjun/Public/Parm_Grasp_Place_Loop
python3 tests/loop/reverse_regrasp_right_handoff2.py


python3 -m tests.reversed_head_observation.execute_taught_tail_handoff \
  --result /home/dainanjun/Public/Parm_Grasp_Place_Loop/output/reversed_head_pose_d435/20260723_150645/result.json \
  --execute


cd /home/dainanjun/Public/Parm_Grasp_Place_Loop

# tests/reversed_head_observation_2/run_workpiece_2.sh full-execute \
#   --handoff-center-m 0.42 0.20 0.12 \
#   --confirm-workpiece-2-reverse-grasp-geometry \
#   --confirm-workpiece-2-handoff-geometry \
#   --confirm-workpiece-2-place-calibration \
#   --restart-planner-for-workpiece-2-collision

/home/dainanjun/Public/Parm_Grasp_Place_Loop/tests/reversed_head_observation_2/run_workpiece_2.sh full-execute \
  --already-left-held \
  --handoff-center-m 0.42 0.10 0.25 \
  --confirm-workpiece-2-handoff-geometry \
  --confirm-workpiece-2-place-calibration
