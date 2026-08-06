uv run python3 -m tests.grasp_weight_check.manual_force_sensor_check \
  --host 127.0.0.1 \
  --hand 2 \
  --empty-max-g 50 \
  --single-min-g 60 \
  --single-max-g 120 \
  --multiple-min-g 140
