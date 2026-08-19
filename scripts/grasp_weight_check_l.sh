uv run python3 -m tools.grasp_weight_check.manual_force_sensor_check \
  --host 127.0.0.1 \
  --hand 1 \
  --empty-max-g 30 \
  --single-min-g 80 \
  --single-max-g 140 \
  --multiple-min-g 180
