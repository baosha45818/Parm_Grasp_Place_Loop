# Module 1/3 runtime tools

This directory contains the project-local runtime dependencies used by the
standalone Module 1 ICP grasp and Module 3 Buffer placement flows.

## Layout

- `sam2-service/`: SAM2/EfficientViT service source, third-party backends and
  checkpoints. Historical output, tests, Git metadata and virtual environments
  are intentionally excluded.
- `graspnet_runtime/`: the minimal GraspNet-derived Planner and D405 Camera
  Broker source closure. GraspNet inference models, checkpoints, jobs, output
  and its 8.9 GB virtual environment are intentionally excluded.
- `calibration/`: only the wrist, Planner bridge and TCP calibration JSON files
  required by Module 1.
- `workpiece_1/workpiece_1.stl`: the CAD model required by the ICP pipeline.
- `planner_runtime.py`, `planner_task_retry.py`,
  `left_motion_right_servo_lock.py`: Module 3 execution support.
- `planner_joint_motion.py`, `go_graspnet.py`, `go_graspnet2.py`: Module 1
  observation-pose motion support.
- `icp.py`: shared ICP solver already owned by the project.

The original `tool/` and `third_party/graspnet/` trees remain in place for
other legacy modules, but Module 1 and Module 3 no longer import code from
them. `graspnet_runtime/LICENSE` preserves the upstream GraspNet license; the
SAM backend licenses remain beside their respective sources.

## SAM2 environment

For a fully independent SAM2 environment, run:

```bash
cd /home/dainanjun/Public/Parm_Grasp_Place_Loop/tools/sam2-service
uv sync --python 3.10
```

Until that local environment is created, the launcher may reuse the existing
legacy SAM2 Python interpreter, while `PYTHONPATH` forces service code and
checkpoints to load from this directory.
