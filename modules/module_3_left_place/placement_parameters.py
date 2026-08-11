"""模块 3 的共享运动参数。

工件1和工件2的正反面放置参数分别在
``workpiece_1_placement_parameters.toml`` 与
``workpiece_2_placement_parameters.toml``。
"""

# 左臂回程位置。
GRASPUP_JOINTS_DEG = [-4.081, -52.722, 3.637, 100.478, -10.051, 53.969, 8.843]
GRASPUP_ABOVE_JOINTS_DEG = [
    -1.311500,
    -60.092500,
    4.404100,
    88.846400,
    -15.384900,
    73.190800,
    12.937800,
]
GRASPUP_ABOVE_PLANNING_TIMEOUT_S = 6.0
DISCARD_BELOW_GRASPUP_M = 0.05

# 从 graspup 上方执行离线规划时的上移量、等待上限及无可行解时的丢弃下移量。
GRASPUP_LIFT_M = 0.10
