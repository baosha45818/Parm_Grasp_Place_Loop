"""桌中 FRONT 放置的共享运动参数。

参数保持 Git ``27254cb`` 旧 Module 3 的组织方式；工件参数分别位于
``workpiece_1_placement_parameters.toml`` 和
``workpiece_2_placement_parameters.toml``。
"""

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
GRASPUP_LIFT_M = 0.10

