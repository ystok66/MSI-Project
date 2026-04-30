"""Fixed scenario assets and validators."""

from .huge_risky_gem_maze_v0 import SPEC as HUGE_RISKY_GEM_MAZE_V0_SPEC
from .mini_explore_loop_v0 import SPEC as MINI_EXPLORE_LOOP_V0_SPEC
from .mini_risk_gate_v0 import SPEC as MINI_RISK_GATE_V0_SPEC
from .mini_risky_maze_suite_v0 import MINI_RISKY_MAZE_SUITE_V0
from .mini_waypoint_bottleneck_v0 import SPEC as MINI_WAYPOINT_BOTTLENECK_V0_SPEC
from .tutor_autonomy_loop_v1 import SPEC as TUTOR_AUTONOMY_LOOP_V1_SPEC
from .tutor_didactic_maze_suite_v1 import TUTOR_DIDACTIC_MAZE_SUITE_V1
from .tutor_principle_door_transfer_v1 import SPEC as TUTOR_PRINCIPLE_DOOR_TRANSFER_V1_SPEC
from .tutor_safety_scaffold_gate_v1 import SPEC as TUTOR_SAFETY_SCAFFOLD_GATE_V1_SPEC
from .validation import summarize_fixed_map_spec, validate_fixed_map_spec

__all__ = [
    "HUGE_RISKY_GEM_MAZE_V0_SPEC",
    "MINI_RISK_GATE_V0_SPEC",
    "MINI_EXPLORE_LOOP_V0_SPEC",
    "MINI_WAYPOINT_BOTTLENECK_V0_SPEC",
    "MINI_RISKY_MAZE_SUITE_V0",
    "TUTOR_SAFETY_SCAFFOLD_GATE_V1_SPEC",
    "TUTOR_AUTONOMY_LOOP_V1_SPEC",
    "TUTOR_PRINCIPLE_DOOR_TRANSFER_V1_SPEC",
    "TUTOR_DIDACTIC_MAZE_SUITE_V1",
    "summarize_fixed_map_spec",
    "validate_fixed_map_spec",
]
