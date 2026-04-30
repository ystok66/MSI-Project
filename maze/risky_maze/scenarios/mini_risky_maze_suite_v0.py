"""MiniRiskyMazeSuite_v0: three compact fixed maps for fast diagnostics.

All maps keep the same symbolic conventions as HugeRiskyGemMaze_v0 while
shrinking the total area to about one quarter of the original map.

Coordinate system:
  - zero-indexed (x, y)
  - origin at top-left
  - x = column
  - y = row
"""

LEGEND = {
    "#": "wall / blocked",
    ".": "safe_floor_type_0",
    ",": "safe_floor_type_1",
    ":": "safe_floor_type_2",
    "r": "danger_type_1_spike_ridge",
    "m": "danger_type_2_toxic_mist",
    "q": "danger_type_3_quicksand",
    "K": "key / useful object",
    "D": "door / pass bottleneck",
    "g": "gem / subgoal",
    "E": "exit",
}

RISK_FEATURE_SPEC = {
    "feature_dim": 12,
    "learner_observation": (
        "The learner does not see r/m/q. "
        "It sees a noisy vector sampled from the latent cell type prototype."
    ),
    "latent_types": {
        ".": {"class": "safe", "prototype_id": "safe_0"},
        ",": {"class": "safe", "prototype_id": "safe_1"},
        ":": {"class": "safe", "prototype_id": "safe_2"},
        "r": {"class": "danger", "prototype_id": "danger_spike_ridge"},
        "m": {"class": "danger", "prototype_id": "danger_toxic_mist"},
        "q": {"class": "danger", "prototype_id": "danger_quicksand"},
    },
    "suggested_noise": {
        "cluster_sigma": 0.45,
        "obs_sigma": 0.35,
        "medium_hard_obs_sigma": 0.50,
    },
}

MINI_RISK_GATE_V0 = {
    "name": "MiniRiskGate_v0",
    "width": 31,
    "height": 19,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "purpose": "warning-gate slice: direct risky corridor vs longer safe detours",
    "map": [
        "###############################",
        "#........##############,,,,,,,#",
        "#..K.....##############,,,,,E,#",
        "#......................,,,,,,,#",
        "#........##############,,,,,,,#",
        "#........###.......####,,,,,,,#",
        "#........###.#####.####,,,,,,,#",
        "####.###.###.#####.####,#####,#",
        "####.###.###.#####.####,#####,#",
        "####.###...rrmmDrrmrr..,#####,#",
        "####.###.##############,#####,#",
        "####.###.##############,#####,#",
        "####.###.#:::qqq:::::##,,g,,,,#",
        "####.###.#:#########:##,,,,,,,#",
        "####.###.#:#########:##,,,,,,,#",
        "####.:::.::::::::::::::,,,,,,,#",
        "#######################,,,,,E,#",
        "#######################,,,,,,,#",
        "###############################",
    ],
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "tasks": {
        "teach": [
            {
                "id": "A_T01_key_danger_gate_NE_exit",
                "start": [2, 2],
                "objectives": [
                    ["pickup", [3, 2]],
                    ["pass", [15, 9]],
                    ["collect_gem", [25, 12]],
                    ["exit", [28, 2]],
                ],
                "time_limit": 90,
                "focus": "warning gate on direct r/m corridor vs safe upper/lower detour",
            },
            {
                "id": "A_T02_south_safe_detour",
                "start": [4, 15],
                "objectives": [["collect_gem", [25, 12]], ["exit", [28, 16]]],
                "time_limit": 70,
                "focus": "safe-detour memory and low-warning route",
            },
            {
                "id": "A_T03_left_to_south_exit",
                "start": [2, 5],
                "objectives": [["collect_gem", [25, 12]], ["exit", [28, 16]]],
                "time_limit": 80,
                "focus": "map reuse from lower corridor; waypoint should be sparse",
            },
        ],
        "eval_same_map_no_tutor": [
            {
                "id": "A_E01_NE_exit_from_left",
                "start": [2, 3],
                "objectives": [["exit", [28, 2]]],
                "time_limit": 75,
                "focus": "does warning/WAIT learn to avoid middle danger corridor?",
            },
            {
                "id": "A_E02_south_gem_exit",
                "start": [6, 15],
                "objectives": [["collect_gem", [25, 12]], ["exit", [28, 16]]],
                "time_limit": 65,
                "focus": "same-map safe-route reuse",
            },
            {
                "id": "A_E03_cross_gate_to_south_exit",
                "start": [24, 3],
                "objectives": [["pickup", [3, 2]], ["exit", [28, 16]]],
                "time_limit": 95,
                "focus": "reverse-direction map memory and risk avoidance",
            },
        ],
    },
}

MINI_EXPLORE_LOOP_V0 = {
    "name": "MiniExploreLoop_v0",
    "width": 31,
    "height": 19,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "purpose": "exploration-loop slice: map memory, reusable loops, q/r/m risk clusters",
    "map": [
        "###############################",
        "#:::::::::############,,,,,,,,#",
        "#::::#::::############,,,#,,E,#",
        "#::g:#::::#..........#,,,,,,,,#",
        "#:::::qqq............#,,,#,,,,#",
        "#::::#::::#............mm#,,,,#",
        "#::::#::::#..######..#,,,#,,,,#",
        "#::::#::::#.....#....#,,,#,,,,#",
        "#:#:###:#:#.....#....#,,,#,,,,#",
        "#::::#::::#....K#.rr..D,,,mmm,#",
        "#::::::::...rrr.#....#,,,#,,,,#",
        "#::::#::::#.....#....#,#,###,,#",
        "#::::#::::#..######..#,,,#,,,,#",
        "#::::#::::#..........#,,,#,,,,#",
        "#::::#::::#............,g,,,,,#",
        "#::::::::............#,,,#,,,,#",
        "#::::#::::############,,,#,,E,#",
        "#:::::::::############,,,,,,,,#",
        "###############################",
    ],
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "tasks": {
        "teach": [
            {
                "id": "B_T01_west_gem_to_east_exit",
                "start": [2, 16],
                "objectives": [["collect_gem", [3, 3]], ["exit", [28, 16]]],
                "time_limit": 100,
                "focus": "large loop exploration; q shortcut vs safe west/central route",
            },
            {
                "id": "B_T02_key_door_east_exit",
                "start": [2, 10],
                "objectives": [["pickup", [15, 9]], ["pass", [22, 9]], ["exit", [28, 2]]],
                "time_limit": 85,
                "focus": "central key-door path; warning against r/m connector",
            },
            {
                "id": "B_T03_east_gem_to_south_exit",
                "start": [28, 3],
                "objectives": [["collect_gem", [24, 14]], ["exit", [28, 16]]],
                "time_limit": 55,
                "focus": "local east room; waypoint should not over-help",
            },
        ],
        "eval_same_map_no_tutor": [
            {
                "id": "B_E01_reverse_east_to_west_gem",
                "start": [28, 16],
                "objectives": [["collect_gem", [3, 3]], ["exit", [28, 2]]],
                "time_limit": 115,
                "focus": "recombines west exploration and east exits",
            },
            {
                "id": "B_E02_central_to_east_gem",
                "start": [15, 15],
                "objectives": [["collect_gem", [24, 14]], ["exit", [28, 16]]],
                "time_limit": 60,
                "focus": "central/east map reuse; avoid m corridor",
            },
            {
                "id": "B_E03_west_to_door_exit",
                "start": [2, 3],
                "objectives": [["pickup", [15, 9]], ["pass", [22, 9]], ["exit", [28, 16]]],
                "time_limit": 90,
                "focus": "door route memory and risk avoidance",
            },
        ],
    },
}

MINI_WAYPOINT_BOTTLENECK_V0 = {
    "name": "MiniWaypointBottleneck_v0",
    "width": 31,
    "height": 19,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "purpose": "waypoint-bottleneck slice: loop/dead-end control and assist-leakage stress test",
    "map": [
        "###############################",
        "#............#####,,,,,,,,,,,,#",
        "#...#........#####,,,,,,,,,K,,#",
        "#............#####,##,#####,#,#",
        "#...#...rrr........rr,,,#,,,,,#",
        "#..#####.###.##.##,,,,,,#,,,,,#",
        "#............##.##,,,,,,,,mm,,#",
        "#............##.##,,,,,,,,,,,,#",
        "####.########.....#########,###",
        "####.########.rDr.#########,E##",
        "####qqq######.....######qqq,###",
        "####.##########.###########,###",
        "#:::.::::::::##.##,,,,,,,,,,,,#",
        "#:::::#::::::##.##,,mmm,,,,,,,#",
        "#:##:#####:#:##.##,,,,,#,,,,,,#",
        "#:::::#::::::::::::,,,,#g,,,,,#",
        "#::g:::::::::#####,#,######,E,#",
        "#::::::::::::#####,,,,,,,,,,,,#",
        "###############################",
    ],
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "tasks": {
        "teach": [
            {
                "id": "C_T01_upper_key_bottleneck_exit",
                "start": [2, 2],
                "objectives": [["pickup", [27, 2]], ["pass", [15, 9]], ["exit", [28, 9]]],
                "time_limit": 85,
                "focus": "waypoint helps find east key and central bottleneck; r lures need warning",
            },
            {
                "id": "C_T02_south_vault_gem_exit",
                "start": [2, 16],
                "objectives": [["collect_gem", [24, 15]], ["exit", [28, 16]]],
                "time_limit": 70,
                "focus": "dead-end/loop control; waypoint should reduce boredom without full pathing",
            },
            {
                "id": "C_T03_central_to_west_gem_NE_exit",
                "start": [15, 15],
                "objectives": [["collect_gem", [3, 16]], ["exit", [28, 9]]],
                "time_limit": 90,
                "focus": "reverse route; map-memory and warning around q corridor",
            },
        ],
        "eval_same_map_no_tutor": [
            {
                "id": "C_E01_east_to_west_gem_exit",
                "start": [28, 2],
                "objectives": [["collect_gem", [3, 16]], ["exit", [28, 16]]],
                "time_limit": 105,
                "focus": "does teach waypoint reveal useful but not over-helped map memory?",
            },
            {
                "id": "C_E02_lower_to_key_exit",
                "start": [4, 16],
                "objectives": [["pickup", [27, 2]], ["exit", [28, 9]]],
                "time_limit": 100,
                "focus": "reuses upper/east key route; tests over-help transfer",
            },
            {
                "id": "C_E03_central_to_east_gem_exit",
                "start": [15, 9],
                "objectives": [["collect_gem", [24, 15]], ["exit", [28, 16]]],
                "time_limit": 70,
                "focus": "local waypoint benefit vs direct eval reuse",
            },
        ],
    },
}

MINI_RISKY_MAZE_SUITE_V0 = {
    "name": "MiniRiskyMazeSuite_v0",
    "description": (
        "Three hand-designed fixed maps, each 31x19 (~1/4 area of "
        "HugeRiskyGemMaze_v0 61x39)."
    ),
    "maps": [
        MINI_RISK_GATE_V0,
        MINI_EXPLORE_LOOP_V0,
        MINI_WAYPOINT_BOTTLENECK_V0,
    ],
}

SPEC_BY_NAME = {
    MINI_RISK_GATE_V0["name"]: MINI_RISK_GATE_V0,
    MINI_EXPLORE_LOOP_V0["name"]: MINI_EXPLORE_LOOP_V0,
    MINI_WAYPOINT_BOTTLENECK_V0["name"]: MINI_WAYPOINT_BOTTLENECK_V0,
}

