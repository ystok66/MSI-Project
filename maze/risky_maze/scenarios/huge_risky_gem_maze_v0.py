"""HugeRiskyGemMaze_v0: hand-designed fixed map and task suite.

Coordinate system:
  - zero-indexed (x, y)
  - origin at top-left
  - x = column
  - y = row
"""

MAP_LINES = [
    "#############################################################",
    "#..........#...###,,,,,,#,,,,,,,###.................#......##",
    "#...K.#........###,,,,,,,,,,,,,,###........#........#..E...##",
    "#.....#....#...###,,#########,,,###........#........#......##",
    "#..##########..###,,,,,,#,,,,#,,###..######.########.####..##",
    "#.....#.......D.r..,,.,,#,,,,,,....D.......#........#..g...##",
    "#..........#...###,,,.,,,,,,,#,,###..mmm...#........#......##",
    "#.....#....#...###,,,.,,,,,,,#K,###.###.########.###.#.....##",
    "#...........g..###,,,.,,,,,.,#,,###.................#......##",
    "#..............######.#####.#######........#...............##",
    "########............r.r.r...#######........................##",
    "########.############.#####.#######.#########.###.###########",
    "########.###########...q..q...q..........####.###.###########",
    "##:::::#.::::::::###....#.......#........###,.,,,.,,,,,#,,,##",
    "##::qqq#:::::#:::###..#.#############.#..###,.,q,#,,q,q#,,,##",
    "##:::::#qqq::#:::###........#.......#....###,.,K,#,,,,,#,,,##",
    "##::###:#####:##:###....#...#...#.g.#....###,.###,#####,##,##",
    "##:::::#:::::#::........#.......#.......D.rr..,,,#,,,,,#,,,##",
    "##::::g#:::::#:::###...####.#######.###..###,,,,,#,,,,,,,,,##",
    "##:::::#:::::::::###........#.......#....###,,r,r#r,,,r#,,,##",
    "##:##:#####:###::###....#...#...#...#....###,##,####,####,,##",
    "##:::::#:::::#::....D...#.......#............,,,,#,,,,,#,,,##",
    "##:::::::::::#:::###..#######.#.######...###,,,,,#,,,g,#,,,##",
    "##:::::#:::::#:::###........#.......#....###,,,,,,,,,,,,,,,##",
    "##::::::::.::#:::###....#...#...#...#....###,,,,,#,.,,,.,,,##",
    "##########.#########.....................##########.###.#####",
    "##########...............r..............r.r.........###.#####",
    "##########.#...################.###...#############...#.#####",
    "#.............#....############.###.###############.###.#####",
    "#.q.q.#.......#q.q.####::::::::.:::.#:::####...........#...##",
    "#.....#.......#....####::::::#::::::#g::####.qq..#.qq..#...##",
    "#..###.#######.###.####::::::#::::::#:::####.....#.........##",
    "#.....#.......#.........:####:######:##.....D.###.#####.##.##",
    "#.....#............####::::::#::::::#:::####.....#.....#...##",
    "#.###.######.####..####::::K:#::::::::::####...........#.E.##",
    "#.........g...#....####:::::::::::::#:::####.##.#####.###..##",
    "#.....#............####::::::#::::::::::####.....#.........##",
    "#############################################################",
    "#############################################################",
]

LEGEND = {
    "#": "wall / blocked",
    ".": "safe_floor_type_0",
    ",": "safe_floor_type_1",
    ":": "safe_floor_type_2",
    "r": "danger_type_1_spike_ridge",
    "m": "danger_type_2_toxic_mist",
    "q": "danger_type_3_quicksand",
    "K": "key / useful object",
    "D": "locked door / bottleneck",
    "g": "gem / subgoal",
    "E": "exit",
}

RISK_FEATURE_SPEC = {
    "feature_dim": 12,
    "learner_observation": (
        "The learner does not see map chars r/m/q. "
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
        "obs_sigma": 0.30,
        "medium_hard_obs_sigma": 0.45,
    },
}

TASKS = {
    "teach": [
        {
            "id": "T01_NW_key_gem_NE_exit",
            "start": [2, 2],
            "objectives": [
                ["pickup", [4, 2]],
                ["pass", [14, 5]],
                ["collect_gem", [12, 8]],
                ["exit", [55, 2]],
            ],
            "time_limit": 170,
            "focus": "top-left map memory; first door; safe detour around r at north connector",
        },
        {
            "id": "T02_NorthLab_to_NE_gem",
            "start": [20, 2],
            "objectives": [
                ["pickup", [30, 7]],
                ["collect_gem", [55, 5]],
                ["exit", [55, 2]],
            ],
            "time_limit": 125,
            "focus": "north lab / NE gallery; toxic mist cluster near y=6; low-risk shortcut learning",
        },
        {
            "id": "T03_WestGarden_to_SE_exit",
            "start": [3, 23],
            "objectives": [["collect_gem", [6, 18]], ["exit", [57, 34]]],
            "time_limit": 165,
            "focus": "west garden; q clusters; central-to-south route discovery",
        },
        {
            "id": "T04_CentralCore_to_SE_exit",
            "start": [21, 13],
            "objectives": [["collect_gem", [34, 16]], ["exit", [57, 34]]],
            "time_limit": 135,
            "focus": "central maze; many corridors; tutor should avoid over-pointing",
        },
        {
            "id": "T05_EastFoundry_to_NE_exit",
            "start": [45, 23],
            "objectives": [
                ["pickup", [47, 15]],
                ["collect_gem", [53, 22]],
                ["exit", [55, 2]],
            ],
            "time_limit": 155,
            "focus": "east foundry; r traps near door; m/q contrast in same region",
        },
        {
            "id": "T06_SWReservoir_to_SE_exit",
            "start": [2, 35],
            "objectives": [["collect_gem", [10, 35]], ["exit", [57, 34]]],
            "time_limit": 150,
            "focus": "bottom corridor memory; mostly map exploration rather than risk",
        },
        {
            "id": "T07_SouthVault_key_door_exit",
            "start": [24, 35],
            "objectives": [
                ["pickup", [27, 34]],
                ["collect_gem", [37, 30]],
                ["pass", [44, 32]],
                ["exit", [57, 34]],
            ],
            "time_limit": 115,
            "focus": "south vault; door waypoint usefulness; assist leakage check",
        },
        {
            "id": "T08_LongDiagonal_multi_gem",
            "start": [2, 2],
            "objectives": [
                ["collect_gem", [34, 16]],
                ["collect_gem", [53, 22]],
                ["exit", [57, 34]],
            ],
            "time_limit": 200,
            "focus": "long-horizon planning; map-memory reuse; tutor wait vs warning vs waypoint",
        },
    ],
    "eval_same_map_no_tutor": [
        {
            "id": "E01_WestGarden_to_NE_exit",
            "start": [3, 22],
            "objectives": [["exit", [55, 2]]],
            "time_limit": 160,
            "focus": "does prior west/north exploration reduce damage and steps?",
        },
        {
            "id": "E02_SE_to_NWgem_NE_exit",
            "start": [53, 34],
            "objectives": [["collect_gem", [12, 8]], ["exit", [55, 2]]],
            "time_limit": 275,
            "focus": "cross-map transfer of route memory",
        },
        {
            "id": "E03_Foundry_to_NorthLab_exit",
            "start": [50, 15],
            "objectives": [["pickup", [30, 7]], ["exit", [55, 2]]],
            "time_limit": 155,
            "focus": "east-to-north transfer; avoid overfitting to teach starts",
        },
        {
            "id": "E04_SW_to_EastGem_SE_exit",
            "start": [10, 35],
            "objectives": [["collect_gem", [53, 22]], ["exit", [57, 34]]],
            "time_limit": 165,
            "focus": "bottom-to-east path planning; risk shortcut vs safe detour",
        },
        {
            "id": "E05_Central_to_SWgem_SE_exit",
            "start": [34, 23],
            "objectives": [["collect_gem", [10, 35]], ["exit", [57, 34]]],
            "time_limit": 200,
            "focus": "central and south route recombination",
        },
        {
            "id": "E06_NW_to_SouthVaultGem_SE_exit",
            "start": [5, 3],
            "objectives": [["collect_gem", [37, 30]], ["exit", [57, 34]]],
            "time_limit": 235,
            "focus": "long-route memory and trap concept transfer",
        },
        {
            "id": "E07_East_to_WestGem_NE_exit",
            "start": [56, 21],
            "objectives": [["collect_gem", [6, 18]], ["exit", [55, 2]]],
            "time_limit": 280,
            "focus": "hard cross-map task; detects whether exploration was useful",
        },
        {
            "id": "E08_SouthVault_to_NE_exit",
            "start": [25, 33],
            "objectives": [["exit", [55, 2]]],
            "time_limit": 140,
            "focus": "new start, known exits; no tutor",
        },
    ],
}

SPEC = {
    "name": "HugeRiskyGemMaze_v0",
    "width": 61,
    "height": 39,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "map": MAP_LINES,
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "tasks": TASKS,
}
