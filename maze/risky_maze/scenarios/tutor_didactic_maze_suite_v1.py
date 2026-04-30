"""TutorDidacticMazeSuite_v1: compact maps for safety/scaffold/transfer studies.

This suite is distinct from MiniRiskyMazeSuite_v0.

- MiniRiskyMazeSuite_v0:
  legacy fast mechanism micro-suite for warning / wait / waypoint debugging.
- TutorDidacticMazeSuite_v1:
  didactic tutor suite targeting safety shield, minimal scaffolding,
  over-help, and success-gated transfer.

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
        "The learner does not see r/m/q. It sees a noisy vector sampled "
        "from the latent cell type prototype."
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
        "obs_sigma": 0.40,
        "medium_hard_obs_sigma": 0.55,
    },
}

TUTOR_SAFETY_SCAFFOLD_GATE_V1 = {
    "name": "TutorSafetyScaffoldGate_v1",
    "width": 31,
    "height": 19,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "purpose": (
        "Safety-hard gate map. No-tutor is tempted by a short r/m/q corridor; "
        "safety warning is necessary, and a minimal waypoint to the lower detour "
        "can save time without revealing the full path."
    ),
    "recommended_config": {
        "hp": 1,
        "warning_policy": "safety_shield_truthful",
        "waypoint_policy": "minimal_frontier_or_detour_bottleneck",
        "consolidation": "success_gated_assist_discounted",
        "hidden_oracle_waypoints": False,
        "success_condition_note": (
            "Teach success should commit the lower detour route; failed teach "
            "should only preserve local risk evidence."
        ),
    },
    "map": [
        "###############################",
        "#........#############,,,,,,,,#",
        "#..K.....#############,,,,,,E,#",
        "#...........#######....,,,g,,,#",
        "#........#############,,,,,,,,#",
        "#........#############,,,,,,,,#",
        "#........#############,,,,,,E,#",
        "#....:...#############,,,:,,,,#",
        "#####:###################:#####",
        "#####:....rrmmqrrmmq.....:#####",
        "#####:###################:#####",
        "#####:#######:###########:#####",
        "#####:#######:###:#######:#####",
        "#####:###:::::###:#######:#####",
        "#####:###########:#######:#####",
        "#####::::::::::D::::::::::#####",
        "###############################",
        "###############################",
        "###############################",
    ],
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "design_checks": {
        "short_risky_path": (
            "From west to east, the shortest path crosses the middle r/m/q corridor."
        ),
        "safe_detour": "A longer lower detour passes through D at (15,15).",
        "teacher_role": (
            "Warning prevents catastrophic middle-gate entry; waypoint should only "
            "point toward the lower detour/frontier, not the final goal."
        ),
        "bug_exposure": [
            "If warning is not triggered before stepping into the gate, no_tutor and weak tutors die.",
            "If waypoint is too strong, assist leakage rises without improving eval.",
            "If success-gated consolidation is missing, eval may succeed even after teach failure.",
        ],
    },
    "tasks": {
        "teach": [
            {
                "id": "S1_T01_short_risky_gate_vs_safe_detour",
                "start": [2, 2],
                "objectives": [["pickup", [3, 2]], ["collect_gem", [26, 3]], ["exit", [28, 2]]],
                "time_limit": 58,
                "focus": (
                    "No tutor tends to take the short risky middle corridor; "
                    "shield warning plus minimal detour waypoint should enable safe completion."
                ),
            },
            {
                "id": "S1_T02_explicit_lower_detour",
                "start": [2, 6],
                "objectives": [["pass", [15, 15]], ["collect_gem", [26, 3]], ["exit", [28, 6]]],
                "time_limit": 66,
                "focus": "Teaches the reusable lower safe detour and D bottleneck.",
            },
            {
                "id": "S1_T03_detour_to_gem_fast",
                "start": [6, 15],
                "objectives": [["collect_gem", [26, 3]], ["exit", [28, 2]]],
                "time_limit": 48,
                "focus": "Checks whether learned detour memory reduces unnecessary waypointing.",
            },
        ],
        "eval_same_map_no_tutor": [
            {
                "id": "S1_E01_reuse_safe_detour_to_gem",
                "start": [2, 3],
                "objectives": [["collect_gem", [26, 3]], ["exit", [28, 2]]],
                "time_limit": 58,
                "focus": "Eval requires reusing the lower detour or learned risk to avoid the middle gate.",
            },
            {
                "id": "S1_E02_lower_detour_reverse",
                "start": [6, 15],
                "objectives": [["pass", [15, 15]], ["collect_gem", [26, 3]], ["exit", [28, 6]]],
                "time_limit": 48,
                "focus": "Tests if D-route memory transfers when tutor is absent.",
            },
            {
                "id": "S1_E03_reverse_crossing",
                "start": [24, 3],
                "objectives": [["pickup", [3, 2]], ["pass", [15, 15]], ["exit", [28, 6]]],
                "time_limit": 96,
                "focus": "Reverse-direction transfer; exposes overfitting to teach start.",
            },
        ],
    },
    "expected_baselines": {
        "no_tutor_mortal": "Low safe-success or death under hp=1 because the middle gate is shorter.",
        "always_warn": "High safety, but may be slower without detour scaffold.",
        "always_waypoint": (
            "High teach completion but high assist leakage; should not be best on eval "
            "if assist-discounted consolidation is active."
        ),
        "minimal_scaffold": (
            "High safety like always_warn, lower regret than warning-only, lower leakage than always_waypoint."
        ),
    },
}

TUTOR_AUTONOMY_LOOP_V1 = {
    "name": "TutorAutonomyLoop_v1",
    "width": 31,
    "height": 19,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "purpose": (
        "Autonomy and over-help map. The task is mostly safe but spatially branching; "
        "over-waypointing can finish teach quickly while reducing useful exploration and eval reuse."
    ),
    "recommended_config": {
        "hp": 2,
        "warning_policy": "safety_shield_truthful",
        "waypoint_policy": "minimal_frontier_or_landmark_only",
        "consolidation": "success_gated_assist_discounted",
        "hidden_oracle_waypoints": False,
        "waypoint_budget": "1-2 per teach episode",
        "success_condition_note": (
            "Routes discovered autonomously should get higher route-graph credit "
            "than waypoint-led routes."
        ),
    },
    "map": [
        "###############################",
        "#::::::::::##########,,,,,,,,,#",
        "#::K:#:::::##########,,,,#,E,,#",
        "#::::::::::::.....,,,,,,,#,,,,#",
        "#:#:###:##:#.......##,##,##,#,#",
        "#::::#::::::::###..##,,,,,,,,,#",
        "#::::#:::::#..###..##,,,,#,,,,#",
        "#::::#:::::#..###..##,,,,#,,,,#",
        "#:#:###:##:#.......##,##,##,#,#",
        "#::::::::::::..D.rrDmmq,,#,,,,#",
        "#::::#qqq::#.......##,,,,,,,,,#",
        "#::::#:::::#..###..##,,,,#,,,,#",
        "#:#:###:##:#..###..##,##,##,#,#",
        "#::::#:::::#..###..##,,mm#mm,,#",
        "#::::#:::::#.......##,,,,#,g,,#",
        "#::::::::::::.....,,,,,,,,,,,,#",
        "#::g:#:::::##########,,,,#,E,,#",
        "#::::::::::##########,,,,,,,,,#",
        "###############################",
    ],
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "design_checks": {
        "branching_loops": "West and east rooms contain multiple loops/dead-ends so WAIT can be useful or wasteful.",
        "moderate_risk": "Risk clusters exist but do not dominate; this map is for hint/waypoint, not primarily safety.",
        "teacher_role": "Waypoint should rescue no-progress/timeout, not trace the whole route.",
        "bug_exposure": [
            "If waypoint is overused, teach steps fall but autonomy credit and useful exploration collapse.",
            "If WAIT is too conservative, map coverage and map reuse are low.",
            "If assist-discounted consolidation is missing, always_waypoint may look artificially strong.",
        ],
    },
    "tasks": {
        "teach": [
            {
                "id": "S2_T01_west_gem_to_ne_exit",
                "start": [2, 16],
                "objectives": [["collect_gem", [3, 16]], ["exit", [27, 2]]],
                "time_limit": 76,
                "focus": "Large west-to-east transfer; useful exploration of hub should matter for eval.",
            },
            {
                "id": "S2_T02_key_to_door_to_south_exit",
                "start": [2, 2],
                "objectives": [["pickup", [3, 2]], ["pass", [15, 9]], ["exit", [27, 16]]],
                "time_limit": 72,
                "focus": "Learns central D/hub relation without requiring dense waypointing.",
            },
            {
                "id": "S2_T03_central_to_east_gem",
                "start": [15, 15],
                "objectives": [["collect_gem", [27, 14]], ["exit", [27, 16]]],
                "time_limit": 44,
                "focus": "Short local task; any waypoint here is likely over-help.",
            },
        ],
        "eval_same_map_no_tutor": [
            {
                "id": "S2_E01_reverse_east_to_west_gem",
                "start": [27, 16],
                "objectives": [["collect_gem", [3, 16]], ["exit", [27, 2]]],
                "time_limit": 104,
                "focus": "Requires reusable west/east route graph, not just following teach waypoints.",
            },
            {
                "id": "S2_E02_central_to_east_gem",
                "start": [15, 15],
                "objectives": [["collect_gem", [27, 14]], ["exit", [27, 16]]],
                "time_limit": 44,
                "focus": "Tests whether local east room memory was discovered or merely pointed through.",
            },
            {
                "id": "S2_E03_west_to_door_exit",
                "start": [2, 3],
                "objectives": [["pickup", [3, 2]], ["pass", [15, 9]], ["exit", [27, 16]]],
                "time_limit": 86,
                "focus": "Reuses central door/hub route under tutor-off eval.",
            },
        ],
    },
    "expected_baselines": {
        "no_tutor_mortal": "Usually completes but with higher steps/no-progress; useful as autonomy baseline.",
        "always_warn": "Similar safety to no_tutor; should not solve no-progress alone.",
        "always_waypoint": "Likely low teach steps but high assist leakage and low autonomy credit.",
        "minimal_scaffold": "Should reduce no-progress while preserving enough autonomous exploration for eval.",
    },
}

TUTOR_PRINCIPLE_DOOR_TRANSFER_V1 = {
    "name": "TutorPrincipleDoorTransfer_v1",
    "width": 31,
    "height": 19,
    "coordinate_system": "zero-indexed (x, y), origin at top-left, x=column, y=row",
    "purpose": (
        "Success-gated transfer map. Teach success should commit a reusable key-door/route principle; "
        "failed teach should leave eval hard. Over-help should complete teach but earn weaker route-memory credit."
    ),
    "recommended_config": {
        "hp": 2,
        "warning_policy": "safety_shield_truthful",
        "waypoint_policy": "minimal_bottleneck_or_frontier",
        "consolidation": "success_gated_assist_discounted",
        "hidden_oracle_waypoints": False,
        "route_graph_commit": "only_on_teach_success",
        "success_condition_note": (
            "If teach fails, only local risk observations should persist; "
            "route graph/key-door principle should not commit."
        ),
    },
    "map": [
        "###############################",
        "#..........#########,,,,,,,,,,#",
        "#..E.......#########,,,,,,,K,,#",
        "#....................,,,,,,,,,#",
        "#..........####.####,,,,,,,,,,#",
        "#..........####.####,,,,,,,,,,#",
        "#..........rrrm.qrrr.,,,,,,,,,#",
        "#..........####.####,,,,,,,,,,#",
        "#####.######.......####,#######",
        "###......###..rDr..###,,,,,,###",
        "#####.######...:...####,#######",
        "#::::.:::::####:####,,,,,,,,,,#",
        "#::::::::::qqqm:rqqq:,,,,,,g,,#",
        "#::::::::::####:####,,,,,,,,,,#",
        "#::::::::::####:####,,,,,,,,,,#",
        "#::::::::::::::::::::,,,,,,,,,#",
        "#::g:::::::#########,,,,,,,E,,#",
        "#::::::::::#########,,,,,,,,,,#",
        "###############################",
    ],
    "legend": LEGEND,
    "risk_feature_spec": RISK_FEATURE_SPEC,
    "design_checks": {
        "principle_dependency": (
            "Key at (27,2), D at (15,9), west gem at (3,16), and exits create a reusable route graph."
        ),
        "success_gating": (
            "Eval tasks reuse the key-door/central-bottleneck relation; failed teach should not commit that relation."
        ),
        "teacher_role": (
            "Warning protects around r/q/m lures; waypoint should point to bottleneck/frontier "
            "only when learner would loop or timeout."
        ),
        "bug_exposure": [
            "If eval succeeds after teach failure, success-gated consolidation is not working.",
            "If oracle waypoint beats minimal scaffold on eval, assist-discounted consolidation is too weak.",
            "If warning blocks all progress near D, warning actionability/scoping is wrong.",
        ],
    },
    "tasks": {
        "teach": [
            {
                "id": "S3_T01_full_key_door_gem_exit",
                "start": [2, 2],
                "objectives": [["pickup", [27, 2]], ["pass", [15, 9]], ["collect_gem", [3, 16]], ["exit", [27, 16]]],
                "time_limit": 96,
                "focus": (
                    "Full teach route; success should commit key-door and central-bottleneck route principle."
                ),
            },
            {
                "id": "S3_T02_south_to_east_gem",
                "start": [2, 16],
                "objectives": [["collect_gem", [27, 12]], ["exit", [27, 16]]],
                "time_limit": 46,
                "focus": "Local but transfer-relevant route; waypoint should be minimal.",
            },
            {
                "id": "S3_T03_ne_to_door_to_west_exit",
                "start": [27, 3],
                "objectives": [["pass", [15, 9]], ["collect_gem", [3, 16]], ["exit", [3, 2]]],
                "time_limit": 68,
                "focus": "Reverse route through D; exposes whether route graph is symmetric or overfit.",
            },
        ],
        "eval_same_map_no_tutor": [
            {
                "id": "S3_E01_reverse_full_transfer",
                "start": [27, 16],
                "objectives": [["pickup", [27, 2]], ["pass", [15, 9]], ["collect_gem", [3, 16]], ["exit", [3, 2]]],
                "time_limit": 92,
                "focus": "Requires successful teach route principle; no tutor in eval.",
            },
            {
                "id": "S3_E02_west_to_key_to_exit",
                "start": [3, 16],
                "objectives": [["pickup", [27, 2]], ["exit", [27, 16]]],
                "time_limit": 78,
                "focus": "Tests route-graph reuse without repeating the exact teach sequence.",
            },
            {
                "id": "S3_E03_central_to_east_gem",
                "start": [15, 9],
                "objectives": [["collect_gem", [27, 12]], ["exit", [27, 16]]],
                "time_limit": 34,
                "focus": "Short eval probe for local D/east-region memory.",
            },
        ],
    },
    "expected_baselines": {
        "no_tutor_mortal": "May sometimes complete, but teach failures should predict poor eval under success-gated commit.",
        "always_warn": "Protects risk but may not reduce no-progress/time pressure.",
        "oracle_waypoint": "Should complete teach, but lower autonomy credit should limit eval transfer.",
        "minimal_scaffold": "Should achieve teach success with higher autonomy credit and lower eval regret.",
    },
}

TUTOR_DIDACTIC_MAZE_SUITE_V1 = {
    "name": "TutorDidacticMazeSuite_v1",
    "description": (
        "Three compact 31x19 fixed maps designed specifically for tutor diagnostics: "
        "safety warning, minimal waypoint scaffolding, over-help/generalization, "
        "and success-gated transfer."
    ),
    "shared_legend": LEGEND,
    "shared_risk_feature_spec": RISK_FEATURE_SPEC,
    "maps": [
        TUTOR_SAFETY_SCAFFOLD_GATE_V1,
        TUTOR_AUTONOMY_LOOP_V1,
        TUTOR_PRINCIPLE_DOOR_TRANSFER_V1,
    ],
}

