"""GTET-L generator smoke test."""
import sys
sys.path.insert(0, ".")

from src.envs.gtet_lattice import generate_gtet_lattice, _bfs_gtet

for diff in ["easy", "medium", "hard"]:
    try:
        gm, cfg, meta, sc = generate_gtet_lattice(seed=42, difficulty=diff)
        gt = meta.gtet_meta
        ct = gm.cell_types
        H, W = ct.shape

        # Check reachability
        start = (H // 2, 1)
        goal = gm.target_pos
        dist = _bfs_gtet(ct, start, goal)

        print(f"=== {diff.upper()} ===")
        print(f"  Grid: {H}x{W}")
        print(f"  Routes: {meta.route_count}")
        print(f"  Stages: {meta.decision_stages}")
        print(f"  Doors: {len(meta.all_door_positions)}")
        print(f"  Belt cells: {len(meta.belt_cells_by_stage[2])}")
        print(f"  Goal cue cells: {len(meta.goal_cue_cells)}")
        print(f"  Tempt cue cells: {len(meta.temptation_cue_cells)}")
        print(f"  Shortest any: {meta.shortest_any}")
        print(f"  Shortest safe: {meta.shortest_safe}")
        print(f"  BFS to goal: {dist}")
        print(f"  Max steps: {cfg.max_steps}")
        print(f"  Reveal order: {gt.subgoal_reveal_order}")
        print(f"  Merge points: {gt.merge_points}")
        print(f"  Overlap records: {len(gt.latent_explanation_overlap)}")
        for o in gt.latent_explanation_overlap:
            print(f"    sg={o['subgoal']} shared={o['n_shared']} "
                  f"goal={o['n_goal']} tempt={o['n_tempt']}")
        print(f"  Goal consistent: { {k: len(v) for k, v in gt.goal_consistent_routes.items()} }")
        print(f"  Tempt preferred: { {k: len(v) for k, v in gt.temptation_preferred_routes.items()} }")
        print()
    except Exception as e:
        import traceback
        print(f"=== {diff.upper()} FAILED ===")
        traceback.print_exc()
        print()

print("Done.")
