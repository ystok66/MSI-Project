"""
Targeted trace audit: 1-seed deterministic investigation of 4 anomalies.

Anomaly 1: hazard_belt 0% SR across all teachers
Anomaly 2: fork_trap unlock_only → TR=75%, Steps=976
Anomaly 3: deadline_gate unlock_only → 5% SR
Anomaly 4: deadline_gate warning → 100% SR (possibly too easy)
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.lattice_v2_runner import LatticeV2Runner
from src.envs.lattice_v2 import _bfs_len
from src.envs.map_generator import CellType
from src.envs.scenario_families import generate_scenario

SEED = 42
runner = LatticeV2Runner()

_CT_NAMES = {0: "NORMAL", 1: "WALL", 2: "HIGH_COST", 3: "RISKY",
             4: "LOCKED_DOOR", 5: "TARGET", 6: "OBJECT_SPAWN"}
def ct_name(v): return _CT_NAMES.get(int(v), f"?{v}")


def banner(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def trace_episode(family, teacher_name, teacher_kw, max_steps_log=20):
    """Run one episode with per-step trace."""
    kw = dict(seed=SEED, latent_mode=True, difficulty="medium", **teacher_kw)
    if family != "baseline_v2":
        kw["scenario_family"] = family
    s = runner.reset(**kw)

    print(f"  Family={family}, Teacher={teacher_name}")
    print(f"  Agent start={s.agent_pos}, Goal={s.goal}")
    print(f"  t_max={s.t_max}")
    print(f"  tutor_mode={s.tutor_mode}, warning_mode={s.warning_mode}")
    print(f"  robot_belief_mode={s.robot_belief_mode}")
    print(f"  intervention_family_mode={s.intervention_family_mode}")
    print(f"  item_drop_enabled={s.item_drop_enabled}")
    print(f"  inventory={s.inventory}")
    print(f"  allowed_interventions={s.allowed_interventions}")
    print(f"  closure_budget={s.closure_budget}")
    print(f"  n_segments={len(s.meta.segments)}")
    for i, seg in enumerate(s.meta.segments):
        print(f"    seg[{i}]: cols={seg.col_start}-{seg.col_end}, "
              f"risky_row={seg.risky_row}, safe_row={seg.safe_row}, "
              f"risky_entry={seg.risky_entry_gate}, "
              f"n_risky={len(seg.risky_cells)}, n_safe={len(seg.safe_cells)}")
    if hasattr(s.meta, 'all_door_positions'):
        print(f"  all_door_positions={s.meta.all_door_positions}")

    step = 0
    while not s.done and step < 200:
        pos_before = s.agent_pos
        runner.step(s)
        step += 1
        if step <= max_steps_log or s.done:
            inv_info = ""
            if s.last_intervention is not None:
                d = s.last_intervention
                inv_info = (f"  → action={d.action}, "
                            f"scores={{W:{d.scores.get('WAIT','-'):.2f}, "
                            f"Wr:{d.scores.get('WARN','-'):.2f}, "
                            f"U:{d.scores.get('UNLOCK','-'):.2f}, "
                            f"I:{d.scores.get('ITEM_DROP','-'):.2f}}}, "
                            f"reason={d.reason}")
            shield_info = ""
            if s.inventory is not None:
                shield_info = f", shield={s.inventory.shield}"
            cell_type = ct_name(s.gridmap.cell_types[s.agent_pos[0], s.agent_pos[1]])
            cell_risk = s.gridmap.true_risk[s.agent_pos[0], s.agent_pos[1]]
            print(f"  t={s.t:3d} pos={s.agent_pos} type={cell_type:8s} "
                  f"risk={cell_risk:.3f}{shield_info}{inv_info}")

    reason = "GOAL" if s.reached_goal else ("DEATH" if not s.survived else "TIMEOUT")
    print(f"  Result: {reason} after {s.steps} steps "
          f"(unlocks={s.unlock_count}, warns={s.warn_count}, "
          f"risky_entered={s.risky_entered}, traps={s.traps_hit})")
    return s


# ──────────────────────────────────────────────────────────────────────
banner("ANOMALY 1: hazard_belt — checking all intervention paths")
# ──────────────────────────────────────────────────────────────────────

# Check map structure first
gm, cfg, meta, sc = generate_scenario("hazard_belt", seed=SEED, latent_mode=False)
print("=== hazard_belt map analysis (latent_mode=False) ===")
belt_seg = meta.segments[1]
print(f"Belt segment: cols={belt_seg.col_start}-{belt_seg.col_end}")
print(f"Belt risky_cells: {belt_seg.risky_cells}")
for r, c in belt_seg.risky_cells:
    print(f"  ({r},{c}): type={ct_name(gm.cell_types[r,c])}, risk={gm.true_risk[r,c]:.3f}")
print(f"Belt safe_cells: {belt_seg.safe_cells}")
print(f"ScenarioConfig: belt_regime={sc.belt_regime}, requires_item={sc.requires_item}")

# Path analysis
safe_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, {belt_seg.risky_entry_gate})
any_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, set())
print(f"Shortest path (any): {any_dist}")
print(f"Shortest path (avoid risky entry): {safe_dist}")
print(f"t_max: {cfg.max_steps}")

teachers_belt = {
    "no_tutor": dict(tutor_mode="none", warning_mode="none"),
    "warning_only": dict(tutor_mode="none", warning_mode="fixed", lambda_lane_warn=5.0),
    "robot_belief": dict(tutor_mode="none", robot_belief_mode=True,
                         intervention_family_mode=True, item_drop_enabled=True,
                         prefix_horizon=5),
}
for t_name, t_kw in teachers_belt.items():
    print(f"\n--- hazard_belt × {t_name} ---")
    trace_episode("hazard_belt", t_name, t_kw)


# ──────────────────────────────────────────────────────────────────────
banner("ANOMALY 2: fork_trap unlock_only — gate blocking check")
# ──────────────────────────────────────────────────────────────────────

gm, cfg, meta, sc = generate_scenario("fork_trap", seed=SEED, latent_mode=False)
print("=== fork_trap map analysis ===")
seg = meta.segments[0]
print(f"Segment: cols={seg.col_start}-{seg.col_end}")
print(f"Risky row={seg.risky_row}, Safe row={seg.safe_row}")
print(f"Risky entry gate={seg.risky_entry_gate}")
print(f"Safe entry gate={seg.safe_entry_gate}")
print(f"t_max: {cfg.max_steps}")

any_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, set())
safe_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, {seg.risky_entry_gate})
print(f"Shortest (any): {any_dist}, Shortest (safe only): {safe_dist}")

# Check if both branches reachable:
risky_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, {seg.safe_entry_gate})
print(f"Shortest (risky only): {risky_dist}")

print("\n--- fork_trap × unlock_only ---")
trace_episode("fork_trap", "unlock_only",
              dict(tutor_mode="time_aware", closure_budget=3),
              max_steps_log=30)


# ──────────────────────────────────────────────────────────────────────
banner("ANOMALY 3: deadline_gate unlock_only — shortcut gate check")
# ──────────────────────────────────────────────────────────────────────

gm, cfg, meta, sc = generate_scenario("deadline_gate", seed=SEED, latent_mode=False)
print("=== deadline_gate map analysis ===")
print(f"Segments: {len(meta.segments)}")
print(f"all_door_positions: {meta.all_door_positions}")
gate = meta.all_door_positions[0]
print(f"Shortcut gate at {gate}: type={ct_name(gm.cell_types[gate[0],gate[1]])}")
print(f"Gate passable? cost={gm.true_cost[gate[0],gate[1]]}")
print(f"ScenarioConfig: gate_mode={sc.gate_mode}")
print(f"t_max: {cfg.max_steps}")

# Safe path (avoid gate)
gates_set = set(meta.all_door_positions)
safe_dist = _bfs_len(gm, gm.agent_start, gm.target_pos, gates_set)
print(f"Safe long path: {safe_dist}")

# Shortcut (open gate)
gm_test = gm
old_ct = gm_test.cell_types[gate[0], gate[1]]
old_cost = gm_test.true_cost[gate[0], gate[1]]
gm_test.cell_types[gate[0], gate[1]] = CellType.NORMAL
gm_test.true_cost[gate[0], gate[1]] = 1.0
shortcut_dist = _bfs_len(gm_test, gm.agent_start, gm.target_pos, set())
gm_test.cell_types[gate[0], gate[1]] = old_ct
gm_test.true_cost[gate[0], gate[1]] = old_cost
print(f"Shortcut path (gate open): {shortcut_dist}")
print(f"Slack: t_max - safe_dist = {cfg.max_steps - safe_dist}")

# Check if runner sees the gate as unlockable
print("\nChecking _find_unlockable_cells behavior:")
passable = np.ones_like(gm.true_cost, dtype=bool)
passable[gm.cell_types == CellType.WALL] = False
passable[gm.cell_types == CellType.LOCKED_DOOR] = False
print(f"  Gate cell passable? {passable[gate[0], gate[1]]}")

# Check risky_entry_gates of each segment
for i, seg in enumerate(meta.segments):
    rg = seg.risky_entry_gate
    print(f"  seg[{i}].risky_entry_gate={rg}, "
          f"type={ct_name(gm.cell_types[rg[0],rg[1]])}, "
          f"passable={passable[rg[0],rg[1]]}")
    # Is this the LOCKED_DOOR?
    if rg == gate:
        print(f"    *** THIS is the shortcut gate — risky_entry_gate matches LOCKED_DOOR")
    else:
        print(f"    (not the shortcut gate)")

print("\n--- deadline_gate × unlock_only ---")
trace_episode("deadline_gate", "unlock_only",
              dict(tutor_mode="time_aware", closure_budget=3),
              max_steps_log=30)

print("\n--- deadline_gate × warning_only ---")
trace_episode("deadline_gate", "warning_only",
              dict(tutor_mode="none", warning_mode="fixed", lambda_lane_warn=5.0),
              max_steps_log=10)


# ──────────────────────────────────────────────────────────────────────
banner("ANOMALY 4: deadline_gate warning 100% — slack analysis")
# ──────────────────────────────────────────────────────────────────────

for diff in ["easy", "medium", "hard"]:
    gm, cfg, meta, sc = generate_scenario("deadline_gate", seed=SEED,
                                           difficulty=diff, latent_mode=False)
    gate = meta.all_door_positions[0]
    gates_set = {gate}
    safe_d = _bfs_len(gm, gm.agent_start, gm.target_pos, gates_set)
    gm.cell_types[gate[0], gate[1]] = CellType.NORMAL
    gm.true_cost[gate[0], gate[1]] = 1.0
    short_d = _bfs_len(gm, gm.agent_start, gm.target_pos, set())
    slack = cfg.max_steps - safe_d
    print(f"  {diff:8s}: safe={safe_d}, shortcut={short_d}, "
          f"t_max={cfg.max_steps}, slack={slack}, "
          f"shortcut_advantage={safe_d - short_d}")
