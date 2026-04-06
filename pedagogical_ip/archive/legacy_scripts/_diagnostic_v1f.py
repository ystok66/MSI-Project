"""Diagnostic: check PlanningTrap TrapValid and RSA warning targeting."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.envs.benchmark_generator import generate_benchmark_map
from src.agents.planner_astar import bounded_astar
from src.teachers.rsa_warning import _build_region_masks

print("=" * 80)
print("  DIAGNOSTIC: PlanningTrap TrapValid + SemanticTrap Warning Regions")
print("=" * 80)

# ── 1. PlanningTrap: Check if bounded A* can find path through unlocked door ──
print("\n── PlanningTrap: TrapValid check ──")
for diff in ["easy", "medium", "hard"]:
    for seed in range(5):
        gm, cfg = generate_benchmark_map("planning_trap", seed * 1000, diff)
        H, W = gm.height, gm.width
        passable = gm.cell_types != 1  # not WALL
        # Door cells are not passable (cell_type = LOCKED_DOOR = 4)
        for dr, dc in gm.door_positions:
            passable[dr, dc] = False

        # Plan WITHOUT unlock (learner's budget)
        risk_mean = np.full((H, W), 0.3)  # PlanningTrap prior
        cost_mean = np.where(passable, 1.0, np.inf)
        cost_var = np.full((H, W), 0.05)
        
        wait_plan = bounded_astar(
            gm.agent_start, gm.object_spawn,
            cost_mean, risk_mean, cost_var,
            budget=cfg.search_budget, lambda_risk=3.0,
            passable_mask=passable,
        )
        wait_reaches = bool(wait_plan and wait_plan[-1] == gm.object_spawn)
        
        # Plan WITH unlock (same budget!)
        unlock_passable = passable.copy()
        unlock_cost = cost_mean.copy()
        for dr, dc in gm.door_positions:
            unlock_passable[dr, dc] = True
            unlock_cost[dr, dc] = 1.0
        
        unlock_plan = bounded_astar(
            gm.agent_start, gm.object_spawn,
            unlock_cost, risk_mean, cost_var,
            budget=cfg.search_budget, lambda_risk=3.0,
            passable_mask=unlock_passable,
        )
        unlock_reaches = bool(unlock_plan and unlock_plan[-1] == gm.object_spawn)
        
        # Also check with budget=30 (what cause_scoring uses)
        unlock_plan_30 = bounded_astar(
            gm.agent_start, gm.object_spawn,
            unlock_cost, risk_mean, cost_var,
            budget=30, lambda_risk=3.0,
            passable_mask=unlock_passable,
        )
        unlock_reaches_30 = bool(unlock_plan_30 and unlock_plan_30[-1] == gm.object_spawn)
        
        trap_valid = (not wait_reaches) and unlock_reaches
        tag = "✓ VALID" if trap_valid else "✗ INVALID"
        trap_valid_30 = (not wait_reaches) and unlock_reaches_30
        tag_30 = "✓" if trap_valid_30 else "✗"
        
        print(f"  {diff:6s} seed={seed}: budget={cfg.search_budget:2d}  "
              f"wait_reaches={wait_reaches}  unlock_reaches(η)={unlock_reaches}  "
              f"unlock_reaches(30)={unlock_reaches_30}  "
              f"TrapValid(η)={tag}  TrapValid(30)={tag_30}")

# ── 2. SemanticTrap: What region does RSA pick and what's its true risk? ──
print("\n── SemanticTrap: RSA warning region analysis ──")
from src.teachers.rsa_warning import select_best_warning, score_utterances
from src.agents.belief import BeliefMap

for diff in ["easy", "medium", "hard"]:
    for seed in range(3):
        gm, cfg = generate_benchmark_map("semantic_trap", seed * 1000, diff)
        H, W = gm.height, gm.width
        
        # Create learner's initial belief (prior)
        risk_mean = np.full((H, W), cfg.prior_risk_mean)
        risk_var = np.full((H, W), cfg.prior_risk_var)
        
        # Score utterances
        scores = score_utterances(
            risk_mean, risk_var, gm.true_risk, gm.agent_start,
        )
        
        best_utt = max(scores, key=scores.get)
        masks = _build_region_masks(H, W)
        
        print(f"\n  {diff:6s} seed={seed}: best_utt={best_utt}")
        for utt in ["LEFT_RISKY", "RIGHT_RISKY", "UPPER_RISKY", "LOWER_RISKY"]:
            mask = masks[utt]
            mean_risk = float(gm.true_risk[mask].mean())
            n_risky = int((gm.true_risk[mask] > 0.1).sum())
            total = int(mask.sum())
            print(f"    {utt:22s} mean_risk={mean_risk:.3f}  "
                  f"risky_cells={n_risky}/{total}  score={scores[utt]:.3f}")
        
        # Show actual risky cells
        risky_pos = list(zip(*np.where(gm.true_risk > 0.1)))
        print(f"    TRUE risky cells: {risky_pos}")

print("\n── Key finding: rho_warn_threshold=0.15 comparison ──")
# Check what threshold would make WarnP nonzero
for diff in ["easy", "medium", "hard"]:
    gm, cfg = generate_benchmark_map("semantic_trap", 0, diff)
    masks = _build_region_masks(gm.height, gm.width)
    best_utt, _ = select_best_warning(
        np.full((10,10), 0.1), np.full((10,10), 0.25),
        gm.true_risk, gm.agent_start,
    )
    if best_utt in masks:
        m = masks[best_utt]
        mean_r = float(gm.true_risk[m].mean())
        print(f"  {diff}: best_utt={best_utt}  region_mean_risk={mean_r:.4f}  "
              f"threshold=0.15  pass={mean_r >= 0.15}")
