"""Diagnostic: check cause scores on PlanningTrap medium/hard."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.envs.benchmark_generator import generate_benchmark_map
from src.agents.planner_astar import bounded_astar
from src.teachers.cause_scoring import compute_cause_scores, compute_success_prob, compute_survival_prob
from src.agents.belief import BeliefMap, apply_rsa_warning
from src.teachers.rsa_warning import select_best_warning, _build_region_masks

for diff in ["easy", "medium", "hard"]:
    gm, cfg = generate_benchmark_map("planning_trap", 0, diff)
    H, W = 10, 10
    p = gm.cell_types != 1
    for d in gm.door_positions:
        p[d] = False
    ld = set(gm.door_positions)

    bm = BeliefMap.from_prior(H, W,
        prior_risk_mean=cfg.prior_risk_mean, prior_risk_var=cfg.prior_risk_var)
    bm.cost_mean[:] = np.where(p, 1.0, np.inf)

    best_utt, _ = select_best_warning(bm.risk_mean, bm.risk_var, gm.true_risk, (0, 0))
    sim_bw = bm.copy()
    apply_rsa_warning(sim_bw, best_utt, 0.5)
    masks = _build_region_masks(H, W)
    wrm = masks.get(best_utt)

    scores = compute_cause_scores(
        est_belief=bm, agent_pos=(0, 0), goal=(5, 9),
        true_risk=gm.true_risk, true_cost=gm.true_cost,
        time_left=cfg.max_steps, risk_budget_left=cfg.risk_budget,
        passable_mask=p, locked_doors=ld, door_positions=gm.door_positions,
        best_utterance=best_utt, sim_belief_warn=sim_bw, warn_region_mask=wrm,
    )
    dom = scores.dominant_cause(3.0)

    wp = bounded_astar((0,0), (5,9), bm.cost_mean, bm.risk_mean, bm.cost_var,
                        budget=30, lambda_risk=3.0, passable_mask=p) or []
    wr = bool(wp and wp[-1] == (5, 9))
    surv = compute_survival_prob(wp, gm.true_risk) if wp else 0.0

    # Unlock path
    up = p.copy()
    uc = bm.cost_mean.copy()
    for d in gm.door_positions:
        up[d] = True
        uc[d] = 1.0
    ulp = bounded_astar((0,0), (5,9), uc, bm.risk_mean, bm.cost_var,
                         budget=30, lambda_risk=3.0, passable_mask=up) or []
    ur = bool(ulp and ulp[-1] == (5, 9))
    usurv = compute_survival_prob(ulp, gm.true_risk) if ulp else 0.0

    risky = gm.true_risk[gm.true_risk > 0.1]
    print(f"{diff}: dominant={dom}")
    print(f"  S_exp={scores.explore:.4f}  S_bel={scores.belief:.4f}  "
          f"S_plan={scores.plan:.4f}  S_haz={scores.hazard:.4f}")
    print(f"  wait: reach={wr} len={len(wp)-1 if wp else 0} survival={surv:.4f}")
    print(f"  unlock: reach={ur} len={len(ulp)-1 if ulp else 0} survival={usurv:.4f}")
    print(f"  risk_budget={cfg.risk_budget}  max_steps={cfg.max_steps}")
    print(f"  risky_vals={risky.tolist() if len(risky) else 'none'}")
    print()
