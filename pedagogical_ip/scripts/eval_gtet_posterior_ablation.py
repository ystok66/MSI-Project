"""GTET-L Posterior Factor Ablation — Real Integration (Exp 1+2+3).

Now factor_mode is wired into the tutor: different factor sets produce
different InterventionConfig weights, leading to different decisions.

Exp 1: Integration sanity — verify action traces differ between FULL and ablated.
Exp 2: ADR measurement — action divergence rate.
Exp 3: Survival/goal comparison + Δ_joint.
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "goal_preference_temptation_entanglement_lattice"

EVAL_FACTORS = ["FULL", "G_THETA", "G_Z", "THETA_Z", "G_ONLY"]


def run_episode(runner, seed, difficulty, factor_mode):
    """Run one GTET-L episode with the specified factor_mode."""
    try:
        s = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY,
            robot_belief_mode=True,
            intervention_family_mode=True,
            item_drop_enabled=True,
            belief_planning_mode=True,
            latent_mode=True,
            patch_radius=2,
            prefix_horizon=5,
            factor_mode=factor_mode,
        )

        interventions = []
        while not s.done:
            s = runner.step(s)
            if s.last_intervention is not None:
                interventions.append(s.last_intervention.action)
            else:
                interventions.append("NONE")

        m = runner.get_metrics(s)
        return {
            "seed": seed,
            "difficulty": difficulty,
            "factor_mode": factor_mode,
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "steps": int(m["steps"]),
            "warn_count": int(m.get("warnings", 0)),
            "unlock_count": int(m.get("unlocks", 0)),
            "interventions": interventions,
            "gtet_log": list(getattr(s, 'gtet_action_log', [])),
            "success": True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "seed": seed, "difficulty": difficulty,
            "factor_mode": factor_mode,
            "survived": False, "reached_goal": False,
            "steps": 0, "warn_count": 0, "unlock_count": 0,
            "interventions": [], "gtet_log": [],
            "success": False,
            "error": str(e)[:200],
        }


def compute_adr(full_interventions, ablated_interventions):
    """Action Divergence Rate: fraction of steps where actions differ."""
    if not full_interventions or not ablated_interventions:
        return 0.0
    n = min(len(full_interventions), len(ablated_interventions))
    if n == 0:
        return 0.0
    disagree = sum(1 for i in range(n)
                   if full_interventions[i] != ablated_interventions[i])
    return disagree / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--difficulty", default="medium")
    args = parser.parse_args()

    runner = LatticeV2Runner()

    print(f"GTET-L Factor Ablation — {args.difficulty}, {args.seeds} seeds")
    print("=" * 65)

    results = {}
    for factor in EVAL_FACTORS:
        results[factor] = []
        for seed in range(args.seeds):
            rec = run_episode(runner, seed, args.difficulty, factor)
            results[factor].append(rec)
            status = "OK" if rec["success"] else "ERR"
            w = rec["warn_count"]
            u = rec["unlock_count"]
            surv = "S" if rec["survived"] else "D"
            print(f"  {factor:10s} seed={seed:2d} {status} {surv} "
                  f"warn={w} unlock={u} steps={rec['steps']}")

    # ── Exp 3: Survival summary ──
    print("\n" + "=" * 65)
    print("Exp 3: Survival & Goal Reach")
    print("-" * 65)
    surv_by = {}
    for factor in EVAL_FACTORS:
        recs = [r for r in results[factor] if r["success"]]
        n = len(recs)
        surv = np.mean([r["survived"] for r in recs]) if n else 0.0
        goal = np.mean([r["reached_goal"] for r in recs]) if n else 0.0
        avg_warn = np.mean([r["warn_count"] for r in recs]) if n else 0.0
        avg_unlock = np.mean([r["unlock_count"] for r in recs]) if n else 0.0
        errs = len([r for r in results[factor] if not r["success"]])
        print(f"  {factor:12s}: surv={surv:.3f} goal={goal:.3f} "
              f"warn={avg_warn:.1f} unlock={avg_unlock:.1f} n={n} err={errs}")
        surv_by[factor] = surv

    # Δ_joint
    full_perf = surv_by.get("FULL", 0.0)
    factored_2way = [surv_by.get(k, 0.0) for k in ["G_THETA", "G_Z", "THETA_Z"]
                     if k in surv_by]
    delta = full_perf - max(factored_2way) if factored_2way else 0.0
    print(f"\nΔ_joint = {delta:+.3f}")
    if delta > 0.01:
        print("→ Full joint posterior shows ADVANTAGE")
    elif delta < -0.01:
        print("→ Factored posterior better (unexpected)")
    else:
        print("→ Approximately equal")

    # ── Exp 2: ADR ──
    print("\n" + "=" * 65)
    print("Exp 2: Action Divergence Rate (ADR)")
    print("-" * 65)
    full_results = results["FULL"]
    for factor in EVAL_FACTORS:
        if factor == "FULL":
            continue
        adr_values = []
        for seed in range(args.seeds):
            full_rec = full_results[seed]
            abl_rec = results[factor][seed]
            if full_rec["success"] and abl_rec["success"]:
                adr = compute_adr(full_rec["interventions"],
                                  abl_rec["interventions"])
                adr_values.append(adr)
        if adr_values:
            mean_adr = np.mean(adr_values)
            max_adr = np.max(adr_values)
            n_nonzero = sum(1 for v in adr_values if v > 0)
            print(f"  {factor:12s}: ADR_mean={mean_adr:.3f} "
                  f"ADR_max={max_adr:.3f} nonzero={n_nonzero}/{len(adr_values)}")
        else:
            print(f"  {factor:12s}: (no valid pairs)")

    # ── Exp 1: Integration sanity ──
    print("\n" + "=" * 65)
    print("Exp 1: Integration Sanity (first seed with action divergence)")
    print("-" * 65)
    found_divergence = False
    for factor in ["G_ONLY", "THETA_ONLY", "G_THETA"]:
        if factor not in results:
            continue
        for seed in range(args.seeds):
            full_rec = full_results[seed]
            abl_rec = results[factor][seed]
            if not (full_rec["success"] and abl_rec["success"]):
                continue
            adr = compute_adr(full_rec["interventions"],
                              abl_rec["interventions"])
            if adr > 0:
                print(f"  Divergence found! seed={seed} factor={factor} ADR={adr:.3f}")
                print(f"    FULL actions[:10]:   {full_rec['interventions'][:10]}")
                print(f"    {factor} actions[:10]: {abl_rec['interventions'][:10]}")
                found_divergence = True
                break
        if found_divergence:
            break
    if not found_divergence:
        print("  WARNING: No action divergence found across all factor modes!")
        print("  → Factor integration may not be reaching tutor decisions.")

    # ── KL/risk diagnostics from GTET log ──
    print("\n" + "=" * 65)
    print("GTET Posterior Diagnostics (factor-restricted)")
    print("-" * 65)
    for factor in EVAL_FACTORS:
        if factor == "FULL":
            continue
        kl_vals = []
        risk_vals = []
        for rec in results[factor]:
            if rec["success"]:
                for entry in rec["gtet_log"]:
                    kl_vals.append(entry["kl_div"])
                    risk_vals.append(entry["risk_bias"])
        if kl_vals:
            print(f"  {factor:12s}: KL_mean={np.mean(kl_vals):.4f} "
                  f"KL_max={np.max(kl_vals):.4f} "
                  f"risk_mean={np.mean(risk_vals):.4f}")
        else:
            print(f"  {factor:12s}: (no log entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
