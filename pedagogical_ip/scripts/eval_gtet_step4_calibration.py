"""GTET-L Step 4 — Baseline & Calibration Sweep.

Work Packages A+B: no_tutor / no_warn baselines + sensitivity calibration.

Baselines:
  - FULL: canonical tutor with full posterior
  - no_tutor: no interventions at all
  - no_warn: UNLOCK + ITEM_DROP only, no WARN
  - G_ONLY: factor-restricted (worst ablation from Step 3b)

Calibration sweep:
  - deadline_slack_final: controls deadline pressure
  - belt_risk: risk per belt cell
  - terminal_belt_fraction: fraction of stage-3 that is belt
"""
import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner

FAMILY = "goal_preference_temptation_entanglement_lattice"


def run_episode(runner, seed, difficulty, factor_mode="FULL",
                tutor_mode_override=None, allowed_override=None,
                user_cfg=None):
    """Run one GTET-L episode."""
    try:
        kw = dict(
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
            user_cfg=user_cfg,
        )
        if allowed_override is not None:
            kw["allowed_interventions"] = allowed_override

        s = runner.reset(**kw)

        while not s.done:
            s = runner.step(s)

        m = runner.get_metrics(s)
        return {
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "steps": int(m["steps"]),
            "warn_count": int(m.get("warnings", 0)),
            "unlock_count": int(m.get("unlocks", 0)),
            "success": True,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "survived": False, "reached_goal": False,
            "steps": 0, "warn_count": 0, "unlock_count": 0,
            "success": False, "error": str(e)[:200],
        }


def run_no_tutor(runner, seed, difficulty, user_cfg=None):
    """Run with no interventions (tutor_mode=none, no robot belief)."""
    try:
        s = runner.reset(
            seed=seed, difficulty=difficulty,
            scenario_family=FAMILY,
            robot_belief_mode=False,
            intervention_family_mode=False,
            item_drop_enabled=False,
            belief_planning_mode=True,
            latent_mode=True,
            patch_radius=2,
            prefix_horizon=5,
            factor_mode="FULL",
            user_cfg=user_cfg,
        )
        while not s.done:
            s = runner.step(s)
        m = runner.get_metrics(s)
        return {
            "survived": bool(m["survived"]),
            "reached_goal": bool(m["reached_goal"]),
            "steps": int(m["steps"]),
            "warn_count": 0, "unlock_count": 0,
            "success": True,
        }
    except Exception as e:
        return {
            "survived": False, "reached_goal": False,
            "steps": 0, "warn_count": 0, "unlock_count": 0,
            "success": False, "error": str(e)[:200],
        }


def summarize(label, recs):
    n = len([r for r in recs if r["success"]])
    if n == 0:
        return f"  {label:20s}: n=0 (all errors)"
    surv = np.mean([r["survived"] for r in recs if r["success"]])
    goal = np.mean([r["reached_goal"] for r in recs if r["success"]])
    warn = np.mean([r["warn_count"] for r in recs if r["success"]])
    unlock = np.mean([r["unlock_count"] for r in recs if r["success"]])
    errs = len(recs) - n
    return (f"  {label:20s}: surv={surv:.3f} goal={goal:.3f} "
            f"warn={warn:.1f} unlock={unlock:.1f} n={n} err={errs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--mode", choices=["baselines", "sweep", "both"],
                        default="both")
    args = parser.parse_args()
    runner = LatticeV2Runner()

    # ═══════════════════════════════════════════════════
    # Work Package B: Baselines on current hard
    # ═══════════════════════════════════════════════════
    if args.mode in ("baselines", "both"):
        print("=" * 70)
        print("Work Package B: Baselines on current GTET-L hard")
        print("=" * 70)

        policies = {
            "FULL": {},
            "G_ONLY": {"factor_mode": "G_ONLY"},
            "no_warn": {"allowed_override": frozenset({"WAIT", "UNLOCK", "ITEM_DROP"})},
            "no_tutor": {"no_tutor": True},
        }
        results_b = {}
        for label, cfg in policies.items():
            recs = []
            for seed in range(args.seeds):
                if cfg.get("no_tutor"):
                    rec = run_no_tutor(runner, seed, "hard")
                else:
                    rec = run_episode(
                        runner, seed, "hard",
                        factor_mode=cfg.get("factor_mode", "FULL"),
                        allowed_override=cfg.get("allowed_override"),
                    )
                recs.append(rec)
            results_b[label] = recs
            print(summarize(label, recs))

        print()

    # ═══════════════════════════════════════════════════
    # Work Package A: Calibration sweep
    # ═══════════════════════════════════════════════════
    if args.mode in ("sweep", "both"):
        print("=" * 70)
        print("Work Package A: GTET-L hard_v2 calibration sweep")
        print("=" * 70)

        # Sweep parameters for final-stage sensitivity
        sweep_configs = [
            # (label, user_cfg overrides)
            ("baseline",          {}),
            ("deadline_tight",    {"deadline_slack_final": 1.05}),
            ("deadline_tighter",  {"deadline_slack_final": 1.00}),
            ("belt_risk_0.4",     {"belt_risk": 0.40}),
            ("belt_risk_0.5",     {"belt_risk": 0.50}),
            ("belt_risk_0.6",     {"belt_risk": 0.60}),
            ("belt_frac_0.3",     {"terminal_belt_fraction": 0.30}),
            ("belt_frac_0.4",     {"terminal_belt_fraction": 0.40}),
            ("tight+risk_0.4",    {"deadline_slack_final": 1.05, "belt_risk": 0.40}),
            ("tight+risk_0.5",    {"deadline_slack_final": 1.05, "belt_risk": 0.50}),
            ("tighter+risk_0.4",  {"deadline_slack_final": 1.00, "belt_risk": 0.40}),
        ]

        for sweep_label, ucfg in sweep_configs:
            print(f"\n--- {sweep_label} ---")
            for policy_label, policy_cfg in [
                ("FULL",     {}),
                ("G_ONLY",   {"factor_mode": "G_ONLY"}),
                ("no_tutor", {"no_tutor": True}),
            ]:
                recs = []
                for seed in range(args.seeds):
                    if policy_cfg.get("no_tutor"):
                        rec = run_no_tutor(runner, seed, "hard",
                                           user_cfg=ucfg if ucfg else None)
                    else:
                        rec = run_episode(
                            runner, seed, "hard",
                            factor_mode=policy_cfg.get("factor_mode", "FULL"),
                            user_cfg=ucfg if ucfg else None,
                        )
                    recs.append(rec)
                print(summarize(policy_label, recs))

    print("\nDone.")


if __name__ == "__main__":
    main()
