"""
Phase 2A — Experiment 1: Basis-Only Shadow.

Compares 4 modes × 15 episodes × 5 seeds on baseline_v2:
  1. linear_fresh:    LatentCostRiskHead, fresh each episode
  2. linear_persist:  LatentCostRiskHead, persisted across episodes
  3. basis_fresh:     StructuredBasisCostRiskHead, fresh each episode
  4. basis_persist:   StructuredBasisCostRiskHead, persisted across episodes

Key questions:
  Q1: basis_fresh ≥ linear_fresh?  (does basis better capture risk/cost?)
  Q2: (basis_persist - basis_fresh) > (linear_persist - linear_fresh)?
      (does basis make cross-episode prior more valuable?)

Usage:
    python scripts/phase2a_basis_shadow.py [--episodes 15] [--seeds 5] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.structured_basis_head import StructuredBasisCostRiskHead


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2A: Basis-only shadow")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="3 episodes, 2 seeds")
    p.add_argument("--output-dir", default="results/phase2a")
    return p.parse_args()


def make_predictor(head_type):
    if head_type == "linear":
        return LatentCostRiskHead(d=4)
    elif head_type == "basis":
        return StructuredBasisCostRiskHead(d=4)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


def run_episode(runner, seed, predictor=None):
    cfg = dict(
        tutor_mode="none",
        warning_mode="none",
        latent_mode=True,
        patch_radius=2,
        prefix_horizon=5,
        belief_planning_mode=True,
        robot_belief_mode=True,
        intervention_family_mode=True,
        item_drop_enabled=True,
        difficulty="medium",
        scenario_family=None,  # baseline_v2
    )
    if predictor is not None:
        cfg["latent_predictor"] = predictor

    try:
        state = runner.reset(seed=seed, **cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)
        lp = state.latent_predictor
        return {
            "survived": metrics["survived"],
            "reached_goal": metrics["reached_goal"],
            "steps": metrics["steps"],
            "n_updates": lp.n_updates if lp else 0,
            "risk_w_norm": float(np.linalg.norm(lp.risk_head.w)) if lp else 0,
            "predictor": lp,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "survived": False,
                "reached_goal": False, "predictor": predictor}


def run_sequence(runner, session_seed, n_episodes, head_type, persist):
    rng = np.random.default_rng(session_seed)
    ep_seeds = rng.integers(0, 100000, size=n_episodes).tolist()

    results = []
    predictor = make_predictor(head_type) if persist else None

    for ep_idx, ep_seed in enumerate(ep_seeds):
        if not persist:
            predictor = make_predictor(head_type)

        m = run_episode(runner, ep_seed, predictor=predictor)
        m["ep_idx"] = ep_idx
        results.append(m)

        if persist and m["success"]:
            predictor = m["predictor"]

    return results


def main():
    args = parse_args()
    n_ep = 3 if args.smoke else args.episodes
    n_seeds = 2 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    # 4 conditions
    conditions = [
        ("linear_fresh",   "linear", False),
        ("linear_persist", "linear", True),
        ("basis_fresh",    "basis",  False),
        ("basis_persist",  "basis",  True),
    ]

    lines = []
    lines.append("Phase 2A - Basis-Only Shadow Evaluation")
    lines.append(f"  episodes={n_ep}, seeds={n_seeds}")
    lines.append("=" * 80)

    all_results = {}

    for cond_name, head_type, persist in conditions:
        cond_results = []
        for s in range(n_seeds):
            seq = run_sequence(runner, s * 1000, n_ep, head_type, persist)
            cond_results.append(seq)
        all_results[cond_name] = cond_results
        lines.append(f"\n--- {cond_name} ---")
        for ep_idx in range(n_ep):
            ep_data = []
            for seq in cond_results:
                if ep_idx < len(seq) and seq[ep_idx].get("success"):
                    ep_data.append(seq[ep_idx])
            if ep_data:
                surv = np.mean([d["survived"] for d in ep_data])
                goal = np.mean([d["reached_goal"] for d in ep_data])
                steps = np.mean([d["steps"] for d in ep_data])
                rw = np.mean([d["risk_w_norm"] for d in ep_data])
                lines.append(f"  ep={ep_idx}: surv={surv:.3f} goal={goal:.3f} "
                             f"steps={steps:.1f} risk_w={rw:.3f} n={len(ep_data)}")

    # Summary
    lines.append(f"\n{'='*80}")
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    hdr = f"{'Condition':>16s} | {'Surv':>6s} | {'Goal':>6s} | {'Steps':>6s} | {'N':>4s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    summary = {}
    for cond_name, _, _ in conditions:
        all_ep = []
        for seq in all_results[cond_name]:
            all_ep.extend([d for d in seq if d.get("success")])
        if all_ep:
            surv = np.mean([d["survived"] for d in all_ep])
            goal = np.mean([d["reached_goal"] for d in all_ep])
            steps = np.mean([d["steps"] for d in all_ep])
            n = len(all_ep)
            summary[cond_name] = {"surv": surv, "goal": goal, "steps": steps}
            lines.append(f"{cond_name:>16s} | {surv:>6.3f} | {goal:>6.3f} | {steps:>6.1f} | {n:>4d}")

    # Late-episode comparison
    half = max(1, n_ep // 2)
    lines.append(f"\n{'='*80}")
    lines.append(f"LATE-EPISODE COMPARISON (ep >= {half})")
    lines.append("=" * 80)

    late_summary = {}
    for cond_name, _, _ in conditions:
        late = []
        for seq in all_results[cond_name]:
            late.extend([d for d in seq if d.get("success") and d["ep_idx"] >= half])
        if late:
            surv = np.mean([d["survived"] for d in late])
            goal = np.mean([d["reached_goal"] for d in late])
            late_summary[cond_name] = {"surv": surv, "goal": goal, "n": len(late)}
            lines.append(f"  {cond_name:>16s}: surv={surv:.3f} goal={goal:.3f} n={len(late)}")

    # Verdict
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    if summary:
        # Q1: basis_fresh vs linear_fresh
        lf = summary.get("linear_fresh", {}).get("surv", 0)
        bf = summary.get("basis_fresh", {}).get("surv", 0)
        lines.append(f"Q1: basis_fresh ({bf:.3f}) vs linear_fresh ({lf:.3f})")
        if bf >= lf:
            lines.append("  → basis ≥ linear in single-episode: PASS")
        else:
            lines.append(f"  → basis < linear by {lf-bf:.3f}: needs investigation")

        # Q2: transfer bonus comparison
        lp = summary.get("linear_persist", {}).get("surv", 0)
        bp = summary.get("basis_persist", {}).get("surv", 0)
        linear_bonus = lp - lf
        basis_bonus = bp - bf
        lines.append(f"Q2: linear transfer bonus = {linear_bonus:.3f}, "
                     f"basis transfer bonus = {basis_bonus:.3f}")
        if basis_bonus > linear_bonus:
            lines.append("  → basis makes transfer MORE valuable: PASS")
        elif basis_bonus >= linear_bonus - 0.02:
            lines.append("  → approximately equal transfer value: NEUTRAL")
        else:
            lines.append("  → basis reduces transfer value: INVESTIGATE")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "basis_shadow.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
