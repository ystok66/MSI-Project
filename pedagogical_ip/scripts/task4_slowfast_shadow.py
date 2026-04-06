"""
Task 4 — SlowFast Transfer Shadow Evaluation.

Runs multi-episode sequences comparing:
  1. canonical: fresh LatentCostRiskHead each episode (no transfer)
  2. persist: same LatentCostRiskHead across episodes (current PRS)
  3. slowfast: SlowFastCostRiskHead with α=0.1

Measures whether slow-fast prior gives early-episode advantage.

Usage:
    python scripts/task4_slowfast_shadow.py [--episodes 10] [--seeds 5] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from copy import deepcopy

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.slow_fast_head import SlowFastCostRiskHead


def parse_args():
    p = argparse.ArgumentParser(description="Task 4: SlowFast transfer shadow")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="3 episodes, 2 seeds")
    p.add_argument("--family", default="baseline_v2",
                   choices=["baseline_v2", "GTET", "DTMB"])
    p.add_argument("--output-dir", default="results/task4")
    return p.parse_args()


FAMILY_MAP = {
    "baseline_v2": None,
    "GTET": "goal_preference_temptation_entanglement_lattice",
    "DTMB": "deep_tree_mixed_bottleneck_lattice",
}


def run_episode(runner, seed, family, predictor=None):
    """Run one episode, return metrics + final predictor state."""
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
        scenario_family=family,
    )
    if predictor is not None:
        cfg["latent_predictor"] = predictor

    try:
        state = runner.reset(seed=seed, **cfg)
        while not state.done:
            state = runner.step(state)
        metrics = runner.get_metrics(state)

        # Capture predictor weights for cross-episode analysis
        lp = state.latent_predictor
        w_risk = lp.risk_head.w.copy() if lp else None
        n_upd = lp.n_updates if lp else 0

        return {
            "survived": metrics["survived"],
            "reached_goal": metrics["reached_goal"],
            "steps": metrics["steps"],
            "n_updates": n_upd,
            "risk_w_norm": float(np.linalg.norm(w_risk)) if w_risk is not None else 0,
            "predictor": lp,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "survived": False,
                "reached_goal": False, "predictor": predictor}


def run_sequence(runner, session_seed, family, n_episodes, mode="fresh"):
    """Run a sequence of episodes with specified transfer strategy."""
    rng = np.random.default_rng(session_seed)
    ep_seeds = rng.integers(0, 100000, size=n_episodes).tolist()

    results = []
    predictor = None

    # Initialize predictor based on mode
    if mode == "slowfast":
        sf_head = SlowFastCostRiskHead(d=4, alpha=0.1)
        predictor = sf_head
    elif mode == "persist":
        predictor = LatentCostRiskHead(d=4)

    for ep_idx, ep_seed in enumerate(ep_seeds):
        if mode == "fresh":
            predictor = None  # fresh each episode
        elif mode == "slowfast":
            sf_head.begin_episode()
            predictor = sf_head
        # persist: reuse same predictor

        m = run_episode(runner, ep_seed, family, predictor=predictor)
        m["ep_idx"] = ep_idx
        m["ep_seed"] = ep_seed
        results.append(m)

        if mode == "slowfast" and m["success"]:
            sf_head.end_episode()
        elif mode == "persist" and m["success"]:
            predictor = m["predictor"]

    # Get slowfast diagnostics
    sf_diag = None
    if mode == "slowfast":
        sf_diag = sf_head.get_diagnostics_summary()

    return results, sf_diag


def main():
    args = parse_args()
    n_ep = 3 if args.smoke else args.episodes
    n_seeds = 2 if args.smoke else args.seeds
    family = FAMILY_MAP[args.family]
    runner = LatticeV2Runner()

    modes = ["fresh", "persist", "slowfast"]
    lines = []
    lines.append("Task 4 - SlowFast Transfer Shadow Evaluation")
    lines.append(f"  family={args.family}, episodes={n_ep}, seeds={n_seeds}")
    lines.append("=" * 80)

    all_results = {}

    for mode in modes:
        mode_results = []
        for session_seed in range(n_seeds):
            seq, sf_diag = run_sequence(
                runner, session_seed * 1000, family, n_ep, mode=mode)
            mode_results.append((seq, sf_diag))

        all_results[mode] = mode_results

        # Aggregate per-episode metrics across seeds
        lines.append(f"\n--- Mode: {mode} ---")
        for ep_idx in range(n_ep):
            ep_data = []
            for seq, _ in mode_results:
                if ep_idx < len(seq) and seq[ep_idx].get("success"):
                    ep_data.append(seq[ep_idx])
            if ep_data:
                surv = np.mean([d["survived"] for d in ep_data])
                goal = np.mean([d["reached_goal"] for d in ep_data])
                steps = np.mean([d["steps"] for d in ep_data])
                w_norm = np.mean([d["risk_w_norm"] for d in ep_data])
                lines.append(f"  ep={ep_idx}: surv={surv:.3f} goal={goal:.3f} "
                             f"steps={steps:.1f} risk_w={w_norm:.3f} n={len(ep_data)}")

        # SlowFast diagnostics
        if mode == "slowfast":
            for i, (_, sf_diag) in enumerate(mode_results):
                if sf_diag:
                    lines.append(f"  SF diag (seed={i}): {sf_diag}")

    # Cross-mode comparison
    lines.append(f"\n{'='*80}")
    lines.append("CROSS-MODE COMPARISON (averaged over all episodes)")
    lines.append("=" * 80)

    hdr = f"{'Mode':>10s} | {'Surv':>6s} | {'Goal':>6s} | {'Steps':>6s} | {'risk_w':>7s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for mode in modes:
        all_ep = []
        for seq, _ in all_results[mode]:
            all_ep.extend([d for d in seq if d.get("success")])
        if all_ep:
            surv = np.mean([d["survived"] for d in all_ep])
            goal = np.mean([d["reached_goal"] for d in all_ep])
            steps = np.mean([d["steps"] for d in all_ep])
            rw = np.mean([d["risk_w_norm"] for d in all_ep])
            lines.append(f"{mode:>10s} | {surv:>6.3f} | {goal:>6.3f} | "
                         f"{steps:>6.1f} | {rw:>7.3f}")

    # Transfer benefit analysis: compare late episodes (ep >= n_ep//2)
    lines.append(f"\n{'='*80}")
    lines.append("LATE-EPISODE COMPARISON (ep >= half)")
    lines.append("=" * 80)
    half = max(1, n_ep // 2)

    for mode in modes:
        late_ep = []
        for seq, _ in all_results[mode]:
            late_ep.extend([d for d in seq if d.get("success") and d["ep_idx"] >= half])
        if late_ep:
            surv = np.mean([d["survived"] for d in late_ep])
            goal = np.mean([d["reached_goal"] for d in late_ep])
            lines.append(f"  {mode:>10s}: surv={surv:.3f} goal={goal:.3f} n={len(late_ep)}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "slowfast_shadow.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
