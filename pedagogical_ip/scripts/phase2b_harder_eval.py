"""
Phase 2B — Basis + SlowFast evaluation on harder baseline_v2.

Task B: Verify basis_fresh < 1.0 AND basis > linear.
Task C: If Task B passes, run α sweep with SlowFast + basis.

Usage:
    python scripts/phase2b_harder_eval.py [--episodes 15] [--seeds 5] [--smoke]
    python scripts/phase2b_harder_eval.py --alpha-sweep --episodes 15 --seeds 5
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.cost_risk_model import LatentCostRiskHead, generate_world_weights
from src.agents.structured_basis_head import StructuredBasisCostRiskHead
from src.agents.slow_fast_head import GenericSlowFastPredictor, SlowFastCostRiskHead
from src.agents.predictor_protocol import predictor_summary


FAMILY = "harder_baseline_v2"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2B: harder baseline eval")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--alpha-sweep", action="store_true",
                   help="Run Task C α sweep (requires Task B to pass)")
    p.add_argument("--output-dir", default="results/phase2b")
    return p.parse_args()


def make_predictor(head_type, alpha=None):
    if head_type == "linear":
        return LatentCostRiskHead(d=4)
    elif head_type == "basis":
        return StructuredBasisCostRiskHead(d=4)
    elif head_type == "slowfast_linear":
        return SlowFastCostRiskHead(d=4, alpha=alpha or 0.1)
    elif head_type == "slowfast_basis":
        return GenericSlowFastPredictor(
            base_factory=lambda: StructuredBasisCostRiskHead(d=4),
            alpha=alpha or 0.1)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")


def run_episode(runner, ep_seed, predictor, session_ww=None):
    """Run one episode on harder_baseline_v2 using native scenario dispatch.
    
    If session_ww is provided, overrides WorldWeights for session-shared regime.
    """
    ucfg = {}
    if session_ww is not None:
        ucfg['world_weights_override'] = session_ww
    
    state = runner.reset(
        seed=ep_seed, latent_mode=True, latent_predictor=predictor,
        tutor_mode="none", warning_mode="none", patch_radius=2,
        prefix_horizon=5, belief_planning_mode=True,
        robot_belief_mode=True, intervention_family_mode=True,
        item_drop_enabled=True, difficulty="medium",
        scenario_family=FAMILY,
        user_cfg=ucfg if ucfg else None,
    )

    while not state.done:
        state = runner.step(state)

    lp = state.latent_predictor
    ps = predictor_summary(lp) if lp else {}
    return {
        "survived": state.survived,
        "reached_goal": state.reached_goal,
        "steps": state.steps,
        "risk_w_norm": ps.get("risk_w_norm", 0),
        "n_updates": ps.get("n_updates", 0),
        "t_max": state.t_max,
    }


def run_sequence(runner, session_seed, n_episodes, head_type, persist, alpha=None):
    """Run a multi-episode sequence (fresh or persist).
    
    Session-shared WorldWeights: all episodes in a session share the same
    feature→risk mapping. This is REQUIRED for transfer to be meaningful.
    """
    rng = np.random.default_rng(session_seed)
    ep_seeds = rng.integers(0, 100000, size=n_episodes).tolist()
    
    # Generate session-level WorldWeights (shared across all episodes)
    ww_rng = np.random.default_rng(session_seed * 7 + 42)
    session_ww = generate_world_weights(ww_rng, d=4)

    results = []
    predictor = make_predictor(head_type, alpha) if persist else None

    for ep_idx, ep_seed in enumerate(ep_seeds):
        if not persist:
            predictor = make_predictor(head_type, alpha)

        # SlowFast: reset fast delta for new episode
        if persist and hasattr(predictor, 'begin_episode'):
            predictor.begin_episode()

        m = run_episode(runner, ep_seed, predictor, session_ww=session_ww)
        m["ep_idx"] = ep_idx
        results.append(m)

        # For SlowFast: end-of-episode slow update
        if persist and hasattr(predictor, 'end_episode'):
            predictor.end_episode()

    return results


def compute_summary(all_results, cond_name, n_ep):
    """Compute overall + early + late summary."""
    all_ep = [d for seq in all_results for d in seq]
    if not all_ep:
        return {}

    surv = np.mean([d["survived"] for d in all_ep])
    goal = np.mean([d["reached_goal"] for d in all_ep])
    steps = np.mean([d["steps"] for d in all_ep])

    early = [d for d in all_ep if d["ep_idx"] < 4]
    late = [d for d in all_ep if d["ep_idx"] >= max(1, n_ep // 2)]

    early_surv = np.mean([d["survived"] for d in early]) if early else 0
    late_surv = np.mean([d["survived"] for d in late]) if late else 0

    return {
        "cond": cond_name,
        "surv": surv, "goal": goal, "steps": steps,
        "early_surv": early_surv, "late_surv": late_surv,
        "n": len(all_ep),
    }


def main():
    args = parse_args()
    n_ep = 3 if args.smoke else args.episodes
    n_seeds = 2 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    lines = []
    lines.append("Phase 2B: Harder Baseline Evaluation")
    lines.append(f"  episodes={n_ep}, seeds={n_seeds}, family={FAMILY}")
    lines.append("=" * 80)

    # ═══ Task B: Basis-only sanity check ═══
    task_b_conditions = [
        ("linear_fresh",   "linear", False, None),
        ("linear_persist", "linear", True,  None),
        ("basis_fresh",    "basis",  False, None),
        ("basis_persist",  "basis",  True,  None),
    ]

    summaries = {}
    for cond_name, head_type, persist, alpha in task_b_conditions:
        cond_results = []
        for s in range(n_seeds):
            seq = run_sequence(runner, s * 1000, n_ep, head_type, persist, alpha)
            cond_results.append(seq)
        sm = compute_summary(cond_results, cond_name, n_ep)
        summaries[cond_name] = sm

        lines.append(f"\n--- {cond_name} ---")
        for ep_idx in range(n_ep):
            ep_data = [seq[ep_idx] for seq in cond_results if ep_idx < len(seq)]
            if ep_data:
                s_rate = np.mean([d["survived"] for d in ep_data])
                g_rate = np.mean([d["reached_goal"] for d in ep_data])
                rw = np.mean([d["risk_w_norm"] for d in ep_data])
                t_max_avg = np.mean([d["t_max"] for d in ep_data])
                lines.append(f"  ep={ep_idx}: surv={s_rate:.3f} goal={g_rate:.3f} "
                             f"risk_w={rw:.3f} t_max={t_max_avg:.0f}")

    # ═══ Task C: α sweep (conditional) ═══
    if args.alpha_sweep:
        alphas = [0.1, 0.2, 0.3, 0.5]
        for alpha in alphas:
            cond_name = f"slowfast_basis_{alpha}"
            cond_results = []
            for s in range(n_seeds):
                seq = run_sequence(runner, s * 1000, n_ep,
                                   "slowfast_basis", True, alpha=alpha)
                cond_results.append(seq)
            sm = compute_summary(cond_results, cond_name, n_ep)
            summaries[cond_name] = sm

            lines.append(f"\n--- {cond_name} ---")
            for ep_idx in range(n_ep):
                ep_data = [seq[ep_idx] for seq in cond_results if ep_idx < len(seq)]
                if ep_data:
                    s_rate = np.mean([d["survived"] for d in ep_data])
                    g_rate = np.mean([d["reached_goal"] for d in ep_data])
                    lines.append(f"  ep={ep_idx}: surv={s_rate:.3f} goal={g_rate:.3f}")

    # ═══ Summary ═══
    lines.append(f"\n{'='*80}")
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    hdr = f"{'Condition':>25s} | {'Surv':>6s} | {'Goal':>6s} | {'Early':>6s} | {'Late':>6s} | {'N':>4s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for cond_name, sm in summaries.items():
        if sm:
            lines.append(f"{cond_name:>25s} | {sm['surv']:>6.3f} | {sm['goal']:>6.3f} | "
                         f"{sm['early_surv']:>6.3f} | {sm['late_surv']:>6.3f} | {sm['n']:>4d}")

    # ═══ Verdict ═══
    lines.append(f"\n{'='*80}")
    lines.append("TASK B VERDICT")
    lines.append("=" * 80)

    lf = summaries.get("linear_fresh", {}).get("surv", 0)
    lp = summaries.get("linear_persist", {}).get("surv", 0)
    bf = summaries.get("basis_fresh", {}).get("surv", 0)
    bp = summaries.get("basis_persist", {}).get("surv", 0)

    lines.append(f"  linear_fresh  = {lf:.3f}")
    lines.append(f"  linear_persist= {lp:.3f}")
    lines.append(f"  basis_fresh   = {bf:.3f}")
    lines.append(f"  basis_persist = {bp:.3f}")
    lines.append("")

    q1_pass = bf > lf + 0.02
    q2_pass = bf < 0.95
    q3_pass = bp > bf + 0.02

    lines.append(f"Q1: basis > linear? {'YES' if q1_pass else 'NO'} "
                 f"(delta={bf-lf:+.3f})")
    lines.append(f"Q2: basis_fresh < 1.0? {'YES' if q2_pass else 'NO'} "
                 f"({bf:.3f})")
    lines.append(f"Q3: basis_persist > basis_fresh? "
                 f"{'YES' if q3_pass else 'NO'} (delta={bp-bf:+.3f})")

    if q1_pass and q2_pass:
        lines.append("\n>>> Harder baseline QUALIFIES for Task C (alpha sweep)")
    elif not q2_pass:
        lines.append("\n>>> basis_fresh still at ceiling — need harder params")
    else:
        lines.append("\n>>> basis not outperforming linear — check feature distributions")

    # α sweep verdict
    if args.alpha_sweep:
        lines.append(f"\n{'='*80}")
        lines.append("TASK C: ALPHA SWEEP")
        lines.append("=" * 80)
        for alpha in [0.1, 0.2, 0.3, 0.5]:
            sf = summaries.get(f"slowfast_basis_{alpha}", {})
            if sf:
                ea = sf['early_surv'] - bf
                la = sf['late_surv'] - bf
                lines.append(f"  alpha={alpha}: surv={sf['surv']:.3f} "
                             f"early={sf['early_surv']:.3f}(+{ea:.3f}) "
                             f"late={sf['late_surv']:.3f}(+{la:.3f})")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "harder_baseline_eval.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
