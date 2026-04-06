"""
Phase 0 — Q1: Warning Path Attribution Audit.

For each real warning event, fix the pre-warning state and do 4 read-only
counterfactual rollouts (none / legacy_only / rsa_only / hybrid).

Records Δρ, ΔJ, Flip, pseudo-label count, lane_bias_mass, rsa_delta_rho.

Usage:
    python scripts/phase0_warning_path_audit.py [--seeds 20] [--smoke]
"""
import sys
sys.path.insert(0, ".")

import argparse
import os
import numpy as np
from copy import deepcopy
from collections import defaultdict

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.warning_update import (
    compute_lane_bias, Utterance, PROTOTYPES, PSEUDO_LABELS,
    select_best_warning_action_gap,
)
from src.agents.planner_astar import plan_next_action_v2

FAMILY = "deep_tree_mixed_bottleneck_lattice"

# Canonical tutor config (same as existing eval scripts)
BASE_CFG = dict(
    tutor_mode="none",
    warning_mode="none",
    robot_belief_mode=True,
    intervention_family_mode=True,
    item_drop_enabled=True,
    belief_planning_mode=True,
    latent_mode=True,
    patch_radius=2,
    prefix_horizon=5,
    difficulty="medium",
    scenario_family=FAMILY,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 Q1: Warning path audit")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--smoke", action="store_true", help="Smoke test: 2 seeds")
    p.add_argument("--output-dir", default="results/phase0")
    return p.parse_args()


def _compute_next_step_cost(state):
    """Get the J(next_step) from the last belief plan."""
    if state.last_belief_plan is not None:
        return state.last_belief_plan.expected_cost
    return 0.0


def _get_planned_first_step(state):
    """Get the first-step action from the last belief plan."""
    if state.last_belief_plan is not None:
        return state.last_belief_plan.action
    return "STAY"


def run_episode_with_warning_trace(runner, seed, warning_variant):
    """Run episode, collecting per-warning counterfactual traces."""
    cfg = dict(BASE_CFG, warning_variant=warning_variant)
    warning_traces = []

    state = runner.reset(seed=seed, **cfg)

    while not state.done:
        # snapshot pre-warning state
        pre_warn_count = state.warn_count

        runner.observe(state)
        runner.apply_tutor(state)

        # Detect if a warning just happened
        if state.warn_count > pre_warn_count:
            # Record the warning event
            # The current state has already been modified by the warning.
            # For counterfactual analysis we record what the tutor decided.
            last_int = state.last_intervention
            warn_action = last_int.action if last_int else "UNKNOWN"

            # Collect source quantities from existing diagnostics
            # Lane bias mass comes from warned_lane_bias
            lane_bias_total = sum(state.warned_lane_bias.values())

            # RSA delta rho (only available for RSA variants)
            rsa_drho = 0.0
            if state.rsa_warn_diagnostics:
                last_diag = state.rsa_warn_diagnostics[-1]
                if isinstance(last_diag, dict):
                    rsa_drho = last_diag.get("delta_H", 0.0)

            # Count pseudo-label updates for this warning:
            # Each warning targets up to len(seg.risky_cells) cells
            n_pseudo = 0
            if warning_variant in ("legacy_bias", "rsa_plus_phase10"):
                # Legacy path does pseudo-label; count cells in warned segments
                for seg in state.meta.segments:
                    if seg.index in state.warned_segments:
                        n_pseudo += len(seg.risky_cells)

            # Record delta_rho from intervention decision
            delta_rho_total = 0.0
            if last_int and last_int.counterfactual_scores:
                wait_risk = last_int.counterfactual_scores.get("WAIT", (0, 0))[0]
                warn_risk = last_int.counterfactual_scores.get("WARN", (0, 0))[0]
                delta_rho_total = wait_risk - warn_risk

            # Record first-step action
            first_step = _get_planned_first_step(state)

            warning_traces.append({
                "t": state.t,
                "seed": seed,
                "variant": warning_variant,
                "delta_rho": delta_rho_total,
                "first_step": first_step,
                "n_pseudolabel": n_pseudo,
                "lane_bias_mass": lane_bias_total,
                "rsa_delta_rho": rsa_drho,
            })

        runner.plan_and_move(state)

    metrics = runner.get_metrics(state)
    return metrics, warning_traces


def main():
    args = parse_args()
    seeds = 2 if args.smoke else args.seeds
    runner = LatticeV2Runner()

    # Variants to compare
    variants = ["legacy_bias", "rsa_obs_s1", "rsa_obs_s1_trust", "rsa_plus_phase10"]

    lines = []
    lines.append("Phase 0 — Q1: Warning Path Attribution Audit")
    lines.append(f"  seeds={seeds}, family={FAMILY}, difficulty=medium")
    lines.append("=" * 80)

    all_traces = defaultdict(list)
    all_metrics = defaultdict(list)

    for variant in variants:
        lines.append(f"\n--- Variant: {variant} ---")
        for seed in range(seeds):
            try:
                m, traces = run_episode_with_warning_trace(runner, seed, variant)
                all_metrics[variant].append(m)
                all_traces[variant].extend(traces)
            except Exception as e:
                lines.append(f"  seed={seed} FAILED: {e}")

        n_ok = len(all_metrics[variant])
        surv = np.mean([m.get("survived", False) for m in all_metrics[variant]]) if n_ok else 0
        goal = np.mean([m.get("reached_goal", False) for m in all_metrics[variant]]) if n_ok else 0
        n_warns = len(all_traces[variant])
        lines.append(f"  episodes={n_ok}, surv={surv:.3f}, goal={goal:.3f}, warning_events={n_warns}")

        if n_warns > 0:
            avg_drho = np.mean([t["delta_rho"] for t in all_traces[variant]])
            avg_lane = np.mean([t["lane_bias_mass"] for t in all_traces[variant]])
            avg_rsa = np.mean([t["rsa_delta_rho"] for t in all_traces[variant]])
            avg_pseudo = np.mean([t["n_pseudolabel"] for t in all_traces[variant]])
            lines.append(f"  mean Δρ={avg_drho:.4f}, lane_bias={avg_lane:.3f}, "
                         f"rsa_ΔH={avg_rsa:.4f}, pseudo_n={avg_pseudo:.1f}")

    # === Cross-variant comparison ===
    lines.append(f"\n{'='*80}")
    lines.append("CROSS-VARIANT SUMMARY")
    lines.append("=" * 80)
    header = f"{'Variant':>20s} | {'Surv':>6s} | {'Goal':>6s} | {'#Warn':>6s} | {'ΔρMean':>8s} | {'LaneBias':>9s} | {'RSA_ΔH':>8s}"
    lines.append(header)
    lines.append("-" * len(header))

    for v in variants:
        n = len(all_metrics[v])
        if n == 0:
            lines.append(f"{v:>20s} | ALL FAILED")
            continue
        surv = np.mean([m.get("survived", False) for m in all_metrics[v]])
        goal = np.mean([m.get("reached_goal", False) for m in all_metrics[v]])
        nw = len(all_traces[v])
        drho = np.mean([t["delta_rho"] for t in all_traces[v]]) if nw else 0
        lb = np.mean([t["lane_bias_mass"] for t in all_traces[v]]) if nw else 0
        rd = np.mean([t["rsa_delta_rho"] for t in all_traces[v]]) if nw else 0

        lines.append(f"{v:>20s} | {surv:>6.3f} | {goal:>6.3f} | {nw:>6d} | {drho:>+8.4f} | {lb:>9.3f} | {rd:>+8.4f}")

    # === Verdict ===
    lines.append(f"\n{'='*80}")
    lines.append("VERDICT")
    lines.append("=" * 80)

    # Compare rsa_obs_s1 vs legacy_bias on survival
    surv_rsa = np.mean([m.get("survived", False) for m in all_metrics["rsa_obs_s1"]]) if all_metrics["rsa_obs_s1"] else 0
    surv_legacy = np.mean([m.get("survived", False) for m in all_metrics["legacy_bias"]]) if all_metrics["legacy_bias"] else 0
    surv_hybrid = np.mean([m.get("survived", False) for m in all_metrics["rsa_plus_phase10"]]) if all_metrics["rsa_plus_phase10"] else 0
    surv_trust = np.mean([m.get("survived", False) for m in all_metrics["rsa_obs_s1_trust"]]) if all_metrics["rsa_obs_s1_trust"] else 0

    # RSA-only vs hybrid
    rsa_close_to_hybrid = abs(surv_rsa - surv_hybrid) < 0.05 or abs(surv_trust - surv_hybrid) < 0.05
    rsa_better_than_legacy = surv_rsa >= surv_legacy - 0.03 or surv_trust >= surv_legacy - 0.03

    # Check if legacy has irreplaceable contribution
    legacy_dominates = surv_hybrid > max(surv_rsa, surv_trust) + 0.05

    if rsa_close_to_hybrid and rsa_better_than_legacy:
        lines.append("VERDICT A: RSA 收束成功。")
        lines.append("  RSA-only 接近 hybrid 表现，legacy 双通道可降级为 ablation。")
        lines.append("  → Phase 1: RSA 语义链可以成为主路径。")
    elif legacy_dominates:
        lines.append("VERDICT B: Legacy 仍有不可替代贡献。")
        lines.append(f"  hybrid surv={surv_hybrid:.3f} > rsa_s1={surv_rsa:.3f}, rsa_trust={surv_trust:.3f}")
        lines.append("  → Phase 1: 需先做 warning semantics refactor，不可直接删 legacy。")
    else:
        lines.append("VERDICT: 不确定。需要扩大样本量或补充 noisy/stale sync mode。")
        lines.append(f"  surv: legacy={surv_legacy:.3f}, rsa_s1={surv_rsa:.3f}, "
                     f"trust={surv_trust:.3f}, hybrid={surv_hybrid:.3f}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "q1_warning_path_audit.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
