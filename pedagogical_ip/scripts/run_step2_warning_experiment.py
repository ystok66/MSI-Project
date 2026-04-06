"""Step 2 Warning Experiment — Phase 2A (Execution-only).

Fixed warn schedule; only the internal warning mechanism varies.

Conditions:
  legacy_bias, rsa_obs_l0, rsa_obs_s1, rsa_obs_s1_trust, rsa_plus_phase10

Families:
  fork_trap, elcb_po, baseline_v2

Headline Metrics:
  SBCR, TBSR, M_true_before/after, ΔH, ΔNLL_local

Diagnostic (not formal):
  response_selectivity_gap  (ΔM_true_necessary - ΔM_true_unnecessary)

Usage:
  python scripts/run_step2_warning_experiment.py --seeds 50
  python scripts/run_step2_warning_experiment.py --seeds 10 --smoke
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.lattice_v2_runner import LatticeV2Runner
from src.agents.rsa_warning_channel import N_HYPOTHESES


# ── Configuration ────────────────────────────────────────────────

VARIANTS = [
    "legacy_bias",
    "rsa_obs_l0",
    "rsa_obs_s1",
    "rsa_obs_s1_trust",
    "rsa_plus_phase10",
]

FAMILIES = ["fork_trap", "elcb_po", "baseline_v2"]
DIFFICULTIES = ["easy", "medium", "hard"]


# ── Metrics collection ──────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Per-episode metrics."""
    seed: int
    family: str
    difficulty: str
    variant: str
    # Behavior
    survived: bool = False
    reached_goal: bool = False
    steps: int = 0
    risky_entered: int = 0
    warn_count: int = 0
    # Belief diagnostics
    rsa_n_updates: int = 0
    rsa_entropy_final: float = 0.0
    delta_H: float = 0.0          # entropy drop from first warning
    M_true_before: float = 0.0    # belief mass on true-hazard before
    M_true_after: float = 0.0     # belief mass on true-hazard after
    # Planner adapter
    risk_delta: float = 0.0       # planner risk adjustment from RSA
    # Action prediction
    delta_nll_local: float = 0.0  # NLL improvement from warning
    nll_before: float = 0.0
    nll_after: float = 0.0
    # Safe branch choice rate (SBCR)
    safe_branch_chosen: bool = False


def _infer_true_hypothesis(meta, seg_index=0) -> str:
    """Infer the ground-truth risk hypothesis from segment topology."""
    if seg_index >= len(meta.segments):
        return "both_safe"
    seg = meta.segments[seg_index]
    if not seg.risky_cells:
        return "both_safe"
    if seg.risky_row <= 2:
        return "left"
    else:
        return "right"


def _run_episode(
    variant: str,
    seed: int,
    family: str,
    difficulty: str = "medium",
) -> EpisodeResult:
    """Run one Phase-2A episode and collect metrics."""
    runner = LatticeV2Runner()
    s = runner.reset(
        seed=seed,
        latent_mode=True,
        warning_mode="fixed",
        scenario_family=family,
        warning_variant=variant,
        difficulty=difficulty,
    )
    while not s.done:
        s = runner.step(s)

    m = runner.get_extended_metrics(s)

    result = EpisodeResult(
        seed=seed, family=family, difficulty=difficulty, variant=variant,
        survived=m["survived"],
        reached_goal=m["reached_goal"],
        steps=m["steps"],
        risky_entered=m["risky_entered"],
        warn_count=m["warnings"],
    )

    # SBCR: did the agent take the safe branch?
    # Check if agent entered any risky cell
    result.safe_branch_chosen = (m["risky_entered"] == 0)

    # RSA diagnostics
    if s.rsa_belief_state is not None:
        result.rsa_n_updates = s.rsa_belief_state.n_updates
        result.rsa_entropy_final = s.rsa_belief_state.entropy()

    if s.rsa_warn_diagnostics:
        diag0 = s.rsa_warn_diagnostics[0]
        result.delta_H = diag0.get("delta_H", 0.0)
        result.risk_delta = diag0.get("risk_delta", 0.0)

        # M_true: belief mass on the true-hazard hypothesis
        true_hyp = _infer_true_hypothesis(s.meta, 0)
        hyp_idx_map = {"left": 0, "right": 1, "both_safe": 2, "ahead": 3}
        true_idx = hyp_idx_map.get(true_hyp, 3)

        prior = diag0.get("prior", [0.25] * 4)
        posterior = diag0.get("posterior", [0.25] * 4)
        result.M_true_before = prior[true_idx] if true_idx < len(prior) else 0.25
        result.M_true_after = posterior[true_idx] if true_idx < len(posterior) else 0.25

        # dNLL_local
        result.delta_nll_local = diag0.get("delta_nll_local", 0.0)
        result.nll_before = diag0.get("nll_before", 0.0)
        result.nll_after = diag0.get("nll_after", 0.0)

    return result


# ── Aggregation ─────────────────────────────────────────────────

def aggregate_results(results: list[EpisodeResult]) -> dict:
    """Group results by (family, difficulty, variant) and compute means."""
    groups = defaultdict(list)
    for r in results:
        key = (r.family, r.difficulty, r.variant)
        groups[key].append(r)

    agg = {}
    for key, rs in groups.items():
        n = len(rs)
        agg[key] = {
            "n": n,
            "SBCR": sum(r.safe_branch_chosen for r in rs) / n,
            "TBSR": sum(r.survived and r.reached_goal for r in rs) / n,
            "survival": sum(r.survived for r in rs) / n,
            "goal": sum(r.reached_goal for r in rs) / n,
            "mean_steps": np.mean([r.steps for r in rs]),
            "mean_risky": np.mean([r.risky_entered for r in rs]),
            "mean_warns": np.mean([r.warn_count for r in rs]),
            "mean_delta_H": np.mean([r.delta_H for r in rs]),
            "mean_M_true_before": np.mean([r.M_true_before for r in rs]),
            "mean_M_true_after": np.mean([r.M_true_after for r in rs]),
            "mean_risk_delta": np.mean([r.risk_delta for r in rs]),
            "mean_entropy_final": np.mean([r.rsa_entropy_final for r in rs]),
            "mean_delta_nll_local": np.mean([r.delta_nll_local for r in rs]),
            "mean_nll_before": np.mean([r.nll_before for r in rs]),
            "mean_nll_after": np.mean([r.nll_after for r in rs]),
        }
    return agg


# ── Reporting ───────────────────────────────────────────────────

def print_headline_table(agg: dict, families: list, difficulties: list,
                         variants: list):
    """Print the headline metrics table."""
    print("\n" + "=" * 100)
    print("Step 2 Phase 2A: Execution-Only Warning Experiment -- Headline Results")
    print("=" * 100)

    for family in families:
        for diff in difficulties:
            print(f"\n-- {family} / {diff} --")
            header = f"{'Variant':<22} {'SBCR':>6} {'TBSR':>6} {'dH':>7} "
            header += f"{'M_true_b':>9} {'M_true_a':>9} {'dNLL_l':>7}"
            print(header)
            print("-" * 72)

            for v in variants:
                key = (family, diff, v)
                if key not in agg:
                    continue
                a = agg[key]
                row = f"{v:<22} {a['SBCR']:>6.3f} {a['TBSR']:>6.3f} "
                row += f"{a['mean_delta_H']:>7.4f} "
                row += f"{a['mean_M_true_before']:>9.4f} {a['mean_M_true_after']:>9.4f} "
                row += f"{a['mean_delta_nll_local']:>7.4f}"
                print(row)


def save_csv(results: list[EpisodeResult], path: Path):
    """Save per-episode results as CSV."""
    import csv
    fields = [
        "seed", "family", "difficulty", "variant",
        "survived", "reached_goal", "steps", "risky_entered",
        "warn_count", "safe_branch_chosen",
        "rsa_n_updates", "rsa_entropy_final",
        "delta_H", "M_true_before", "M_true_after", "risk_delta",
        "delta_nll_local", "nll_before", "nll_after",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"Saved {len(results)} rows -> {path}")


def save_report(agg: dict, families: list, difficulties: list,
                variants: list, path: Path, elapsed: float, n_seeds: int):
    """Save markdown report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Step 2 Phase 2A: Execution-Only Warning Experiment\n\n")
        f.write(f"**Seeds**: {n_seeds} per cell | ")
        f.write(f"**Elapsed**: {elapsed:.1f}s\n\n")
        f.write("## Headline Metrics\n\n")
        f.write("| Family | Diff | Variant | SBCR | TBSR | dH | M_true_b | M_true_a | dNLL_l |\n")
        f.write("|--------|------|---------|------|------|----|----------|----------|--------|\n")

        for family in families:
            for diff in difficulties:
                for v in variants:
                    key = (family, diff, v)
                    if key not in agg:
                        continue
                    a = agg[key]
                    f.write(f"| {family} | {diff} | {v} | "
                            f"{a['SBCR']:.3f} | {a['TBSR']:.3f} | "
                            f"{a['mean_delta_H']:.4f} | "
                            f"{a['mean_M_true_before']:.4f} | "
                            f"{a['mean_M_true_after']:.4f} | "
                            f"{a['mean_delta_nll_local']:.4f} |\n")

        # Response selectivity diagnostic
        f.write("\n## Diagnostic: Response Selectivity Gap\n\n")
        f.write("*(Not a formal metric in Phase 2A -- SelGap deferred to Phase 2B)*\n\n")
        f.write("| Family | Variant | dM_true |\n")
        f.write("|--------|---------|--------|\n")
        for family in families:
            for v in variants:
                key = (family, "medium", v)
                if key not in agg:
                    continue
                a = agg[key]
                delta_m = a["mean_M_true_after"] - a["mean_M_true_before"]
                f.write(f"| {family} | {v} | {delta_m:.4f} |\n")

    print(f"Report -> {path}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2 Phase 2A experiment")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 3 seeds, medium only, 3 variants")
    parser.add_argument("--families", type=str, default=None,
                        help="Comma-separated family list")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variant list")
    parser.add_argument("--outdir", type=str, default="results/step2_phase2a")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = list(range(3))
        families = ["fork_trap", "baseline_v2"]
        difficulties = ["medium"]
        variants = ["legacy_bias", "rsa_obs_s1", "rsa_plus_phase10"]
    else:
        seeds = list(range(args.seeds))
        families = args.families.split(",") if args.families else FAMILIES
        difficulties = DIFFICULTIES
        variants = args.variants.split(",") if args.variants else VARIANTS

    total = len(seeds) * len(families) * len(difficulties) * len(variants)
    print(f"Phase 2A: {total} episodes "
          f"({len(seeds)} seeds × {len(families)} families × "
          f"{len(difficulties)} diffs × {len(variants)} variants)")

    results = []
    t0 = time.time()
    done = 0

    for family in families:
        for diff in difficulties:
            for variant in variants:
                for seed in seeds:
                    try:
                        r = _run_episode(variant, seed, family, diff)
                        results.append(r)
                    except Exception as e:
                        print(f"  ERROR: {family}/{diff}/{variant}/s{seed}: {e}",
                              file=sys.stderr)
                    done += 1
                    if done % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"  [{done}/{total}] {elapsed:.0f}s elapsed",
                              file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)}/{total} episodes in {elapsed:.1f}s")

    # Aggregate and report
    agg = aggregate_results(results)
    print_headline_table(agg, families, difficulties, variants)
    save_csv(results, out / "phase2a_episodes.csv")
    save_report(agg, families, difficulties, variants,
                out / "phase2a_report.md", elapsed, len(seeds))


if __name__ == "__main__":
    main()
