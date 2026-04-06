"""Step 2 Phase 2B: Closed-loop Warning Experiment.

Connects Step 1 best micro policy (robot_belief + intervention_family)
with Step 2 warning variants. The micro decides WHEN to warn;
the warning_variant controls WHAT the warning does internally.

Conditions:
  best_micro + legacy_bias
  best_micro + rsa_obs_s1
  (ablation: best_micro + rsa_obs_s1_trust, rsa_plus_phase10)

Families: fork_trap, elcb_po, baseline_v2

Headline Metrics: SelGap, SBCR, TBSR, dM_true, dH, dNLL_local
  SelGap = WarnRate_necessary - WarnRate_unnecessary

Usage:
  python scripts/run_step2_phase2b.py --seeds 50
  python scripts/run_step2_phase2b.py --seeds 5 --smoke
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.lattice_v2_runner import LatticeV2Runner


# -- Configuration ----------------------------------------------------

VARIANTS = ["legacy_bias", "rsa_obs_s1"]
ABLATION_VARIANTS = ["rsa_obs_s1_trust", "rsa_plus_phase10"]
FAMILIES = ["fork_trap", "elcb_po", "baseline_v2"]


# -- Metrics -----------------------------------------------------------

@dataclass
class ClosedLoopResult:
    """Per-episode metrics for Phase 2B."""
    seed: int
    family: str
    variant: str
    # Behavior
    survived: bool = False
    reached_goal: bool = False
    steps: int = 0
    risky_entered: int = 0
    safe_branch_chosen: bool = False
    # Micro decisions
    warn_count: int = 0
    unlock_count: int = 0
    # SelGap components
    necessary_warn: int = 0
    unnecessary_warn: int = 0
    necessary_segments: int = 0
    unnecessary_segments: int = 0
    # Belief diagnostics
    delta_H: float = 0.0
    M_true_before: float = 0.0
    M_true_after: float = 0.0
    delta_nll_local: float = 0.0


def _classify_segment_necessity(seg, gridmap):
    """Classify a segment as 'necessary' (warn needed) or 'unnecessary'.

    A warning is 'necessary' if the segment has risky cells where the
    agent could plausibly die. Otherwise it's 'unnecessary'.
    """
    if not seg.risky_cells:
        return "unnecessary"
    # Check if any risky cell has high risk
    total_risk = 0
    for rc in seg.risky_cells:
        if hasattr(gridmap, 'true_risk'):
            total_risk += float(gridmap.true_risk[rc[0], rc[1]])
    avg_risk = total_risk / max(len(seg.risky_cells), 1)
    return "necessary" if avg_risk > 0.1 else "unnecessary"


def _run_closed_loop(variant, seed, family, difficulty="medium"):
    """Run one closed-loop episode with best micro + given warning variant."""
    runner = LatticeV2Runner()
    s = runner.reset(
        seed=seed,
        latent_mode=True,
        scenario_family=family,
        difficulty=difficulty,
        # Step 1 best micro config
        tutor_mode="none",
        robot_belief_mode=True,
        intervention_family_mode=True,
        item_drop_enabled=True,
        prefix_horizon=5,
        # Step 2 warning variant
        warning_variant=variant,
    )
    while not s.done:
        s = runner.step(s)

    m = runner.get_extended_metrics(s)

    result = ClosedLoopResult(
        seed=seed, family=family, variant=variant,
        survived=m["survived"],
        reached_goal=m["reached_goal"],
        steps=m["steps"],
        risky_entered=m["risky_entered"],
        safe_branch_chosen=(m["risky_entered"] == 0),
        warn_count=m["warnings"],
        unlock_count=m.get("unlock_count", 0),
    )

    # SelGap: classify warnings by necessity
    for seg in s.meta.segments:
        necessity = _classify_segment_necessity(seg, s.gridmap)
        warned = seg.index in s.warned_segments
        if necessity == "necessary":
            result.necessary_segments += 1
            if warned:
                result.necessary_warn += 1
        else:
            result.unnecessary_segments += 1
            if warned:
                result.unnecessary_warn += 1

    # RSA diagnostics
    if s.rsa_warn_diagnostics:
        diag0 = s.rsa_warn_diagnostics[0]
        result.delta_H = diag0.get("delta_H", 0.0)
        result.delta_nll_local = diag0.get("delta_nll_local", 0.0)

        # M_true
        seg = s.meta.segments[0] if s.meta.segments else None
        if seg:
            true_hyp = "left" if seg.risky_row <= 2 else "right"
            hyp_idx = 0 if true_hyp == "left" else 1
            prior = diag0.get("prior", [0.25] * 4)
            posterior = diag0.get("posterior", [0.25] * 4)
            result.M_true_before = prior[hyp_idx] if hyp_idx < len(prior) else 0.25
            result.M_true_after = posterior[hyp_idx] if hyp_idx < len(posterior) else 0.25

    return result


# -- Aggregation -------------------------------------------------------

def aggregate(results):
    """Group by (family, variant) and compute headline metrics."""
    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.variant)].append(r)

    agg = {}
    for key, rs in groups.items():
        n = len(rs)
        total_nec = sum(r.necessary_segments for r in rs)
        total_unnec = sum(r.unnecessary_segments for r in rs)
        warn_nec = sum(r.necessary_warn for r in rs)
        warn_unnec = sum(r.unnecessary_warn for r in rs)
        wr_nec = warn_nec / max(total_nec, 1)
        wr_unnec = warn_unnec / max(total_unnec, 1)

        warned_rs = [r for r in rs if r.warn_count > 0]
        agg[key] = {
            "n": n,
            "SBCR": sum(r.safe_branch_chosen for r in rs) / n,
            "TBSR": sum(r.survived and r.reached_goal for r in rs) / n,
            "survival": sum(r.survived for r in rs) / n,
            "SelGap": wr_nec - wr_unnec,
            "WR_nec": wr_nec,
            "WR_unnec": wr_unnec,
            "mean_warns": np.mean([r.warn_count for r in rs]),
            "mean_unlocks": np.mean([r.unlock_count for r in rs]),
            "mean_dH": np.mean([r.delta_H for r in rs]) if rs else 0,
            "mean_dM": np.mean([r.M_true_after - r.M_true_before for r in rs]) if rs else 0,
            "mean_dNLL": np.mean([r.delta_nll_local for r in rs]) if rs else 0,
        }
    return agg


# -- Reporting ---------------------------------------------------------

def print_results(agg, families, variants):
    print("\n" + "=" * 90)
    print("Step 2 Phase 2B: Closed-Loop Warning Experiment")
    print("=" * 90)

    for fam in families:
        print(f"\n-- {fam} --")
        print(f"  {'Variant':<22} {'SBCR':>5} {'TBSR':>5} {'SelGap':>7} "
              f"{'WR_n':>5} {'WR_u':>5} {'Warn':>5} {'dM':>7} {'dNLL':>7}")
        print("  " + "-" * 75)
        for v in variants:
            k = (fam, v)
            if k not in agg:
                continue
            a = agg[k]
            print(f"  {v:<22} {a['SBCR']:>5.2f} {a['TBSR']:>5.2f} "
                  f"{a['SelGap']:>7.3f} {a['WR_nec']:>5.2f} {a['WR_unnec']:>5.2f} "
                  f"{a['mean_warns']:>5.1f} {a['mean_dM']:>7.4f} {a['mean_dNLL']:>7.4f}")


def save_csv(results, path):
    import csv
    fields = [
        "seed", "family", "variant",
        "survived", "reached_goal", "steps", "risky_entered",
        "safe_branch_chosen", "warn_count", "unlock_count",
        "necessary_warn", "unnecessary_warn",
        "necessary_segments", "unnecessary_segments",
        "delta_H", "M_true_before", "M_true_after", "delta_nll_local",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"Saved {len(results)} rows -> {path}")


def save_report(agg, families, variants, path, elapsed, n_seeds):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Step 2 Phase 2B: Closed-Loop Warning Experiment\n\n")
        f.write(f"**Seeds**: {n_seeds} | **Elapsed**: {elapsed:.1f}s\n\n")
        f.write("## Headline Metrics\n\n")
        f.write("| Family | Variant | SBCR | TBSR | SelGap | WR_nec | WR_unnec | Warns | dM | dNLL |\n")
        f.write("|--------|---------|------|------|--------|--------|----------|-------|----|------|\n")
        for fam in families:
            for v in variants:
                k = (fam, v)
                if k not in agg: continue
                a = agg[k]
                f.write(f"| {fam} | {v} | {a['SBCR']:.3f} | {a['TBSR']:.3f} | "
                        f"{a['SelGap']:.3f} | {a['WR_nec']:.3f} | {a['WR_unnec']:.3f} | "
                        f"{a['mean_warns']:.1f} | {a['mean_dM']:.4f} | {a['mean_dNLL']:.4f} |\n")

        # Promotion decision
        f.write("\n## Promotion Decision\n\n")
        for fam in families:
            legacy_k = (fam, "legacy_bias")
            s1_k = (fam, "rsa_obs_s1")
            if legacy_k not in agg or s1_k not in agg:
                continue
            lg = agg[legacy_k]
            s1 = agg[s1_k]
            f.write(f"### {fam}\n\n")
            sbcr_ok = s1['SBCR'] >= lg['SBCR'] - 0.05
            tbsr_ok = s1['TBSR'] >= lg['TBSR'] - 0.10
            sel_ok = s1['SelGap'] >= lg['SelGap'] - 0.05
            dm_ok = s1['mean_dM'] >= lg['mean_dM']
            f.write(f"- SBCR: {s1['SBCR']:.3f} vs {lg['SBCR']:.3f} {'PASS' if sbcr_ok else 'FAIL'}\n")
            f.write(f"- TBSR: {s1['TBSR']:.3f} vs {lg['TBSR']:.3f} {'PASS' if tbsr_ok else 'FAIL'}\n")
            f.write(f"- SelGap: {s1['SelGap']:.3f} vs {lg['SelGap']:.3f} {'PASS' if sel_ok else 'FAIL'}\n")
            f.write(f"- dM_true: {s1['mean_dM']:.4f} vs {lg['mean_dM']:.4f} {'PASS' if dm_ok else 'FAIL'}\n\n")

    print(f"Report -> {path}")


# -- Main --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2 Phase 2B experiment")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 5 seeds, 2 variants, 2 families")
    parser.add_argument("--ablation", action="store_true",
                        help="Include ablation variants")
    parser.add_argument("--outdir", type=str, default="results/step2_phase2b")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds = list(range(5))
        families = ["fork_trap", "baseline_v2"]
        variants = ["legacy_bias", "rsa_obs_s1"]
    else:
        seeds = list(range(args.seeds))
        families = FAMILIES
        variants = VARIANTS + (ABLATION_VARIANTS if args.ablation else [])

    total = len(seeds) * len(families) * len(variants)
    print(f"Phase 2B: {total} episodes "
          f"({len(seeds)} seeds x {len(families)} families x {len(variants)} variants)")

    results = []
    t0 = time.time()
    done = 0

    for fam in families:
        for v in variants:
            for seed in seeds:
                try:
                    r = _run_closed_loop(v, seed, fam)
                    results.append(r)
                except Exception as e:
                    print(f"  ERROR: {fam}/{v}/s{seed}: {e}", file=sys.stderr)
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total}] {elapsed:.0f}s", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)}/{total} in {elapsed:.1f}s")

    agg = aggregate(results)
    print_results(agg, families, variants)
    save_csv(results, out / "phase2b_episodes.csv")
    save_report(agg, families, variants, out / "phase2b_report.md", elapsed, len(seeds))


if __name__ == "__main__":
    main()
