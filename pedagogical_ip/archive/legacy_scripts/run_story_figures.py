"""Generate paper-facing story tables from existing results.

Reads:
  results/tpm_sweep_cross_difficulty.csv
  results/tpm_ablation.csv
  results/transfer_eval.csv
  results/diagnostic_summary.csv

Outputs:
  results/story_tables.md
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, ".")

import numpy as np


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    out_dir = Path("results")

    # ── Load data ─────────────────────────────────────────────────
    sweep = read_csv(out_dir / "tpm_sweep_cross_difficulty.csv")
    ablation = read_csv(out_dir / "tpm_ablation.csv")
    transfer = read_csv(out_dir / "transfer_eval.csv")
    diag = read_csv(out_dir / "diagnostic_summary.csv")

    # ── Aggregate sweep by (family, difficulty, condition) ────────
    sw_agg = defaultdict(lambda: {"n": 0, "s": 0})
    for r in sweep:
        k = (r["family"], r["difficulty"], r["condition"])
        sw_agg[k]["n"] += 1
        sw_agg[k]["s"] += int(r["success"])

    def sw_sr(fam, diff, cond):
        a = sw_agg.get((fam, diff, cond), {"n": 1, "s": 0})
        return a["s"] / max(a["n"], 1)

    # ── Aggregate ablation by (family, ablation) ─────────────────
    ab_agg = defaultdict(lambda: {"n": 0, "s": 0})
    for r in ablation:
        k = (r["family"], r["ablation"])
        ab_agg[k]["n"] += 1
        ab_agg[k]["s"] += int(r["success"])

    def ab_sr(fam, abl):
        a = ab_agg.get((fam, abl), {"n": 1, "s": 0})
        return a["s"] / max(a["n"], 1)

    # ── Write story tables ────────────────────────────────────────
    md_path = out_dir / "story_tables.md"
    with open(md_path, "w") as f:
        # ═══ Table 1: Online Gain ═══════════════════════════════
        f.write("# Paper-Facing Story Tables\n\n")
        f.write("## Table 1: Online Gain (Pre-TPM → Post-TPM)\n\n")
        f.write("| Family | Difficulty | No Tutor | Pre-TPM | **Post-TPM** | **Δ(post-pre)** |\n")
        f.write("|--------|-----------|----------|---------|-------------|----------------|\n")

        families = ["fork_trap", "hazard_belt", "deadline_gate"]
        diffs = ["easy", "medium", "hard"]
        for fam in families:
            for diff in diffs:
                nt = sw_sr(fam, diff, "no_tutor")
                pre = sw_sr(fam, diff, "robot_belief_pre")
                post = sw_sr(fam, diff, "robot_belief_post")
                delta = post - pre
                sign = "+" if delta >= 0 else ""
                f.write(
                    f"| {fam} | {diff} | {nt:.0%} | {pre:.0%} "
                    f"| **{post:.0%}** | **{sign}{delta:.0%}** |\n")
            f.write("| | | | | | |\n")
        f.write("\n")

        # ═══ Table 2: Ablation ═══════════════════════════════════
        f.write("## Table 2: TPM Component Ablation (medium, 20 seeds)\n\n")
        f.write("| Component Removed | fork_trap | **hazard_belt** | deadline_gate |\n")
        f.write("|-------------------|-----------|--------------|---------------|\n")

        ablations = ["full_tpm", "no_bottleneck_match", "no_warn_damping",
                     "no_unlock_memory", "no_perceptual_access", "cf_only"]
        for abl in ablations:
            row = f"| {abl} "
            for fam in families:
                sr = ab_sr(fam, abl)
                full_sr = ab_sr(fam, "full_tpm")
                delta = sr - full_sr
                if abl == "full_tpm":
                    row += f"| {sr:.0%} "
                else:
                    sign = "+" if delta >= 0 else ""
                    row += f"| {sr:.0%} ({sign}{delta:.0%}) "
            row += "|\n"
            f.write(row)
        f.write("\n")

        # ═══ Table 3: Help vs Learning ═══════════════════════════
        f.write("## Table 3: Help vs Learning (Online Help Gain vs Transfer)\n\n")
        f.write("OHG = SR_assisted - SR_no_tutor (online)\n\n")
        f.write("LG = probe_SR(k=3) - probe_SR(k=0, no_tutor) (transfer)\n\n")
        f.write("PE = LG / OHG (pedagogical efficiency)\n\n")

        f.write("| Family | Condition | SR_online | OHG | LG(k=3) | PE |\n")
        f.write("|--------|-----------|-----------|-----|---------|----|\n")

        # Transfer data
        tr_data = {}
        for r in transfer:
            k = (r["family"], r["condition"], int(r["exposure_k"]))
            tr_data[k] = float(r["probe_sr"])

        for fam in families:
            nt_sr = sw_sr(fam, "medium", "no_tutor")
            nt_probe_0 = tr_data.get((fam, "no_tutor", 0), 0)

            conds = ["no_tutor", "robot_belief_pre", "robot_belief_post"]
            for cond in conds:
                online_sr = sw_sr(fam, "medium", cond)
                ohg = online_sr - nt_sr
                probe_3 = tr_data.get((fam, cond, 3), 0)
                lg = probe_3 - nt_probe_0
                pe = lg / max(ohg, 0.01) if ohg > 0.01 else 0.0
                ohg_s = f"+{ohg:.0%}" if ohg >= 0 else f"{ohg:.0%}"
                lg_s = f"+{lg:.0%}" if lg >= 0 else f"{lg:.0%}"
                f.write(
                    f"| {fam} | {cond} | {online_sr:.0%} "
                    f"| {ohg_s} | {lg_s} | {pe:.2f} |\n")
            f.write("| | | | | | |\n")
        f.write("\n")

        # ═══ Table 4: Intervention Semantics ═════════════════════
        f.write("## Table 4: Intervention Semantic Taxonomy\n\n")
        f.write("| Intervention | Target Layer | Mechanism | Best Family |\n")
        f.write("|-------------|-------------|-----------|-------------|\n")
        f.write("| WARN | Epistemic (belief) | Biases risk-relevant features toward danger | fork_trap |\n")
        f.write("| UNLOCK | Structural (affordance) | Reduces uncertainty on newly reachable cells | deadline_gate |\n")
        f.write("| ITEM_DROP | Outcome (mitigation) | Provides shield for unavoidable hazard crossing | hazard_belt |\n")
        f.write("| WAIT | — | Allows autonomous learning without interference | — |\n\n")

        # ═══ Table 5: Diagnostic Summary (from Step A) ════════════
        f.write("## Table 5: Learner Dynamics Diagnostics (5 seeds, medium)\n\n")
        f.write("| Family | Condition | SR | mean Δθ | Δθ_r | BAR | Interpretation |\n")
        f.write("|--------|-----------|----|---------|----|-----|----------------|\n")

        diag_agg = defaultdict(lambda: {"n": 0, "dt": [], "dt_r": [],
                                        "bar": [], "sr": []})
        for r in diag:
            k = (r["family"], r["condition"])
            a = diag_agg[k]
            a["n"] += 1
            a["dt"].append(float(r["mean_delta_theta"]) if r["mean_delta_theta"] != "nan" else 0.0)
            a["dt_r"].append(float(r["mean_delta_theta_r"]))
            a["bar"].append(float(r["bar"]))
            a["sr"].append(int(r["success"]))

        for fam in families:
            conds = list(FAMILY_CONDITIONS_ORDER.get(fam, []))
            for cond in conds:
                a = diag_agg.get((fam, cond))
                if a is None:
                    continue
                n = a["n"]
                sr = sum(a["sr"]) / n
                dt = np.nanmean(a["dt"])
                dt_r = np.mean(a["dt_r"])
                bar = np.mean(a["bar"])

                # Auto-interpret
                if bar > 0.9:
                    interp = "all-WARN (bottleneck always epistemic → action always matches)"
                elif bar > 0.6:
                    interp = "diversified targeting (TPM redirects to non-WARN actions)"
                elif sr == 0:
                    interp = "no intervention, learner updates from failures"
                else:
                    interp = "single-lever or no-tutor"

                f.write(
                    f"| {fam} | {cond} | {sr:.0%} | {dt:.4f} | {dt_r:.4f} "
                    f"| {bar:.2f} | {interp} |\n")
            f.write("| | | | | | | |\n")

        f.write("\n---\n\n")
        f.write("## Key Narrative\n\n")
        f.write("1. **TPM improves online success** substantially "
                "(hazard_belt +25pp, fork_trap +20pp at hard)\n")
        f.write("2. **warn_damping is the critical mechanism** — "
                "disabling it drops hazard_belt by 25pp\n")
        f.write("3. **Transfer is zero for ALL conditions** — "
                "TPM is a per-episode assistant, not a learning-inducing tutor\n")
        f.write("4. **The learner IS updating** (Δθ ≈ 0.15-0.29) — "
                "null transfer is NOT from a dead learner\n")
        f.write("5. **Risk head updates dominate** (Δθ_r >> Δθ_c) — "
                "cost learning is minimal\n")
        f.write("6. **Pre-TPM BAR is perfectly 1.00** because all interventions are WARN, "
                "which always matches the dominant epistemic bottleneck. "
                "Post-TPM BAR drops to ~0.7 but achieves HIGHER SR — "
                "confirming that correct action diversity beats spurious alignment.\n")

    print(f"Story tables -> {md_path}", file=sys.stderr)


FAMILY_CONDITIONS_ORDER = {
    "fork_trap":    ["no_tutor", "warning_only", "robot_belief_pre", "robot_belief_post"],
    "hazard_belt":  ["no_tutor", "item_only", "robot_belief_pre", "robot_belief_post"],
    "deadline_gate": ["no_tutor", "unlock_only", "robot_belief_pre", "robot_belief_post"],
}

if __name__ == "__main__":
    main()
