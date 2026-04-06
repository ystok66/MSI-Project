"""C1+C2+C3: Decision-Aware Tutor + Urgency + Param Sweep.

C1: Validate DIG metric alignment with SBCR (better than entropy IG)
C2: Test learning_aware_v2 with urgency (should warn earlier)
C3: Sweep λ_D, λ_U, λ_M, λ_C over small grid
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal,
    neutralize_identity_features,
)
from src.envs.observation_mask import make_observation_mask
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch
from src.metrics.decision_info import (
    compute_branch_posteriors, compute_all_decision_metrics,
)
from src.teachers.learning_aware_policy_v2 import (
    LearningAwarePolicyV2, TutorV2Config,
)

FAM = "elcb_po"
DIFF = "medium"
out = Path("results")
out.mkdir(exist_ok=True)


def apply_fix(gm, meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def vis_candidates(sc, obs_r=2):
    fk = sc.fork_cell
    ma = make_observation_mask(sc.branch_a_cells, fk, obs_r)
    mb = make_observation_mask(sc.branch_b_cells, fk, obs_r)
    va = [c for c, m in zip(sc.branch_a_cells, ma) if m > 0.5]
    vb = [c for c, m in zip(sc.branch_b_cells, mb) if m > 0.5]
    mg = sc.merge_cell
    return [
        BranchCandidate(0, va, len(va), fk, mg, (1, fk[1]), (1, mg[1])),
        BranchCandidate(1, vb, len(vb), fk, mg, (3, fk[1]), (3, mg[1])),
    ]


def train_base(seeds, use_warn=False, oracle=False, lp=None, lib=None,
                scorer=None, tutor_v2=None, strategy="none"):
    if lp is None:
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    if lib is None:
        lib = BranchConceptLibrary()
    if scorer is None:
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)

    for seed in seeds:
        gm, _, meta, sc = generate_scenario(FAM, seed, DIFF, latent_mode=True)
        fb, ww = apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss)
        lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        # Decide intervention
        do_warn = False
        if strategy == "always_warn" or strategy == "oracle":
            do_warn = True
        elif strategy == "always_wait":
            do_warn = False
        elif strategy == "v2" and tutor_v2 is not None:
            action, _ = tutor_v2.decide(sc, fb, lp, lib, scorer)
            do_warn = (action == "WARN")
        elif oracle:
            do_warn = True

        if do_warn:
            for r, c in sc.risky_cells:
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
            ss2 = summarize_branch(sc.safe_cells, fb, fv, lp)
            sr2 = summarize_branch(sc.risky_cells, fb, fv, lp)
            lib.update("safe_branch", ss2)
            lib.update("risky_branch", sr2)
            scorer.update(build_scorer_input(ss2, lib), 1.0)
            scorer.update(build_scorer_input(sr2, lib), 0.0)

    return lp, lib, scorer


def probe_metrics(probe_seeds, lp, lib, scorer):
    """Probe: returns per-seed (chose_safe, DIG, BR_pre, BR_post, DCG)."""
    records = []
    for ps in probe_seeds:
        gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
        fb, _ = apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)
        passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
        tie_rng = np.random.default_rng(ps + 777)

        cands = vis_candidates(sc, 2)

        # Pre-warning posteriors (visible only)
        s_a_pre = summarize_branch(cands[0].cells, fb, fv, lp)
        s_b_pre = summarize_branch(cands[1].cells, fb, fv, lp)
        pa_pre, pb_pre = compute_branch_posteriors(
            s_a_pre, s_b_pre, scorer, build_scorer_input, lib)

        # Post-warning posteriors (full branch)
        s_a_post = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        s_b_post = summarize_branch(sc.branch_b_cells, fb, fv, lp)
        pa_post, pb_post = compute_branch_posteriors(
            s_a_post, s_b_post, scorer, build_scorer_input, lib)

        dm = compute_all_decision_metrics(pa_pre, pb_pre, pa_post, pb_post,
                                           sc.oracle_safe_branch_id)

        # Actual branch choice
        best, _ = choose_branch(
            cands, fb, fv, lp, passable, lib, scorer,
            lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
        chose_safe = int(best.branch_id == sc.oracle_safe_branch_id)

        records.append({
            "seed": ps, "chose_safe": chose_safe,
            "BR_pre": dm["BR_pre"], "BR_post": dm["BR_post"],
            "DIG": dm["DIG"], "DCG": dm["DCG"],
        })
    return records


# ══════════════════════════════════════════════════════════════
# C1: Validate DIG alignment
# ══════════════════════════════════════════════════════════════
def c1_dig_validation():
    print("C1: DIG Validation", file=sys.stderr)
    train_seeds = list(range(40))
    probe_seeds = list(range(100, 150))

    configs = {
        "no_tutor":    {"oracle": False, "use_warn": False},
        "oracle_warn": {"oracle": True,  "use_warn": True},
    }

    results = {}
    for name, cfg in configs.items():
        lp, lib, scorer = train_base(
            train_seeds, oracle=cfg["oracle"], strategy="oracle" if cfg["oracle"] else "always_wait")
        recs = probe_metrics(probe_seeds, lp, lib, scorer)
        sbcr = np.mean([r["chose_safe"] for r in recs])
        mean_dig = np.mean([r["DIG"] for r in recs])
        mean_br_pre = np.mean([r["BR_pre"] for r in recs])
        mean_br_post = np.mean([r["BR_post"] for r in recs])
        mean_dcg = np.mean([r["DCG"] for r in recs])
        results[name] = {
            "SBCR": round(float(sbcr), 3),
            "DIG": round(float(mean_dig), 4),
            "BR_pre": round(float(mean_br_pre), 4),
            "BR_post": round(float(mean_br_post), 4),
            "DCG": round(float(mean_dcg), 3),
        }
    return results


# ══════════════════════════════════════════════════════════════
# C2: Urgency-aware v2 tutor
# ══════════════════════════════════════════════════════════════
def c2_urgency_test():
    print("C2: Urgency-Aware v2", file=sys.stderr)
    K_VALUES = [0, 1, 3, 10, 30]
    PROBE_SEEDS = list(range(100, 150))
    N_BOOT = 200

    strategies = {
        "always_wait": ("always_wait", None),
        "always_warn": ("always_warn", None),
        "v1_conservative": ("v2", TutorV2Config(lambda_c=0.1, lambda_u=0.0, lambda_m=0.0, lambda_d=0.5)),
        "v2_urgency":      ("v2", TutorV2Config()),  # defaults: λ_U=1.5, λ_M=1.0, λ_D=2.0
        "oracle":          ("oracle", None),
    }

    all_results = []
    all_stats = {}

    for sname, (strat, cfg) in strategies.items():
        tutor = LearningAwarePolicyV2(cfg) if cfg else (
            LearningAwarePolicyV2(TutorV2Config()) if strat == "v2" else None)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        lib = BranchConceptLibrary()
        scorer = BranchScorerProbe(lr=0.05, l2=0.01)
        k_done = 0

        for k_target in K_VALUES:
            batch = list(range(k_done, k_target))
            if batch:
                lp, lib, scorer = train_base(
                    batch, lp=lp, lib=lib, scorer=scorer,
                    tutor_v2=tutor, strategy=strat)
            k_done = k_target

            per = []
            for ps in PROBE_SEEDS:
                gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
                fb, _ = apply_fix(gm, meta, sc)
                fv = np.full_like(fb, 0.3)
                passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
                tie_rng = np.random.default_rng(ps + 777)
                cands = vis_candidates(sc, 2)
                best, _ = choose_branch(
                    cands, fb, fv, lp, passable, lib, scorer,
                    lambda_b=1.0, score_mode="hybrid", tie_rng=tie_rng)
                per.append(int(best.branch_id == sc.oracle_safe_branch_id))

            per = np.array(per)
            sbcr = float(np.mean(per))
            br = np.random.default_rng(k_target * 100 + 9999)
            bm = [float(np.mean(per[br.integers(0, len(per), len(per))]))
                  for _ in range(N_BOOT)]
            all_results.append({
                "strategy": sname, "k": k_target,
                "SBCR": round(sbcr, 3),
                "CI_lo": round(float(np.percentile(bm, 2.5)), 3),
                "CI_hi": round(float(np.percentile(bm, 97.5)), 3),
            })

        if tutor:
            all_stats[sname] = {
                "warn_rate": round(tutor.warn_rate, 3),
                "warn_count": tutor.warn_count,
                "wait_count": tutor.wait_count,
            }
        else:
            total = 30
            wc = total if strat in ("always_warn", "oracle") else 0
            all_stats[sname] = {"warn_rate": round(wc / total, 3),
                                 "warn_count": wc, "wait_count": total - wc}

    return all_results, all_stats


# ══════════════════════════════════════════════════════════════
# C3: Parameter sweep
# ══════════════════════════════════════════════════════════════
def c3_param_sweep():
    print("C3: Param Sweep", file=sys.stderr)
    PROBE_SEEDS = list(range(100, 150))
    train_seeds = list(range(30))

    configs = {
        "baseline":   TutorV2Config(lambda_d=2.0, lambda_u=1.5, lambda_m=1.0, lambda_c=0.05),
        "high_dig":   TutorV2Config(lambda_d=4.0, lambda_u=1.5, lambda_m=1.0, lambda_c=0.05),
        "high_urg":   TutorV2Config(lambda_d=2.0, lambda_u=3.0, lambda_m=2.0, lambda_c=0.05),
        "low_cost":   TutorV2Config(lambda_d=2.0, lambda_u=1.5, lambda_m=1.0, lambda_c=0.01),
        "aggressive": TutorV2Config(lambda_d=4.0, lambda_u=3.0, lambda_m=2.0, lambda_c=0.01),
    }

    results = []
    for cname, cfg in configs.items():
        tutor = LearningAwarePolicyV2(cfg)
        lp, lib, scorer = train_base(
            train_seeds, tutor_v2=tutor, strategy="v2")
        per = []
        for ps in PROBE_SEEDS:
            gm, _, meta, sc = generate_scenario(FAM, ps, DIFF, latent_mode=True)
            fb, _ = apply_fix(gm, meta, sc)
            fv = np.full_like(fb, 0.3)
            passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
            trng = np.random.default_rng(ps + 777)
            cands = vis_candidates(sc, 2)
            best, _ = choose_branch(
                cands, fb, fv, lp, passable, lib, scorer,
                lambda_b=1.0, score_mode="hybrid", tie_rng=trng)
            per.append(int(best.branch_id == sc.oracle_safe_branch_id))

        sbcr = round(float(np.mean(per)), 3)
        wr = round(tutor.warn_rate, 3)
        pe = round(sbcr / max(wr, 0.01), 2)
        results.append({
            "config": cname, "SBCR": sbcr, "warn_rate": wr, "PE": pe,
            "params": "D={} U={} M={} C={}".format(
                cfg.lambda_d, cfg.lambda_u, cfg.lambda_m, cfg.lambda_c),
        })

    return results


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    c1 = c1_dig_validation()
    c2_res, c2_stats = c2_urgency_test()
    c3 = c3_param_sweep()

    with open(out / "decision_aware_report.md", "w") as f:
        f.write("# Decision-Aware Tutor Report\n\n")

        # C1
        f.write("## C1: DIG Validation\n\n")
        f.write("| Condition | SBCR | DIG | BR_pre | BR_post | DCG |\n")
        f.write("|-----------|------|-----|--------|---------|-----|\n")
        for n, d in c1.items():
            f.write("| {} | {:.0%} | {:.4f} | {:.4f} | {:.4f} | {:.3f} |\n".format(
                n, d["SBCR"], d["DIG"], d["BR_pre"], d["BR_post"], d["DCG"]))

        # C2
        f.write("\n## C2: Urgency-Aware v2\n\n")
        f.write("### Intervention Statistics\n\n")
        f.write("| Strategy | Warn Rate | Warns | Waits |\n")
        f.write("|----------|-----------|-------|-------|\n")
        for sn, st in c2_stats.items():
            f.write("| {} | {:.0%} | {} | {} |\n".format(
                sn, st["warn_rate"], st["warn_count"], st["wait_count"]))

        f.write("\n### Training-Probe (50 seeds, 95% CI)\n\n")
        f.write("| Strategy | k=0 | k=1 | k=3 | k=10 | k=30 |\n")
        f.write("|----------|-----|-----|-----|------|------|\n")
        for sn in ["always_wait", "v1_conservative", "v2_urgency", "always_warn", "oracle"]:
            rows = [r for r in c2_res if r["strategy"] == sn]
            cells = ["{:.0%} [{:.0%},{:.0%}]".format(r["SBCR"], r["CI_lo"], r["CI_hi"])
                     for r in rows]
            f.write("| {} | {} |\n".format(sn, " | ".join(cells)))

        f.write("\n### Learning Gain LG(k)\n\n")
        for sn in ["always_wait", "v1_conservative", "v2_urgency", "always_warn", "oracle"]:
            rows = [r for r in c2_res if r["strategy"] == sn]
            k0 = rows[0]["SBCR"]
            parts = ["k={}:{:+.0%}".format(r["k"], r["SBCR"] - k0) for r in rows[1:]]
            f.write("- **{}**: {}\n".format(sn, ", ".join(parts)))

        # C3
        f.write("\n## C3: Parameter Sweep\n\n")
        f.write("| Config | Params | SBCR | WarnRate | PE |\n")
        f.write("|--------|--------|------|---------|----|\n")
        for r in c3:
            f.write("| {} | {} | {:.0%} | {:.0%} | {:.2f} |\n".format(
                r["config"], r["params"], r["SBCR"], r["warn_rate"], r["PE"]))

    print("Report -> results/decision_aware_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)
