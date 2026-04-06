"""Step 1: Ablation Detail — redundancy questions.

Targeted ablations:
  1. LearnGain: behavior_loss vs entropy_reduction
  2. DepCost: simple (p_blind only) vs full (+ ν̂ increment)
  3. p_self: two-outcome vs three-outcome (variant C)
  4. p_self: variant A (fusion) vs B (predictive) vs C (three-outcome)
  5. κ̂-aware micro vs 3D-only (research ablation)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import PROBE_NAMES, LESSON_V2_BY_NAME
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.micro_bayes_shadow import (
    MicroBayesShadow, LearnGainVariant, DepCostVariant,
)
from src.teachers.p_self_posterior_shadow import PSelfMode, compute_p_self_posterior
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

TARGET_LESSONS = []
for name in ["tic_rescue_heavy", "tic_self_discovery", "ppmrb_self_discovery",
              "false_suppression", "ppmrb_standard", "tic_standard"]:
    if name in LESSON_V2_BY_NAME:
        TARGET_LESSONS.append(LESSON_V2_BY_NAME[name])

N_STEPS = 20
N_SEEDS = 6


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_ablation(name, mb_kwargs, p_self_mode_str, theta, seed):
    """Run a single ablation config over N_STEPS."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState()
    m.snapshot()

    mb = MicroBayesShadow(agent_params=AP, **mb_kwargs)
    results = []

    for step_i in range(N_STEPS):
        les = TARGET_LESSONS[step_i % len(TARGET_LESSONS)]
        ub = {p: 0.4 for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(
            les, step_i + seed * 10000, theta, ub, rng)
        ep, spec, gm, cfg_e, meta, sc = et
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(3):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        lib = BranchConceptLibrary()
        scr = BranchScorerProbe(lr=0.05, l2=0.01)
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe", ss); lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)

        dc = getattr(sc, 'commit_depth', 3)
        dr = getattr(sc, 'reveal_depth', 2)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        subtype = getattr(ep, 'subtype', '')

        # Compute p_self
        ps_mode = PSelfMode(p_self_mode_str)
        ps_result = compute_p_self_posterior(
            ps_mode, dc, dr,
            tau_hat=m.tau, nu_hat=m.nu, gamma_gen_hat=m.gamma_gen,
        )
        p_self = ps_result["p_self"]
        p_fail = ps_result["p_fail"]

        # Compute delta_s, dvoi
        from src.envs.observation_mask import make_observation_mask
        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, 2)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, 2)
        vis_a = [c for c, mm in zip(sc.branch_a_cells, mask_a) if mm > 0.5]
        vis_b = [c for c, mm in zip(sc.branch_b_cells, mask_b) if mm > 0.5]
        sa = summarize_branch(vis_a, fb, fv, lp)
        sb = summarize_branch(vis_b, fb, fv, lp)
        sa2 = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        sb2 = summarize_branch(sc.branch_b_cells, fb, fv, lp)
        delta_s = max(abs(sa2[0] - sb2[0]) - abs(sa[0] - sb[0]), 0)
        dvoi = max(float(1.0/(1.0+np.exp(-abs(sa2[0]-sb2[0])))) -
                   float(1.0/(1.0+np.exp(-abs(sa[0]-sb[0])))), 0)

        has_self_ev = (2 >= dc - 1) or p_self > 0.5
        novelty = 0.3 if subtype in ("beneficial_novelty",) else 0.0
        self_ev = 0.7 if has_self_ev else 0.3

        from src.agents.behavior_probes import BEHAVIOR_ZONES
        zones = BEHAVIOR_ZONES.get("safe", {})

        act, dose, info = mb.score(
            m, delta_s, dvoi, tempt, risk,
            p_self, p_fail, subtype, has_self_ev, zones,
            novelty, self_ev,
        )

        # Agent choice
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b)
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a)
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng)
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose > 0
        self_disc = correct and not warned and p_self > 0.5

        # Update m
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if p_self < 0.5: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5:
            m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15)
        m.snapshot()

        results.append({
            "correct": correct, "warned": warned,
            "self_disc": self_disc, "p_self": p_self,
            "p_fail": p_fail,
            "p_undecided": ps_result.get("p_undecided", 0),
            "subtype": subtype,
        })

    return results, m


ABLATIONS = {
    # LearnGain ablation
    "LG1_behavior_loss": (
        {"learn_gain_variant": LearnGainVariant.BEHAVIOR_LOSS,
         "dep_cost_variant": DepCostVariant.FULL},
        "baseline"),
    "LG2_entropy_reduction": (
        {"learn_gain_variant": LearnGainVariant.ENTROPY_REDUCTION,
         "dep_cost_variant": DepCostVariant.FULL},
        "baseline"),

    # DepCost ablation
    "DC_simple": (
        {"dep_cost_variant": DepCostVariant.SIMPLE},
        "baseline"),
    "DC_full": (
        {"dep_cost_variant": DepCostVariant.FULL},
        "baseline"),

    # p_self variant ablation (all with micro_bayes)
    "PS_baseline": ({}, "baseline"),
    "PS_A_fusion": ({}, "posterior_A"),
    "PS_B_predictive": ({}, "posterior_B"),
    "PS_C_three_outcome": ({}, "posterior_C"),

    # κ̂-aware (research only)
    "kappa_aware": (
        {"use_kappa_aware": True, "kappa_hat": 1.5},
        "baseline"),
    "kappa_off": (
        {"use_kappa_aware": False},
        "baseline"),
}


def main():
    print("═══ Step 1: Ablation Detail ═══\n", file=sys.stderr)
    L = ["# Step 1: Ablation Detail\n\n"]

    metrics = {}
    for abl_name, (mb_kw, ps_mode) in ABLATIONS.items():
        all_results = []
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                res, m_final = run_ablation(abl_name, mb_kw, ps_mode, th, sid)
                all_results.extend(res)
            print(f"  {abl_name} / {th} done", file=sys.stderr)

        tbsr = np.mean([r["correct"] for r in all_results])
        wr = np.mean([r["warned"] for r in all_results])
        sd = np.mean([r["self_disc"] for r in all_results])
        brier = np.mean([(r["p_self"] - (1.0 if r["self_disc"] else 0.0))**2
                          for r in all_results])
        p_und = np.mean([r.get("p_undecided", 0) for r in all_results])
        metrics[abl_name] = {
            "tbsr": round(tbsr, 4), "warn_rate": round(wr, 4),
            "sd_rate": round(sd, 4), "brier": round(brier, 4),
            "p_undecided_mean": round(p_und, 4),
        }

    # Table: All ablations
    L.append("## Ablation Results\n\n")
    L.append("| Ablation | TBSR | WarnRate | SD_Rate | Brier | p_undecided |\n")
    L.append("|:---------|:----:|:-------:|:-------:|:-----:|:----------:|\n")
    for abl, m in metrics.items():
        L.append(f"| {abl} | {m['tbsr']:.4f} | {m['warn_rate']:.4f} | "
                 f"{m['sd_rate']:.4f} | {m['brier']:.4f} | {m['p_undecided_mean']:.4f} |\n")

    # Verdicts
    L.append("\n## Verdicts\n\n")

    # Q1: LearnGain
    lg1 = metrics["LG1_behavior_loss"]
    lg2 = metrics["LG2_entropy_reduction"]
    L.append(f"> **LearnGain**: behavior_loss TBSR={lg1['tbsr']:.4f} vs "
             f"entropy_reduction TBSR={lg2['tbsr']:.4f}\n")
    if abs(lg1["tbsr"] - lg2["tbsr"]) < 0.02:
        L.append("> → Approximately equivalent. Keep behavior_loss (simpler).\n\n")
    else:
        L.append("> → Meaningful difference detected.\n\n")

    # Q2: DepCost
    dcs = metrics["DC_simple"]
    dcf = metrics["DC_full"]
    L.append(f"> **DepCost**: simple WR={dcs['warn_rate']:.4f} vs "
             f"full WR={dcf['warn_rate']:.4f}\n")
    if abs(dcs["warn_rate"] - dcf["warn_rate"]) < 0.02:
        L.append("> → ν̂ increment adds no discriminative value. Use simple.\n\n")
    else:
        L.append("> → ν̂ increment provides additional signal.\n\n")

    # Q3: p_self variants
    ps_b = metrics["PS_B_predictive"]["brier"]
    ps_a = metrics["PS_A_fusion"]["brier"]
    ps_c = metrics["PS_C_three_outcome"]["brier"]
    ps_base = metrics["PS_baseline"]["brier"]
    L.append(f"> **p_self variants** Brier: baseline={ps_base:.4f}, A={ps_a:.4f}, "
             f"B={ps_b:.4f}, C={ps_c:.4f}\n")
    best = min([(ps_base, "baseline"), (ps_a, "A"), (ps_b, "B"), (ps_c, "C")])
    L.append(f"> → Best calibration: **{best[1]}** (Brier={best[0]:.4f})\n\n")

    # Q4: p_fail = 1 - p_self wrong?
    p_und = metrics["PS_C_three_outcome"]["p_undecided_mean"]
    L.append(f"> **Three-outcome model**: mean p_undecided = {p_und:.4f}\n")
    if p_und > 0.05:
        L.append("> → p_fail = 1-p_self is **too coarse** — significant undecided mass.\n\n")
    else:
        L.append("> → p_fail ≈ 1-p_self is adequate for current scenarios.\n\n")

    # Q5: κ̂-aware
    ka = metrics["kappa_aware"]
    ko = metrics["kappa_off"]
    L.append(f"> **κ̂-aware**: TBSR={ka['tbsr']:.4f} vs 3D-only={ko['tbsr']:.4f}\n")
    if ka["tbsr"] > ko["tbsr"] + 0.02:
        L.append("> → κ̂ provides net benefit. Consider for future promotion.\n")
    else:
        L.append("> → κ̂ does not help. Keep 3D-only as default.\n")

    rpt = out / "step1_ablation_detail.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
