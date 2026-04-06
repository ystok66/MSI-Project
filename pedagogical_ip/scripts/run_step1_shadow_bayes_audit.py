"""Step 1: Shadow Bayes Audit — 5-arm comparison.

Five experimental arms:
  A: canonical BCICTv4 + baseline p_self
  B: old Phase 7 shadow (EPU + belief-horizon blend)
  C: micro_bayes_shadow only (canonical p_self)
  D: p_self_posterior_shadow only (canonical micro)
  E: micro_bayes_shadow + p_self_posterior_shadow (both)

Arms C/D/E use REPLACE mode: shadow scorer controls actual decisions.

Required subtypes:
  self_discovery_teach, self_discovery_needed, boundary_obs,
  warn_rescue, false_suppression_cost

Metrics:
  Calibration (Brier, ECE), SelGap, TBSR, APD, OTR, Δν̂/n_int
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
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES, LESSON_V2_BY_NAME
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# Target lessons — cover all 5 required subtypes
TARGET_LESSONS = []
for name in ["tic_rescue_heavy", "warn_symmetric_rescue",          # warn_rescue
              "tic_self_discovery", "ppmrb_self_discovery",         # self_discovery_teach/needed
              "false_suppression", "beneficial_novelty",            # false_suppression_cost
              "ppmrb_standard", "tic_standard"]:                    # boundary_obs
    if name in LESSON_V2_BY_NAME:
        TARGET_LESSONS.append(LESSON_V2_BY_NAME[name])

# Arm definitions
ARMS = {
    "A": {"micro_policy_mode": "canonical", "p_self_mode": "baseline",
           "use_epu_shadow": False, "use_belief_horizon_pself": False},
    "B": {"micro_policy_mode": "canonical", "p_self_mode": "baseline",
           "use_epu_shadow": True, "use_belief_horizon_pself": True},
    "C": {"micro_policy_mode": "micro_bayes_shadow", "p_self_mode": "baseline",
           "use_epu_shadow": False, "use_belief_horizon_pself": False},
    "D": {"micro_policy_mode": "canonical", "p_self_mode": "posterior_B",
           "use_epu_shadow": False, "use_belief_horizon_pself": False},
    "E": {"micro_policy_mode": "micro_bayes_shadow", "p_self_mode": "posterior_B",
           "use_epu_shadow": False, "use_belief_horizon_pself": False},
}

N_SESSIONS = 3
N_STEPS = 25
N_SEEDS = 8


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def make_tutor(arm_cfg):
    return BCICTv4(
        agent_params=AP,
        use_dose=False,
        micro_policy_mode=arm_cfg["micro_policy_mode"],
        p_self_mode=arm_cfg["p_self_mode"],
        use_epu_shadow=arm_cfg.get("use_epu_shadow", False),
        use_belief_horizon_pself=arm_cfg.get("use_belief_horizon_pself", False),
    )


def sim_step(m, observer, theta, lesson, step_idx, seed, rng, tutor):
    ub = {p: 0.4 + 0.1 * (step_idx / N_STEPS) for p in PROBE_NAMES}
    et = generate_episode_from_lesson_v2(
        lesson, step_idx + seed * 10000, theta, ub, rng)
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
    lib.update("safe", ss)
    lib.update("risky", sr)
    scr.update(build_scorer_input(ss, lib), 1.0)
    scr.update(build_scorer_input(sr, lib), 0.0)

    act, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

    dc = getattr(sc, 'commit_depth', 3)
    dr = getattr(sc, 'reveal_depth', 2)
    p_self = estimate_self_discovery_prob(dc, dr)
    risk = getattr(sc, 'risk_level', 0.3)
    tempt = getattr(sc, 'temptation_strength', 0.0)
    subtype = getattr(ep, 'subtype', '')

    bas = BranchAttributes(
        safety_score=float(ss[0]), risk_penalty=0.1,
        temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                          else sc.tempt_score_b)
    bar = BranchAttributes(
        safety_score=float(sr[0]), risk_penalty=risk,
        temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                          else sc.tempt_score_a)

    ac = sample_factored_choice([bas, bar], theta, m, AP, rng,
                                [0.0, 0.0], [False, False])
    correct = (ac == sc.oracle_safe_branch_id)
    warned = dose > 0
    self_disc = correct and not warned and p_self > 0.5

    # Update m_t
    if warned:
        m.update_trust(warn_helpful=(risk > 0.25 and correct))
        if p_self < 0.5:
            m.update_dependence(blind_obey=True)
        m.update_gamma_gen(sustained_pressure=True)
    else:
        if self_disc:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
    if not correct and tempt > 0.5:
        m.update_gamma_spec(tempt_error=True)
    m.update_risk(risk if not correct else 0.05, 0.15)
    m.snapshot()

    risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
    ev = ObsEvent(
        episode_id=seed, step_id=step_idx, subtype=subtype,
        theta_post=theta, dose=dose, warned=warned,
        follow_warn=(warned and correct),
        d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
        risk_hat=risk_hat, lure=tempt,
        agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
        self_discovery=self_disc,
    )
    observer.update(ev)

    # Determine if WARN was necessary
    need_warn = (risk > 0.3 and p_self < 0.4)

    return {
        "correct": correct, "warned": warned,
        "self_disc": self_disc, "subtype": subtype,
        "p_self": p_self, "p_fail": 1.0 - p_self,
        "need_warn": need_warn,
        "info": info,
    }


def run_arm(arm_name, arm_cfg, theta, seed):
    rng = np.random.default_rng(seed * 10000)
    all_results = []

    for sess_k in range(N_SESSIONS):
        m = FactoredInternalizationState()
        m.snapshot()
        observer = A1MtObserverFrozen()
        observer.reset()
        tutor = make_tutor(arm_cfg)

        for step_i in range(N_STEPS):
            les = TARGET_LESSONS[step_i % len(TARGET_LESSONS)]
            result = sim_step(m, observer, theta, les, step_i + sess_k * 100,
                              seed, rng, tutor)
            all_results.append(result)

        est = observer.get_estimate()
        all_results.append({
            "_summary": True,
            "session": sess_k,
            "nu_hat_T": est.get("nu", 0),
            "tau_hat_T": est.get("tau", 0),
            "gg_hat_T": est.get("gamma_gen", 0),
            "nu_T": m.nu,
            "n_warns": tutor.warn_count,
            "n_waits": tutor.wait_count,
        })

    return all_results


def compute_metrics(results, arm_name):
    """Compute metrics from list of step results."""
    steps = [r for r in results if not r.get("_summary")]
    summaries = [r for r in results if r.get("_summary")]

    if not steps:
        return {}

    # TBSR: fraction correct (proxy for finishing before timeout)
    tbsr = np.mean([r["correct"] for r in steps])

    # WarnRate
    warn_rate = np.mean([r["warned"] for r in steps])

    # SelGap
    necessary = [r for r in steps if r["need_warn"]]
    unnecessary = [r for r in steps if not r["need_warn"]]
    wr_nec = np.mean([r["warned"] for r in necessary]) if necessary else 0
    wr_unnec = np.mean([r["warned"] for r in unnecessary]) if unnecessary else 0
    sel_gap = wr_nec - wr_unnec

    # Self-discovery rate
    sd_rate = np.mean([r["self_disc"] for r in steps])

    # Calibration: Brier score for p_self
    brier_pself = np.mean([
        (r["p_self"] - (1.0 if r["self_disc"] else 0.0)) ** 2
        for r in steps
    ])

    # ECE for p_self (5 bins)
    bins = [[] for _ in range(5)]
    for r in steps:
        b = min(int(r["p_self"] * 5), 4)
        actual = 1.0 if r["self_disc"] else 0.0
        bins[b].append((r["p_self"], actual))
    ece = 0.0
    total = len(steps)
    for i, b in enumerate(bins):
        if not b:
            continue
        conf = np.mean([x[0] for x in b])
        acc = np.mean([x[1] for x in b])
        ece += (len(b) / total) * abs(acc - conf)

    # OTR: mean terminal ν̂
    nu_T_mean = np.mean([s["nu_hat_T"] for s in summaries]) if summaries else 0
    n_warns_mean = np.mean([s["n_warns"] for s in summaries]) if summaries else 0
    delta_nu_per_int = (nu_T_mean - 0.1) / max(n_warns_mean, 1)

    # Per-subtype breakdown
    subtype_metrics = {}
    for st in ["self_discovery_teach", "self_discovery_needed", "boundary_obs",
                "warn_rescue", "false_suppression_cost", "warn_trap",
                "temptation_repeat", "wait_clean"]:
        st_steps = [r for r in steps if r["subtype"] == st]
        if st_steps:
            subtype_metrics[st] = {
                "n": len(st_steps),
                "correct": np.mean([r["correct"] for r in st_steps]),
                "warned": np.mean([r["warned"] for r in st_steps]),
                "sd_rate": np.mean([r["self_disc"] for r in st_steps]),
            }

    return {
        "tbsr": round(tbsr, 4),
        "warn_rate": round(warn_rate, 4),
        "sel_gap": round(sel_gap, 4),
        "sd_rate": round(sd_rate, 4),
        "brier_pself": round(brier_pself, 4),
        "ece_pself": round(ece, 4),
        "nu_T_mean": round(nu_T_mean, 4),
        "delta_nu_per_int": round(delta_nu_per_int, 4),
        "n_warns_mean": round(n_warns_mean, 1),
        "subtype": subtype_metrics,
    }


def main():
    print("═══ Step 1: Shadow Bayes 5-Arm Audit ═══\n", file=sys.stderr)
    L = ["# Step 1: Shadow Bayes 5-Arm Audit\n\n"]

    all_metrics = {}
    for arm_name, arm_cfg in ARMS.items():
        arm_results = []
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                results = run_arm(arm_name, arm_cfg, th, sid)
                arm_results.extend(results)
            print(f"  {arm_name} / {th} done", file=sys.stderr)
        metrics = compute_metrics(arm_results, arm_name)
        all_metrics[arm_name] = metrics

    # Table 1: Overall metrics
    L.append("## Overall Metrics\n\n")
    L.append("| Arm | TBSR | WarnRate | SelGap | SD_Rate | Brier_pself | ECE_pself | ν̂_T | Δν̂/int |\n")
    L.append("|:---:|:----:|:-------:|:------:|:-------:|:----------:|:--------:|:---:|:------:|\n")
    for arm in ARMS:
        m = all_metrics[arm]
        L.append(f"| {arm} | {m['tbsr']:.4f} | {m['warn_rate']:.4f} | "
                 f"{m['sel_gap']:.4f} | {m['sd_rate']:.4f} | "
                 f"{m['brier_pself']:.4f} | {m['ece_pself']:.4f} | "
                 f"{m['nu_T_mean']:.4f} | {m['delta_nu_per_int']:.4f} |\n")

    # Table 2: Deltas vs. baseline (A)
    L.append("\n## Deltas vs. Baseline (A)\n\n")
    L.append("| Arm | ΔTBSR | ΔSelGap | ΔSD_Rate | ΔBrier | ΔECE | ΔOTR(ν̂) |\n")
    L.append("|:---:|:-----:|:-------:|:--------:|:-----:|:----:|:-------:|\n")
    bm = all_metrics["A"]
    for arm in ["B", "C", "D", "E"]:
        m = all_metrics[arm]
        L.append(f"| {arm} | {m['tbsr']-bm['tbsr']:+.4f} | "
                 f"{m['sel_gap']-bm['sel_gap']:+.4f} | "
                 f"{m['sd_rate']-bm['sd_rate']:+.4f} | "
                 f"{m['brier_pself']-bm['brier_pself']:+.4f} | "
                 f"{m['ece_pself']-bm['ece_pself']:+.4f} | "
                 f"{m['nu_T_mean']-bm['nu_T_mean']:+.4f} |\n")

    # Table 3: Per-subtype breakdown (priority subtypes)
    priority_st = ["self_discovery_teach", "self_discovery_needed",
                    "boundary_obs", "warn_rescue", "false_suppression_cost"]
    L.append("\n## Per-Subtype Breakdown\n\n")
    for st in priority_st:
        L.append(f"### {st}\n\n")
        L.append("| Arm | n | Correct | WarnRate | SD_Rate |\n")
        L.append("|:---:|:-:|:-------:|:-------:|:-------:|\n")
        for arm in ARMS:
            sm = all_metrics[arm].get("subtype", {}).get(st, {})
            if sm:
                L.append(f"| {arm} | {sm['n']} | {sm['correct']:.3f} | "
                         f"{sm['warned']:.3f} | {sm['sd_rate']:.3f} |\n")
            else:
                L.append(f"| {arm} | 0 | — | — | — |\n")
        L.append("\n")

    # Verdict
    L.append("## Verdict\n\n")

    m_c = all_metrics["C"]
    m_d = all_metrics["D"]
    m_e = all_metrics["E"]

    # Q1: micro_bayes_shadow only (C vs A)
    c_vs_a_selgap = m_c["sel_gap"] - bm["sel_gap"]
    L.append(f"> **Q1 (C vs A): micro_bayes only** — ΔSelGap={c_vs_a_selgap:+.4f}, "
             f"ΔTBSR={m_c['tbsr']-bm['tbsr']:+.4f}\n")
    if abs(c_vs_a_selgap) < 0.02 and abs(m_c["tbsr"] - bm["tbsr"]) < 0.02:
        L.append("> Likely **narrative unification only**, not behavioral improvement\n\n")
    else:
        L.append("> Shows **behavioral impact**\n\n")

    # Q2: p_self posterior only (D vs A)
    d_vs_a_selgap = m_d["sel_gap"] - bm["sel_gap"]
    d_vs_a_brier = m_d["brier_pself"] - bm["brier_pself"]
    L.append(f"> **Q2 (D vs A): p_self_posterior only** — ΔSelGap={d_vs_a_selgap:+.4f}, "
             f"ΔBrier={d_vs_a_brier:+.4f}\n")
    if d_vs_a_brier < -0.01:
        L.append("> **p_self posterior improves calibration** — likely worth keeping\n\n")
    else:
        L.append("> p_self posterior does not improve calibration\n\n")

    # Q3: both (E vs C)
    e_vs_c_selgap = m_e["sel_gap"] - m_c["sel_gap"]
    L.append(f"> **Q3 (E vs C): adding p_self to micro_bayes** — ΔSelGap={e_vs_c_selgap:+.4f}\n\n")

    # Promote criteria
    promote = True
    if m_e["tbsr"] < bm["tbsr"] - 0.02:
        L.append("> ⚠️ ΔTBSR < 0: does not meet promote criterion\n")
        promote = False
    if m_e["sel_gap"] <= bm["sel_gap"]:
        L.append("> ⚠️ ΔSelGap ≤ 0: does not meet promote criterion\n")
        promote = False
    if m_e["nu_T_mean"] > bm["nu_T_mean"] + 0.03:
        L.append("> ⚠️ ΔOTR significantly worse\n")
        promote = False
    if promote:
        L.append("> **✅ Arm E meets promote criteria**\n")
    else:
        L.append("> **⏸️ Continue as shadow-only, investigate further**\n")

    rpt = out / "step1_shadow_bayes_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
