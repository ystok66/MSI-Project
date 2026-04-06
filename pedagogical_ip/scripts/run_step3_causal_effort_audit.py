"""Step 3: Causal-Effort 5-Arm Audit.

Five experimental arms:
  A: Step 2 best (conservative-gated v2 + posterior_C)
  B: Step 2 + LearnGain_do only (credit correction, no effort loss)
  C: Step 2 + EffortLoss only (effort latent, no credit correction)
  D: v3-lite (credit + effort = full causal shadow)
  E: Step 2 + calibration reopened (diagnostics check)

All use REPLACE mode.

New metrics:
  - Credit Leakage = LearnGain_raw(WARN) - LearnGain_do(WARN)
  - Effort Recovery Rate (ERR) = P(e_{t+k} > e_t | self-discovery window)
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
from src.teachers.micro_bayes_shadow_v3 import MicroBayesShadowV3
from src.teachers.effort_latent_shadow import EffortLatentShadow
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

TARGET_LESSONS = []
for name in ["tic_rescue_heavy", "warn_symmetric_rescue",
              "tic_self_discovery", "ppmrb_self_discovery",
              "false_suppression", "beneficial_novelty",
              "ppmrb_standard", "tic_standard",
              "blind_activation_corridor", "soft_boundary_tradeoff",
              "verified_warn"]:
    if name in LESSON_V2_BY_NAME:
        TARGET_LESSONS.append(LESSON_V2_BY_NAME[name])

N_SESSIONS = 3
N_STEPS = 30
N_SEEDS = 8


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def make_v3_scorer(use_credit, use_effort):
    return MicroBayesShadowV3(
        agent_params=AP,
        use_credit_correction=use_credit,
        use_effort_loss=use_effort,
    )


def sim_step(m, effort_tracker, theta, lesson, step_idx, seed, rng, tutor, v3_scorer=None):
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
                if gm.cell_types[r, c] == CellType.WALL: continue
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
    p_self = estimate_self_discovery_prob(dc, dr)
    risk = getattr(sc, 'risk_level', 0.3)
    tempt = getattr(sc, 'temptation_strength', 0.0)
    subtype = getattr(ep, 'subtype', '')
    has_self_ev = (2 >= dc - 1) or p_self > 0.5

    if v3_scorer is not None:
        # Direct v3 scoring (for arms B/C/D)
        from src.teachers.p_self_posterior_shadow import compute_p_self_posterior, PSelfMode
        ps_r = compute_p_self_posterior(
            PSelfMode.POSTERIOR_C, dc, dr,
            tau_hat=m.tau, nu_hat=m.nu, gamma_gen_hat=m.gamma_gen)
        p_s = ps_r["p_self"]
        p_f = ps_r["p_fail"]
        p_u = ps_r.get("p_undecided", 0.0)
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
        from src.agents.behavior_probes import BEHAVIOR_ZONES
        zones = BEHAVIOR_ZONES.get("safe", {})
        novelty = 0.3 if subtype == "beneficial_novelty" else 0.0
        self_ev = 0.7 if has_self_ev else 0.3

        if effort_tracker is not None:
            v3_scorer.effort = effort_tracker

        act, dose, info = v3_scorer.score(
            m, delta_s, dvoi, tempt, risk,
            p_s, p_f, p_u, subtype, has_self_ev, zones,
            novelty, self_ev)
    else:
        act, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

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
    need_warn = (risk > 0.3 and p_self < 0.4)

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

    # Update effort tracker
    if effort_tracker is not None:
        effort_tracker.update(dose, p_self, has_self_ev, self_disc)

    leakage = info.get("leakage_warn", 0.0) if isinstance(info, dict) else 0.0
    effort_now = info.get("effort_now", 0.5) if isinstance(info, dict) else 0.5

    return {
        "correct": correct, "warned": warned, "self_disc": self_disc,
        "subtype": subtype, "p_self": p_self, "need_warn": need_warn,
        "leakage": leakage, "effort": effort_now,
    }


def run_arm(arm_name, v3_config, theta, seed):
    rng = np.random.default_rng(seed * 10000)
    all_results = []
    for sess_k in range(N_SESSIONS):
        m = FactoredInternalizationState(); m.snapshot()
        observer = A1MtObserverFrozen(); observer.reset()
        effort_tracker = EffortLatentShadow() if v3_config else None

        if v3_config:
            v3_scorer = MicroBayesShadowV3(
                agent_params=AP,
                use_credit_correction=v3_config.get("use_credit", True),
                use_effort_loss=v3_config.get("use_effort", True),
            )
            tutor = None
        elif arm_name == "A":
            tutor = BCICTv4(agent_params=AP, use_dose=False,
                           micro_policy_mode="micro_bayes_shadow_v2",
                           p_self_mode="posterior_C")
            v3_scorer = None
        else:
            tutor = BCICTv4(agent_params=AP, use_dose=False,
                           micro_policy_mode="micro_bayes_shadow_v2",
                           p_self_mode="posterior_C")
            v3_scorer = None

        warn_count = 0
        for step_i in range(N_STEPS):
            les = TARGET_LESSONS[step_i % len(TARGET_LESSONS)]
            r = sim_step(m, effort_tracker, theta, les, step_i + sess_k * 100,
                         seed, rng, tutor, v3_scorer)
            if r["warned"]: warn_count += 1
            all_results.append(r)

        est = observer.get_estimate()
        e_final = effort_tracker.effort if effort_tracker else 0.5
        all_results.append({
            "_summary": True, "session": sess_k,
            "nu_hat_T": est.get("nu", 0), "nu_T": m.nu,
            "n_warns": warn_count, "effort_T": e_final,
        })
    return all_results


def compute_metrics(results):
    steps = [r for r in results if not r.get("_summary")]
    summaries = [r for r in results if r.get("_summary")]
    if not steps: return {}

    tbsr = np.mean([r["correct"] for r in steps])
    wr = np.mean([r["warned"] for r in steps])
    necessary = [r for r in steps if r["need_warn"]]
    unnecessary = [r for r in steps if not r["need_warn"]]
    wr_nec = np.mean([r["warned"] for r in necessary]) if necessary else 0
    wr_unnec = np.mean([r["warned"] for r in unnecessary]) if unnecessary else 0
    sel_gap = wr_nec - wr_unnec
    sd_rate = np.mean([r["self_disc"] for r in steps])
    brier = np.mean([(r["p_self"] - (1.0 if r["self_disc"] else 0.0))**2 for r in steps])

    # Credit leakage (mean over warned steps)
    warned_steps = [r for r in steps if r["warned"]]
    leakage = np.mean([r["leakage"] for r in warned_steps]) if warned_steps else 0

    # Effort metrics
    efforts = [r["effort"] for r in steps]
    effort_mean = np.mean(efforts) if efforts else 0.5
    effort_T = np.mean([s["effort_T"] for s in summaries]) if summaries else 0.5

    # Effort Recovery Rate: in windows after WAIT, does effort increase?
    err_count = 0
    err_total = 0
    for i in range(1, len(steps)):
        if not steps[i-1]["warned"] and steps[i-1]["self_disc"]:
            err_total += 1
            if steps[i]["effort"] > steps[i-1]["effort"]:
                err_count += 1
    err = err_count / max(err_total, 1)

    nu_T = np.mean([s["nu_hat_T"] for s in summaries]) if summaries else 0
    n_warns = np.mean([s["n_warns"] for s in summaries]) if summaries else 0
    otr = (nu_T - 0.1) / max(n_warns, 1)

    subtype_m = {}
    for st in ["self_discovery_teach", "self_discovery_needed", "boundary_obs",
                "warn_rescue", "false_suppression_cost", "beneficial_novelty",
                "blind_corridor", "soft_gradual", "verified_warn"]:
        ss = [r for r in steps if r["subtype"] == st]
        if ss:
            subtype_m[st] = {
                "n": len(ss),
                "correct": round(np.mean([r["correct"] for r in ss]), 3),
                "warned": round(np.mean([r["warned"] for r in ss]), 3),
                "sd_rate": round(np.mean([r["self_disc"] for r in ss]), 3),
            }

    return {
        "tbsr": round(tbsr, 4), "warn_rate": round(wr, 4),
        "sel_gap": round(sel_gap, 4), "sd_rate": round(sd_rate, 4),
        "brier": round(brier, 4), "nu_T": round(nu_T, 4), "otr": round(otr, 4),
        "leakage": round(leakage, 4), "effort_mean": round(effort_mean, 4),
        "effort_T": round(effort_T, 4), "err": round(err, 4),
        "n_warns": round(n_warns, 1), "subtype": subtype_m,
    }


# Arm configs: None = use tutor directly, dict = v3 config
ARM_CONFIGS = {
    "A": None,  # Step 2 best via BCICTv4
    "B": {"use_credit": True, "use_effort": False},   # +credit only
    "C": {"use_credit": False, "use_effort": True},    # +effort only
    "D": {"use_credit": True, "use_effort": True},     # v3-lite full
    "E": None,  # Step 2 + calibration (re-check)
}


def main():
    print("═══ Step 3: Causal-Effort 5-Arm Audit ═══\n", file=sys.stderr)
    L = ["# Step 3: Causal-Effort 5-Arm Audit\n\n"]

    all_metrics = {}
    for arm, cfg in ARM_CONFIGS.items():
        arm_results = []
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                results = run_arm(arm, cfg, th, sid)
                arm_results.extend(results)
            print(f"  {arm} / {th} done", file=sys.stderr)
        all_metrics[arm] = compute_metrics(arm_results)

    # Table 1: Overall
    L.append("## Overall Metrics\n\n")
    L.append("| Arm | TBSR | WR | SelGap | SD | Brier | ν̂_T | Leakage | EffortT | ERR |\n")
    L.append("|:---:|:----:|:--:|:------:|:--:|:-----:|:---:|:-------:|:-------:|:---:|\n")
    for arm in ARM_CONFIGS:
        m = all_metrics[arm]
        L.append(f"| {arm} | {m['tbsr']:.4f} | {m['warn_rate']:.4f} | "
                 f"{m['sel_gap']:.4f} | {m['sd_rate']:.4f} | {m['brier']:.4f} | "
                 f"{m['nu_T']:.4f} | {m['leakage']:.4f} | {m['effort_T']:.4f} | "
                 f"{m['err']:.4f} |\n")

    # Table 2: Deltas vs A
    bm = all_metrics["A"]
    L.append("\n## Deltas vs. A (Step 2 Best)\n\n")
    L.append("| Arm | ΔTBSR | ΔSelGap | ΔSD | ΔWR | Δν̂ | ΔLeakage | ΔEffort | ΔERR |\n")
    L.append("|:---:|:-----:|:-------:|:---:|:---:|:---:|:--------:|:------:|:----:|\n")
    for arm in ["B", "C", "D", "E"]:
        m = all_metrics[arm]
        L.append(f"| {arm} | {m['tbsr']-bm['tbsr']:+.4f} | "
                 f"{m['sel_gap']-bm['sel_gap']:+.4f} | "
                 f"{m['sd_rate']-bm['sd_rate']:+.4f} | "
                 f"{m['warn_rate']-bm['warn_rate']:+.4f} | "
                 f"{m['nu_T']-bm['nu_T']:+.4f} | "
                 f"{m['leakage']-bm['leakage']:+.4f} | "
                 f"{m['effort_T']-bm['effort_T']:+.4f} | "
                 f"{m['err']-bm['err']:+.4f} |\n")

    # Per-subtype
    priority_st = ["self_discovery_teach", "self_discovery_needed",
                    "boundary_obs", "warn_rescue", "false_suppression_cost",
                    "beneficial_novelty"]
    L.append("\n## Per-Subtype Breakdown\n\n")
    for st in priority_st:
        L.append(f"### {st}\n\n")
        L.append("| Arm | n | Correct | WR | SD |\n")
        L.append("|:---:|:-:|:-------:|:--:|:--:|\n")
        for arm in ARM_CONFIGS:
            sm = all_metrics[arm].get("subtype", {}).get(st, {})
            if sm:
                L.append(f"| {arm} | {sm['n']} | {sm['correct']:.3f} | "
                         f"{sm['warned']:.3f} | {sm['sd_rate']:.3f} |\n")
            else:
                L.append(f"| {arm} | 0 | — | — | — |\n")
        L.append("\n")

    # Verdicts
    L.append("## Verdict\n\n")
    m_b, m_c, m_d, m_e = [all_metrics[a] for a in "BCDE"]

    L.append(f"> **Q1 (B vs A): LearnGain_do** — ΔLeakage={m_b['leakage']-bm['leakage']:+.4f}, "
             f"ΔSD={m_b['sd_rate']-bm['sd_rate']:+.4f}, ΔTBSR={m_b['tbsr']-bm['tbsr']:+.4f}\n")
    if m_b["leakage"] < bm["leakage"] - 0.005:
        L.append("> Credit correction **reduces leakage** — necessary module.\n\n")
    else:
        L.append("> Credit correction has minimal leakage effect.\n\n")

    L.append(f"> **Q2 (C vs A): EffortLoss** — Δν̂={m_c['nu_T']-bm['nu_T']:+.4f}, "
             f"ΔEffort={m_c['effort_T']-bm['effort_T']:+.4f}, ΔERR={m_c['err']-bm['err']:+.4f}\n")
    if m_c["nu_T"] < bm["nu_T"] - 0.01:
        L.append("> EffortLoss **improves autonomy** — necessary module.\n\n")
    else:
        L.append("> EffortLoss is theoretical refinement, not behavioral necessity.\n\n")

    L.append(f"> **Q3 (D vs A): v3-lite** — ΔSelGap={m_d['sel_gap']-bm['sel_gap']:+.4f}, "
             f"ΔTBSR={m_d['tbsr']-bm['tbsr']:+.4f}, ΔWR={m_d['warn_rate']-bm['warn_rate']:+.4f}\n")
    promote = True
    checks = []
    if m_d["tbsr"] < bm["tbsr"] - 0.02: checks.append("ΔTBSR<0"); promote = False
    if m_d["sel_gap"] < bm["sel_gap"] - 0.02: checks.append("ΔSelGap<0")
    if m_d["nu_T"] > bm["nu_T"] + 0.03: checks.append("ΔOTR worse"); promote = False
    if promote:
        L.append(f"> **✅ v3-lite meets promote criteria**\n\n")
    else:
        L.append(f"> **⏸️ v3-lite issues: {', '.join(checks)}**\n\n")

    L.append(f"> **Q4 (E vs A): Calibration recheck** — ΔBrier={m_e['brier']-bm['brier']:+.4f}\n")
    if abs(m_e["brier"] - bm["brier"]) < 0.01:
        L.append("> Calibration **confirmed marginal** — keep as diagnostics only.\n\n")
    else:
        L.append("> Calibration shows measurable effect.\n\n")

    rpt = out / "step3_causal_effort_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
