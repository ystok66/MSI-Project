"""T2 Exp-2C: TIC-v4 Longitudinal Audit — 3-Arm.

Three arms:
  - reset: each session fresh
  - persistent_nohook: carry-over, no curriculum hook
  - persistent_needhook: carry-over + profile-aware need bonus

TIC-v4 subtypes tested:
  - self_discovery_needed
  - false_suppression_cost
  - sparse_valid_advice
  - sparse_invalid_advice
  - beneficial_novelty
  - verified_warn

Key metrics:
  1. Per-subtype WarnRate (warn-necessary vs warn-unnecessary)
  2. SelGap, TBSR, APD
  3. E^calib_s = |m̂_T^(s) - m_T^(s)|_W
  4. E^drift_s = |(m̂_T^(s)-m̂_T^(s-1)) - (m_T^(s)-m_T^(s-1))|_W
  5. SatRate_d per dimension (lockout / washout detection)
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
from src.curriculum.curriculum_controller_v13 import CurriculumControllerV13
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.teachers.profile_state import ProfileState, SessionSummary
from src.teachers.profile_bootstrap import (
    bootstrap_observer, bootstrap_agent_state, finalize_session, make_need_hook,
)
from src.teachers.profile_manager import ProfileManager
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# TIC-v4 lessons (5 subtypes) + TIC (for baseline context)
TICV4_LESSONS = [l for l in LESSON_CATALOG_V2
                 if l.family in ("TIC-v4", "TIC")]
# Pure TIC-v4 for subtype analysis
TICV4_ONLY = [l for l in LESSON_CATALOG_V2 if l.family == "TIC-v4"]

N_SESSIONS = 5
N_STEPS_PER_SESSION = 25
N_SEEDS = 8

# Subtype → warn-necessary classification
WARN_NECESSARY = {"verified_warn", "warn_rescue"}
WARN_UNNECESSARY = {"beneficial_novelty", "false_suppression_cost",
                    "self_discovery_needed", "sparse_invalid_advice"}


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def sim_step(m, observer, theta, lesson, step_idx, seed, rng,
             tutor_active=True):
    """Simulate one teaching step."""
    ub = {p: 0.4 + 0.1 * (step_idx / N_STEPS_PER_SESSION) for p in PROBE_NAMES}
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
    lib = BranchConceptLibrary(); scr = BranchScorerProbe(lr=0.05, l2=0.01)
    ss = summarize_branch(sc.safe_cells, fb, fv, lp)
    sr = summarize_branch(sc.risky_cells, fb, fv, lp)
    lib.update("safe", ss); lib.update("risky", sr)
    scr.update(build_scorer_input(ss, lib), 1.0)
    scr.update(build_scorer_input(sr, lib), 0.0)

    if tutor_active:
        tutor = BCICTv4(agent_params=AP, use_dose=False)
        act, dose, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m)
    else:
        act, dose = "WAIT", 0.0

    dc = getattr(sc, 'commit_depth', 3)
    dr = getattr(sc, 'reveal_depth', 2)
    p_self = estimate_self_discovery_prob(dc, dr)
    risk = getattr(sc, 'risk_level', 0.3)
    tempt = getattr(sc, 'temptation_strength', 0.0)

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

    risk_hat = float(lp.predict_risk(
        sr[0:4] if len(sr) >= 4 else np.zeros(4)))
    ev = ObsEvent(
        episode_id=seed, step_id=step_idx, subtype=ep.subtype,
        theta_post=theta,
        dose=dose, warned=warned, follow_warn=(warned and correct),
        d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
        risk_hat=risk_hat, lure=tempt,
        agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
        self_discovery=self_disc,
    )
    observer.update(ev)

    return {
        "correct": correct, "warned": warned, "risk": risk,
        "self_disc": self_disc, "act": act,
        "subtype": lesson.subtype, "lesson": lesson.name,
    }


def run_longitudinal(theta, seed, mode="reset"):
    """Run N_SESSIONS of TIC-v4 mixed curriculum."""
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    pm = ProfileManager()
    lid = f"L_{seed}_{theta}_{mode}"

    all_sessions = []
    prev_m_hat = None
    prev_m_true = None

    for sess_k in range(N_SESSIONS):
        if mode != "reset" and pm.session_count(lid) > 0:
            prev = pm.latest(lid)
            bootstrap_observer(observer, prev)
            bootstrap_agent_state(m, prev)
        elif sess_k > 0 and mode == "reset":
            m = FactoredInternalizationState(); m.snapshot()
            observer = A1MtObserverFrozen(); observer.reset()

        ctrl = CurriculumControllerV13(theta=theta)
        ctrl.reset_session(budget=15.0)
        if mode == "persistent_needhook" and pm.session_count(lid) > 0:
            z_bar = pm.probe_weakness_summary(lid)
            hook = make_need_hook(z_bar, lambda_need=0.3)
            ctrl.install_profile_hook(hook)

        ss = SessionSummary()
        subtype_warn = defaultdict(int)
        subtype_total = defaultdict(int)
        step_results = []

        # Mix lessons: cycle through TIC-v4 + TIC
        for step_i in range(N_STEPS_PER_SESSION):
            les = TICV4_LESSONS[step_i % len(TICV4_LESSONS)]
            r = sim_step(m, observer, theta, les, step_i + sess_k * 100,
                        seed, rng, tutor_active=True)
            step_results.append(r)
            sub = r["subtype"]
            subtype_total[sub] += 1
            if r["warned"]:
                subtype_warn[sub] += 1
                ss.n_warn += 1
            else:
                ss.n_wait += 1

        # Compute per-subtype WarnRate
        for sub in subtype_total:
            ss.warn_rate_by_subtype[sub] = (
                subtype_warn[sub] / max(subtype_total[sub], 1))
        ss.subtype_counts = dict(subtype_total)
        ss.n_steps = N_STEPS_PER_SESSION

        # Probe means from observer estimate proxy
        est = observer.get_estimate()
        ss.probe_means = {
            "RC": min(est.get("tau", 0.3) * 1.5, 1.0),
            "TR": min(est.get("tau", 0.3), 1.0),
            "EP": max(1.0 - est.get("nu", 0.1) * 2, 0.0),
            "VA": min(est.get("tau", 0.3) * 1.2, 1.0),
            "IA": max(1.0 - est.get("gamma_gen", 0.0) * 2, 0.0),
        }

        # Finalize
        profile = finalize_session(observer, m, sess_k, theta, lid, ss)
        pm.finalize_session(lid, profile)

        # Current terminal states
        m_hat = dict(est)
        m_true = m.as_dict

        # Calibration error: |m̂_T - m_T|_W
        cal_dims = ["tau", "nu", "gamma_gen"]
        e_calib = {d: abs(m_hat.get(d, 0) - m_true.get(d, 0))
                   for d in cal_dims}
        e_calib_overall = float(np.mean(list(e_calib.values())))

        # Drift tracking error: |(Δm̂) - (Δm)| across sessions
        e_drift = {}
        if prev_m_hat is not None:
            for d in cal_dims:
                dm_hat = m_hat.get(d, 0) - prev_m_hat.get(d, 0)
                dm_true = m_true.get(d, 0) - prev_m_true.get(d, 0)
                e_drift[d] = abs(dm_hat - dm_true)
        e_drift_overall = float(np.mean(list(e_drift.values()))) if e_drift else 0.0

        # Saturation rate per dimension
        SAT_EPS = 0.05
        sat = {}
        for d in cal_dims:
            v = m_hat.get(d, 0.5)
            sat[d] = 1 if (v <= 0.0 + SAT_EPS or v >= 1.0 - SAT_EPS) else 0

        # Success metrics
        n = len(step_results)
        success = sum(1 for r in step_results if r["correct"]) / max(n, 1)
        wr_nec = sum(1 for r in step_results
                     if r["warned"] and r["subtype"] in WARN_NECESSARY
                     ) / max(sum(1 for r in step_results
                                if r["subtype"] in WARN_NECESSARY), 1)
        wr_unnec = sum(1 for r in step_results
                       if r["warned"] and r["subtype"] in WARN_UNNECESSARY
                       ) / max(sum(1 for r in step_results
                                  if r["subtype"] in WARN_UNNECESSARY), 1)
        selgap = wr_nec - wr_unnec

        all_sessions.append({
            "session_idx": sess_k,
            "success": round(success, 4),
            "warn_rate_by_subtype": dict(ss.warn_rate_by_subtype),
            "selgap": round(selgap, 4),
            "wr_necessary": round(wr_nec, 4),
            "wr_unnecessary": round(wr_unnec, 4),
            "e_calib": e_calib,
            "e_calib_overall": round(e_calib_overall, 6),
            "e_drift": e_drift,
            "e_drift_overall": round(e_drift_overall, 6),
            "sat": sat,
            "m_hat": dict(m_hat),
            "m_true": dict(m_true),
        })

        prev_m_hat = dict(m_hat)
        prev_m_true = dict(m_true)

    return all_sessions


def main():
    print("═══ T2 Exp-2C: TIC-v4 Longitudinal Audit ═══\n", file=sys.stderr)
    L = ["# T2 Exp-2C: TIC-v4 Longitudinal Audit — 3-Arm\n\n"]

    results = {}
    for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
        per_session = defaultdict(lambda: defaultdict(list))
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                sessions = run_longitudinal(th, sid, mode=mode)
                for s in sessions:
                    k = s["session_idx"]
                    per_session[(th, k)]["success"].append(s["success"])
                    per_session[(th, k)]["selgap"].append(s["selgap"])
                    per_session[(th, k)]["wr_nec"].append(s["wr_necessary"])
                    per_session[(th, k)]["wr_unnec"].append(s["wr_unnecessary"])
                    per_session[(th, k)]["e_calib"].append(s["e_calib_overall"])
                    per_session[(th, k)]["e_drift"].append(s["e_drift_overall"])
                    for d in ["tau", "nu", "gamma_gen"]:
                        per_session[(th, k)][f"sat_{d}"].append(s["sat"].get(d, 0))
                    for sub in s["warn_rate_by_subtype"]:
                        per_session[(th, k)][f"wr_{sub}"].append(
                            s["warn_rate_by_subtype"][sub])
            print(f"  {mode} / {th} done", file=sys.stderr)
        results[mode] = dict(per_session)

    # ═══ Table 1: Success + SelGap by Session ═══
    for th in ["safe", "shiny"]:
        L.append(f"\n## θ = {th}\n\n")
        L.append("### Success, SelGap, WarnRate\n\n")
        L.append("| Sess | Mode | Success | SelGap | WR_nec | WR_unnec |\n")
        L.append("|:----:|:----:|:-------:|:------:|:------:|:--------:|\n")
        for k in range(N_SESSIONS):
            for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
                d = results[mode].get((th, k), {})
                su = float(np.mean(d.get("success", [0])))
                sg = float(np.mean(d.get("selgap", [0])))
                wn = float(np.mean(d.get("wr_nec", [0])))
                wu = float(np.mean(d.get("wr_unnec", [0])))
                L.append(f"| {k} | {mode} | {su:.3f} | {sg:.3f} | "
                         f"{wn:.3f} | {wu:.3f} |\n")

    # ═══ Table 2: Per-Subtype WarnRate (final session) ═══
    L.append("\n## Per-Subtype WarnRate (Final Session)\n\n")
    subtypes = ["verified_warn", "warn_rescue", "sparse_valid_advice",
                "sparse_invalid_advice", "beneficial_novelty",
                "false_suppression_cost", "self_discovery_needed",
                "temptation_repeat"]
    L.append("| θ | Subtype | reset | nohook | needhook |\n")
    L.append("|:-:|:-------:|:-----:|:------:|:--------:|\n")
    for th in ["safe", "shiny"]:
        for sub in subtypes:
            vals = []
            for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
                d = results[mode].get((th, N_SESSIONS - 1), {})
                vals.append(float(np.mean(d.get(f"wr_{sub}", [0]))))
            L.append(f"| {th} | {sub} | {vals[0]:.3f} | "
                     f"{vals[1]:.3f} | {vals[2]:.3f} |\n")

    # ═══ Table 3: Calibration & Drift ═══
    L.append("\n## Calibration Error & Drift Tracking\n\n")
    L.append("| θ | Sess | Mode | E_calib | E_drift |\n")
    L.append("|:-:|:----:|:----:|:-------:|:-------:|\n")
    for th in ["safe", "shiny"]:
        for k in range(N_SESSIONS):
            for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
                d = results[mode].get((th, k), {})
                ec = float(np.mean(d.get("e_calib", [0])))
                ed = float(np.mean(d.get("e_drift", [0])))
                L.append(f"| {th} | {k} | {mode} | {ec:.4f} | {ed:.4f} |\n")

    # ═══ Table 4: Saturation Rate ═══
    L.append("\n## Saturation Rate (Lock/Wash Detection)\n\n")
    L.append("| θ | Mode | SatRate_τ | SatRate_ν | SatRate_γ |\n")
    L.append("|:-:|:----:|:---------:|:---------:|:---------:|\n")
    for th in ["safe", "shiny"]:
        for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
            sat_tau, sat_nu, sat_gg = [], [], []
            for k in range(N_SESSIONS):
                d = results[mode].get((th, k), {})
                sat_tau.extend(d.get("sat_tau", [0]))
                sat_nu.extend(d.get("sat_nu", [0]))
                sat_gg.extend(d.get("sat_gamma_gen", [0]))
            st = float(np.mean(sat_tau)) if sat_tau else 0
            sn = float(np.mean(sat_nu)) if sat_nu else 0
            sg = float(np.mean(sat_gg)) if sat_gg else 0
            L.append(f"| {th} | {mode} | {st:.3f} | {sn:.3f} | {sg:.3f} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check 1: persistent ≥ reset on success (Phase B proxy = later sessions)
    pass_success = 0
    for th in ["safe", "shiny"]:
        dr = results["reset"].get((th, N_SESSIONS-1), {})
        dp = results["persistent_nohook"].get((th, N_SESSIONS-1), {})
        sr = float(np.mean(dr.get("success", [0])))
        sp = float(np.mean(dp.get("success", [0])))
        if sp >= sr - 0.02:
            pass_success += 1
    L.append(f"> Success maintained: persistent ≥ reset in {pass_success}/2 θ\n")

    # Check 2: warn-unnecessary doesn't spike
    pass_warn = 0
    for th in ["safe", "shiny"]:
        dr = results["reset"].get((th, N_SESSIONS-1), {})
        dp = results["persistent_needhook"].get((th, N_SESSIONS-1), {})
        wr = float(np.mean(dr.get("wr_unnec", [0])))
        wp = float(np.mean(dp.get("wr_unnec", [0])))
        if wp <= wr + 0.05:
            pass_warn += 1
    L.append(f"> Warn-unnecessary no spike: {pass_warn}/2 θ\n")

    # Check 3: calibration not growing
    cal_growing = False
    for th in ["safe", "shiny"]:
        d0 = results["persistent_needhook"].get((th, 0), {})
        dN = results["persistent_needhook"].get((th, N_SESSIONS-1), {})
        c0 = float(np.mean(d0.get("e_calib", [0])))
        cN = float(np.mean(dN.get("e_calib", [0])))
        if cN > c0 + 0.05:
            cal_growing = True
    L.append(f"> Calibration error growing: {'⚠️ YES' if cal_growing else '✅ NO'}\n")

    # Check 4: saturation check
    sat_alert = False
    for th in ["safe", "shiny"]:
        for mode in ["persistent_nohook", "persistent_needhook"]:
            for k in range(N_SESSIONS):
                d = results[mode].get((th, k), {})
                for dim in ["sat_tau", "sat_nu", "sat_gamma_gen"]:
                    rate = float(np.mean(d.get(dim, [0])))
                    if rate > 0.5:
                        sat_alert = True
    L.append(f"> Saturation alert: {'⚠️ YES' if sat_alert else '✅ NO'}\n")

    # Check 5: needhook vs nohook differentiation
    hook_diff = 0
    for th in ["safe", "shiny"]:
        dn = results["persistent_nohook"].get((th, N_SESSIONS-1), {})
        dh = results["persistent_needhook"].get((th, N_SESSIONS-1), {})
        sn = float(np.mean(dn.get("success", [0])))
        sh = float(np.mean(dh.get("success", [0])))
        sgn = float(np.mean(dn.get("selgap", [0])))
        sgh = float(np.mean(dh.get("selgap", [0])))
        if sh > sn + 0.01 or sgh > sgn + 0.01:
            hook_diff += 1
    L.append(f"> needhook > nohook differentiation: {hook_diff}/2 θ\n")

    # Overall
    all_ok = pass_success >= 1 and pass_warn >= 1 and not cal_growing and not sat_alert
    if all_ok:
        if hook_diff >= 1:
            L.append("> **✅ Profile consumption validated: carry-over + hook both contribute**\n")
        else:
            L.append("> **✅ Carry-over validated. Hook not yet differentiated — "
                     "conclude Task 2 as 'carry-over is main effect'**\n")
    else:
        L.append("> **⚠️ Issues found — investigate before closing Task 2**\n")

    rpt = out / "t2_exp2c_ticv4_longitudinal.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
