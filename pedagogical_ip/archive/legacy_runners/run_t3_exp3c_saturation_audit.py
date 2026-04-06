"""T3 Exp-3C: ν/γ_gen Saturation Audit — Curriculum-Conditioned.

4 curriculum conditions:
  - no_tutor: tutor always WAITs
  - rescue_heavy: tic_rescue_heavy + warn_symmetric_rescue
  - self_disc: tic_self_discovery + beneficial_novelty + false_suppression
  - mixed: full TIC + TIC-v4 catalog

Key question: Is saturation from state dynamics × curriculum interaction?

Metrics:
  - SatRate_d per dimension
  - Terminal state distribution
  - ν/γ_gen trajectory
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

# Curriculum groups
RESCUE_HEAVY = [LESSON_V2_BY_NAME[n] for n in
                ["tic_rescue_heavy", "warn_symmetric_rescue", "blind_activation_corridor"]]
SELF_DISC = [LESSON_V2_BY_NAME[n] for n in
             ["tic_self_discovery", "beneficial_novelty", "false_suppression",
              "ppmrb_self_discovery"]]
MIXED = [l for l in LESSON_CATALOG_V2 if l.family in ("TIC", "TIC-v4", "ACTIVE")]

N_SESSIONS = 4
N_STEPS = 30
N_SEEDS = 10
SAT_EPS = 0.05


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def sim_step(m, observer, theta, lesson, step_idx, seed, rng,
             tutor_active=True):
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

    risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
    ev = ObsEvent(
        episode_id=seed, step_id=step_idx, subtype=ep.subtype,
        theta_post=theta, dose=dose, warned=warned,
        follow_warn=(warned and correct),
        d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
        risk_hat=risk_hat, lure=tempt,
        agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
        self_discovery=self_disc,
    )
    observer.update(ev)
    return {"correct": correct, "warned": warned}


def run_curriculum(theta, seed, curriculum_name, lessons, tutor_on=True):
    rng = np.random.default_rng(seed * 10000)
    all_sessions = []

    for sess_k in range(N_SESSIONS):
        m = FactoredInternalizationState(); m.snapshot()
        observer = A1MtObserverFrozen(); observer.reset()

        nu_trace = []
        gg_trace = []

        for step_i in range(N_STEPS):
            les = lessons[step_i % len(lessons)]
            sim_step(m, observer, theta, les, step_i + sess_k * 100, seed, rng,
                    tutor_active=tutor_on)
            nu_trace.append(m.nu)
            gg_trace.append(m.gamma_gen)

        est = observer.get_estimate()
        m_true = m.as_dict

        sat = {}
        for d, val in [("tau", est.get("tau", 0.3)),
                       ("nu", est.get("nu", 0.1)),
                       ("gamma_gen", est.get("gamma_gen", 0.0))]:
            if d == "nu":
                sat[d] = 1 if val >= 0.5 - SAT_EPS or val <= SAT_EPS else 0
            elif d == "gamma_gen":
                sat[d] = 1 if val >= 0.5 - SAT_EPS or val <= SAT_EPS else 0
            else:
                sat[d] = 1 if val >= 1.0 - SAT_EPS or val <= SAT_EPS else 0

        all_sessions.append({
            "session_idx": sess_k,
            "sat": sat,
            "nu_T": m_true.get("nu", 0),
            "gg_T": m_true.get("gamma_gen", 0),
            "nu_hat_T": est.get("nu", 0),
            "gg_hat_T": est.get("gamma_gen", 0),
            "tau_hat_T": est.get("tau", 0),
            "nu_mid": float(np.mean(nu_trace[10:20])),
            "gg_mid": float(np.mean(gg_trace[10:20])),
        })
    return all_sessions


def main():
    print("═══ T3 Exp-3C: Saturation Audit ═══\n", file=sys.stderr)
    L = ["# T3 Exp-3C: ν/γ_gen Saturation Audit\n\n"]

    curricula = [
        ("no_tutor", MIXED, False),
        ("rescue_heavy", RESCUE_HEAVY, True),
        ("self_disc", SELF_DISC, True),
        ("mixed", MIXED, True),
    ]

    results = {}
    for cur_name, lessons, tutor_on in curricula:
        per_session = defaultdict(lambda: defaultdict(list))
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                sessions = run_curriculum(th, sid, cur_name, lessons, tutor_on)
                for s in sessions:
                    k = s["session_idx"]
                    per_session[(th, k)]["sat_nu"].append(s["sat"]["nu"])
                    per_session[(th, k)]["sat_gg"].append(s["sat"]["gamma_gen"])
                    per_session[(th, k)]["sat_tau"].append(s["sat"]["tau"])
                    per_session[(th, k)]["nu_T"].append(s["nu_T"])
                    per_session[(th, k)]["gg_T"].append(s["gg_T"])
                    per_session[(th, k)]["nu_hat"].append(s["nu_hat_T"])
                    per_session[(th, k)]["gg_hat"].append(s["gg_hat_T"])
                    per_session[(th, k)]["nu_mid"].append(s["nu_mid"])
                    per_session[(th, k)]["gg_mid"].append(s["gg_mid"])
            print(f"  {cur_name} / {th} done", file=sys.stderr)
        results[cur_name] = dict(per_session)

    # Table 1: SatRate by curriculum (aggregated)
    L.append("## SatRate by Curriculum (All Sessions Aggregated)\n\n")
    L.append("| θ | Curriculum | SatRate_ν | SatRate_γ | SatRate_τ |\n")
    L.append("|:-:|:----------:|:---------:|:---------:|:---------:|\n")
    for th in ["safe", "shiny"]:
        for cur_name, _, _ in curricula:
            sn, sg, st = [], [], []
            for k in range(N_SESSIONS):
                d = results[cur_name].get((th, k), {})
                sn.extend(d.get("sat_nu", []))
                sg.extend(d.get("sat_gg", []))
                st.extend(d.get("sat_tau", []))
            L.append(f"| {th} | {cur_name} | {np.mean(sn):.3f} | "
                     f"{np.mean(sg):.3f} | {np.mean(st):.3f} |\n")

    # Table 2: Terminal state means
    L.append("\n## Terminal State Means (Final Session)\n\n")
    L.append("| θ | Curriculum | ν_T | γ_gen_T | ν̂_T | γ̂_gen_T |\n")
    L.append("|:-:|:----------:|:---:|:-------:|:---:|:-------:|\n")
    for th in ["safe", "shiny"]:
        for cur_name, _, _ in curricula:
            d = results[cur_name].get((th, N_SESSIONS-1), {})
            L.append(f"| {th} | {cur_name} | "
                     f"{np.mean(d.get('nu_T', [0])):.3f} | "
                     f"{np.mean(d.get('gg_T', [0])):.3f} | "
                     f"{np.mean(d.get('nu_hat', [0])):.3f} | "
                     f"{np.mean(d.get('gg_hat', [0])):.3f} |\n")

    # Table 3: Mid-session trajectory (steps 10-20)
    L.append("\n## Mid-Session Trajectory (Steps 10-20, Final Session)\n\n")
    L.append("| θ | Curriculum | ν_mid | γ_gen_mid |\n")
    L.append("|:-:|:----------:|:-----:|:---------:|\n")
    for th in ["safe", "shiny"]:
        for cur_name, _, _ in curricula:
            d = results[cur_name].get((th, N_SESSIONS-1), {})
            L.append(f"| {th} | {cur_name} | "
                     f"{np.mean(d.get('nu_mid', [0])):.3f} | "
                     f"{np.mean(d.get('gg_mid', [0])):.3f} |\n")

    # Verdict
    L.append("\n## Verdict\n\n")

    # Check: rescue_heavy > self_disc in saturation?
    rescue_vs_disc = 0
    for th in ["safe", "shiny"]:
        sr_nu, sd_nu = [], []
        for k in range(N_SESSIONS):
            dr = results["rescue_heavy"].get((th, k), {})
            ds = results["self_disc"].get((th, k), {})
            sr_nu.extend(dr.get("sat_nu", []))
            sd_nu.extend(ds.get("sat_nu", []))
        if np.mean(sr_nu) > np.mean(sd_nu) + 0.05:
            rescue_vs_disc += 1
    L.append(f"> rescue_heavy SatRate_ν > self_disc: {rescue_vs_disc}/2 θ\n")

    # Check: no_tutor has lower saturation?
    tutor_off_lower = 0
    for th in ["safe", "shiny"]:
        snt, snm = [], []
        for k in range(N_SESSIONS):
            dt = results["no_tutor"].get((th, k), {})
            dm = results["mixed"].get((th, k), {})
            snt.extend(dt.get("sat_gg", []))
            snm.extend(dm.get("sat_gg", []))
        if np.mean(snt) < np.mean(snm) - 0.05:
            tutor_off_lower += 1
    L.append(f"> no_tutor SatRate_γ < mixed: {tutor_off_lower}/2 θ\n")

    # Check: self_disc ν significantly lower?
    disc_nu_lower = 0
    for th in ["safe", "shiny"]:
        d_disc = results["self_disc"].get((th, N_SESSIONS-1), {})
        d_mix = results["mixed"].get((th, N_SESSIONS-1), {})
        nu_disc = np.mean(d_disc.get("nu_T", [0]))
        nu_mix = np.mean(d_mix.get("nu_T", [0]))
        if nu_disc < nu_mix - 0.02:
            disc_nu_lower += 1
    L.append(f"> self_disc ν_T < mixed: {disc_nu_lower}/2 θ\n")

    if rescue_vs_disc >= 1 or tutor_off_lower >= 1:
        L.append("> **✅ Saturation is curriculum-dependent: rescue/tutor inflates ν/γ_gen, "
                 "self-discovery preserves them**\n")
    else:
        L.append("> **⚠️ Saturation appears state-dynamics intrinsic, not curriculum-modulated**\n")

    rpt = out / "t3_exp3c_saturation_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
