"""T3 Exp-3A: Calibrated Persistence — reset vs raw vs calibrated.

Three arms:
  - reset: each session fresh
  - persistent_raw: η=1, no calibration (Task 2 baseline)
  - persistent_calibrated: η=1, use_calibration=True, λ_c=0.3

Focus: E_calib trend (should flatten under calibration).
Must not lose carry-over WarnRate benefit.
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
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.teachers.profile_state import ProfileState, SessionSummary
from src.teachers.profile_bootstrap import (
    bootstrap_observer, bootstrap_agent_state, finalize_session,
)
from src.teachers.profile_manager import ProfileManager
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

TIC_LESSONS = [l for l in LESSON_CATALOG_V2 if l.family in ("TIC", "TIC-v4")]
N_SESSIONS = 6
N_STEPS = 25
N_SEEDS = 8


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def sim_step(m, observer, theta, lesson, step_idx, seed, rng):
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

    tutor = BCICTv4(agent_params=AP, use_dose=False)
    act, dose, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m)

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
    return {"correct": correct, "warned": warned, "act": act}


def run_sessions(theta, seed, mode="reset"):
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    pm = ProfileManager()
    lid = f"L_{seed}_{theta}_{mode}"
    all_sessions = []

    for sess_k in range(N_SESSIONS):
        if mode != "reset" and pm.session_count(lid) > 0:
            prev = pm.latest(lid)
            use_cal = (mode == "persistent_calibrated")
            bootstrap_observer(observer, prev,
                             use_calibration=use_cal, lambda_c=0.3)
            bootstrap_agent_state(m, prev)
        elif sess_k > 0 and mode == "reset":
            m = FactoredInternalizationState(); m.snapshot()
            observer = A1MtObserverFrozen(); observer.reset()

        ss = SessionSummary()
        for step_i in range(N_STEPS):
            les = TIC_LESSONS[step_i % len(TIC_LESSONS)]
            r = sim_step(m, observer, theta, les, step_i + sess_k * 100, seed, rng)
            if r["warned"]: ss.n_warn += 1
            else: ss.n_wait += 1
        ss.n_steps = N_STEPS

        profile = finalize_session(observer, m, sess_k, theta, lid, ss)
        pm.finalize_session(lid, profile)

        est = observer.get_estimate()
        m_true = m.as_dict
        cal_dims = ["tau", "nu", "gamma_gen"]
        e_calib = {d: abs(est.get(d, 0) - m_true.get(d, 0)) for d in cal_dims}
        e_calib_overall = float(np.mean(list(e_calib.values())))

        all_sessions.append({
            "session_idx": sess_k,
            "warn_rate": ss.warn_rate,
            "e_calib": e_calib_overall,
            "e_calib_per_dim": e_calib,
            "m_hat": dict(est),
            "m_true": dict(m_true),
            "success": sum(1 for _ in range(1)) / 1,  # placeholder
        })
    return all_sessions


def main():
    print("═══ T3 Exp-3A: Calibrated Persistence ═══\n", file=sys.stderr)
    L = ["# T3 Exp-3A: Calibrated Persistence\n\n"]

    results = {}
    for mode in ["reset", "persistent_raw", "persistent_calibrated"]:
        per_session = defaultdict(lambda: defaultdict(list))
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                sessions = run_sessions(th, sid, mode=mode)
                for s in sessions:
                    k = s["session_idx"]
                    per_session[(th, k)]["wr"].append(s["warn_rate"])
                    per_session[(th, k)]["e_calib"].append(s["e_calib"])
                    for d in ["tau", "nu", "gamma_gen"]:
                        per_session[(th, k)][f"ec_{d}"].append(
                            s["e_calib_per_dim"][d])
            print(f"  {mode} / {th} done", file=sys.stderr)
        results[mode] = dict(per_session)

    # Table 1: E_calib by session
    L.append("## Calibration Error (E_calib) by Session\n\n")
    L.append("| θ | Sess | reset | raw | calibrated | cal < raw? |\n")
    L.append("|:-:|:----:|:-----:|:---:|:----------:|:----------:|\n")
    for th in ["safe", "shiny"]:
        for k in range(N_SESSIONS):
            er = float(np.mean(results["reset"].get((th, k), {}).get("e_calib", [0])))
            ep = float(np.mean(results["persistent_raw"].get((th, k), {}).get("e_calib", [0])))
            ec = float(np.mean(results["persistent_calibrated"].get((th, k), {}).get("e_calib", [0])))
            better = "✅" if ec < ep - 0.005 else ("≈" if abs(ec - ep) <= 0.005 else "❌")
            L.append(f"| {th} | {k} | {er:.4f} | {ep:.4f} | {ec:.4f} | {better} |\n")

    # Table 2: WarnRate by session (preserve carry-over benefit)
    L.append("\n## WarnRate by Session\n\n")
    L.append("| θ | Sess | reset | raw | calibrated |\n")
    L.append("|:-:|:----:|:-----:|:---:|:----------:|\n")
    for th in ["safe", "shiny"]:
        for k in range(N_SESSIONS):
            wr = float(np.mean(results["reset"].get((th, k), {}).get("wr", [0])))
            wp = float(np.mean(results["persistent_raw"].get((th, k), {}).get("wr", [0])))
            wc = float(np.mean(results["persistent_calibrated"].get((th, k), {}).get("wr", [0])))
            L.append(f"| {th} | {k} | {wr:.3f} | {wp:.3f} | {wc:.3f} |\n")

    # Table 3: E_calib per dimension (final session)
    L.append("\n## Per-Dimension Calibration (Final Session)\n\n")
    L.append("| θ | Mode | E_τ | E_ν | E_γ |\n")
    L.append("|:-:|:----:|:---:|:---:|:---:|\n")
    for th in ["safe", "shiny"]:
        for mode in ["reset", "persistent_raw", "persistent_calibrated"]:
            d = results[mode].get((th, N_SESSIONS - 1), {})
            et = float(np.mean(d.get("ec_tau", [0])))
            en = float(np.mean(d.get("ec_nu", [0])))
            eg = float(np.mean(d.get("ec_gamma_gen", [0])))
            L.append(f"| {th} | {mode} | {et:.4f} | {en:.4f} | {eg:.4f} |\n")

    # Verdict
    L.append("\n## Verdict\n\n")
    # Check 1: calibrated E_calib < raw E_calib at final session
    cal_better = 0
    for th in ["safe", "shiny"]:
        ep = float(np.mean(results["persistent_raw"].get((th, N_SESSIONS-1), {}).get("e_calib", [0])))
        ec = float(np.mean(results["persistent_calibrated"].get((th, N_SESSIONS-1), {}).get("e_calib", [0])))
        if ec < ep - 0.005:
            cal_better += 1
    L.append(f"> E_calib reduction: calibrated < raw in {cal_better}/2 θ\n")

    # Check 2: WarnRate not lost
    wr_ok = 0
    for th in ["safe", "shiny"]:
        wr = float(np.mean(results["reset"].get((th, N_SESSIONS-1), {}).get("wr", [0])))
        wc = float(np.mean(results["persistent_calibrated"].get((th, N_SESSIONS-1), {}).get("wr", [0])))
        if wc <= wr + 0.01:
            wr_ok += 1
    L.append(f"> WarnRate preserved: calibrated ≤ reset in {wr_ok}/2 θ\n")

    # Check 3: E_calib growth slope
    slope_flat = 0
    for th in ["safe", "shiny"]:
        ec_0 = float(np.mean(results["persistent_calibrated"].get((th, 0), {}).get("e_calib", [0])))
        ec_N = float(np.mean(results["persistent_calibrated"].get((th, N_SESSIONS-1), {}).get("e_calib", [0])))
        if ec_N - ec_0 < 0.03:
            slope_flat += 1
    L.append(f"> E_calib slope flattened: {slope_flat}/2 θ\n")

    if cal_better >= 1 and wr_ok >= 1:
        L.append("> **✅ Calibrated persistence reduces E_calib without losing carry-over**\n")
    else:
        L.append("> **⚠️ Calibration needs further investigation**\n")

    rpt = out / "t3_exp3a_calibrated_persistence.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
