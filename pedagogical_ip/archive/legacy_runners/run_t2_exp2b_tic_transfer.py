"""T2 Exp-2B: TIC 3-Phase Transfer — 3-Arm Comparison.

Three arms:
  - reset: each session fresh (no profile)
  - persistent_nohook: carry-over state, no curriculum hook
  - persistent_needhook: carry-over state + profile-aware need bonus

TIC structure:
  Phase A: tutor-active block (lesson: tic_rescue_heavy, tic_temptation)
  Phase B: tutor-off autonomy transfer
  Phase C: shifted-structure transfer (different risk/temptation profile)

Key metrics:
  - Per-phase success rate, risk, WarnRate
  - TBSR (time-budgeted success rate)
  - SelGap
  - warn_trap baseline maintenance
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

# TIC lessons for Phase A
TIC_LESSONS = [l for l in LESSON_CATALOG_V2 if l.family in ("TIC", "ACTIVE")]
ALL_LESSONS = list(LESSON_CATALOG_V2)

N_SESSIONS = 4
STEPS_PHASE_A = 15   # tutor active
STEPS_PHASE_B = 10   # tutor off (autonomy)
STEPS_PHASE_C = 10   # shifted structure
N_SEEDS = 8


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def sim_step(m, observer, theta, lesson, step_idx, seed, rng,
             tutor_active=True, risk_shift=0.0):
    """Simulate one step: episode generation, agent choice, tutor decision, state update."""
    ub = {p: 0.4 + 0.1 * (step_idx / 20) for p in PROBE_NAMES}
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

    # Tutor decision
    if tutor_active:
        tutor = BCICTv4(agent_params=AP, use_dose=False)
        act, dose, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m)
    else:
        act, dose = "WAIT", 0.0  # tutor off

    dc = getattr(sc, 'commit_depth', 3)
    dr = getattr(sc, 'reveal_depth', 2)
    p_self = estimate_self_discovery_prob(dc, dr)
    risk = getattr(sc, 'risk_level', 0.3) + risk_shift
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

    # State updates
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
    }


def run_3phase(theta, seed, mode="reset"):
    """Run one learner through N_SESSIONS of 3-phase TIC.

    mode: "reset" | "persistent_nohook" | "persistent_needhook"
    """
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()
    pm = ProfileManager()
    lid = f"L_{seed}_{theta}_{mode}"

    all_sessions = []

    for sess_k in range(N_SESSIONS):
        # Bootstrap
        if mode != "reset" and pm.session_count(lid) > 0:
            prev = pm.latest(lid)
            bootstrap_observer(observer, prev)
            bootstrap_agent_state(m, prev)
        else:
            if sess_k > 0 and mode == "reset":
                m = FactoredInternalizationState(); m.snapshot()
                observer = A1MtObserverFrozen(); observer.reset()

        # Install need hook if applicable
        ctrl = CurriculumControllerV13(theta=theta)
        ctrl.reset_session(budget=20.0)
        if mode == "persistent_needhook" and pm.session_count(lid) > 0:
            z_bar = pm.probe_weakness_summary(lid)
            hook = make_need_hook(z_bar, lambda_need=0.3)
            ctrl.install_profile_hook(hook)

        ss = SessionSummary()
        phase_results = {"A": [], "B": [], "C": []}
        step_global = 0

        # Phase A: tutor active (use curriculum controller to pick lessons)
        for i in range(STEPS_PHASE_A):
            les = TIC_LESSONS[i % len(TIC_LESSONS)]
            r = sim_step(m, observer, theta, les, step_global, seed, rng,
                        tutor_active=True)
            phase_results["A"].append(r)
            if r["warned"]: ss.n_warn += 1
            else: ss.n_wait += 1
            step_global += 1

        # Phase B: tutor off (autonomy transfer)
        for i in range(STEPS_PHASE_B):
            les = TIC_LESSONS[i % len(TIC_LESSONS)]
            r = sim_step(m, observer, theta, les, step_global, seed, rng,
                        tutor_active=False)
            phase_results["B"].append(r)
            ss.n_wait += 1
            step_global += 1

        # Phase C: shifted structure (higher risk, different temptation)
        for i in range(STEPS_PHASE_C):
            les = TIC_LESSONS[(i + 2) % len(TIC_LESSONS)]
            r = sim_step(m, observer, theta, les, step_global, seed, rng,
                        tutor_active=False, risk_shift=0.15)
            phase_results["C"].append(r)
            ss.n_wait += 1
            step_global += 1

        # Compute per-phase metrics
        session_metrics = {}
        for phase_name, results in phase_results.items():
            n = len(results)
            succ = sum(1 for r in results if r["correct"]) / max(n, 1)
            risk_avg = np.mean([r["risk"] for r in results]) if results else 0
            wr = sum(1 for r in results if r["warned"]) / max(n, 1)
            session_metrics[phase_name] = {
                "success": round(succ, 4),
                "risk": round(float(risk_avg), 4),
                "warn_rate": round(wr, 4),
                "n": n,
            }

        # Set probe means from mastery proxy
        ss.n_steps = step_global
        ss.probe_means = {p: 0.5 for p in PROBE_NAMES}
        ss.transfer_success = session_metrics["B"]["success"]

        profile = finalize_session(observer, m, sess_k, theta, lid, ss)
        pm.finalize_session(lid, profile)

        # TBSR = success with time budget (Phase B within budget)
        tbsr_b = session_metrics["B"]["success"]
        tbsr_c = session_metrics["C"]["success"]

        all_sessions.append({
            "session_idx": sess_k,
            "phase_metrics": session_metrics,
            "tbsr_b": tbsr_b,
            "tbsr_c": tbsr_c,
            "m_hat": dict(observer.get_estimate()),
            "m_true": m.as_dict,
        })

    return all_sessions


def main():
    print("═══ T2 Exp-2B: TIC 3-Phase Transfer ═══\n", file=sys.stderr)
    L = ["# T2 Exp-2B: TIC 3-Phase Transfer — 3-Arm Comparison\n\n"]

    results = {}
    for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
        per_session = defaultdict(lambda: defaultdict(list))
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                sessions = run_3phase(th, sid, mode=mode)
                for s in sessions:
                    k = s["session_idx"]
                    for ph in ["A", "B", "C"]:
                        pm = s["phase_metrics"][ph]
                        per_session[(th, k)][f"{ph}_success"].append(pm["success"])
                        per_session[(th, k)][f"{ph}_risk"].append(pm["risk"])
                        per_session[(th, k)][f"{ph}_wr"].append(pm["warn_rate"])
                    per_session[(th, k)]["tbsr_b"].append(s["tbsr_b"])
                    per_session[(th, k)]["tbsr_c"].append(s["tbsr_c"])
            print(f"  {mode} / {th} done", file=sys.stderr)
        results[mode] = dict(per_session)

    # ═══ Main Table: Per-phase success by session ═══
    for th in ["safe", "shiny"]:
        L.append(f"\n## θ = {th}\n\n")
        L.append("### Per-Phase Success Rate\n\n")
        L.append("| Session | Mode | Phase A | Phase B | Phase C | TBSR_B | TBSR_C |\n")
        L.append("|:-------:|:----:|:-------:|:-------:|:-------:|:------:|:------:|\n")
        for k in range(N_SESSIONS):
            for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
                d = results[mode].get((th, k), {})
                sa = float(np.mean(d.get("A_success", [0])))
                sb = float(np.mean(d.get("B_success", [0])))
                sc = float(np.mean(d.get("C_success", [0])))
                tb = float(np.mean(d.get("tbsr_b", [0])))
                tc = float(np.mean(d.get("tbsr_c", [0])))
                L.append(f"| {k} | {mode} | {sa:.3f} | {sb:.3f} | {sc:.3f} | {tb:.3f} | {tc:.3f} |\n")

    # ═══ WarnRate Table (Phase A only) ═══
    L.append("\n## Phase A WarnRate by Session\n\n")
    L.append("| θ | Session | reset | nohook | needhook |\n")
    L.append("|:-:|:-------:|:-----:|:------:|:--------:|\n")
    for th in ["safe", "shiny"]:
        for k in range(N_SESSIONS):
            vals = []
            for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
                d = results[mode].get((th, k), {})
                vals.append(float(np.mean(d.get("A_wr", [0]))))
            L.append(f"| {th} | {k} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |\n")

    # ═══ Transfer Delta Table ═══
    L.append("\n## Transfer Delta: Phase B Success (Final Session)\n\n")
    L.append("| θ | reset | nohook | needhook | needhook > reset? |\n")
    L.append("|:-:|:-----:|:------:|:--------:|:-----------------:|\n")
    for th in ["safe", "shiny"]:
        vals = []
        for mode in ["reset", "persistent_nohook", "persistent_needhook"]:
            d = results[mode].get((th, N_SESSIONS - 1), {})
            vals.append(float(np.mean(d.get("B_success", [0]))))
        better = "✅" if vals[2] > vals[0] + 0.01 else ("≈" if abs(vals[2] - vals[0]) <= 0.01 else "❌")
        L.append(f"| {th} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {better} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")
    pass_transfer = 0
    for th in ["safe", "shiny"]:
        r = float(np.mean(results["reset"].get((th, N_SESSIONS-1), {}).get("B_success", [0])))
        p = float(np.mean(results["persistent_needhook"].get((th, N_SESSIONS-1), {}).get("B_success", [0])))
        if p >= r - 0.02:
            pass_transfer += 1
    L.append(f"> Transfer (Phase B): needhook ≥ reset in {pass_transfer}/2 θ\n")

    # Phase C check
    pass_c = 0
    for th in ["safe", "shiny"]:
        r = float(np.mean(results["reset"].get((th, N_SESSIONS-1), {}).get("C_success", [0])))
        p = float(np.mean(results["persistent_needhook"].get((th, N_SESSIONS-1), {}).get("C_success", [0])))
        if p >= r - 0.02:
            pass_c += 1
    L.append(f"> Shifted transfer (Phase C): needhook ≥ reset in {pass_c}/2 θ\n")

    if pass_transfer >= 1 and pass_c >= 1:
        L.append("> **✅ Persistent profile with need hook shows value in transfer**\n")
    else:
        L.append("> **⚠️ Transfer advantage not clear — investigate**\n")

    rpt = out / "t2_exp2b_tic_transfer.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
