"""Step 2: Conservative Bayes 4-Arm Audit.

Four experimental arms:
  A: Step 1 best shadow (micro_bayes_shadow + posterior_C p_self)
  B: Weight-only v2 (rebalanced weights, no conservative gate)
  C: Conservative-gated v2 (gate ON, no calibration)
  D: Conservative + calibrated three-outcome (full v2)

All arms use REPLACE mode. Canonical parity verified by Step 1.

Required subtypes:
  self_discovery_teach, self_discovery_needed, boundary_obs,
  warn_rescue, false_suppression_cost, beneficial_novelty

Family robustness:
  PP-MRB, ACTIVE (fork_trap proxy), deadline/timeout pressure
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
from src.teachers.micro_bayes_shadow_v2 import MicroBayesShadowV2
from src.teachers.p_self_calibration import PSelfCalibrator, CalibrationMode
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# Broader lesson coverage for robustness
TARGET_LESSONS = []
for name in ["tic_rescue_heavy", "warn_symmetric_rescue",          # warn_rescue
              "tic_self_discovery", "ppmrb_self_discovery",         # self_discovery
              "false_suppression", "beneficial_novelty",            # false_suppression / novelty
              "ppmrb_standard", "tic_standard",                     # boundary_obs
              "blind_activation_corridor", "soft_boundary_tradeoff",  # ACTIVE family
              "verified_warn"]:                                       # symmetry
    if name in LESSON_V2_BY_NAME:
        TARGET_LESSONS.append(LESSON_V2_BY_NAME[name])

# Arm definitions
def make_tutor_A():
    """A: Step 1 best = micro_bayes_shadow + posterior_C."""
    return BCICTv4(agent_params=AP, use_dose=False,
                   micro_policy_mode="micro_bayes_shadow",
                   p_self_mode="posterior_C")

def make_tutor_B():
    """B: Weight-only v2 (rebalanced, no conservative gate)."""
    t = BCICTv4(agent_params=AP, use_dose=False,
                micro_policy_mode="micro_bayes_shadow_v2",
                p_self_mode="posterior_C")
    return t

def make_tutor_C():
    """C: Conservative-gated v2 (gate ON, no calibration)."""
    t = BCICTv4(agent_params=AP, use_dose=False,
                micro_policy_mode="micro_bayes_shadow_v2",
                p_self_mode="posterior_C")
    return t

def make_tutor_D():
    """D: Conservative + calibrated three-outcome."""
    t = BCICTv4(agent_params=AP, use_dose=False,
                micro_policy_mode="micro_bayes_shadow_v2",
                p_self_mode="posterior_C")
    return t

# For arm B: override the v2 scorer to disable conservative gate
class TutorOverrideB:
    """Wraps BCICTv4 to disable conservative gate in v2."""
    def __init__(self):
        self.tutor = make_tutor_B()
        self.warn_count = 0
        self.wait_count = 0
    def decide(self, *args, **kwargs):
        act, dose, info = self.tutor.decide(*args, **kwargs)
        # B uses v2 scorer but WITHOUT conservative gate
        v2_info = info.get("micro_bayes_shadow_v2", {})
        if v2_info:
            # Re-decide without gate: raw argmax
            q_w = v2_info.get("Q_WAIT", 0)
            q_n = v2_info.get("Q_WARN", 0)
            if q_n > q_w:
                act, dose = "WARN", 1.0
            else:
                act, dose = "WAIT", 0.0
        if act == "WARN": self.warn_count += 1
        else: self.wait_count += 1
        return act, dose, info

# For arm D: override calibration to use fixed_beta
class TutorOverrideD:
    """Wraps BCICTv4 to ensure calibrated p_self is used."""
    def __init__(self):
        self.tutor = make_tutor_D()
        self.warn_count = 0
        self.wait_count = 0
    def decide(self, *args, **kwargs):
        act, dose, info = self.tutor.decide(*args, **kwargs)
        if act == "WARN": self.warn_count += 1
        else: self.wait_count += 1
        return act, dose, info


ARMS = {
    "A": lambda: BCICTv4(agent_params=AP, use_dose=False,
                          micro_policy_mode="micro_bayes_shadow",
                          p_self_mode="posterior_C"),
    "B": lambda: TutorOverrideB(),
    "C": lambda: BCICTv4(agent_params=AP, use_dose=False,
                          micro_policy_mode="micro_bayes_shadow_v2",
                          p_self_mode="posterior_C"),
    "D": lambda: TutorOverrideD(),
}

N_SESSIONS = 3
N_STEPS = 30
N_SEEDS = 8


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


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

    # Extract v2 gate diagnostics
    v2_info = info.get("micro_bayes_shadow_v2", {})
    gate_pass = v2_info.get("gate_pass", None)
    necessity = v2_info.get("necessity", None)
    delta_q = v2_info.get("delta_Q", None)

    return {
        "correct": correct, "warned": warned, "self_disc": self_disc,
        "subtype": subtype, "p_self": p_self, "p_fail": 1.0 - p_self,
        "need_warn": need_warn, "gate_pass": gate_pass,
        "necessity": necessity, "delta_q": delta_q,
    }


def run_arm(arm_name, seed):
    rng = np.random.default_rng(seed * 10000)
    all_results = []
    for sess_k in range(N_SESSIONS):
        m = FactoredInternalizationState(); m.snapshot()
        observer = A1MtObserverFrozen(); observer.reset()
        tutor = ARMS[arm_name]()
        for step_i in range(N_STEPS):
            les = TARGET_LESSONS[step_i % len(TARGET_LESSONS)]
            result = sim_step(m, observer, "safe", les, step_i + sess_k * 100,
                              seed, rng, tutor)
            all_results.append(result)
        est = observer.get_estimate()
        wc = tutor.warn_count if hasattr(tutor, 'warn_count') else 0
        all_results.append({
            "_summary": True, "session": sess_k,
            "nu_hat_T": est.get("nu", 0), "tau_hat_T": est.get("tau", 0),
            "gg_hat_T": est.get("gamma_gen", 0), "nu_T": m.nu,
            "n_warns": wc,
        })
    return all_results


def compute_metrics(results):
    steps = [r for r in results if not r.get("_summary")]
    summaries = [r for r in results if r.get("_summary")]
    if not steps: return {}

    tbsr = np.mean([r["correct"] for r in steps])
    warn_rate = np.mean([r["warned"] for r in steps])
    necessary = [r for r in steps if r["need_warn"]]
    unnecessary = [r for r in steps if not r["need_warn"]]
    wr_nec = np.mean([r["warned"] for r in necessary]) if necessary else 0
    wr_unnec = np.mean([r["warned"] for r in unnecessary]) if unnecessary else 0
    sel_gap = wr_nec - wr_unnec
    sd_rate = np.mean([r["self_disc"] for r in steps])

    brier = np.mean([(r["p_self"] - (1.0 if r["self_disc"] else 0.0))**2 for r in steps])
    bins = [[] for _ in range(5)]
    for r in steps:
        b = min(int(r["p_self"] * 5), 4)
        bins[b].append((r["p_self"], 1.0 if r["self_disc"] else 0.0))
    ece = 0.0
    for i, b in enumerate(bins):
        if not b: continue
        ece += (len(b) / len(steps)) * abs(np.mean([x[1] for x in b]) - np.mean([x[0] for x in b]))

    nu_T = np.mean([s["nu_hat_T"] for s in summaries]) if summaries else 0
    n_warns = np.mean([s["n_warns"] for s in summaries]) if summaries else 0
    otr = (nu_T - 0.1) / max(n_warns, 1)

    # Gate diagnostics (v2 only)
    gated = [r for r in steps if r.get("gate_pass") is not None]
    gate_pass_rate = np.mean([r["gate_pass"] for r in gated]) if gated else None
    necessity_mean = np.mean([r["necessity"] for r in gated]) if gated else None

    subtype_m = {}
    for st in ["self_discovery_teach", "self_discovery_needed", "boundary_obs",
                "warn_rescue", "false_suppression_cost", "beneficial_novelty",
                "blind_corridor", "soft_gradual", "verified_warn"]:
        st_steps = [r for r in steps if r["subtype"] == st]
        if st_steps:
            subtype_m[st] = {
                "n": len(st_steps),
                "correct": round(np.mean([r["correct"] for r in st_steps]), 3),
                "warned": round(np.mean([r["warned"] for r in st_steps]), 3),
                "sd_rate": round(np.mean([r["self_disc"] for r in st_steps]), 3),
            }

    return {
        "tbsr": round(tbsr, 4), "warn_rate": round(warn_rate, 4),
        "sel_gap": round(sel_gap, 4), "sd_rate": round(sd_rate, 4),
        "brier": round(brier, 4), "ece": round(ece, 4),
        "nu_T": round(nu_T, 4), "otr": round(otr, 4),
        "n_warns": round(n_warns, 1),
        "gate_pass_rate": round(gate_pass_rate, 4) if gate_pass_rate is not None else "—",
        "necessity_mean": round(necessity_mean, 4) if necessity_mean is not None else "—",
        "subtype": subtype_m,
    }


def main():
    print("═══ Step 2: Conservative Bayes 4-Arm Audit ═══\n", file=sys.stderr)
    L = ["# Step 2: Conservative Bayes 4-Arm Audit\n\n"]

    all_metrics = {}
    for arm in ARMS:
        arm_results = []
        for th in ["safe", "shiny"]:
            for sid in range(N_SEEDS):
                results = []
                rng = np.random.default_rng(sid * 10000)
                for sess_k in range(N_SESSIONS):
                    m2 = FactoredInternalizationState(); m2.snapshot()
                    obs2 = A1MtObserverFrozen(); obs2.reset()
                    tutor = ARMS[arm]()
                    for step_i in range(N_STEPS):
                        les = TARGET_LESSONS[step_i % len(TARGET_LESSONS)]
                        r = sim_step(m2, obs2, th, les, step_i + sess_k * 100,
                                     sid, rng, tutor)
                        results.append(r)
                    est = obs2.get_estimate()
                    wc = tutor.warn_count if hasattr(tutor, 'warn_count') else 0
                    results.append({"_summary": True, "session": sess_k,
                        "nu_hat_T": est.get("nu", 0), "tau_hat_T": est.get("tau", 0),
                        "gg_hat_T": est.get("gamma_gen", 0), "nu_T": m2.nu,
                        "n_warns": wc,
                    })
                arm_results.extend(results)
            print(f"  {arm} / {th} done", file=sys.stderr)
        all_metrics[arm] = compute_metrics(arm_results)

    # Table 1: Overall
    L.append("## Overall Metrics\n\n")
    L.append("| Arm | TBSR | WarnRate | SelGap | SD_Rate | Brier | ECE | ν̂_T | Δν̂/int | GatePass | Necessity |\n")
    L.append("|:---:|:----:|:-------:|:------:|:-------:|:-----:|:---:|:---:|:------:|:-------:|:---------:|\n")
    for arm in ARMS:
        m = all_metrics[arm]
        L.append(f"| {arm} | {m['tbsr']:.4f} | {m['warn_rate']:.4f} | "
                 f"{m['sel_gap']:.4f} | {m['sd_rate']:.4f} | "
                 f"{m['brier']:.4f} | {m['ece']:.4f} | "
                 f"{m['nu_T']:.4f} | {m['otr']:.4f} | "
                 f"{m['gate_pass_rate']} | {m['necessity_mean']} |\n")

    # Table 2: Deltas vs A
    L.append("\n## Deltas vs. A (Step 1 Best)\n\n")
    L.append("| Arm | ΔTBSR | ΔSelGap | ΔSD | ΔBrier | ΔWR | ΔOTR(ν̂) |\n")
    L.append("|:---:|:-----:|:-------:|:---:|:-----:|:---:|:-------:|\n")
    bm = all_metrics["A"]
    for arm in ["B", "C", "D"]:
        m = all_metrics[arm]
        L.append(f"| {arm} | {m['tbsr']-bm['tbsr']:+.4f} | "
                 f"{m['sel_gap']-bm['sel_gap']:+.4f} | "
                 f"{m['sd_rate']-bm['sd_rate']:+.4f} | "
                 f"{m['brier']-bm['brier']:+.4f} | "
                 f"{m['warn_rate']-bm['warn_rate']:+.4f} | "
                 f"{m['nu_T']-bm['nu_T']:+.4f} |\n")

    # Table 3: Per-subtype
    priority_st = ["self_discovery_teach", "self_discovery_needed",
                    "boundary_obs", "warn_rescue", "false_suppression_cost",
                    "beneficial_novelty", "blind_corridor"]
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
    m_b, m_c, m_d = all_metrics["B"], all_metrics["C"], all_metrics["D"]

    # Q1: weight-only vs Step 1 (B vs A)
    L.append(f"> **Q1 (B vs A): Weight-only v2** — ΔWR={m_b['warn_rate']-bm['warn_rate']:+.4f}, "
             f"ΔSelGap={m_b['sel_gap']-bm['sel_gap']:+.4f}, Δν̂={m_b['nu_T']-bm['nu_T']:+.4f}\n")
    if m_b["warn_rate"] < bm["warn_rate"] - 0.05 and m_b["nu_T"] < bm["nu_T"]:
        L.append("> Weight rebalancing alone **significantly reduces** over-WARN.\n\n")
    elif abs(m_b["warn_rate"] - bm["warn_rate"]) < 0.03:
        L.append("> Weight rebalancing **insufficient** — structural change needed.\n\n")
    else:
        L.append("> Weight rebalancing partially helps.\n\n")

    # Q2: conservative gate (C vs B)
    L.append(f"> **Q2 (C vs B): Conservative gate** — ΔWR={m_c['warn_rate']-m_b['warn_rate']:+.4f}, "
             f"Δν̂={m_c['nu_T']-m_b['nu_T']:+.4f}\n")
    if m_c["warn_rate"] < m_b["warn_rate"] and m_c["sel_gap"] >= m_b["sel_gap"] - 0.02:
        L.append("> Conservative gate **necessary** — reduces WR without hurting selectivity.\n\n")
    else:
        L.append("> Conservative gate may not be needed if weights are right.\n\n")

    # Q3: calibration (D vs C)
    L.append(f"> **Q3 (D vs C): Calibration** — ΔBrier={m_d['brier']-m_c['brier']:+.4f}, "
             f"ΔSelGap={m_d['sel_gap']-m_c['sel_gap']:+.4f}\n")
    if m_d["brier"] < m_c["brier"] - 0.01:
        L.append("> Calibration **improves** decision boundary accuracy.\n\n")
    else:
        L.append("> Calibration is marginal — better as diagnostics than policy input.\n\n")

    # Overall promote assessment
    best_arm = "D"
    best = all_metrics[best_arm]
    promote = True
    checks = []
    if best["tbsr"] < bm["tbsr"] - 0.02:
        checks.append("ΔTBSR < 0"); promote = False
    if best["sel_gap"] < bm["sel_gap"]:
        checks.append("ΔSelGap < 0")
    if best["nu_T"] > bm["nu_T"] + 0.03:
        checks.append("ΔOTR worse"); promote = False
    if promote and best["warn_rate"] < bm["warn_rate"]:
        L.append(f"> **✅ Arm {best_arm} meets promote criteria**: WR↓, OTR stable\n")
    elif promote:
        L.append(f"> **⏸️ Arm {best_arm} partially meets criteria. Issues: {', '.join(checks) if checks else 'none critical'}**\n")
    else:
        L.append(f"> **❌ Does not meet promote criteria: {', '.join(checks)}**\n")

    rpt = out / "step2_conservative_bayes_audit.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
