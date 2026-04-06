"""CCT-v1 Experiment: Curriculum-Control Tutor.

Exp A: 5 curriculum strategies × 2θ × 8 seeds, 5-phase transfer + MCA.
Exp B: Actionability audit (PCR, AM, curriculum change rate).
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.teaching_internalization_corridor_v4 import (
    generate_tic_v4_session, generate_tic_v4_scenario,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal, neutralize_identity_features,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.behavior_probes import all_probes
from src.agents.trainable_bridge import TrainableBridge
from src.curriculum.lesson_library import LESSON_CATALOG, LESSON_BY_NAME
from src.curriculum.curriculum_controller_v1 import CurriculumControllerV1
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.actionability import policy_change_rate, curriculum_change_rate

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)

def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def _subtype_to_lesson(subtype):
    mapping = {
        "temptation_repeat": "tic_temptation",
        "self_discovery_teach": "ppmrb_self_discovery",
        "warn_rescue": "tic_rescue_heavy",
        "boundary_obs": "ppmrb_standard",
        "verified_warn": "verified_warn",
        "self_discovery_needed": "tic_self_discovery",
        "false_suppression_cost": "false_suppression",
        "sparse_valid_advice": "sparse_valid_advice",
        "sparse_invalid_advice": "sparse_invalid_advice",
        "beneficial_novelty": "beneficial_novelty",
    }
    return LESSON_BY_NAME.get(mapping.get(subtype, "ppmrb_standard"))


def fixed_curriculum(strategy, rng):
    """Return a fixed lesson name for static strategies."""
    if strategy == "ppmrb_only":
        return rng.choice(["ppmrb_standard", "ppmrb_self_discovery"])
    elif strategy == "tic_heavy":
        return rng.choice(["tic_rescue_heavy", "tic_temptation"])
    elif strategy == "mixed_random":
        return rng.choice([l.name for l in LESSON_CATALOG])
    elif strategy == "self_disc_heavy":
        return rng.choice(["ppmrb_self_discovery", "tic_self_discovery",
                           "beneficial_novelty", "false_suppression"])
    return "ppmrb_standard"


def run_session(strategy, theta, seed=0):
    sess = generate_tic_v4_session(seed * 1000 + abs(hash(theta)) % 1000, theta)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    micro_tutor = BCICTv4(agent_params=AP)
    bridge = TrainableBridge()
    m = FactoredInternalizationState()
    m.snapshot()

    cct = None
    if strategy == "cct_v1":
        cct = CurriculumControllerV1(bridge=bridge, theta=theta)

    rng_curr = np.random.default_rng(seed * 7 + 31)
    traces = {"A": [], "B": [], "C": [], "D": [], "E": []}
    micro_decisions = []
    curriculum_seq = []
    n_warns = 0

    for ep in sess.episodes:
        gm, cfg, meta, sc = generate_tic_v4_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.cue_layout_seed + 9999)
        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL: continue
                    z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss); lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
            risk_penalty=0.1)
        bar = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
            risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

        phase = ep.phase; subtype = ep.subtype
        dose = 0.0; give_advice = False; advice_correct = True

        # ---- Macro layer: select lesson context ----
        if phase == "A":
            if strategy == "cct_v1":
                lesson, _, _ = cct.select_lesson(m)
                curriculum_seq.append(lesson.name)
                # Use lesson dose profile to inform micro-tutor
                dose_hint = lesson.dose_profile
            else:
                lesson_name = fixed_curriculum(strategy, rng_curr)
                curriculum_seq.append(lesson_name)
                lesson = LESSON_BY_NAME.get(lesson_name)
                dose_hint = lesson.dose_profile if lesson else 0.5

            # ---- Micro layer ----
            action, dose, _ = micro_tutor.decide(sc, fb, lp, lib, scorer, 2, m)
            # Blend micro decision with lesson dose profile
            if dose_hint < 0.2 and dose > 0:
                # Lesson says don't warn, micro says warn → override to soft
                dose = min(dose, 0.5)
            elif dose_hint > 0.8 and dose == 0:
                # Lesson says warn hard, micro says wait → keep wait (micro veto)
                pass
            micro_decisions.append("WARN" if dose > 0 else "WAIT")
            if dose > 0: n_warns += 1
        elif phase == "C":
            if rng.random() < 0.5: give_advice = True; advice_correct = True
        elif phase == "D":
            if rng.random() < 0.5: give_advice = True; advice_correct = False

        if phase == "A":
            wb = [0.3 * dose, -0.3 * dose]
        elif give_advice:
            wb = ([0.3, -0.3] if advice_correct == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3])
        else:
            wb = [0.0, 0.0]
        nf = [False, False]
        if subtype == "beneficial_novelty":
            nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb, nf)

        chose_risky = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
        has_self_ev = (ep.d_commit > ep.d_reveal + 1)
        if phase == "A" and dose > 0:
            m.update_trust(warn_helpful=(ep.d_commit <= ep.d_reveal))
            if not has_self_ev:
                old_nu = m.nu; m.update_dependence(blind_obey=True)
                m.nu = old_nu + dose * (m.nu - old_nu)
            old_gg = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True)
            m.gamma_gen = old_gg + dose * (m.gamma_gen - old_gg)
        elif phase == "A" and not chose_risky:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
        elif phase in ("C", "D") and give_advice:
            if phase == "C" and not chose_risky: m.update_trust(warn_helpful=True)
            if phase == "D" and chose_risky: m.update_dependence(blind_obey=True)
            elif phase == "D" and not chose_risky and has_self_ev:
                m.update_dependence(self_discovery=True)
        if chose_risky and bar.temptation_score > 0.5:
            m.update_gamma_spec(tempt_error=True)
        if subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky:
            m.update_gamma_spec(false_suppression=True)
        m.snapshot()

        # Train bridge
        probes = all_probes(m, AP, theta)
        bridge.update(m, probes, sc.risk_level if hasattr(sc, 'risk_level') else 0.3,
                      bar.temptation_score, 0.3 if subtype == "beneficial_novelty" else 0.0,
                      0.7 if has_self_ev else 0.3)

        correct = chose_risky if subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        mca = correct
        if subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
        if subtype in ("sparse_valid_advice",) and correct and m.nu > 0.3: mca = False
        traces[phase].append({"correct": correct, "mca": mca, "subtype": subtype})

    def rate(ph, key="correct"):
        t = traces.get(ph, [])
        return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None

    probes = all_probes(m, AP, theta)
    otr = overteach_rate_v2(m)
    cur_summ = cct.curriculum_summary() if cct else {"counts": {}, "sequence": curriculum_seq}

    return {
        "B": rate("B"), "C": rate("C"), "D": rate("D"), "E": rate("E"),
        "mca_E": rate("E", "mca"), "mca_C": rate("C", "mca"),
        "wr": round(n_warns / max(len(traces.get("A", [])), 1), 3),
        "EP": probes["EP"], "VA": probes["VA"], "IA": probes["IA"],
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gg": round(m.gamma_gen, 3), "otr": otr["total"],
        "micro_decisions": micro_decisions,
        "curriculum_seq": curriculum_seq,
        "unique_lessons": len(set(curriculum_seq)),
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ CCT-v1: Curriculum Control ═══\n", file=sys.stderr)
    strategies = ["ppmrb_only", "tic_heavy", "mixed_random",
                  "self_disc_heavy", "cct_v1"]
    lines = ["# CCT-v1: Curriculum-Control Tutor\n\n"]

    # Exp A: 5-Phase
    lines.append("## Experiment A: 5-Phase Transfer\n\n")
    lines.append("| θ | Curriculum | WR | **B** | **C** | MCA_C | **D** | **E** | MCA_E |\n")
    lines.append("|---|-----------|----|----|----|----|----|----|----|\n")

    all_r = []
    for theta in ["safe", "shiny"]:
        for s in strategies:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in [
                "B", "C", "D", "E", "mca_E", "mca_C",
                "wr", "EP", "VA", "IA", "tau", "nu", "gg", "otr", "unique_lessons"]}
            a["theta"] = theta; a["strategy"] = s
            a["_micro"] = [r.get("micro_decisions", []) for r in rs]
            a["_curseq"] = [r.get("curriculum_seq", []) for r in rs]
            all_r.append(a)
            lines.append("| {} | {} | {} | **{}** | **{}** | {} | **{}** | **{}** | {} |\n".format(
                theta, s, sf(a["wr"]),
                sf(a["B"]), sf(a["C"]), sf(a["mca_C"]),
                sf(a["D"]), sf(a["E"]), sf(a["mca_E"])))
            print(f"  {theta}×{s}: B={sf(a['B'])} C={sf(a['C'])} E={sf(a['E'])} "
                  f"MCA_E={sf(a['mca_E'])} ν={sf(a['nu'],'{:.2f}')} "
                  f"lessons={sf(a['unique_lessons'],'{:.0f}')}",
                  file=sys.stderr)

    # State + curriculum
    lines.append("\n## State + Curriculum\n\n")
    lines.append("| θ | Curriculum | τ | ν | **τ-ν** | γg | OTR | #Unique |\n")
    lines.append("|---|-----------|---|---|---------|----|----|--------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s][0]
            gap = round(r["tau"] - r["nu"], 3) if r["tau"] and r["nu"] else None
            lines.append("| {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                theta, s,
                sf(r["tau"], "{:.3f}"), sf(r["nu"], "{:.3f}"),
                sf(gap, "{:+.3f}"), sf(r["gg"], "{:.3f}"),
                sf(r["otr"], "{:.3f}"), sf(r["unique_lessons"], "{:.0f}")))

    # Probes
    lines.append("\n## Probes\n\n")
    lines.append("| θ | Curriculum | EP | VA | IA |\n")
    lines.append("|---|-----------|----|----|----|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s][0]
            lines.append("| {} | {} | {} | {} | {} |\n".format(
                theta, s,
                sf(r["EP"], "{:.3f}"), sf(r["VA"], "{:.3f}"), sf(r["IA"], "{:.3f}")))

    # Exp B: Actionability
    lines.append("\n## Experiment B: Actionability Audit\n\n")
    lines.append("| θ | Comparison | micro_PCR | curriculum_CR |\n")
    lines.append("|---|-----------|-----------|---------------|\n")
    for theta in ["safe", "shiny"]:
        base = [x for x in all_r if x["theta"] == theta and x["strategy"] == "mixed_random"][0]
        cct = [x for x in all_r if x["theta"] == theta and x["strategy"] == "cct_v1"][0]
        # Average PCR across seeds
        pcrs = []
        ccrs = []
        for i in range(min(len(base["_micro"]), len(cct["_micro"]))):
            p = policy_change_rate(cct["_micro"][i], base["_micro"][i])
            c = curriculum_change_rate(cct["_curseq"][i], base["_curseq"][i])
            pcrs.append(p); ccrs.append(c)
        avg_pcr = round(np.mean(pcrs), 3) if pcrs else None
        avg_ccr = round(np.mean(ccrs), 3) if ccrs else None
        lines.append("| {} | cct_v1 vs mixed | {} | {} |\n".format(
            theta, sf(avg_pcr), sf(avg_ccr)))

        # TIC-heavy vs CCT-v1
        tic = [x for x in all_r if x["theta"] == theta and x["strategy"] == "tic_heavy"][0]
        pcrs2 = []
        ccrs2 = []
        for i in range(min(len(tic["_micro"]), len(cct["_micro"]))):
            pcrs2.append(policy_change_rate(cct["_micro"][i], tic["_micro"][i]))
            ccrs2.append(curriculum_change_rate(cct["_curseq"][i], tic["_curseq"][i]))
        lines.append("| {} | cct_v1 vs tic_heavy | {} | {} |\n".format(
            theta, sf(round(np.mean(pcrs2), 3) if pcrs2 else None),
            sf(round(np.mean(ccrs2), 3) if ccrs2 else None)))

    with open(out / "curriculum_control_report.md", "w") as f:
        f.writelines(lines)
    print(f"\nReport -> results/curriculum_control_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
