"""CCT-v2 + AEG-v1: Closed-Loop Curriculum Control.

Phase A: macro selects lesson → AEG generates episode → micro acts with budget.
Phase B-E: fixed transfer (not macro-controlled).
5 strategies × 2θ × 8 seeds.
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

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
from src.curriculum.curriculum_controller_v2 import (
    CurriculumControllerV2, DoseBudgetTracker,
)
from src.curriculum.adaptive_episode_generator import (
    generate_episode_from_lesson, generate_transfer_episode,
)
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.metrics.teaching_zone_v2 import overteach_rate_v2
from src.metrics.actionability_v2 import (
    lesson_fidelity, episode_realization_change_rate,
    micro_policy_change_rate,
)

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


def run_session(strategy, theta, seed=0):
    rng_main = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    micro_tutor = BCICTv4(agent_params=AP)
    bridge = TrainableBridge()
    m = FactoredInternalizationState()
    m.snapshot()
    budget = DoseBudgetTracker()

    cct = None
    if strategy == "cct_v2":
        cct = CurriculumControllerV2(bridge=bridge, theta=theta)

    traces = {"A": [], "B": [], "C": [], "D": [], "E": []}
    micro_decisions = []
    ep_params_list = []
    lf_scores = []
    n_warns, n_soft, n_budget_blocked = 0, 0, 0
    idx = 0

    # ─── Phase A: 10 macro-controlled episodes ───
    for _ in range(10):
        # Macro: select lesson
        if strategy == "cct_v2":
            lesson, _, _ = cct.select_lesson(m)
        elif strategy == "ppmrb_only":
            lesson = LESSON_BY_NAME[rng_main.choice(["ppmrb_standard", "ppmrb_self_discovery"])]
        elif strategy == "tic_heavy":
            lesson = LESSON_BY_NAME[rng_main.choice(["tic_rescue_heavy", "tic_temptation"])]
        elif strategy == "mixed_random":
            lesson = rng_main.choice(LESSON_CATALOG)
        elif strategy == "self_disc_heavy":
            lesson = LESSON_BY_NAME[rng_main.choice(
                ["ppmrb_self_discovery", "tic_self_discovery", "beneficial_novelty", "false_suppression"])]
        else:
            lesson = LESSON_CATALOG[0]

        # AEG: generate episode from lesson
        ep_params, spec, gm, cfg, meta, sc = generate_episode_from_lesson(
            lesson, idx, theta, rng_main)
        if cct:
            cct.record_realization(ep_params)
        ep_params_list.append(ep_params)
        lf_scores.append(lesson_fidelity(ep_params, lesson))

        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(spec.cue_layout_seed + 9999)
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

        # Micro: decide with dose budget constraint
        budget.reset(ep_params)
        action, raw_dose, _ = micro_tutor.decide(sc, fb, lp, lib, scorer, 2, m)
        feasible = budget.feasible_doses()
        dose = raw_dose if raw_dose in feasible else max(d for d in feasible if d <= raw_dose)
        if raw_dose > dose:
            n_budget_blocked += 1
        budget.consume(dose)

        if dose > 0: n_warns += 1
        if 0 < dose < 1: n_soft += 1
        micro_decisions.append(f"{'WARN' if dose >= 1 else 'SOFT' if dose > 0 else 'WAIT'}_{ep_params.subtype}")

        wb = [0.3 * dose, -0.3 * dose]
        nf = [False, False]
        if ep_params.subtype == "beneficial_novelty":
            nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb, nf)

        chose_risky = (ac != sc.oracle_safe_branch_id)
        m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
        has_self_ev = (spec.d_commit > spec.d_reveal + 1)
        if dose > 0:
            m.update_trust(warn_helpful=(spec.d_commit <= spec.d_reveal))
            if not has_self_ev:
                old_nu = m.nu; m.update_dependence(blind_obey=True)
                m.nu = old_nu + dose * (m.nu - old_nu)
            old_gg = m.gamma_gen; m.update_gamma_gen(sustained_pressure=True)
            m.gamma_gen = old_gg + dose * (m.gamma_gen - old_gg)
        elif not chose_risky:
            m.update_dependence(self_discovery=True)
            m.update_gamma_gen(successful_exploration=True)
        if chose_risky and bar.temptation_score > 0.5:
            m.update_gamma_spec(tempt_error=True)
        if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky:
            m.update_gamma_spec(false_suppression=True)
        m.snapshot()

        probes = all_probes(m, AP, theta)
        bridge.update(m, probes, sc.risk_level if hasattr(sc, 'risk_level') else 0.3,
                      bar.temptation_score, ep_params.novelty,
                      0.7 if has_self_ev else 0.3)

        correct = chose_risky if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
        mca = correct
        if ep_params.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
        traces["A"].append({"correct": correct, "mca": mca, "subtype": ep_params.subtype})
        idx += 1

    # ─── Phases B-E: transfer episodes (not macro-controlled) ───
    for phase, n_ep in [("B", 4), ("C", 4), ("D", 4), ("E", 4)]:
        for _ in range(n_ep):
            ep_params, spec, gm, cfg, meta, sc = generate_transfer_episode(
                phase, idx, theta, rng_main)
            ep_params_list.append(ep_params)
            fb, ww = apply_fix(meta, sc)
            fv = np.full_like(fb, 0.3)
            rng = np.random.default_rng(spec.cue_layout_seed + 9999)
            for _ in range(5):
                for r in range(gm.height):
                    for c in range(gm.width):
                        if gm.cell_types[r, c] == CellType.WALL: continue
                        z = fb[r, c]; lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
            ss = summarize_branch(sc.safe_cells, fb, fv, lp)
            sr = summarize_branch(sc.risky_cells, fb, fv, lp)
            bas = BranchAttributes(safety_score=float(ss[0]),
                temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b,
                risk_penalty=0.1)
            bar = BranchAttributes(safety_score=float(sr[0]),
                temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a,
                risk_penalty=sc.risk_level if hasattr(sc, 'risk_level') else 0.4)

            give_advice = False; advice_correct = True
            if phase == "C" and rng.random() < 0.5: give_advice = True; advice_correct = True
            elif phase == "D" and rng.random() < 0.5: give_advice = True; advice_correct = False
            if give_advice:
                wb = ([0.3, -0.3] if advice_correct == (sc.oracle_safe_branch_id == 0) else [-0.3, 0.3])
            else:
                wb = [0.0, 0.0]
            nf = [False, False]
            if ep_params.subtype == "beneficial_novelty":
                nf = [False, True] if sc.oracle_safe_branch_id == 1 else [True, False]
            ac = sample_factored_choice([bas, bar], theta, m, AP, rng, wb, nf)

            chose_risky = (ac != sc.oracle_safe_branch_id)
            m.update_risk(sc.risk_level if hasattr(sc, 'risk_level') and chose_risky else 0.05, 0.15)
            has_self_ev = (spec.d_commit > spec.d_reveal + 1)
            if phase in ("C", "D") and give_advice:
                if phase == "C" and not chose_risky: m.update_trust(warn_helpful=True)
                if phase == "D" and chose_risky: m.update_dependence(blind_obey=True)
                elif phase == "D" and not chose_risky and has_self_ev:
                    m.update_dependence(self_discovery=True)
            if chose_risky and bar.temptation_score > 0.5:
                m.update_gamma_spec(tempt_error=True)
            if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") and not chose_risky:
                m.update_gamma_spec(false_suppression=True)
            m.snapshot()

            correct = chose_risky if ep_params.subtype in ("false_suppression_cost", "beneficial_novelty") else (ac == sc.oracle_safe_branch_id)
            mca = correct
            if ep_params.subtype == "beneficial_novelty" and correct and m.gamma_gen > 0.3: mca = False
            traces[phase].append({"correct": correct, "mca": mca, "subtype": ep_params.subtype})
            idx += 1

    def rate(ph, key="correct"):
        t = traces.get(ph, [])
        return sum(1 for x in t if x[key]) / max(len(t), 1) if t else None

    probes = all_probes(m, AP, theta)
    otr = overteach_rate_v2(m)
    avg_lf = round(np.mean(lf_scores), 3) if lf_scores else None

    return {
        "B": rate("B"), "C": rate("C"), "D": rate("D"), "E": rate("E"),
        "mca_E": rate("E", "mca"), "mca_C": rate("C", "mca"),
        "wr": round(n_warns / 10, 3),
        "budget_blocked": n_budget_blocked,
        "LF": avg_lf,
        "EP": probes["EP"], "VA": probes["VA"], "IA": probes["IA"],
        "tau": round(m.tau, 3), "nu": round(m.nu, 3),
        "gg": round(m.gamma_gen, 3), "otr": otr["total"],
        "micro_decisions": micro_decisions,
        "ep_params": ep_params_list[:10],  # Phase A only
        "subtypes_a": [t["subtype"] for t in traces["A"]],
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ CCT-v2 + AEG-v1 ═══\n", file=sys.stderr)
    strategies = ["ppmrb_only", "tic_heavy", "mixed_random",
                  "self_disc_heavy", "cct_v2"]
    lines = ["# CCT-v2 + AEG-v1: Closed-Loop Curriculum\n\n"]

    lines.append("## 5-Phase Transfer\n\n")
    lines.append("| θ | Curriculum | WR | LF | **B** | **C** | MCA_C | **D** | **E** | MCA_E |\n")
    lines.append("|---|-----------|---|----|----|----|----|----|----|----|\n")

    all_r = []
    for theta in ["safe", "shiny"]:
        for s in strategies:
            rs = [run_session(s, theta, sid) for sid in range(8)]
            a = {k: avg(rs, k) for k in [
                "B", "C", "D", "E", "mca_E", "mca_C", "wr", "LF",
                "budget_blocked", "EP", "VA", "IA",
                "tau", "nu", "gg", "otr"]}
            a["theta"] = theta; a["strategy"] = s
            a["_micro"] = [r.get("micro_decisions", []) for r in rs]
            a["_ep_params"] = [r.get("ep_params", []) for r in rs]
            a["_subtypes_a"] = [r.get("subtypes_a", []) for r in rs]
            all_r.append(a)
            lines.append("| {} | {} | {} | {} | **{}** | **{}** | {} | **{}** | **{}** | {} |\n".format(
                theta, s, sf(a["wr"]), sf(a["LF"], "{:.3f}"),
                sf(a["B"]), sf(a["C"]), sf(a["mca_C"]),
                sf(a["D"]), sf(a["E"]), sf(a["mca_E"])))
            print(f"  {theta}×{s}: B={sf(a['B'])} C={sf(a['C'])} D={sf(a['D'])} "
                  f"E={sf(a['E'])} LF={sf(a['LF'],'{:.2f}')} ν={sf(a['nu'],'{:.2f}')}",
                  file=sys.stderr)

    # State
    lines.append("\n## State\n\n")
    lines.append("| θ | Curriculum | τ | ν | **τ-ν** | γg | OTR | BdgBlk |\n")
    lines.append("|---|-----------|---|---|---------|----|----|-------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s][0]
            gap = round(r["tau"] - r["nu"], 3) if r["tau"] and r["nu"] else None
            lines.append("| {} | {} | {} | {} | **{}** | {} | {} | {} |\n".format(
                theta, s,
                sf(r["tau"], "{:.3f}"), sf(r["nu"], "{:.3f}"),
                sf(gap, "{:+.3f}"), sf(r["gg"], "{:.3f}"),
                sf(r["otr"], "{:.3f}"), sf(r["budget_blocked"], "{:.0f}")))

    # Actionability
    lines.append("\n## Closed-Loop Actionability\n\n")
    lines.append("| θ | vs | ERCR | micro_PCR |\n")
    lines.append("|---|---|------|----------|\n")
    for theta in ["safe", "shiny"]:
        base = [x for x in all_r if x["theta"] == theta and x["strategy"] == "mixed_random"][0]
        cct = [x for x in all_r if x["theta"] == theta and x["strategy"] == "cct_v2"][0]
        ercrs, pcrs = [], []
        for i in range(min(len(base["_ep_params"]), len(cct["_ep_params"]))):
            if base["_ep_params"][i] and cct["_ep_params"][i]:
                ercrs.append(episode_realization_change_rate(
                    cct["_ep_params"][i], base["_ep_params"][i]))
            pcrs.append(micro_policy_change_rate(
                cct["_micro"][i], base["_micro"][i]))
        lines.append("| {} | cct_v2 vs mixed | {} | {} |\n".format(
            theta,
            sf(round(np.mean(ercrs), 3) if ercrs else None),
            sf(round(np.mean(pcrs), 3) if pcrs else None)))

        tic = [x for x in all_r if x["theta"] == theta and x["strategy"] == "tic_heavy"][0]
        ercrs2, pcrs2 = [], []
        for i in range(min(len(tic["_ep_params"]), len(cct["_ep_params"]))):
            if tic["_ep_params"][i] and cct["_ep_params"][i]:
                ercrs2.append(episode_realization_change_rate(
                    cct["_ep_params"][i], tic["_ep_params"][i]))
            pcrs2.append(micro_policy_change_rate(
                cct["_micro"][i], tic["_micro"][i]))
        lines.append("| {} | cct_v2 vs tic_heavy | {} | {} |\n".format(
            theta,
            sf(round(np.mean(ercrs2), 3) if ercrs2 else None),
            sf(round(np.mean(pcrs2), 3) if pcrs2 else None)))

    # Subtype distribution
    lines.append("\n## Phase A Subtype Distribution\n\n")
    lines.append("| θ | Curriculum | Subtypes (top 3) |\n")
    lines.append("|---|-----------|------------------|\n")
    for theta in ["safe", "shiny"]:
        for s in strategies:
            r = [x for x in all_r if x["theta"] == theta and x["strategy"] == s][0]
            all_subs = [st for seeds in r["_subtypes_a"] for st in seeds]
            if all_subs:
                from collections import Counter
                top3 = Counter(all_subs).most_common(3)
                top_str = ", ".join(f"{k}({v})" for k, v in top3)
            else:
                top_str = "—"
            lines.append(f"| {theta} | {s} | {top_str} |\n")

    with open(out / "curriculum_control_v2_report.md", "w") as f:
        f.writelines(lines)
    print(f"\nReport -> results/curriculum_control_v2_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
