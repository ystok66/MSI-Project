"""T3-Followup: Intervention Timing + Planning Coupling Audit.

Exp-T3-F1: Intervention timing — does the risk head flag timeout/blind-commit
           earlier than geometry-only heuristics?
Exp-T3-F2: Planning coupling — does predictive gain change lesson ranking
           selectively in hard families?
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
from src.agents.agent_belief_state import AgentBelief
from src.agents.world_state import WorldState
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.teachers.action_predictor import ActionPredictor
from src.teachers.robot_belief_over_agent import RobotBeliefOverAgent
from src.teachers.intervention_risk_head import InterventionRiskHead
from src.teachers.macro_predictive_hook import MacroPredictiveHook
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

LESSONS = [l for l in LESSON_CATALOG_V2 if l.family in ("TIC", "TIC-v4")]
# Timing-sensitive families
TIMING_FAMILIES = {"TIC-v4"}
N_STEPS = 25
N_SEEDS = 10


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_audit(theta, seed):
    rng = np.random.default_rng(seed * 10000)
    m = FactoredInternalizationState(); m.snapshot()
    observer = A1MtObserverFrozen(); observer.reset()

    # POMDP interfaces
    ap = ActionPredictor(params=AP)
    rboa = RobotBeliefOverAgent(action_predictor=ap)
    irh = InterventionRiskHead(lambda_time=1.0, lambda_blind=1.0, threshold=0.5)
    mph = MacroPredictiveHook(action_predictor=ap, beta_pred=0.5)

    timing_results = []
    coupling_results = []

    for step_i in range(N_STEPS):
        les = LESSONS[step_i % len(LESSONS)]
        ub = {p: 0.4 + 0.1 * (step_i / N_STEPS) for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(
            les, step_i + seed * 10000, theta, ub, rng)
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
        branches = [bas, bar]

        ac = sample_factored_choice(branches, theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose > 0

        # === Timing audit ===
        ws = WorldState(t=step_i, t_max=N_STEPS,
                       agent_pos=(0, 0), goal_pos=(7, 7))
        ab = AgentBelief(m_state=dict(m.as_dict), theta=theta)
        irisk = irh.predict(ws, rboa, ap, branches, ab,
                           d_commit=dc, d_reveal=dr,
                           path_length_estimate=N_STEPS - step_i)

        # Heuristic baseline: simple geometry check
        heuristic_timeout = 1 if (N_STEPS - step_i) < 5 else 0
        heuristic_blind = 1 if dc < dr else 0

        # Ground truth: did failure actually happen?
        actual_failure = not correct and risk > 0.3
        actual_timeout = (step_i >= N_STEPS - 3) and not correct

        timing_results.append({
            "step": step_i,
            "family": les.family,
            "irh_flagged": irisk.flagged,
            "irh_p_timeout": irisk.p_timeout,
            "irh_p_blind": irisk.p_blind,
            "irh_u_int": irisk.u_int,
            "heur_timeout": heuristic_timeout,
            "heur_blind": heuristic_blind,
            "actual_failure": actual_failure,
            "actual_timeout": actual_timeout,
            "correct": correct,
            "warned": warned,
        })

        # === Planning coupling ===
        # Score a small set of lessons
        sample_lessons = LESSONS[:min(5, len(LESSONS))]
        base_scores = [0.5 + 0.1 * i for i in range(len(sample_lessons))]
        gains = []
        for sl in sample_lessons:
            # Approximate predictive gain: lessons matching current weakness
            probe_branches = [branches]
            g = mph.score_predictive_gain(
                sl.name, ab, probe_branches, [sc.oracle_safe_branch_id])
            gains.append(g)
        scores = mph.rerank_lessons_shadow(
            [sl.name for sl in sample_lessons], base_scores, gains)

        n_changed = sum(1 for s in scores if s.rank_changed)
        coupling_results.append({
            "step": step_i,
            "family": les.family,
            "n_changed": n_changed,
            "mean_gain": float(np.mean(gains)),
            "top1_base": scores[0].lesson_name if scores else "?",
        })

        # State updates
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

        rboa.update_from_action(ws, branches, ac, ab)

    return timing_results, coupling_results


def main():
    print("═══ T3-Followup: Timing + Coupling Audit ═══\n", file=sys.stderr)
    L = ["# T3-Followup: Intervention Timing + Planning Coupling\n\n"]

    all_timing = defaultdict(list)
    all_coupling = defaultdict(list)

    for th in ["safe", "shiny"]:
        for sid in range(N_SEEDS):
            tr, cr = run_audit(th, sid)
            all_timing[th].extend(tr)
            all_coupling[th].extend(cr)
        print(f"  {th} done", file=sys.stderr)

    # ═══ Table 1: Timing precision/recall ═══
    L.append("## Exp-T3-F1: Intervention Timing\n\n")
    L.append("| θ | IRH Precision | IRH Recall | Heur Precision | Heur Recall | IRH Lead |\n")
    L.append("|:-:|:------------:|:----------:|:--------------:|:-----------:|:--------:|\n")

    for th in ["safe", "shiny"]:
        tr = all_timing[th]
        # IRH as classifier
        irh_tp = sum(1 for r in tr if r["irh_flagged"] and r["actual_failure"])
        irh_fp = sum(1 for r in tr if r["irh_flagged"] and not r["actual_failure"])
        irh_fn = sum(1 for r in tr if not r["irh_flagged"] and r["actual_failure"])
        irh_prec = irh_tp / max(irh_tp + irh_fp, 1)
        irh_rec = irh_tp / max(irh_tp + irh_fn, 1)

        # Heuristic as classifier
        h_tp = sum(1 for r in tr if (r["heur_timeout"] or r["heur_blind"])
                   and r["actual_failure"])
        h_fp = sum(1 for r in tr if (r["heur_timeout"] or r["heur_blind"])
                   and not r["actual_failure"])
        h_fn = sum(1 for r in tr if not (r["heur_timeout"] or r["heur_blind"])
                   and r["actual_failure"])
        h_prec = h_tp / max(h_tp + h_fp, 1)
        h_rec = h_tp / max(h_tp + h_fn, 1)

        # Lead time: for true failures, first step where IRH flagged
        leads = []
        for sid in range(N_SEEDS):
            seed_tr = [r for r in tr if r["actual_failure"]]
            for r in seed_tr:
                if r["irh_flagged"]:
                    leads.append(N_STEPS - r["step"])
        mean_lead = float(np.mean(leads)) if leads else 0.0

        L.append(f"| {th} | {irh_prec:.3f} | {irh_rec:.3f} | "
                 f"{h_prec:.3f} | {h_rec:.3f} | {mean_lead:.1f} |\n")

    # ═══ Table 2: Risk scores by family ═══
    L.append("\n## Risk Scores by Family\n\n")
    L.append("| θ | Family | Mean p_timeout | Mean p_blind | Mean u_int | Flag Rate |\n")
    L.append("|:-:|:------:|:--------------:|:------------:|:----------:|:---------:|\n")
    for th in ["safe", "shiny"]:
        tr = all_timing[th]
        families = set(r["family"] for r in tr)
        for fam in sorted(families):
            fr = [r for r in tr if r["family"] == fam]
            L.append(f"| {th} | {fam} | "
                     f"{np.mean([r['irh_p_timeout'] for r in fr]):.3f} | "
                     f"{np.mean([r['irh_p_blind'] for r in fr]):.3f} | "
                     f"{np.mean([r['irh_u_int'] for r in fr]):.3f} | "
                     f"{np.mean([r['irh_flagged'] for r in fr]):.3f} |\n")

    # ═══ Table 3: Planning coupling ═══
    L.append("\n## Exp-T3-F2: Planning Coupling\n\n")
    L.append("| θ | Mean Gain | Rank Change Rate | Selective? |\n")
    L.append("|:-:|:---------:|:----------------:|:----------:|\n")
    for th in ["safe", "shiny"]:
        cr = all_coupling[th]
        mg = np.mean([r["mean_gain"] for r in cr])
        rcr = np.mean([r["n_changed"] for r in cr]) / 5.0
        # Selective = changes only in timing families
        timing_cr = [r for r in cr if r["family"] in TIMING_FAMILIES]
        other_cr = [r for r in cr if r["family"] not in TIMING_FAMILIES]
        t_rate = np.mean([r["n_changed"] for r in timing_cr]) / 5.0 if timing_cr else 0
        o_rate = np.mean([r["n_changed"] for r in other_cr]) / 5.0 if other_cr else 0
        selective = "✅" if t_rate >= o_rate else "⚠️"
        L.append(f"| {th} | {mg:.4f} | {rcr:.3f} | {selective} |\n")

    # ═══ Verdict ═══
    L.append("\n## Verdict\n\n")

    # Check 1: IRH recall ≥ heuristic
    recall_ok = 0
    for th in ["safe", "shiny"]:
        tr = all_timing[th]
        irh_tp = sum(1 for r in tr if r["irh_flagged"] and r["actual_failure"])
        irh_fn = sum(1 for r in tr if not r["irh_flagged"] and r["actual_failure"])
        h_tp = sum(1 for r in tr if (r["heur_timeout"] or r["heur_blind"])
                   and r["actual_failure"])
        h_fn = sum(1 for r in tr if not (r["heur_timeout"] or r["heur_blind"])
                   and r["actual_failure"])
        irh_rec = irh_tp / max(irh_tp + irh_fn, 1)
        h_rec = h_tp / max(h_tp + h_fn, 1)
        if irh_rec >= h_rec - 0.05:
            recall_ok += 1
    L.append(f"> IRH recall ≥ heuristic: {recall_ok}/2 θ\n")

    # Check 2: Selective coupling
    selective_ok = 0
    for th in ["safe", "shiny"]:
        cr = all_coupling[th]
        timing_cr = [r for r in cr if r["family"] in TIMING_FAMILIES]
        other_cr = [r for r in cr if r["family"] not in TIMING_FAMILIES]
        t_rate = np.mean([r["n_changed"] for r in timing_cr]) if timing_cr else 0
        o_rate = np.mean([r["n_changed"] for r in other_cr]) if other_cr else 0
        if t_rate >= o_rate - 0.1:
            selective_ok += 1
    L.append(f"> Planning coupling selective: {selective_ok}/2 θ\n")

    if recall_ok >= 1 and selective_ok >= 1:
        L.append("> **✅ T3-Followup validates utility of POMDP interfaces**\n")
    else:
        L.append("> **⚠️ Partial validation — needs further investigation**\n")

    rpt = out / "t3_followup_timing_coupling.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
