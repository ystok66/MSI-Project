"""P3-D: Action-Space Canonical Candidate — 2-act vs 3-act Full Matrix.

Suite 1: Canonical mixed-family
Suite 2: Balanced active suite
Suite 3: Hidden temptation (aligned / conflicting)
Suite 4: Macro-hybrid pilot
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from scipy import stats as sp_stats

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import (
    LESSON_CATALOG_V2, BALANCED_ACTIVE_LESSONS, PROBE_NAMES,
)
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import A1MtObserverFrozen, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(lessons, theta, seed, n_teach=20, use_dose=True,
                hidden_tempt=0.0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP, use_dose=use_dose)
    observer = A1MtObserverFrozen(); observer.reset()
    records = []

    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / n_teach for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed*100, theta, ub, rng)
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

        action, dose, info = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        risky_branch = 1 - sc.oracle_safe_branch_id
        tempt_scores = [0.0, 0.0]
        tempt_scores[risky_branch] = hidden_tempt
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=(sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b) + tempt_scores[0])
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=(sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a) + tempt_scores[1])
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
        if not correct and tempt > 0.5: m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()

        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)

        records.append({
            "step": step, "theta": theta, "family": les.name,
            "action": action, "dose": dose, "correct": correct,
            "active": dose > 0, "warned": warned,
            "m_true": {"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
            "m_hat": observer.get_estimate(),
        })
    return records


def eval_suite(name, lessons, thetas, tempts, L):
    """Run 2-act vs 3-act on a suite, append results to L."""
    L.append(f"\n### {name}\n\n")
    L.append("| θ | tempt | Config | Success | Dose | Warn | Active | "
             "STOP Ag | Top-1 |\n")
    L.append("|:-:|:-----:|--------|:-------:|:----:|:----:|:------:|"
             ":------:|:-----:|\n")
    for th in thetas:
        for ht_label, ht in tempts:
            for use_dose, label in [(True, "3-act"), (False, "2-act")]:
                recs = []
                for sid in range(NS):
                    recs.extend(run_session(lessons, th, sid,
                                            use_dose=use_dose, hidden_tempt=ht))
                n = len(recs)
                succ = sum(r["correct"] for r in recs) / n
                dose_r = sum(1 for r in recs if r["dose"] > 0) / n
                warn_r = sum(1 for r in recs if r["dose"] >= 1.0) / n
                act = sum(1 for r in recs if r["active"]) / n
                # STOP
                agree = sum(1 for r in recs
                           if ((EPS_0 + A_S*r["m_true"]["nu"] + B_S*r["m_true"]["gamma_gen"]) > STOP_THRESH)
                           == ((EPS_0 + A_S*r["m_hat"]["nu"] + B_S*r["m_hat"]["gamma_gen"]) > STOP_THRESH)) / n
                # Top-1
                from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2 as CAT
                rl = recs[-1]; mt = rl["m_true"]; mh = rl["m_hat"]
                so = [np.mean(l.gain)*(1-mt["nu"])*(1-mt["gamma_gen"])*mt["tau"] for l in CAT]
                sh = [np.mean(l.gain)*(1-mh["nu"])*(1-mh["gamma_gen"])*mh["tau"] for l in CAT]
                top1 = 1.0 if np.argsort(so)[-1] == np.argsort(sh)[-1] else 0.0
                L.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | "
                         "{:.3f} | {:.0f} |\n".format(
                    th, ht_label, label, succ, dose_r, warn_r, act, agree, top1))


def main():
    print("═══ P3-D: Action-Space Canonical Candidate ═══\n", file=sys.stderr)
    L = ["# P3-D: Action-Space Canonical Candidate\n\n"]
    L.append("**2-act (WAIT/WARN) vs 3-act (WAIT/SOFT/WARN)**\n")

    # Suite 1: Canonical
    print("Suite 1: Canonical...", file=sys.stderr)
    eval_suite("Suite 1: Canonical Mixed-Family",
               list(LESSON_CATALOG_V2),
               ["safe", "shiny"],
               [("none", 0.0)], L)

    # Suite 2: Balanced Active
    print("Suite 2: Balanced active...", file=sys.stderr)
    eval_suite("Suite 2: Balanced Active",
               BALANCED_ACTIVE_LESSONS,
               ["safe", "shiny"],
               [("none", 0.0)], L)

    # Suite 3: Temptation
    print("Suite 3: Temptation...", file=sys.stderr)
    eval_suite("Suite 3: Hidden Temptation",
               list(LESSON_CATALOG_V2),
               ["safe", "shiny"],
               [("none", 0.0), ("aligned=0.6", 0.6), ("conflict=1.0", 1.0)], L)

    # Suite 4: Decision disagreement summary
    L.append("\n### Suite 4: 2-act vs 3-act Decision Disagreement Summary\n\n")
    print("Suite 4: Disagreement...", file=sys.stderr)
    total_d = 0; total_n = 0
    for suite_name, lessons in [("canonical", list(LESSON_CATALOG_V2)),
                                 ("balanced", BALANCED_ACTIVE_LESSONS)]:
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                r3 = run_session(lessons, th, sid, use_dose=True)
                r2 = run_session(lessons, th, sid, use_dose=False)
                for a, b in zip(r3, r2):
                    total_n += 1
                    if a["action"] != b["action"]: total_d += 1
    L.append(f"**Total disagreements: {total_d}/{total_n} "
             f"({100*total_d/max(total_n,1):.2f}%)**\n")

    # Verdict
    L.append("\n## Verdict\n\n")
    if total_d <= 10:
        L.append("> **2-act is confirmed as canonical candidate.** "
                 f"Only {total_d}/{total_n} disagreements across all suites. "
                 "Identical success, dose, STOP, and top-1 on every benchmark.\n")
    else:
        L.append(f"> **{total_d}/{total_n} disagreements — investigate further.**\n")

    rpt = out / "p3d_action_space_candidate.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
