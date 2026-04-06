"""P4-A: γ_spec Verification — temptation-specific generalization tracking.

Exp 1: γ_spec trajectory under varying temptation
Exp 2: ν contamination check (ν should NOT absorb temptation)
Exp 3: Correlation: γ_spec vs true γ_spec
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

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
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15
ALL_LESSONS = list(LESSON_CATALOG_V2)


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_tempt_session(theta, seed, hidden_tempt=0.0, n_teach=20):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = A1MtObserverFrozen(); observer.reset()
    records = []

    for step in range(n_teach):
        les = ALL_LESSONS[step % len(ALL_LESSONS)]
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
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            lure=tempt + hidden_tempt,  # effective lure includes hidden tempt
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen,
                    "gamma_spec": m.gamma_spec},
        )
        observer.update(ev)

        records.append({
            "step": step, "theta": theta, "hidden_tempt": hidden_tempt,
            "correct": correct, "lure": tempt + hidden_tempt,
            "gamma_spec_true": m.gamma_spec,
            "gamma_spec_hat": observer.gamma_spec_hat,
            "nu_hat": observer.nu_hat,
            "nu_true": m.nu,
            "gamma_gen_hat": observer.gamma_gen_hat,
            "gamma_gen_true": m.gamma_gen,
        })
    return records


def main():
    print("═══ P4-A: γ_spec Verification ═══\n", file=sys.stderr)
    L = ["# P4-A: γ_spec Verification\n\n"]

    # ─── Exp 1: γ_spec trajectory under varying temptation ─
    L.append("## Exp 1: γ̂_spec Trajectory by Temptation Level\n\n")
    L.append("| θ | Tempt | γ̂_spec(final) | γ_spec(true,final) | "
             "ν̂(final) | ν(true,final) | γ̂_gen(final) |\n")
    L.append("|:-:|:-----:|:-------------:|:------------------:|"
             ":--------:|:-------------:|:------------:|\n")
    print("Exp 1: Trajectories...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        for ht_label, ht in [("none", 0.0), ("0.3", 0.3), ("0.6", 0.6), ("1.0", 1.0)]:
            gs_finals = []; gs_true = []; nu_finals = []; nu_true = []
            gg_finals = []
            for sid in range(NS):
                recs = run_tempt_session(th, sid, hidden_tempt=ht)
                gs_finals.append(recs[-1]["gamma_spec_hat"])
                gs_true.append(recs[-1]["gamma_spec_true"])
                nu_finals.append(recs[-1]["nu_hat"])
                nu_true.append(recs[-1]["nu_true"])
                gg_finals.append(recs[-1]["gamma_gen_hat"])
            L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                th, ht_label,
                np.mean(gs_finals), np.mean(gs_true),
                np.mean(nu_finals), np.mean(nu_true),
                np.mean(gg_finals)))

    # ─── Exp 2: ν contamination check ────────────────────
    L.append("\n## Exp 2: ν Contamination Check\n\n")
    L.append("Does ν̂ change when temptation increases? (It shouldn't.)\n\n")
    L.append("| θ | ν̂(tempt=0) | ν̂(tempt=0.6) | ν̂(tempt=1.0) | "
             "Δν̂(0→1.0) |\n")
    L.append("|:-:|:----------:|:------------:|:------------:|"
             ":----------:|\n")
    print("\nExp 2: ν contamination...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        nu_by_tempt = {}
        for ht in [0.0, 0.6, 1.0]:
            nus = []
            for sid in range(NS):
                recs = run_tempt_session(th, sid, hidden_tempt=ht)
                nus.append(recs[-1]["nu_hat"])
            nu_by_tempt[ht] = np.mean(nus)
        delta = nu_by_tempt[1.0] - nu_by_tempt[0.0]
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
            th, nu_by_tempt[0.0], nu_by_tempt[0.6], nu_by_tempt[1.0], delta))

    # ─── Exp 3: γ_spec monotonicity ─────────────────────
    L.append("\n## Exp 3: γ̂_spec Monotonicity\n\n")
    L.append("Does γ̂_spec increase with resistance (higher tempt + correct choices)?\n\n")
    L.append("| θ | Tempt | Mean γ̂_spec | Monotone? |\n")
    L.append("|:-:|:-----:|:-----------:|:---------:|\n")
    print("\nExp 3: Monotonicity...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        prev = -1
        for ht_label, ht in [("0.0", 0.0), ("0.3", 0.3), ("0.6", 0.6), ("1.0", 1.0)]:
            vals = []
            for sid in range(NS):
                recs = run_tempt_session(th, sid, hidden_tempt=ht)
                vals.append(recs[-1]["gamma_spec_hat"])
            mean_gs = np.mean(vals)
            mono = "✅" if mean_gs >= prev - 0.001 else "❌"
            L.append("| {} | {} | {:.4f} | {} |\n".format(th, ht_label, mean_gs, mono))
            prev = mean_gs

    rpt = out / "p4a_gamma_spec_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
