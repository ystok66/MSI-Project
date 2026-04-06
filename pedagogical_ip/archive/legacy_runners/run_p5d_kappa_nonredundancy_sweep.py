"""P5-D: κ̂ Non-Redundancy Proof + β Sweep.

Part 1: ΔR² (does κ̂ add explanatory power beyond 4D?)
Part 2: Conditional correlation (κ̂ vs γ̂_spec | θ, temptation)
Part 3: β sweep for macro bonus (0, 0.02, 0.05, 0.1, 0.2)
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
ALL_LESSONS = list(LESSON_CATALOG_V2)
CAT = list(LESSON_CATALOG_V2)
RISK_FAMILIES = {"tic_rescue_heavy", "blind_activation_corridor",
                 "warn_symmetric_rescue"}
EPS_0 = 0.30; A_S = 0.15; B_S = 0.10; STOP_THRESH = 0.35


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(theta, seed, n_teach=20, hidden_tempt=0.0):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
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
        action, dose, info = BCICTv4(agent_params=AP, use_dose=False).decide(
            sc, fb, lp, lib, scr, 2, m)
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        eff_lure = tempt + hidden_tempt
        risky_branch = 1 - sc.oracle_safe_branch_id
        ts = [0.0, 0.0]; ts[risky_branch] = hidden_tempt
        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=(sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b) + ts[0])
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=(sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a) + ts[1])
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
        risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat, lure=eff_lure,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        observer.update(ev)
        est = observer.get_estimate()
        records.append({
            "step": step, "theta": theta, "family": les.name,
            "risk": risk, "risk_hat": risk_hat,
            "delta_risk": risk - risk_hat,
            "tempt": eff_lure,
            "est": est, "correct": correct,
        })
    return records


def linear_r2(X, y):
    """OLS R² for X (n×d) predicting y (n,)."""
    X = np.array(X); y = np.array(y)
    if X.ndim == 1: X = X.reshape(-1, 1)
    X = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        return 0.0


def main():
    print("═══ P5-D: Non-Redundancy + β Sweep ═══\n", file=sys.stderr)
    L = ["# P5-D: κ̂ Non-Redundancy Proof + β Sweep\n\n"]

    # ═══ Part 1: ΔR² ═════════════════════════════════════
    L.append("## Part 1: ΔR² — Incremental Explanatory Power\n\n")
    print("Part 1: ΔR²...", file=sys.stderr)

    all_recs = []
    for th in ["safe", "shiny"]:
        for ht_label, ht in [("none", 0.0), ("tempt", 0.6)]:
            for sid in range(NS):
                recs = run_session(th, sid, hidden_tempt=ht)
                for r in recs:
                    r["condition"] = f"{th}_{ht_label}"
                all_recs.extend(recs)

    # Extract arrays
    y = np.array([r["delta_risk"] for r in all_recs])
    X_4d = np.array([[r["est"]["tau"], r["est"]["nu"],
                       r["est"]["gamma_gen"], r["est"]["gamma_spec"]]
                      for r in all_recs])
    X_5d = np.array([[r["est"]["tau"], r["est"]["nu"],
                       r["est"]["gamma_gen"], r["est"]["gamma_spec"],
                       r["est"]["kappa"]]
                      for r in all_recs])

    r2_4d = linear_r2(X_4d, y)
    r2_5d = linear_r2(X_5d, y)
    delta_r2 = r2_5d - r2_4d

    L.append("| Model | R² | ΔR² |\n")
    L.append("|-------|:--:|:---:|\n")
    L.append(f"| 4D (τ̂,ν̂,γ̂_gen,γ̂_spec) | {r2_4d:.6f} | — |\n")
    L.append(f"| 5D (+κ̂) | {r2_5d:.6f} | **{delta_r2:+.6f}** |\n\n")

    # Per-condition ΔR²
    L.append("### Per-Condition ΔR²\n\n")
    L.append("| Condition | R²(4D) | R²(5D) | ΔR² |\n")
    L.append("|-----------|:------:|:------:|:---:|\n")
    conditions = sorted(set(r["condition"] for r in all_recs))
    for cond in conditions:
        cr = [r for r in all_recs if r["condition"] == cond]
        yc = np.array([r["delta_risk"] for r in cr])
        x4 = np.array([[r["est"]["tau"], r["est"]["nu"],
                         r["est"]["gamma_gen"], r["est"]["gamma_spec"]] for r in cr])
        x5 = np.array([[r["est"]["tau"], r["est"]["nu"],
                         r["est"]["gamma_gen"], r["est"]["gamma_spec"],
                         r["est"]["kappa"]] for r in cr])
        r4 = linear_r2(x4, yc); r5 = linear_r2(x5, yc)
        L.append(f"| {cond} | {r4:.6f} | {r5:.6f} | {r5-r4:+.6f} |\n")

    # ═══ Part 2: Conditional Correlation ══════════════════
    L.append("\n## Part 2: Conditional Corr(κ̂, γ̂_spec)\n\n")
    print("\nPart 2: Conditional corr...", file=sys.stderr)

    L.append("| Condition | Raw Corr | Partial Corr (residualized) |\n")
    L.append("|-----------|:--------:|:---------------------------:|\n")
    for cond in conditions:
        cr = [r for r in all_recs if r["condition"] == cond]
        kaps = np.array([r["est"]["kappa"] for r in cr])
        gsps = np.array([r["est"]["gamma_spec"] for r in cr])
        # Raw
        raw_c = sp_stats.pearsonr(kaps, gsps)[0] if len(set(kaps)) > 1 else 0
        # Partial: residualize both on (τ̂, ν̂, γ̂_gen)
        X_base = np.array([[r["est"]["tau"], r["est"]["nu"],
                             r["est"]["gamma_gen"]] for r in cr])
        X_b = np.column_stack([np.ones(len(X_base)), X_base])
        try:
            b_k = np.linalg.lstsq(X_b, kaps, rcond=None)[0]
            b_g = np.linalg.lstsq(X_b, gsps, rcond=None)[0]
            res_k = kaps - X_b @ b_k
            res_g = gsps - X_b @ b_g
            part_c = sp_stats.pearsonr(res_k, res_g)[0]
        except Exception:
            part_c = float('nan')
        L.append(f"| {cond} | {raw_c:.4f} | {part_c:.4f} |\n")

    # ═══ Part 3: β Sweep ═════════════════════════════════
    L.append("\n## Part 3: Macro κ-Bonus β Sweep\n\n")
    print("\nPart 3: β sweep...", file=sys.stderr)

    betas = [0.0, 0.02, 0.05, 0.1, 0.2]
    L.append("| β | Risk Rank Shift | Top-1 Agree | Kendall τ | "
             "STOP Agree |\n")
    L.append("|:-:|:---------------:|:-----------:|:---------:|"
             ":----------:|\n")

    for beta in betas:
        shifts_all = []; t1_ok = 0; kts = []; stop_ok = 0; n = 0
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs = run_session(th, sid)
                rl = recs[-1]; est = rl["est"]
                # Oracle scores
                so = [np.mean(l.gain)*(1-est["nu"])*(1-est["gamma_gen"])*est["tau"]
                      for l in CAT]
                # Base scores
                sb = list(so)
                # Bonus scores
                sk = []
                for i, l in enumerate(CAT):
                    s = sb[i]
                    if l.name in RISK_FAMILIES:
                        s += beta * (1 - est["kappa"])
                    sk.append(s)
                n += 1
                if np.argsort(so)[-1] == np.argsort(sk)[-1]: t1_ok += 1
                kt, _ = sp_stats.kendalltau(so, sk)
                kts.append(kt)
                # Risk rank shift
                rank_base = list(np.argsort(sb)[::-1])
                rank_bonus = list(np.argsort(sk)[::-1])
                for i, l in enumerate(CAT):
                    if l.name in RISK_FAMILIES:
                        shifts_all.append(rank_base.index(i) - rank_bonus.index(i))
                # STOP
                s_o = EPS_0 + A_S * est["nu"] + B_S * est["gamma_gen"]
                if beta == 0:
                    s_i = s_o
                else:
                    s_i = s_o  # STOP not modified by κ bonus
                if (s_o > STOP_THRESH) == (s_i > STOP_THRESH): stop_ok += 1

        avg_shift = np.mean(shifts_all) if shifts_all else 0
        L.append("| {:.2f} | {:+.1f} | {:.0f}% | {:.4f} | {:.0f}% |\n".format(
            beta, avg_shift, 100*t1_ok/n, np.mean(kts), 100*stop_ok/n))

    # ═══ Verdict ═════════════════════════════════════════
    L.append("\n## Verdict\n\n")
    L.append(f"> **ΔR² = {delta_r2:+.6f}**: κ̂ adds ")
    if delta_r2 > 0.001:
        L.append("meaningful incremental explanatory power beyond 4D.\n\n")
    elif delta_r2 > 0:
        L.append("small but positive incremental power.\n\n")
    else:
        L.append("no incremental power (investigate).\n\n")

    L.append("> **β sweep**: See table above for optimal β range.\n")

    rpt = out / "p5d_kappa_nonredundancy_sweep.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
