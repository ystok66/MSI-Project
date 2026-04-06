"""Step 3: Shadow Observer Evaluation — main diagnostic experiment.

Exp-1A: Global replay comparison (RMSE, MAE, 90% coverage, event-NLL)
Exp-1B: Directional event response verification
Exp-2A: Dependence inflation attribution

Uses lesson-catalog families: warn_rescue, self_discovery_needed,
  boundary_obs, false_suppression_cost.

Usage:
  python scripts/run_step3_shadow_observer_eval.py
  python scripts/run_step3_shadow_observer_eval.py --n_grid 64
  python scripts/run_step3_shadow_observer_eval.py --n_grid 96 --n_seeds 3
"""

from __future__ import annotations
import sys, os, argparse, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from src.teachers.internalization_observer import ObsEvent
from src.teachers.a1mt_observer_shadow_bridge import ShadowObserverBridge
from src.teachers.a1mt_observer_shadow_types import DIM_NAMES
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob


AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)

# Phase 1 families (high-signal subtypes)
PHASE1_FAMILIES = ["warn_rescue", "self_discovery_needed",
                   "boundary_obs", "false_suppression_cost"]


def _apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(lessons, theta, seed, n_teach, bridge, hidden_tempt=0.0):
    """Run one teaching session, feeding events to the bridge."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState()
    m.snapshot()
    bridge.reset()

    m_true_trace = []
    per_step = []

    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / n_teach for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed * 100, theta, ub, rng)
        ep, spec, gm, cfg_e, meta, sc = et
        fb, ww = _apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
        for _ in range(3):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))
        lib = BranchConceptLibrary()
        scr = BranchScorerProbe(lr=0.05, l2=0.01)
        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe", ss)
        lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)

        tutor = BCICTv4(agent_params=AP, use_dose=False)
        act, dose, _ = tutor.decide(sc, fb, lp, lib, scr, 2, m)

        dc = getattr(sc, 'commit_depth', 3)
        dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        eff_lure = tempt + hidden_tempt
        risky_branch = 1 - sc.oracle_safe_branch_id

        bas = BranchAttributes(
            safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=(sc.tempt_score_a if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_b))
        bar = BranchAttributes(
            safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=(sc.tempt_score_b if sc.oracle_safe_branch_id == 0
                              else sc.tempt_score_a))
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng,
                                    [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose > 0
        self_disc = correct and not warned and p_self > 0.5

        # True state updates
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

        risk_hat = float(lp.predict_risk(sr[0:4] if len(sr) >= 4 else np.zeros(4)))

        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose, warned=warned, follow_warn=(warned and correct),
            warn_correct=(warned and correct and risk > 0.25),
            warn_wrong=(warned and correct and risk <= 0.25),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk,
            risk_hat=risk_hat,
            lure=eff_lure, agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc,
            beneficial_novelty=(self_disc and not warned),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen,
                    "gamma_spec": m.gamma_spec, "kappa": m.kappa},
        )
        bridge.step(ev)
        m_true_trace.append(m.as_dict)

        per_step.append({
            "step": step, "subtype": ep.subtype, "warned": warned,
            "self_disc": self_disc, "correct": correct,
            "p_self": p_self, "dose": dose,
        })

    return m_true_trace, per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grid", type=int, default=32)
    parser.add_argument("--n_seeds", type=int, default=15)
    parser.add_argument("--n_teach", type=int, default=20)
    parser.add_argument("--c_bnd", type=float, default=20.0)
    parser.add_argument("--sigma_kappa", type=float, default=0.1)
    args = parser.parse_args()

    out = Path("results/step3_shadow_observer")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lines = []
    lines.append(f"# Step 3: Shadow Observer Diagnostics (n_grid={args.n_grid})\n\n")

    # ── Filter lessons to Phase 1 families ────────────────────
    all_lessons = list(LESSON_CATALOG_V2)
    phase1_lessons = [l for l in all_lessons
                      if any(f in l.subtype for f in PHASE1_FAMILIES)]
    if not phase1_lessons:
        phase1_lessons = all_lessons[:8]  # fallback
    print(f"Using {len(phase1_lessons)} lessons from {len(PHASE1_FAMILIES)} families",
          file=sys.stderr)

    # ══════════════════════════════════════════════════════════
    #  Exp-1A: Global Replay (RMSE, MAE, Coverage, NLL)
    # ══════════════════════════════════════════════════════════
    lines.append("## Exp-1A: Global Replay Comparison\n\n")
    print("Exp-1A: Global replay...", file=sys.stderr)

    lines.append(f"| Dim | RMSE_shadow | RMSE_frozen | MAE_shadow | MAE_frozen | "
                 f"Cov90_shadow | NLL_shadow |\n")
    lines.append("|-----|------------|------------|-----------|-----------|"
                 "------------|------------|\n")

    all_diags = []
    for theta in ["safe", "shiny"]:
        for seed in range(args.n_seeds):
            bridge = ShadowObserverBridge(
                shadow_mode="prob", n_grid=args.n_grid,
                c_bnd=args.c_bnd, sigma_kappa=args.sigma_kappa)
            run_session(phase1_lessons, theta, seed, args.n_teach, bridge)
            diag = bridge.get_diagnostics()
            all_diags.append(diag)

    # Aggregate across seeds/thetas
    for dim in DIM_NAMES:
        rmse_s = np.mean([d.rmse.get(dim, 0) for d in all_diags])
        rmse_f = np.mean([d.rmse_frozen.get(dim, 0) for d in all_diags])
        mae_s = np.mean([d.mae.get(dim, 0) for d in all_diags])
        mae_f = np.mean([d.mae_frozen.get(dim, 0) for d in all_diags])
        cov = np.mean([d.coverage_90.get(dim, 0) for d in all_diags])
        nll = np.mean([d.mean_event_nll for d in all_diags])
        lines.append(f"| {dim} | {rmse_s:.4f} | {rmse_f:.4f} | {mae_s:.4f} | "
                     f"{mae_f:.4f} | {cov:.3f} | {nll:.4f} |\n")
        # Check: did shadow improve?
        delta = rmse_f - rmse_s
        tag = "BETTER" if delta > 0.001 else ("SAME" if abs(delta) < 0.001 else "WORSE")
        print(f"  {dim}: RMSE shadow={rmse_s:.4f} frozen={rmse_f:.4f} "
              f"({tag}) Cov90={cov:.3f}", file=sys.stderr)

    # ══════════════════════════════════════════════════════════
    #  Exp-1B: Directional Event Responses
    # ══════════════════════════════════════════════════════════
    lines.append("\n## Exp-1B: Directional Event Responses\n\n")
    print("\nExp-1B: Directional responses...", file=sys.stderr)

    all_responses = {}
    for theta in ["safe", "shiny"]:
        for seed in range(args.n_seeds):
            bridge = ShadowObserverBridge(
                shadow_mode="prob", n_grid=args.n_grid,
                c_bnd=args.c_bnd, sigma_kappa=args.sigma_kappa)
            run_session(phase1_lessons, theta, seed, args.n_teach, bridge)
            resp = bridge.directional_responses()
            for key, val in resp.items():
                all_responses.setdefault(key, []).extend(
                    [val["mean"]] * val["n"])

    expected_signs = {
        "delta_tau_trust+": "+",
        "delta_tau_trust-": "-",
        "delta_nu_blind": "+",
        "delta_nu_selfdisc": "-",
        "delta_gg_pressure": "+",
        "delta_gg_explore+": "-",
        "delta_gs_lure": "+",
    }

    lines.append("| Event | Expected | Mean Delta | N | Correct? |\n")
    lines.append("|-------|----------|-----------|---|----------|\n")
    n_correct = 0
    n_total = 0
    for key, sign in expected_signs.items():
        vals = all_responses.get(key, [])
        if vals:
            mean_d = np.mean(vals)
            ok = (mean_d > 0 and sign == "+") or (mean_d < 0 and sign == "-")
            n_correct += int(ok)
            n_total += 1
            lines.append(f"| {key} | {sign} | {mean_d:+.6f} | {len(vals)} "
                         f"| {'YES' if ok else 'NO'} |\n")
            print(f"  {key}: mean={mean_d:+.6f} (n={len(vals)}) "
                  f"{'OK' if ok else 'WRONG'}", file=sys.stderr)
        else:
            lines.append(f"| {key} | {sign} | N/A | 0 | N/A |\n")
            n_total += 1

    lines.append(f"\n**Directional correctness: {n_correct}/{n_total}**\n\n")

    # ══════════════════════════════════════════════════════════
    #  Exp-2A: Dependence Inflation Attribution
    # ══════════════════════════════════════════════════════════
    lines.append("## Exp-2A: Dependence Inflation Attribution\n\n")
    print("\nExp-2A: Dependence inflation...", file=sys.stderr)

    # Track nu deltas by condition
    nu_deltas_warned = []
    nu_deltas_no_warn = []
    nu_deltas_selfdisc = []

    for theta in ["safe", "shiny"]:
        for seed in range(args.n_seeds):
            bridge = ShadowObserverBridge(
                shadow_mode="prob", n_grid=args.n_grid,
                c_bnd=args.c_bnd, sigma_kappa=args.sigma_kappa)
            _, per_step = run_session(
                phase1_lessons, theta, seed, args.n_teach, bridge)
            sh = bridge.get_shadow_history()

            for t in range(1, len(sh)):
                nu_delta = sh[t].mean("nu") - sh[t-1].mean("nu")
                if per_step[t]["warned"]:
                    nu_deltas_warned.append(nu_delta)
                elif per_step[t]["self_disc"]:
                    nu_deltas_selfdisc.append(nu_delta)
                else:
                    nu_deltas_no_warn.append(nu_delta)

    lines.append("| Condition | Mean dNu | N | Std |\n")
    lines.append("|-----------|---------|---|-----|\n")
    for name, vals in [("warned (blind)", nu_deltas_warned),
                       ("no_warn (neutral)", nu_deltas_no_warn),
                       ("self_discovery", nu_deltas_selfdisc)]:
        if vals:
            lines.append(f"| {name} | {np.mean(vals):+.6f} | {len(vals)} "
                         f"| {np.std(vals):.6f} |\n")
            print(f"  {name}: dNu={np.mean(vals):+.6f} (n={len(vals)})",
                  file=sys.stderr)
        else:
            lines.append(f"| {name} | N/A | 0 | N/A |\n")

    # ── Frozen vs shadow nu comparison ────────────────────────
    lines.append("\n### Frozen vs Shadow nu trajectory\n\n")
    bridge = ShadowObserverBridge(
        shadow_mode="prob", n_grid=args.n_grid,
        c_bnd=args.c_bnd, sigma_kappa=args.sigma_kappa)
    run_session(phase1_lessons, "safe", 0, args.n_teach, bridge)
    fh = bridge.get_frozen_history()
    sh = bridge.get_shadow_history()
    lines.append("| Step | nu_frozen | nu_shadow | nu_shadow_std |\n")
    lines.append("|------|----------|----------|---------------|\n")
    for t in range(min(len(fh), len(sh))):
        nf = fh[t].get("nu", 0)
        ns = sh[t].mean("nu")
        ns_std = sh[t].posteriors["nu"].std
        lines.append(f"| {t} | {nf:.4f} | {ns:.4f} | {ns_std:.4f} |\n")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t0
    lines.append(f"\n---\n\n**Elapsed**: {elapsed:.1f}s | "
                 f"**n_grid**: {args.n_grid} | **seeds**: {args.n_seeds}\n")

    rpt = out / f"step3_eval_g{args.n_grid}.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nReport -> {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
