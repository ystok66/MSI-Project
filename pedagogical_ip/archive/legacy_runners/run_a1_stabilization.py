"""A1 stabilization: blind audit + A2 comparison + confidence calibration + hybrid dry-run.

Exp A: p_self activation audit (warn rate, blind count, timing regimes)
Exp B: Blind definition comparison (A1 original vs A2 expanded)
Exp C: Confidence calibration (A0 vs A1 vs A2)
Exp D: Hybrid dry-run (α=0.25/0.5/0.75/1.0)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np

from src.agents.stochastic_agent_policy import BranchAttributes, AgentPolicyParams
from src.agents.internalization_state_v3 import (
    FactoredInternalizationState, sample_factored_choice,
)
from src.agents.behavior_probes import all_probes
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.curriculum.lesson_library_v2 import LESSON_CATALOG_V2, PROBE_NAMES
from src.curriculum.adaptive_episode_generator_v2 import generate_episode_from_lesson_v2
from src.teachers.internalization_control_tutor_v4 import BCICTv4
from src.teachers.internalization_observer import (
    RuleBasedMtObserver, A1MtObserver, A2MtObserver, ObsEvent,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 12
ALL_LESSONS = list(LESSON_CATALOG_V2)


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(lessons, theta, seed, n_teach=12, observer=None):
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    if observer is None: observer = A1MtObserver()
    observer.reset()
    records = []
    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / max(n_teach, 1) for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed * 100, theta, ub, rng)
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
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        dc = getattr(sc, 'commit_depth', 3); dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)
        bas = BranchAttributes(safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b)
        bar = BranchAttributes(safety_score=float(sr[0]), risk_penalty=risk,
            temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a)
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose_oracle > 0; follow_warn = warned and correct
        has_self_ev = p_self > 0.5
        self_disc = correct and not warned and has_self_ev
        bn = ep.subtype == "beneficial_novelty" and correct
        fs = ep.subtype == "false_suppression_cost" and not correct
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if not has_self_ev: m.update_dependence(blind_obey=True)
            m.update_gamma_gen(sustained_pressure=True)
        else:
            if self_disc:
                m.update_dependence(self_discovery=True)
                m.update_gamma_gen(successful_exploration=True)
        if not correct and tempt > 0.5: m.update_gamma_spec(tempt_error=True)
        m.update_risk(risk if not correct else 0.05, 0.15); m.snapshot()
        probes = all_probes(m, AP, theta) if step % 2 == 0 else {}
        ev = ObsEvent(
            episode_id=seed, step_id=step, subtype=ep.subtype, theta_post=theta,
            dose=dose_oracle, warned=warned, follow_warn=follow_warn,
            warn_correct=(warned and risk > 0.25), warn_wrong=(warned and risk <= 0.25),
            d_commit=dc, d_reveal=dr, p_self=p_self, risk=risk, lure=tempt,
            agent_choice=ac, oracle_safe=sc.oracle_safe_branch_id,
            self_discovery=self_disc, false_suppression=fs, beneficial_novelty=bn,
            probe_VA=probes.get("VA"), probe_IA=probes.get("IA"), probe_EP=probes.get("EP"),
            m_true={"tau": m.tau, "nu": m.nu, "gamma_gen": m.gamma_gen},
        )
        snap = observer.update(ev)
        m_hat = m.copy(); m_hat.tau = observer.tau_hat
        m_hat.nu = observer.nu_hat; m_hat.gamma_gen = observer.gamma_gen_hat
        a_infer, _, info_infer = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat)
        disagree = (a_oracle != a_infer)
        if hasattr(observer, 'record_action_agreement'):
            observer.record_action_agreement(not disagree)
        rec = observer.to_log_record(step, ev, snap, a_oracle, a_infer,
            info_oracle.get("Q", 0), info_infer.get("Q", 0))
        rec["conf"] = observer.get_confidence()
        rec["warned"] = warned; rec["follow_warn"] = follow_warn
        rec["dose"] = dose_oracle; rec["p_self"] = p_self
        records.append(rec)
    return records


def analyze(records):
    if not records:
        return {"tau": {"mae": 0, "corr": 0}, "nu": {"mae": 0, "corr": 0},
                "gamma_gen": {"mae": 0, "corr": 0}, "ADR": 0}
    def _m(tv, hv):
        if len(tv) < 2: return {"mae": 0, "corr": 0}
        mae = float(np.mean(np.abs(np.array(tv) - np.array(hv))))
        corr = float(np.corrcoef(tv, hv)[0, 1]) if np.std(tv) > 1e-6 else 0.0
        return {"mae": round(mae, 4), "corr": round(corr, 4)}
    tt=[r["m_true"]["tau"] for r in records if r.get("m_true")]
    th=[r["m_hat"]["tau"] for r in records if r.get("m_true")]
    nt=[r["m_true"]["nu"] for r in records if r.get("m_true")]
    nh=[r["m_hat"]["nu"] for r in records if r.get("m_true")]
    gt=[r["m_true"]["gamma_gen"] for r in records if r.get("m_true")]
    gh=[r["m_hat"]["gamma_gen"] for r in records if r.get("m_true")]
    adr = sum(1 for r in records if r["disagree"]) / max(len(records), 1)
    return {"tau": _m(tt, th), "nu": _m(nt, nh), "gamma_gen": _m(gt, gh),
            "ADR": round(adr, 3)}


def main():
    print("═══ A1 Stabilization ═══\n", file=sys.stderr)
    L = ["# A1 Stabilization & Blind Channel Audit\n\n"]

    # ─── Exp A: p_self activation audit ──────────────────
    L.append("## Exp A: p_self Activation Audit (A1)\n\n")
    L.append("| Metric | Value |\n|--------|------:|\n")
    print("Exp A: p_self audit...", file=sys.stderr)
    recs_all = []
    for th in ["safe", "shiny"]:
        for sid in range(NS):
            recs_all.extend(run_session(ALL_LESSONS, th, sid, observer=A1MtObserver()))
    warn_rate = sum(1 for r in recs_all if r.get("warned")) / max(len(recs_all), 1)
    follow_rate = sum(1 for r in recs_all if r.get("follow_warn")) / max(len(recs_all), 1)
    dose_pos = sum(1 for r in recs_all if r.get("dose", 0) > 0) / max(len(recs_all), 1)
    blind_vals = [r["events"]["blind"] for r in recs_all if r.get("events")]
    sd_vals = [r["events"]["selfdisc"] for r in recs_all if r.get("events")]
    pself_vals = [r.get("p_self", 0.5) for r in recs_all]
    L.append(f"| warn_rate | {warn_rate:.4f} |\n")
    L.append(f"| follow_warn_rate | {follow_rate:.4f} |\n")
    L.append(f"| dose>0_rate | {dose_pos:.4f} |\n")
    L.append(f"| mean_blind (A1) | {np.mean(blind_vals):.6f} |\n")
    L.append(f"| mean_selfdisc | {np.mean(sd_vals):.4f} |\n")
    L.append(f"| mean_p_self | {np.mean(pself_vals):.4f} |\n")
    L.append(f"| n_blind>0 | {sum(1 for v in blind_vals if v>0)} / {len(blind_vals)} |\n")
    print(f"  warn={warn_rate:.3f} follow={follow_rate:.3f} dose>0={dose_pos:.3f} "
          f"blind={np.mean(blind_vals):.6f}", file=sys.stderr)

    # ─── Exp B: Blind definition comparison ──────────────
    L.append("\n## Exp B: Blind Definition — A1 vs A2\n\n")
    L.append("| Observer | MAE_ν | Corr_ν | Blind>0 | Mean blind | ADR |\n")
    L.append("|----------|:-----:|:------:|:-------:|:----------:|:---:|\n")
    print("\nExp B: Blind comparison...", file=sys.stderr)
    for obs_name, obs_cls in [("A1", A1MtObserver), ("A2", A2MtObserver)]:
        recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, observer=obs_cls()))
        m = analyze(recs)
        bv = [r["events"]["blind"] for r in recs if r.get("events")]
        n_pos = sum(1 for v in bv if v > 0)
        L.append("| {} | {} | {} | {} / {} | {:.6f} | {} |\n".format(
            obs_name, m["nu"]["mae"], m["nu"]["corr"], n_pos, len(bv),
            np.mean(bv), m["ADR"]))
        print(f"  {obs_name}: Corr_nu={m['nu']['corr']} blind>0={n_pos}/{len(bv)}", file=sys.stderr)

    # ─── Exp C: Confidence calibration ───────────────────
    L.append("\n## Exp C: Confidence Calibration (A0 vs A1 vs A2)\n\n")
    L.append("| Observer | Dim | Corr(conf, −|err|) |\n")
    L.append("|----------|-----|:---:|\n")
    print("\nExp C: Confidence...", file=sys.stderr)
    for obs_name, obs_cls in [("A0", RuleBasedMtObserver), ("A1", A1MtObserver), ("A2", A2MtObserver)]:
        recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, n_teach=16, observer=obs_cls()))
        for dim in ["tau", "nu", "gamma_gen"]:
            confs = [r["conf"].get(dim, 0.5) for r in recs if r.get("conf")]
            errs = [abs(r["m_true"][dim]-r["m_hat"][dim]) for r in recs if r.get("m_true")]
            if len(confs)==len(errs) and len(confs)>2:
                corr = float(np.corrcoef(confs, [-e for e in errs])[0,1]) if np.std(confs)>1e-6 else 0
                L.append(f"| {obs_name} | {dim} | {corr:.4f} |\n")

    # ─── Exp D: Hybrid dry-run ───────────────────────────
    L.append("\n## Exp D: Hybrid Dry-Run (A1)\n\n")
    L.append("| α | MAE_τ | MAE_ν | MAE_γ | micro_ADR | Δε_stop |\n")
    L.append("|:-:|:-----:|:-----:|:-----:|:---------:|:-------:|\n")
    print("\nExp D: Hybrid dry-run...", file=sys.stderr)
    a_s = 0.15; b_s = 0.10; eps_0 = 0.30
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs.extend(run_session(ALL_LESSONS, th, sid, observer=A1MtObserver()))
        # Compute hybrid m
        mae_t = []; mae_n = []; mae_g = []; eps_o=[]; eps_i=[]
        for r in recs:
            if not r.get("m_true"): continue
            mt = r["m_true"]; mh = r["m_hat"]
            hyb_tau = (1-alpha)*mt["tau"] + alpha*mh["tau"]
            hyb_nu = (1-alpha)*mt["nu"] + alpha*mh["nu"]
            hyb_gg = (1-alpha)*mt["gamma_gen"] + alpha*mh["gamma_gen"]
            mae_t.append(abs(mt["tau"] - hyb_tau))
            mae_n.append(abs(mt["nu"] - hyb_nu))
            mae_g.append(abs(mt["gamma_gen"] - hyb_gg))
            eps_o.append(eps_0 + a_s*mt["nu"] + b_s*mt["gamma_gen"])
            eps_i.append(eps_0 + a_s*hyb_nu + b_s*hyb_gg)
        delta_eps = np.mean(np.abs(np.array(eps_i) - np.array(eps_o)))
        adr = analyze(recs)["ADR"]
        L.append("| {} | {:.4f} | {:.4f} | {:.4f} | {} | {:.4f} |\n".format(
            alpha, np.mean(mae_t), np.mean(mae_n), np.mean(mae_g), adr, delta_eps))
        print(f"  α={alpha}: MAE_tau={np.mean(mae_t):.4f} Δε={delta_eps:.4f}", file=sys.stderr)

    rpt = out / "a1_stabilization_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
