"""Shadow Observer Experiment: targeted + mixed + robustness.

Exp A: Targeted identification (per-family, per-dimension)
Exp B: Mixed-family generalization
Exp C: Robustness sweeps (noise, boundary, θ)
"""
import sys, json
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
from src.teachers.internalization_observer import RuleBasedMtObserver, ObsEvent
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import generate_world_weights_orthogonal, neutralize_identity_features
from src.metrics.self_discovery import estimate_self_discovery_prob

out = Path("results"); out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)
NS = 15

# Families grouped by target dimension
DIM_FAMILIES = {
    "tau": ["tic_rescue_heavy", "sparse_valid_advice", "verified_warn"],
    "nu": ["sparse_invalid_advice", "self_discovery_needed", "ppmrb_self_discovery"],
    "gamma_gen": ["beneficial_novelty", "false_suppression_cost", "tic_self_discovery"],
}
LESSON_MAP = {l.name: l for l in LESSON_CATALOG_V2}


def apply_fix(meta, sc):
    rng_w = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng_w, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def run_session(lessons, theta, seed, n_teach=6, noise_scale=0.0):
    """Run a multi-step teaching session, tracking observer vs true m."""
    rng = np.random.default_rng(seed * 1000 + abs(hash(theta)) % 1000)
    m = FactoredInternalizationState(); m.snapshot()
    tutor = BCICTv4(agent_params=AP)
    observer = RuleBasedMtObserver(); observer.reset()

    records = []
    for step in range(n_teach):
        les = lessons[step % len(lessons)]
        ub = {p: 0.4 + 0.1 * step / n_teach for p in PROBE_NAMES}
        et = generate_episode_from_lesson_v2(les, step + seed * 100, theta, ub, rng)
        ep, spec, gm, cfg_e, meta, sc = et
        fb, ww = apply_fix(meta, sc)
        if noise_scale > 0:
            fb = fb + rng.normal(0, noise_scale, fb.shape)
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
        lib.update("safe", ss); lib.update("risky", sr)
        scr.update(build_scorer_input(ss, lib), 1.0)
        scr.update(build_scorer_input(sr, lib), 0.0)

        # Tutor decides with true m
        a_oracle, dose_oracle, info_oracle = tutor.decide(sc, fb, lp, lib, scr, 2, m)
        dc = getattr(sc, 'commit_depth', 3)
        dr = getattr(sc, 'reveal_depth', 2)
        p_self = estimate_self_discovery_prob(dc, dr)
        risk = getattr(sc, 'risk_level', 0.3)
        tempt = getattr(sc, 'temptation_strength', 0.0)

        # Simulate agent choice
        bas = BranchAttributes(safety_score=float(ss[0]), risk_penalty=0.1,
            temptation_score=sc.tempt_score_a if sc.oracle_safe_branch_id == 0 else sc.tempt_score_b)
        bar = BranchAttributes(safety_score=float(sr[0]),
            risk_penalty=risk, temptation_score=sc.tempt_score_b if sc.oracle_safe_branch_id == 0 else sc.tempt_score_a)
        ac = sample_factored_choice([bas, bar], theta, m, AP, rng, [0.0, 0.0], [False, False])
        correct = (ac == sc.oracle_safe_branch_id)
        warned = dose_oracle > 0
        follow_warn = warned and correct
        has_self_ev = p_self > 0.5
        self_disc = correct and not warned and has_self_ev
        bn = ep.subtype == "beneficial_novelty" and correct
        fs = ep.subtype == "false_suppression_cost" and not correct

        # Update true m
        if warned:
            m.update_trust(warn_helpful=(risk > 0.25 and correct))
            if not has_self_ev:
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

        # Probes (every other step)
        probes = all_probes(m, AP, theta) if step % 2 == 0 else {}

        # Observer event
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

        # Shadow decision with m̂
        m_hat = m.copy()
        m_hat.tau = observer.tau_hat
        m_hat.nu = observer.nu_hat
        m_hat.gamma_gen = observer.gamma_gen_hat
        a_infer, dose_infer, info_infer = tutor.decide(sc, fb, lp, lib, scr, 2, m_hat)

        rec = observer.to_log_record(
            step, ev, snap, a_oracle, a_infer,
            info_oracle.get("Q", 0), info_infer.get("Q", 0))
        records.append(rec)

    return records


def analyze_records(records, label=""):
    """Compute MAE, Corr, ADR from records."""
    if not records:
        return {}
    tau_true = [r["m_true"]["tau"] for r in records if r.get("m_true")]
    tau_hat = [r["m_hat"]["tau"] for r in records if r.get("m_true")]
    nu_true = [r["m_true"]["nu"] for r in records if r.get("m_true")]
    nu_hat = [r["m_hat"]["nu"] for r in records if r.get("m_true")]
    gg_true = [r["m_true"]["gamma_gen"] for r in records if r.get("m_true")]
    gg_hat = [r["m_hat"]["gamma_gen"] for r in records if r.get("m_true")]

    def _metrics(true_vals, hat_vals):
        if len(true_vals) < 2:
            return {"mae": 0, "corr": 0}
        mae = np.mean(np.abs(np.array(true_vals) - np.array(hat_vals)))
        corr = np.corrcoef(true_vals, hat_vals)[0, 1] if np.std(true_vals) > 1e-6 else 0.0
        return {"mae": round(float(mae), 4), "corr": round(float(corr), 4)}

    adr = sum(1 for r in records if r["disagree"]) / max(len(records), 1)

    return {
        "tau": _metrics(tau_true, tau_hat),
        "nu": _metrics(nu_true, nu_hat),
        "gamma_gen": _metrics(gg_true, gg_hat),
        "ADR": round(adr, 3),
        "n": len(records),
    }


def main():
    print("═══ Shadow Observer Experiment ═══\n", file=sys.stderr)
    L = ["# Shadow Observer Results\n\n"]
    L.append(f"> Seeds: {NS}\n\n")

    # ─── Exp A: Targeted ─────────────────────────────────
    L.append("## Exp A: Targeted Identification\n\n")
    L.append("| Target | Family | MAE_τ | Corr_τ | MAE_ν | Corr_ν | MAE_γ | Corr_γ | ADR |\n")
    L.append("|--------|--------|:-----:|:------:|:-----:|:------:|:-----:|:------:|:---:|\n")
    print("Exp A: Targeted...", file=sys.stderr)

    for dim, families in DIM_FAMILIES.items():
        lessons = [LESSON_MAP[n] for n in families if n in LESSON_MAP]
        if not lessons:
            continue
        all_recs = []
        for th in ["safe", "shiny"]:
            for sid in range(NS):
                recs = run_session(lessons, th, sid)
                all_recs.extend(recs)
        m = analyze_records(all_recs)
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            dim, "+".join(families[:2]),
            m["tau"]["mae"], m["tau"]["corr"],
            m["nu"]["mae"], m["nu"]["corr"],
            m["gamma_gen"]["mae"], m["gamma_gen"]["corr"],
            m["ADR"]))
        print(f"  {dim}: MAE_tau={m['tau']['mae']} Corr_tau={m['tau']['corr']} "
              f"MAE_nu={m['nu']['mae']} ADR={m['ADR']}", file=sys.stderr)

    # ─── Exp B: Mixed ────────────────────────────────────
    L.append("\n## Exp B: Mixed-Family Generalization\n\n")
    L.append("| θ | MAE_τ | Corr_τ | MAE_ν | Corr_ν | MAE_γ | Corr_γ | ADR |\n")
    L.append("|---|:-----:|:------:|:-----:|:------:|:-----:|:------:|:---:|\n")
    print("\nExp B: Mixed...", file=sys.stderr)

    all_lessons = list(LESSON_MAP.values())
    for th in ["safe", "shiny"]:
        all_recs = []
        for sid in range(NS):
            recs = run_session(all_lessons, th, sid)
            all_recs.extend(recs)
        m = analyze_records(all_recs)
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            th, m["tau"]["mae"], m["tau"]["corr"],
            m["nu"]["mae"], m["nu"]["corr"],
            m["gamma_gen"]["mae"], m["gamma_gen"]["corr"],
            m["ADR"]))
        print(f"  {th}: MAE_tau={m['tau']['mae']} ADR={m['ADR']}", file=sys.stderr)

    # ─── Exp C: Robustness ───────────────────────────────
    L.append("\n## Exp C: Robustness Sweeps\n\n")

    # C1: Noise
    L.append("### C1: Noise Sweep\n\n")
    L.append("| Noise | MAE_τ | MAE_ν | MAE_γ | ADR |\n")
    L.append("|:-----:|:-----:|:-----:|:-----:|:---:|\n")
    print("\nExp C1: Noise...", file=sys.stderr)
    for noise in [0.0, 0.3, 0.5]:
        all_recs = []
        for sid in range(min(NS, 8)):
            recs = run_session(all_lessons, "safe", sid, noise_scale=noise)
            all_recs.extend(recs)
        m = analyze_records(all_recs)
        L.append("| {} | {} | {} | {} | {} |\n".format(
            noise, m["tau"]["mae"], m["nu"]["mae"], m["gamma_gen"]["mae"], m["ADR"]))

    # C2: θ sweep
    L.append("\n### C2: θ Sweep\n\n")
    L.append("| θ | MAE_τ | MAE_ν | MAE_γ | ADR |\n")
    L.append("|---|:-----:|:-----:|:-----:|:---:|\n")
    print("Exp C2: θ sweep...", file=sys.stderr)
    for th in ["safe", "shiny"]:
        all_recs = []
        for sid in range(NS):
            recs = run_session(all_lessons, th, sid)
            all_recs.extend(recs)
        m = analyze_records(all_recs)
        L.append("| {} | {} | {} | {} | {} |\n".format(
            th, m["tau"]["mae"], m["nu"]["mae"], m["gamma_gen"]["mae"], m["ADR"]))

    rpt = out / "shadow_observer_results.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.writelines(L)
    # Also write JSONL log
    logf = out / "shadow_observer_log.jsonl"
    with open(logf, "w") as f:
        pass  # truncate
    print(f"\nReport → {rpt}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
