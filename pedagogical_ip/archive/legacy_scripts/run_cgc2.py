"""CGC-v2 Experiment: Factor-vector goal + exact posterior.

6 conditions × 2θ on train pool + held-out generalization.

Conditions:
  1. v1_1_persistent (preference-only, baseline)
  2. joint_v2_coupled (discrete goals)
  3. cajt_v3_full (calibrated discrete)
  4. factor_exact (GoalFactorPosterior, no calibration)
  5. factor_cajt (GoalFactorPosterior + calibrated confidence)
  6. oracle

Metrics: ExactGoalAcc, FactorAcc, SelGap, subtype WR, HeldOutCompAcc
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np

from src.envs.compositional_goal_corridor_v2 import (
    generate_cgc2_session, generate_cgc2_scenario,
    CGC2_SUBTYPES, TRAIN_POOL, HELDOUT_POOL,
)
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal, neutralize_identity_features,
)
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
)
from src.agents.goal_factor_posterior import GoalFactorPosterior, compute_factor_likelihood
from src.teachers.persistent_tutor_v1_1 import PersistentTutorV1_1
from src.teachers.joint_tutor_v2 import JointTutorV2
from src.teachers.calibrated_adaptive_joint_tutor_v3 import CAJTv3
from src.metrics.calibrated_confidence import calibrate_posterior, calibrated_confidence
from src.metrics.change_detection import (
    compute_surprisal, compute_drift_score, compute_adaptive_rho, apply_adaptive_diffusion,
)
from src.metrics.self_discovery import estimate_self_discovery_prob, estimate_failure_if_wait
from src.envs.observation_mask import make_observation_mask

out = Path("results")
out.mkdir(exist_ok=True)
AP = AgentPolicyParams(beta=4.0, epsilon=0.1, lambda_theta=1.0)


def sf(v, fmt="{:.0%}"):
    return "—" if v is None else fmt.format(v)


def apply_fix(meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))))


class FactorTutor:
    """Tutor using GoalFactorPosterior for decisions."""

    def __init__(self, ap, calibrate=False, T=0.6):
        self.ap = ap
        self.factor_post = GoalFactorPosterior(K=4, forgetting_rate=0.0)
        self.calibrate = calibrate
        self.T = T
        self.warn_count = 0
        self.wait_count = 0
        self.last_drift = 0.0
        self.last_rho = 0.005

    def _cal_table(self):
        if self.calibrate:
            return calibrate_posterior(self.factor_post.log_table, self.T)
        return self.factor_post.table

    def _confidence(self):
        ct = self._cal_table()
        return calibrated_confidence(ct)

    def decide(self, sc, fb, lp, lib, scorer, obs=2):
        fv = np.full_like(fb, 0.3)
        dc = getattr(sc, 'commit_depth', obs + 1)
        dr = getattr(sc, 'reveal_depth', 3)
        delta = dc - dr
        p_self = estimate_self_discovery_prob(dc, dr)
        p_fail = estimate_failure_if_wait(dc, dr)

        fork = sc.fork_cell
        mask_a = make_observation_mask(sc.branch_a_cells, fork, obs)
        mask_b = make_observation_mask(sc.branch_b_cells, fork, obs)
        vis_a = [c for c, m in zip(sc.branch_a_cells, mask_a) if m > 0.5]
        vis_b = [c for c, m in zip(sc.branch_b_cells, mask_b) if m > 0.5]

        sa = summarize_branch(vis_a, fb, fv, lp)
        sb = summarize_branch(vis_b, fb, fv, lp)
        sa2 = summarize_branch(sc.branch_a_cells, fb, fv, lp)
        sb2 = summarize_branch(sc.branch_b_cells, fb, fv, lp)
        delta_s = max(abs(sa2[0] - sb2[0]) - abs(sa[0] - sb[0]), 0)
        dvoi = max(_sigmoid(abs(sa2[0] - sb2[0])) - _sigmoid(abs(sa[0] - sb[0])), 0)

        C_t = self._confidence()
        O_wait = p_self * _sigmoid((delta - 1.0) / 1.5)
        S_obs = _sigmoid((dc - 2.0) / 1.5)

        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        joint_ent = self.factor_post.entropy
        max_ent = self.factor_post.max_entropy
        unc = joint_ent / max(max_ent, 1e-6)

        # Experience warmup: suppress unc-driven warn until we have enough obs
        n_obs = self.factor_post.observation_count
        warmup = _sigmoid((n_obs - 3) / 1.5)  # ramps 0→1 over ~4 observations
        # Autonomy bonus: high when few observations (prefer WAIT to gather info)
        autonomy_bonus = max(1.0 - warmup, 0.0) * 2.0

        # Q equations — unc penalty only kicks in after warmup
        Q_warn = (1.0 * delta_s + 2.0 * dvoi + 1.5 * (1 - p_self)
                  + 1.5 * warmup * (1 - C_t) * unc
                  + 1.5 * warmup * tempt_str * unc
                  + 1.0 * self.last_drift * max(delta_s, dvoi)
                  - 0.05)
        Q_wait = (2.0 * p_self * delta_s - 1.5 * p_fail
                  + 3.5 * C_t * O_wait
                  + 3.0 * S_obs * (1.0 - warmup)  # early: strong wait bias
                  + autonomy_bonus)

        action = "WARN" if Q_warn > Q_wait else "WAIT"
        if action == "WARN":
            self.warn_count += 1
        else:
            self.wait_count += 1
        diag = {"Q_warn": round(Q_warn, 4), "Q_wait": round(Q_wait, 4),
                "C_t": round(C_t, 4), "D_t": round(self.last_drift, 4),
                "warmup": round(warmup, 4), "n_obs": n_obs}
        return action, diag

    def observe_agent_choice(self, chosen_idx, branches):
        # Adaptive diffusion
        ct = self._cal_table()
        from src.agents.goal_factor_posterior import FACTOR_VALUES, PREFERENCE_TYPES as PT
        all_y = self.factor_post._all_y
        lik_arr = np.array([
            compute_factor_likelihood(chosen_idx, branches, y, p, self.ap)
            for y in all_y for p in PT
        ])
        surp = compute_surprisal(ct.ravel(), lik_arr)
        self.last_drift = compute_drift_score(surp)
        self.last_rho = compute_adaptive_rho(self.last_drift)
        if self.calibrate:
            self.factor_post.log_table = apply_adaptive_diffusion(
                self.factor_post.log_table, self.last_rho)
        self.factor_post.update_from_choice(chosen_idx, branches, self.ap)


def run_session(strategy, theta, session, use_heldout=False):
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)

    if strategy == "v1_1": tutor = PersistentTutorV1_1(agent_params=AP)
    elif strategy == "joint_v2": tutor = JointTutorV2(agent_params=AP)
    elif strategy == "cajt_v3": tutor = CAJTv3(agent_params=AP)
    elif strategy == "factor_exact": tutor = FactorTutor(AP, calibrate=False)
    elif strategy == "factor_cajt": tutor = FactorTutor(AP, calibrate=True)
    else: tutor = None

    traces = []
    for ep in session.episodes:
        gm, cfg, meta, sc = generate_cgc2_scenario(ep)
        fb, ww = apply_fix(meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(ep.seed + 9999)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

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
            risk_penalty=0.4)
        branches = [bas, bar]
        ac = sample_branch_choice(branches, theta, AP, rng)

        do_warn = False
        diag = {}
        if strategy == "oracle":
            do_warn = (ep.d_commit < ep.d_reveal)
        elif strategy == "v1_1":
            tutor.reset_stats()
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
        elif tutor is not None:
            action, diag = tutor.decide(sc, fb, lp, lib, scorer, 2)
            do_warn = (action == "WARN")
            if hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(ac, branches)

        # Factor accuracy
        fact_acc = None
        exact_acc = None
        if isinstance(tutor, FactorTutor):
            fact_acc = tutor.factor_post.factor_accuracy(ep.goal_vec)
            pred_y = tutor.factor_post.predicted_goal_vec
            exact_acc = 1.0 if pred_y == ep.goal_vec else 0.0

        traces.append({
            "subtype": ep.subtype, "warned": do_warn,
            "agent_safe": (ac == 0), "goal_vec": ep.goal_vec,
            "fact_acc": fact_acc, "exact_acc": exact_acc,
        })

    n = len(traces)
    if n == 0:
        return {}
    sbcr = sum(1 for t in traces if t["agent_safe"]) / n
    wr = sum(1 for t in traces if t["warned"]) / n
    sw = {}
    for st in CGC2_SUBTYPES:
        eps = [t for t in traces if t["subtype"] == st]
        sw[st] = sum(1 for t in eps if t["warned"]) / len(eps) if eps else None
    sg = None
    if sw.get("conflict") is not None and sw.get("aligned") is not None:
        sg = sw["conflict"] - sw["aligned"]

    fa_vals = [t["fact_acc"] for t in traces if t["fact_acc"] is not None]
    ea_vals = [t["exact_acc"] for t in traces if t["exact_acc"] is not None]

    return {
        "sbcr": round(sbcr, 3), "wr": round(wr, 3),
        "sg": round(sg, 3) if sg is not None else None,
        "wr_aln": round(sw.get("aligned", 0), 3) if sw.get("aligned") is not None else None,
        "wr_cnf": round(sw.get("conflict", 0), 3) if sw.get("conflict") is not None else None,
        "wr_bnd": round(sw.get("boundary_obs", 0), 3) if sw.get("boundary_obs") is not None else None,
        "wr_dcy": round(sw.get("decoy", 0), 3) if sw.get("decoy") is not None else None,
        "fact_acc": round(np.mean(fa_vals), 3) if fa_vals else None,
        "exact_acc": round(np.mean(ea_vals), 3) if ea_vals else None,
    }


def avg(rs, k):
    vs = [r[k] for r in rs if r.get(k) is not None]
    return round(np.mean(vs), 3) if vs else None


def main():
    print("═══ CGC-v2 Experiment ═══\n", file=sys.stderr)
    strategies = ["v1_1", "joint_v2", "cajt_v3", "factor_exact", "factor_cajt", "oracle"]
    thetas = ["safe", "shiny"]
    lines = ["# CGC-v2: Factor-Vector Goal Experiment\n\n"]

    # ── Train pool ──
    print("Train compositions...", file=sys.stderr)
    lines.append("## Train Compositions (12 vectors)\n\n")
    lines.append("| θ | Strategy | SBCR | WR | WR(aln) | WR(cnf) | **SelGap** | FactorAcc | ExactAcc |\n")
    lines.append("|---|----------|------|----|---------|---------|-----------|-----------|----------|\n")
    train_results = []
    for theta in thetas:
        for s in strategies:
            rs = []
            for sid in range(6):
                sess = generate_cgc2_session(sid * 1000 + abs(hash(theta)) % 1000,
                                              12, theta, False)
                r = run_session(s, theta, sess)
                rs.append(r)
            a = {k: avg(rs, k) for k in ["sbcr", "wr", "sg", "wr_aln", "wr_cnf",
                                           "wr_bnd", "wr_dcy", "fact_acc", "exact_acc"]}
            a["theta"] = theta; a["strategy"] = s
            train_results.append(a)
            lines.append("| {} | {} | {} | {} | {} | {} | **{}** | {} | {} |\n".format(
                theta, s, sf(a["sbcr"]), sf(a["wr"]),
                sf(a["wr_aln"]), sf(a["wr_cnf"]),
                sf(a["sg"], "{:.3f}"),
                sf(a["fact_acc"], "{:.3f}"), sf(a["exact_acc"], "{:.3f}")))
            print(f"  {theta} × {s}: SG={sf(a['sg'], '{:.3f}')} FA={sf(a['fact_acc'], '{:.3f}')}",
                  file=sys.stderr)

    # SelGap comparison
    lines.append("\n### SelGap Comparison\n\n")
    lines.append("| θ | v1.1 | joint_v2 | cajt_v3 | factor_exact | **factor_cajt** | oracle |\n")
    lines.append("|---|------|----------|---------|-------------|----------------|--------|\n")
    for theta in thetas:
        vals = {}
        for s in strategies:
            r = [x for x in train_results if x["theta"] == theta and x["strategy"] == s]
            vals[s] = r[0]["sg"] if r else None
        lines.append("| {} | {} | {} | {} | {} | **{}** | {} |\n".format(
            theta, sf(vals["v1_1"], "{:.3f}"), sf(vals["joint_v2"], "{:.3f}"),
            sf(vals["cajt_v3"], "{:.3f}"), sf(vals["factor_exact"], "{:.3f}"),
            sf(vals["factor_cajt"], "{:.3f}"), sf(vals["oracle"], "{:.3f}")))

    # ── Held-out compositions ──
    print("\nHeld-out compositions...", file=sys.stderr)
    lines.append("\n## Held-Out Compositions (6 novel vectors)\n\n")
    lines.append("| θ | Strategy | SBCR | SelGap | FactorAcc | ExactAcc |\n")
    lines.append("|---|----------|------|--------|-----------|----------|\n")
    for theta in thetas:
        for s in ["factor_cajt", "cajt_v3", "v1_1", "oracle"]:
            rs = []
            for sid in range(6):
                sess = generate_cgc2_session(sid * 1000 + abs(hash(theta)) % 1000 + 500,
                                              12, theta, True)
                r = run_session(s, theta, sess, True)
                rs.append(r)
            a = {k: avg(rs, k) for k in ["sbcr", "sg", "fact_acc", "exact_acc"]}
            lines.append("| {} | {} | {} | {} | {} | {} |\n".format(
                theta, s, sf(a["sbcr"]), sf(a["sg"], "{:.3f}"),
                sf(a["fact_acc"], "{:.3f}"), sf(a["exact_acc"], "{:.3f}")))
            print(f"  HELDOUT {theta} × {s}: SG={sf(a['sg'], '{:.3f}')} FA={sf(a['fact_acc'], '{:.3f}')}",
                  file=sys.stderr)

    with open(out / "cgc2_report.md", "w") as f:
        f.writelines(lines)
    print("\nReport -> results/cgc2_report.md", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
