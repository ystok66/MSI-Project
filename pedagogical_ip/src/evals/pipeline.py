"""K5 — Unified Experiment Pipeline.

Config-driven experiment runner that unifies all experiment scripts.
Supports: family × planner × tutor × agent_policy × latent_type × sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, ".")

from src.envs.scenario_families import generate_scenario
from src.envs.map_generator import CellType
from src.envs.semantic_subspace import (
    generate_world_weights_orthogonal,
    neutralize_identity_features,
)
from src.envs.observation_mask import make_observation_mask
from src.agents.cost_risk_model import LatentCostRiskHead
from src.agents.branch_summary import summarize_branch
from src.agents.branch_concepts import BranchConceptLibrary
from src.agents.branch_scorer_probe import BranchScorerProbe, build_scorer_input
from src.agents.stochastic_agent_policy import (
    BranchAttributes, AgentPolicyParams, sample_branch_choice,
)
from src.planner.branch_candidates import BranchCandidate
from src.planner.branch_reranker import choose_branch


@dataclass
class ExperimentConfig:
    """Unified config for any experiment."""
    name: str = "default"
    family: str = "elcb_po"
    family_kwargs: dict = field(default_factory=dict)
    tutor_type: str = "v4"           # always_wait|always_warn|v4|pref_v2|goal_v1|oracle
    agent_type: str = "deterministic"  # deterministic|stochastic
    agent_beta: float = 4.0
    agent_epsilon: float = 0.1
    obs_radius: int = 2
    train_seeds: list = field(default_factory=lambda: list(range(50)))
    probe_seeds: list = field(default_factory=lambda: list(range(100, 160)))
    n_boot: int = 200
    difficulty: str = "medium"


def _apply_fix(gm, meta, sc):
    rng = np.random.default_rng(42)
    ww = generate_world_weights_orthogonal(rng, d=4)
    allb = list(sc.branch_a_cells) + list(sc.branch_b_cells)
    fb = neutralize_identity_features(meta.cell_features, allb, 0.5)
    return fb, ww


def _vis_candidates(sc, obs_r):
    fk, mg = sc.fork_cell, sc.merge_cell
    ma = make_observation_mask(sc.branch_a_cells, fk, obs_r)
    mb = make_observation_mask(sc.branch_b_cells, fk, obs_r)
    va = [c for c, m in zip(sc.branch_a_cells, ma) if m > 0.5]
    vb = [c for c, m in zip(sc.branch_b_cells, mb) if m > 0.5]
    return [
        BranchCandidate(0, va, len(va), fk, mg, (1, fk[1]), (1, mg[1])),
        BranchCandidate(1, vb, len(vb), fk, mg, (3, fk[1]), (3, mg[1])),
    ]


def _make_tutor(cfg: ExperimentConfig):
    if cfg.tutor_type == "v4":
        from src.teachers.learning_aware_policy_v4 import LearningAwarePolicyV4
        return LearningAwarePolicyV4()
    elif cfg.tutor_type == "pref_v2":
        from src.teachers.preference_aware_policy_v2 import PreferenceAwarePolicyV2
        return PreferenceAwarePolicyV2(
            agent_params=AgentPolicyParams(beta=cfg.agent_beta, epsilon=cfg.agent_epsilon))
    elif cfg.tutor_type == "goal_v1":
        from src.teachers.goal_preference_aware_policy_v1 import GoalPreferenceAwarePolicyV1
        return GoalPreferenceAwarePolicyV1(
            agent_params=AgentPolicyParams(beta=cfg.agent_beta, epsilon=cfg.agent_epsilon))
    return None


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run a single experiment condition and return metrics."""
    lp = LatentCostRiskHead(d=4, risk_supervision="oracle_visited")
    lib = BranchConceptLibrary()
    scorer = BranchScorerProbe(lr=0.05, l2=0.01)
    tutor = _make_tutor(cfg)
    warns, waits = 0, 0
    pref_correct, pref_total = 0, 0
    goal_correct, goal_total = 0, 0
    agent_safe_count, agent_total = 0, 0
    agent_params = AgentPolicyParams(beta=cfg.agent_beta, epsilon=cfg.agent_epsilon)

    for seed in cfg.train_seeds:
        gm, _, meta, sc = generate_scenario(
            cfg.family, seed, cfg.difficulty,
            latent_mode=True, **cfg.family_kwargs)
        fb, ww = _apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)
        rng = np.random.default_rng(seed + 5000)

        for _ in range(5):
            for r in range(gm.height):
                for c in range(gm.width):
                    if gm.cell_types[r, c] == CellType.WALL:
                        continue
                    z = fb[r, c]
                    lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z))

        ss = summarize_branch(sc.safe_cells, fb, fv, lp)
        sr = summarize_branch(sc.risky_cells, fb, fv, lp)
        lib.update("safe_branch", ss)
        lib.update("risky_branch", sr)
        scorer.update(build_scorer_input(ss, lib), 1.0)
        scorer.update(build_scorer_input(sr, lib), 0.0)

        # Agent choice (stochastic or deterministic)
        tempt_str = getattr(sc, 'temptation_strength', 0.0)
        ba_safe = BranchAttributes(safety_score=float(ss[0]),
            temptation_score=getattr(sc, 'tempt_score_a', 0.1)
            if getattr(sc, 'oracle_safe_branch_id', 0) == 0
            else getattr(sc, 'tempt_score_b', tempt_str * 0.8),
            risk_penalty=0.1)
        ba_risky = BranchAttributes(safety_score=float(sr[0]),
            temptation_score=getattr(sc, 'tempt_score_b', tempt_str * 0.8)
            if getattr(sc, 'oracle_safe_branch_id', 0) == 0
            else getattr(sc, 'tempt_score_a', 0.1),
            risk_penalty=0.4)
        branches = [ba_safe, ba_risky]

        if cfg.agent_type == "stochastic":
            true_theta = getattr(sc, 'latent_preference', 'neutral')
            agent_choice = sample_branch_choice(branches, true_theta, agent_params, rng)
            agent_total += 1
            if agent_choice == 0:
                agent_safe_count += 1
        else:
            agent_choice = 0

        # Tutor decision
        do_warn = False
        if cfg.tutor_type in ("always_warn", "oracle"):
            do_warn = True
        elif cfg.tutor_type == "always_wait":
            do_warn = False
        elif tutor is not None:
            if hasattr(tutor, 'decide'):
                action, _ = tutor.decide(sc, fb, lp, lib, scorer, cfg.obs_radius)
                do_warn = (action == "WARN")
            if cfg.tutor_type in ("pref_v2", "goal_v1") and hasattr(tutor, 'observe_agent_choice'):
                tutor.observe_agent_choice(agent_choice, branches)

        if do_warn:
            warns += 1
            for r, c in sc.risky_cells:
                z = fb[r, c]
                lp.update_from_outcome(z, ww.true_cost(z), ww.true_risk(z), weight=1.0)
            ss2 = summarize_branch(sc.safe_cells, fb, fv, lp)
            sr2 = summarize_branch(sc.risky_cells, fb, fv, lp)
            lib.update("safe_branch", ss2)
            lib.update("risky_branch", sr2)
            scorer.update(build_scorer_input(ss2, lib), 1.0)
            scorer.update(build_scorer_input(sr2, lib), 0.0)
        else:
            waits += 1

        # Track latent accuracy
        if cfg.tutor_type == "pref_v2" and hasattr(tutor, 'pref_posterior'):
            pref_total += 1
            true_theta = getattr(sc, 'latent_preference', 'neutral')
            if tutor.pref_posterior.predicted_type == true_theta:
                pref_correct += 1
        if cfg.tutor_type == "goal_v1" and hasattr(tutor, 'goal_posterior'):
            goal_total += 1
            true_g = getattr(sc, 'latent_goal', 'default')
            if tutor.goal_posterior.predicted_type == true_g:
                goal_correct += 1

    # Probe
    per_seed = []
    for ps in cfg.probe_seeds:
        gm, _, meta, sc = generate_scenario(
            cfg.family, ps, cfg.difficulty,
            latent_mode=True, **cfg.family_kwargs)
        fb, _ = _apply_fix(gm, meta, sc)
        fv = np.full_like(fb, 0.3)
        passable = np.ones((fb.shape[0], fb.shape[1]), dtype=bool)
        trng = np.random.default_rng(ps + 777)
        cands = _vis_candidates(sc, cfg.obs_radius)
        best, _ = choose_branch(
            cands, fb, fv, lp, passable, lib, scorer,
            lambda_b=1.0, score_mode="hybrid", tie_rng=trng)
        per_seed.append(int(best.branch_id == sc.oracle_safe_branch_id))

    per_seed = np.array(per_seed)
    sbcr = float(np.mean(per_seed))
    brng = np.random.default_rng(42)
    bm = [float(np.mean(per_seed[brng.integers(0, len(per_seed), len(per_seed))]))
           for _ in range(cfg.n_boot)]
    total = warns + waits

    return {
        "name": cfg.name, "family": cfg.family, "tutor": cfg.tutor_type,
        "agent": cfg.agent_type,
        "SBCR": round(sbcr, 3),
        "CI_lo": round(float(np.percentile(bm, 2.5)), 3),
        "CI_hi": round(float(np.percentile(bm, 97.5)), 3),
        "warn_rate": round(warns / max(total, 1), 3),
        "pref_acc": round(pref_correct / max(pref_total, 1), 3) if pref_total > 0 else None,
        "goal_acc": round(goal_correct / max(goal_total, 1), 3) if goal_total > 0 else None,
        "agent_safe_rate": round(agent_safe_count / max(agent_total, 1), 3) if agent_total > 0 else None,
    }


def write_report(results: list[dict], path: Path, title: str = "Experiment"):
    """Write unified markdown report."""
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        has_pref = any(r.get("pref_acc") is not None for r in results)
        has_goal = any(r.get("goal_acc") is not None for r in results)
        has_agent = any(r.get("agent_safe_rate") is not None for r in results)

        cols = ["Name", "Family", "Tutor", "SBCR", "CI", "WarnRate"]
        if has_pref:
            cols.append("PrefAcc")
        if has_goal:
            cols.append("GoalAcc")
        if has_agent:
            cols.append("Agent%Safe")

        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")

        for r in results:
            row = [
                r["name"], r["family"], r["tutor"],
                "{:.0%}".format(r["SBCR"]),
                "[{:.0%},{:.0%}]".format(r["CI_lo"], r["CI_hi"]),
                "{:.0%}".format(r["warn_rate"]),
            ]
            if has_pref:
                row.append("{:.0%}".format(r["pref_acc"]) if r.get("pref_acc") is not None else "—")
            if has_goal:
                row.append("{:.0%}".format(r["goal_acc"]) if r.get("goal_acc") is not None else "—")
            if has_agent:
                row.append("{:.0%}".format(r["agent_safe_rate"]) if r.get("agent_safe_rate") is not None else "—")
            f.write("| " + " | ".join(row) + " |\n")
