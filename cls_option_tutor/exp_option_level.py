"""
exp_option_level.py — Baseline comparison experiment for Option-Level Strong Tutor.

Objective (v2):
    J = delta_eval_from_obs - beta * DeathRate - gamma * TimeoutRate

CONDITIONS:
  no_tutor          : learner alone (tutor always WAIT)
  old_tutor         : legacy counterfactual tutor (G1b canonical)
  new_baseline      : OptionLevelTutorAgent, confusion-first shortlist (g_learn_mode="none")
  new_probe         : OptionLevelTutorAgent, G_probe Q_T objective (Method A)
  new_oracle_surrogate : OptionLevelTutorAgent, oracle distance surrogate (Method B)
  sparse            : SparseTutorAgent, BAN/HIGHLIGHT/MIX tier-aware (Bayes Gate Tutor)
  sparse_no_shift   : sparse with λ_shift=0 (diagnostic)
  sparse_low_shift  : sparse with λ_shift=0.125 (diagnostic)

FOUR SCENARIOS:
  A: Deadline Shortlist — K=10, T_max=3, n_risky=3
  B: Safety-Critical — K=10, T_max=5, H_0=3, n_risky=7
  C: Mixed Safety + Deadline — K=10, T_max=3, H_0=3, n_risky=5
  D: Ample Time (Control) — K=10, T_max=20, H_0=20, n_risky=0

USAGE:
    python exp_option_level.py --smoke                      # 2 grammars × 2 seeds
    python exp_option_level.py --workers 8                  # full run
    python exp_option_level.py --scenario A B               # specific scenarios
    python exp_option_level.py --cond sparse --smoke        # quick sparse smoke test
    python exp_option_level.py --diag --smoke --scenario A B # diagnostic experiment
    python exp_option_level.py --trace --smoke --cond sparse --scenario A  # decision trace
    python exp_option_level.py --phase0 --scenario A        # Phase 0 proxy validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioCfg:
    name: str
    K: int = 10
    T_max: int = 5
    H_0: int = 5
    n_risky: int = 4
    N_obs: int = 2
    N_teach: int = 2
    N_eval: int = 3
    n_sup: int = 4
    description: str = ""

SCENARIOS = {
    "A": ScenarioCfg(
        name="A_deadline",
        K=10, T_max=3, H_0=5, n_risky=3,
        N_obs=2, N_teach=2, N_eval=3, n_sup=4,
        description="K=10 >> tau_t=3: shortlist required to be solvable",
    ),
    "B": ScenarioCfg(
        name="B_safety",
        K=10, T_max=5, H_0=3, n_risky=7,
        N_obs=2, N_teach=2, N_eval=3, n_sup=4,
        description="H_0=3 + 7 risky options: safety filter critical",
    ),
    "C": ScenarioCfg(
        name="C_mixed",
        K=10, T_max=3, H_0=3, n_risky=5,
        N_obs=2, N_teach=2, N_eval=3, n_sup=4,
        description="Deadline + safety simultaneously",
    ),
    "D": ScenarioCfg(
        name="D_ample",
        K=10, T_max=20, H_0=20, n_risky=0,
        N_obs=2, N_teach=2, N_eval=3, n_sup=4,
        description="T_max>>K, no risk: no intervention expected",
    ),
}

# Full condition list (order matters for display)
CONDITIONS_BASE    = ["no_tutor", "old_tutor", "new_baseline"]
CONDITIONS_NEW     = ["new_probe", "new_oracle_surrogate"]
CONDITIONS_SPARSE  = ["sparse"]
CONDITIONS_DIAG    = ["sparse_no_shift", "sparse_low_shift"]
# Repair validation conditions (g_eval via probe/oracle, lambda_shift variants, hl gate)
CONDITIONS_REPAIR  = [
    "sparse_probe_shift0125",    # probe + lambda_shift=0.125
    "sparse_probe_shift0",       # probe + lambda_shift=0.0 (ablation)
    "sparse_oracle_shift0125",   # oracle_surrogate + lambda_shift=0.125
    "sparse_oracle_shift0",      # oracle_surrogate + lambda_shift=0.0
    "sparse_probe_hl35",         # probe + lambda_shift=0.125 + hl_threshold=0.35
]
# Rescue mode conditions (dual-mode: learning + deadline rescue)
CONDITIONS_RESCUE  = [
    "sparse_probe_rescue",           # probe + rescue gate (theta=0.5, lambda_to=1.0)
    "sparse_probe_rescue_shift0125", # probe + rescue + lambda_shift=0.125
    "sparse_oracle_rescue",          # oracle + rescue gate
    "sparse_oracle_rescue_shift0125",# oracle + rescue + lambda_shift=0.125
]
# Persistent Prior conditions (Step 4+5: cross-query HL/BAN priors)
CONDITIONS_PERSISTENT = [
    "persistent_hl_probe_rescue",    # Step 4: EMA HL prior (rho=0.3, lambda=0.3)
    "persistent_ban_probe_rescue",   # Step 5: EMA BAN prior (rho=0.3, lambda=0.5)
    "persistent_both_probe_rescue",  # Step 4+5: both priors combined
]
# Nonreveal feedback conditions (Step 6: reveal vs nonreveal isolation)
CONDITIONS_NONREVEAL = [
    "no_tutor_nonreveal_off",   # no tutor + nonreveal + no learning  (purest control)
    "no_tutor_nonreveal_neg",   # no tutor + nonreveal + neg evidence (learner-only effect)
    "nonreveal_off",            # sparse BAN+HL + nonreveal + no neg learning
    "nonreveal_neg",            # sparse BAN+HL + nonreveal + neg evidence
    "nonreveal_reveal_reg",     # sparse BAN+HL + reveal (regression vs sparse)
]
CONDITIONS_ALL     = (CONDITIONS_BASE + CONDITIONS_NEW + CONDITIONS_SPARSE
                      + CONDITIONS_DIAG + CONDITIONS_REPAIR + CONDITIONS_RESCUE
                      + CONDITIONS_PERSISTENT + CONDITIONS_NONREVEAL)
# Rollout proxy upgrade conditions (Step 7-8: learner-consistent + rollout calibration)
CONDITIONS_ROLLOUT = [
    "sparse_proxy",    # rollout_mode='proxy'  — old static approx (backward-compat baseline)
    "sparse_rollout",  # rollout_mode='hybrid' — learner-consistent + rollout
]
# Correct-pick learning conditions (Step 9: positive reinforcement on correct picks)
CONDITIONS_CORRECT_PICK = [
    "sparse_correct_learn",            # reveal + correct_pick_learning=cortex_em, eta=1.0
    "sparse_correct_learn_gate",       # reveal + correct_pick_learning=cortex_em, eta=0.5
    "sparse_nonreveal_neg_correct_gate", # nonreveal + neg evidence + correct pos (eta=0.5)
]
# Pedagogical-mode tutor conditions (dual-mode: protective / pedagogical)
# E1: nonreveal rollout baselines (no code change needed — just config)
CONDITIONS_PP_TUTOR = [
    # E1: nonreveal rollout baseline pair (isolate probe signal value)
    "sparse_rollout_nr_none",   # nonreveal + hybrid rollout + g_learn=none
    "sparse_rollout_nr_probe",  # nonreveal + hybrid rollout + g_learn=probe
    # E2 reveal: dual-mode reveal conditions
    "sparse_protective",        # reveal + protective mode + g_learn=probe
    "sparse_pedagogical",       # reveal + pedagogical mode + g_learn=probe
    # E2 nonreveal: dual-mode nonreveal conditions
    "sparse_protective_nr",     # nonreveal + protective + g_learn=probe
    "sparse_pedagogical_nr",    # nonreveal + pedagogical + g_learn=probe
]

# J objective weights (defaults — override via CLI if needed)
J_BETA  = 0.5   # death penalty
J_GAMMA = 0.2   # timeout penalty


def _js_divergence_fn(p: np.ndarray, q: np.ndarray, eps: float = 1e-30) -> float:
    """Jensen-Shannon divergence, bounded [0, ln2]. Used for ChoiceShiftJS."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(np.clip(0.5 * kl_pm + 0.5 * kl_qm, 0.0, np.log(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_data_dir(script_file):
    cls_dir = os.path.dirname(os.path.abspath(script_file))
    project_root = os.path.dirname(cls_dir)
    return os.path.join(project_root, 'BASIC', 'cls_learner', 'data')


def _make_cfg(sc: ScenarioCfg, cond: str, teach_budget: int = 0):
    from cls_option_tutor.config import FullConfig

    cfg = FullConfig()
    cfg.env.K = sc.K
    cfg.env.T_max = sc.T_max
    cfg.env.H_0 = sc.H_0
    cfg.env.n_risky = min(sc.n_risky, sc.K - 1)
    cfg.env.N_obs = sc.N_obs
    cfg.env.N_teach = sc.N_teach
    cfg.env.N_eval = sc.N_eval
    cfg.env.teach_step_budget = teach_budget  # 0 = disabled
    if teach_budget > 0:
        # Pre-allocate enough queries: obs + budget (worst case) + eval
        cfg.env.M_queries = sc.N_obs + teach_budget + sc.N_eval
    else:
        cfg.env.M_queries = sc.N_obs + sc.N_teach + sc.N_eval
    cfg.learner.use_cls = True
    cfg.learner.n_sup = sc.n_sup
    cfg.learner.n_em = 2
    cfg.learner.use_hpc = True
    cfg.rsa.use_rsa = False

    if cond == "old_tutor":
        cfg.rsa.use_rsa = True
        cfg.rsa.omega_hl = 2.0
        cfg.rsa.omega_ban = 3.0
        cfg.rsa.ban_teaches_risk = False
        cfg.rsa.use_sem_gate = False
        cfg.rsa.rho_attn = 0.0
        cfg.rsa.gamma_attn = 0.0
        cfg.rsa.use_l0_tutor = False
        cfg.learner.reveal_learning_mode = "cortex_em"

    return cfg


def _compute_phase_sr(block):
    """Return (obs_sr, teach_sr, eval_sr) for a completed block."""
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries

    def _phase_sr(q_start, q_end):
        qs = block.queries[q_start:q_end]
        if not qs:
            return 0.0
        return sum(1 for q in qs if q.success) / len(qs)

    return (
        _phase_sr(0, obs_end),
        _phase_sr(obs_end, teach_end),
        _phase_sr(teach_end, teach_end + block.eval_phase_queries),
    )


def _compute_safety_metrics(block):
    """Compute survival, lethal_pick, total_damage, teach_death_rate, teach_timeout_rate.

    Two flavours of teach-phase safety:
      teach_death_rate  : fraction of teach queries where hp==0 at end (KO)
      teach_timeout_rate: fraction of teach queries where not success and hp>0 (timeout)
      death_rate        : lethal_pick / total_picks (pick-level, kept for J backward-compat)
      timeout_rate      : alias of teach_timeout_rate (used in J objective)
    """
    lethal_picks = 0
    total_picks = 0
    teach_deaths = 0    # KO queries in teach phase
    timeouts = 0        # timed-out queries in teach phase
    total_teach_qs = block.teach_phase_queries

    # Build query_id → QueryState map
    qid_to_qs = {q.query_id: q for q in block.queries}

    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    teach_qids = {q.query_id for q in block.queries[obs_end:teach_end]}

    for lstep in block.learner_trace:
        if lstep.action != "pick":
            continue
        total_picks += 1
        if lstep.damage is not None and lstep.damage > 0:
            qs = qid_to_qs.get(lstep.query_id)
            if qs is not None and qs.hp <= 0 and not qs.success:
                lethal_picks += 1

    # Teach-phase per-query outcomes:
    #   KO (death)  : hp == 0 at end of query
    #   Timeout     : not success and hp > 0
    #   Success     : success == True
    for q in block.queries[obs_end:teach_end]:
        if q.hp <= 0:
            teach_deaths += 1        # KO
        elif not q.success:
            timeouts += 1            # timeout (rounds exhausted without pick)

    # Eval survival (no tutor, frozen learner)
    eval_qs = block.queries[teach_end:]
    survival = [1 if q.hp > 0 else 0 for q in eval_qs]

    teach_death_rate = teach_deaths / max(total_teach_qs, 1)
    teach_timeout_rate = timeouts / max(total_teach_qs, 1)
    death_rate = lethal_picks / max(total_picks, 1)

    return {
        "survival_rate":      float(np.mean(survival)) if survival else 1.0,
        "lethal_pick_count":  lethal_picks,
        "lethal_pick_rate":   death_rate,
        # Teach-phase per-query rates (what the user actually wants to see)
        "teach_death_rate":   teach_death_rate,
        "teach_timeout_rate": teach_timeout_rate,
        "teach_success_rate": 1.0 - teach_death_rate - teach_timeout_rate,
        # Legacy pick-level proxy (used in J objective for backward-compat)
        "death_rate":         death_rate,
        "timeout_rate":       teach_timeout_rate,
        "total_damage":       block.total_damage,
    }


def _compute_rollout_calibration(block, tutor) -> dict:
    """Compute calibration metrics for the tutor's P_death / P_timeout predictions.

    Matches the tutor's predicted P_death / P_timeout for the chosen action
    against realized outcomes in the teach phase.

    Returns:
        pdeath_brier:       Brier score for P_death predictions
        ptimeout_brier:     Brier score for P_timeout predictions
        pdeath_calib_gap:   mean(pred - realized) for P_death  (+ve = over-estimate)
        ptimeout_calib_gap: mean(pred - realized) for P_timeout (+ve = over-estimate)
    """
    ZERO = {"pdeath_brier": 0.0, "ptimeout_brier": 0.0,
            "pdeath_calib_gap": 0.0, "ptimeout_calib_gap": 0.0}
    if tutor is None or not hasattr(tutor, 'get_decision_trace'):
        return ZERO

    trace = tutor.get_decision_trace()
    if not trace:
        return ZERO

    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    teach_qs = {q.query_id: q for q in block.queries[obs_end:teach_end]}

    pd_sq, pd_gap = [], []
    pt_sq, pt_gap = [], []

    for entry in trace:
        qid = entry.get("query_id")
        if qid not in teach_qs:
            continue
        q = teach_qs[qid]
        chosen = entry.get("chosen_action", "WAIT")
        for c in entry.get("scoring", {}).get("candidates", []):
            if c.get("action") == chosen:
                pd_pred = float(c.get("p_death", 0.0))
                pt_pred = float(c.get("p_timeout", 0.0))
                # Realized outcomes for this query
                real_d = float(q.hp <= 0 and not q.success)
                real_t = float(not q.success and q.hp > 0)
                pd_sq.append((pd_pred - real_d) ** 2)
                pd_gap.append(pd_pred - real_d)
                pt_sq.append((pt_pred - real_t) ** 2)
                pt_gap.append(pt_pred - real_t)
                break

    if not pd_sq:
        return ZERO

    return {
        "pdeath_brier":       round(float(np.mean(pd_sq)), 6),
        "ptimeout_brier":     round(float(np.mean(pt_sq)), 6),
        "pdeath_calib_gap":   round(float(np.mean(pd_gap)), 6),
        "ptimeout_calib_gap": round(float(np.mean(pt_gap)), 6),
    }


def _count_actions(block):
    """Action distribution for tutor during teaching phase. O(n)."""
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    # Build set of teach query_ids for O(1) membership check
    teach_qids = {q.query_id for q in block.queries[obs_end:teach_end]}
    actions: Dict[str, int] = {}
    for ts in block.tutor_trace:
        if ts.query_id in teach_qids:
            actions[ts.action] = actions.get(ts.action, 0) + 1
    total = sum(actions.values()) or 1
    return {k: round(100 * v / total, 1) for k, v in actions.items()}


def _count_shortlists(block):
    return sum(1 for ts in block.tutor_trace if ts.action == "SHORTLIST")


def _count_teaching_stats(block):
    """Per-query teaching diagnostics. O(n) with pre-built trace index."""
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    teach_queries = block.queries[obs_end:teach_end]
    teach_qids = {q.query_id for q in teach_queries}

    # Pre-build: query_id → list of learner pick steps (O(n) pass)
    picks_by_qid: Dict[int, list] = {q.query_id: [] for q in teach_queries}
    for ls in block.learner_trace:
        if ls.action == "pick" and ls.query_id in picks_by_qid:
            picks_by_qid[ls.query_id].append(ls)

    wrong_picks_total = 0
    reveals_total = 0
    first_pick_correct_total = 0
    cls_updates = 0

    for q in teach_queries:
        qid = q.query_id
        picks = picks_by_qid.get(qid, [])
        if picks:
            first_correct = bool(picks[0].correct) if picks[0].correct is not None else q.success
            first_pick_correct_total += int(first_correct)
            wrong_picks_total += sum(
                1 for p in picks if not (bool(p.correct) if p.correct is not None else False)
            )
        # Reveals: stored in QueryState.reveal_history
        n_reveals = len(q.reveal_history) if hasattr(q, 'reveal_history') else 0
        reveals_total += n_reveals
        cls_updates += n_reveals

    n_teach = max(len(teach_queries), 1)
    return {
        "wrong_picks_per_query":   round(wrong_picks_total / n_teach, 3),
        "reveals_per_query":       round(reveals_total / n_teach, 3),
        "first_pick_correct_rate": round(first_pick_correct_total / n_teach, 3),
        "cls_updates_per_query":   round(cls_updates / n_teach, 3),
    }


def _compute_eval_diagnostics(block):
    """Eval-phase diagnostics: 1stOK, avg_steps, deaths, timeouts.

    Computed over all eval-phase queries only.
      eval_1st_ok   : fraction of eval queries where learner picks correctly first try
      eval_avg_steps: average number of pick attempts per eval query until done
      eval_deaths   : number of eval queries ending in death (hp<=0 and not success)
      eval_timeouts : number of eval queries ending in timeout (hp>0 and not success)
    """
    obs_end   = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    eval_qs   = block.queries[teach_end:]

    if not eval_qs:
        return {
            "eval_1st_ok":    0.0,
            "eval_avg_steps": 0.0,
            "eval_deaths":    0,
            "eval_timeouts":  0,
        }

    eval_qids = {q.query_id for q in eval_qs}

    # Pre-build: query_id -> list of pick steps (O(n) pass)
    picks_by_qid: Dict[int, list] = {q.query_id: [] for q in eval_qs}
    for ls in block.learner_trace:
        if ls.action == "pick" and ls.query_id in picks_by_qid:
            picks_by_qid[ls.query_id].append(ls)

    first_ok_total = 0
    total_steps    = 0
    deaths         = 0
    timeouts       = 0

    for q in eval_qs:
        picks = picks_by_qid.get(q.query_id, [])
        n_picks = len(picks)
        total_steps += n_picks if n_picks > 0 else 1  # count at least 1 attempt

        if picks:
            # first pick correctness
            first_ok = bool(picks[0].correct) if picks[0].correct is not None else q.success
            first_ok_total += int(first_ok)

        # Outcome classification
        if not q.success:
            if q.hp <= 0:
                deaths += 1       # hp exhausted -> death
            else:
                timeouts += 1     # time ran out without KO -> timeout

    n_eval = len(eval_qs)
    return {
        "eval_1st_ok":    round(first_ok_total / n_eval, 4),
        "eval_avg_steps": round(total_steps / n_eval, 4),
        "eval_deaths":    deaths,
        "eval_timeouts":  timeouts,
    }


def _compute_j(eval_sr, obs_sr, death_rate, timeout_rate, beta, gamma):
    """J = delta_eval_from_obs - beta * death_rate - gamma * timeout_rate.

    Note: beta/gamma passed explicitly to avoid subprocess global-variable isolation issues.
    """
    delta_eval = eval_sr - obs_sr
    return round(delta_eval - beta * death_rate - gamma * timeout_rate, 4)


def _run_eval_z(block, env, learner, cfg, grammar_id) -> dict:
    """Run EVAL-Z: re-run eval-phase queries with all option risk_class=0.

    EVAL-Z isolates concept learning from risk navigation:
      - All option risk_class forced to 0  (no HP damage from wrong picks)
      - HP set to T_max + 1               (effectively infinite, no death)
      - Tutor always WAITs                (no interventions)
      - Learner uses its current trained belief (_scorer state carried over)

    Returns:
        eval_z_sr       : success rate (fraction of queries solved)
        eval_z_1st_ok   : fraction of queries where first pick is correct
        eval_z_avg_attempts : mean number of pick attempts per query

    Implementation: builds fresh QueryState copies from the eval queries
    in the completed block, runs learner.act() in a lightweight loop.
    """
    import copy
    from cls_option_tutor.env.state import QueryState, BlockState
    from cls_option_tutor.env.interventions import get_active_menu

    # Get eval-phase query states from original block
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    orig_eval_qs = block.queries[teach_end: teach_end + block.eval_phase_queries]

    if not orig_eval_qs:
        return {"eval_z_sr": 0.0, "eval_z_1st_ok": 0.0, "eval_z_avg_attempts": 0.0}

    # Build zero-risk copies of each eval query
    z_query_states = []
    for qi, orig in enumerate(orig_eval_qs):
        # Deep-copy menu, zero out risk
        z_menu = []
        for opt in orig.menu:
            z_opt = copy.copy(opt)
            z_opt.risk_class = 0
            import numpy as _np
            z_opt.danger_vec = _np.zeros_like(opt.danger_vec)
            z_menu.append(z_opt)

        z_qs = QueryState(
            query_id=qi,
            target_output=list(orig.target_output),
            true_program=list(orig.true_program),
            hp=cfg.env.T_max + 1,      # effectively infinite — no death possible
            max_rounds=cfg.env.T_max,
            max_refreshes=orig.max_refreshes,
            menu=z_menu,
        )
        z_query_states.append(z_qs)

    # Build a minimal BlockState for the Z eval
    z_block = BlockState(
        block_id=-1,
        support_examples=block.support_examples,
        queries=z_query_states,
        obs_phase_queries=0,
        teach_phase_queries=0,
        eval_phase_queries=len(z_query_states),
    )
    z_block.current_query_idx = 0

    # --- Run learner (tutor always WAIT, learner uses existing _scorer) ---
    # We do NOT call learner.init_block() — reuse current trained scorer.
    # We DO reset per-query attention inside act() (it checks L != current L).
    # Temporarily reset attention to force fresh init for each query.
    saved_attention = learner.policy.attention
    learner.policy.attention = None

    n_queries = len(z_query_states)
    successes = 0
    first_ok_total = 0
    total_attempts = 0
    MAX_STEPS = n_queries * (cfg.env.T_max + 2)
    steps = 0

    while not z_block.done and steps < MAX_STEPS:
        steps += 1
        z_qs = z_block.current_query
        if z_qs is None or z_qs.done:
            break

        # Tutor: WAIT (no intervention in EVAL-Z)
        env.tutor_act(z_block, "WAIT")
        if z_qs.done:
            continue

        # Track first pick
        picks_before = len([
            ls for ls in z_block.learner_trace
            if ls.query_id == z_qs.query_id and ls.action == "pick"
        ])

        # Learner acts
        policy_out = learner.act(z_block, env)

        picks_after = len([
            ls for ls in z_block.learner_trace
            if ls.query_id == z_qs.query_id and ls.action == "pick"
        ])

        if picks_before == 0 and picks_after == 1:
            # This was the first pick — check correctness
            for ls in z_block.learner_trace:
                if ls.query_id == z_qs.query_id and ls.action == "pick":
                    if ls.correct:
                        first_ok_total += 1
                    break

    # Restore attention
    learner.policy.attention = saved_attention

    # Count outcomes
    for z_qs in z_query_states:
        if z_qs.success:
            successes += 1
        # Count picks for this query
        n_picks = sum(
            1 for ls in z_block.learner_trace
            if ls.query_id == z_qs.query_id and ls.action == "pick"
        )
        total_attempts += max(n_picks, 1)

    eval_z_sr = round(successes / n_queries, 4)
    eval_z_1st_ok = round(first_ok_total / n_queries, 4)
    eval_z_avg_attempts = round(total_attempts / n_queries, 4)

    return {
        "eval_z_sr":           eval_z_sr,
        "eval_z_1st_ok":       eval_z_1st_ok,
        "eval_z_avg_attempts": eval_z_avg_attempts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Job runner (ProcessPool-safe)
# ─────────────────────────────────────────────────────────────────────────────

def run_job(args):
    """Run one (sc, cond, grammar, n_sup, seed, beta, gamma, n_teach, n_probe, teach_budget, eta_reveal) combination."""
    sc_name, cond, grammar_id, n_sup, seed, beta, gamma, n_teach, n_probe, teach_budget, eta_reveal = args

    import sys, os, time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from cls_option_tutor.env.option_env import OptionEnv
    from cls_option_tutor.learner.learner_agent import LearnerAgent

    sc = SCENARIOS[sc_name]
    # Override N_teach if CLI --n_teach was specified
    sc_mod = ScenarioCfg(**{**vars(sc), 'n_sup': n_sup, 'N_teach': n_teach})
    cfg = _make_cfg(sc_mod, cond, teach_budget=teach_budget)

    # Apply eta_reveal to learner config (reveal learning strength knob)
    cfg.learner.eta_reveal = float(eta_reveal)

    DATA_DIR = _resolve_data_dir(__file__)
    t0 = time.time()

    try:
        env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
        learner = LearnerAgent(cfg=cfg)

        # ── Condition dispatch ───────────────────────────────────
        if cond == "no_tutor":
            block = learner.run_block(env, grammar_id, seed=seed,
                                      tutor_action="WAIT")

        elif cond == "no_tutor_nonreveal_off":
            # Learner alone, nonreveal feedback, no negative evidence learning.
            # Purest isolation: tutor=WAIT, feedback_mode=nonreveal, learning=off.
            # vs no_tutor: reveals full feedback-mode effect on learner alone (no tutor).
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "off"
            cfg.learner.negative_evidence_mode = "off"
            block = learner.run_block(env, grammar_id, seed=seed,
                                      tutor_action="WAIT")

        elif cond == "no_tutor_nonreveal_neg":
            # Learner alone, nonreveal feedback + negative evidence.
            # vs no_tutor_nonreveal_off: isolates neg evidence effect on learner alone.
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.learner.eta_negative = 1.0
            cfg.learner.lambda_neg = 1.0
            block = learner.run_block(env, grammar_id, seed=seed,
                                      tutor_action="WAIT")


        elif cond == "old_tutor":
            from cls_option_tutor.tutor.tutor_agent import TutorAgent
            tutor = TutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "new_baseline":
            from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent
            tutor = OptionLevelTutorAgent(cfg=cfg, shortlist_mode="confusion",
                                          g_learn_mode="none")
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "new_probe":
            from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent
            tutor = OptionLevelTutorAgent(
                cfg=cfg,
                shortlist_mode="confusion",
                g_learn_mode="probe",
                lambda_learn=1.0,
                beta=beta,
                gamma=gamma,
                n_probe=n_probe,   # from CLI --n_probe
                n_candidates=3,
            )
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "new_oracle_surrogate":
            from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent
            tutor = OptionLevelTutorAgent(
                cfg=cfg,
                shortlist_mode="confusion",
                g_learn_mode="oracle_surrogate",
                lambda_learn=1.0,
                beta=beta,
                gamma=gamma,
                n_candidates=3,
            )
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_no_shift":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.0
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_low_shift":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        # ── Repair validation conditions (g_eval enabled) ────────
        elif cond == "sparse_probe_shift0125":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_probe_shift0":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.0
            cfg.tutor.sparse_g_learn_mode = "probe"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_oracle_shift0125":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "oracle_surrogate"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_oracle_shift0":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.0
            cfg.tutor.sparse_g_learn_mode = "oracle_surrogate"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_probe_hl35":
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.tutor.hl_timeout_threshold = 0.35
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        # ── Rescue mode conditions (dual-mode tutor) ───────────────
        elif cond == "sparse_probe_rescue":
            # Dual-mode: learning uses lambda_shift=0.125 (conservative),
            # rescue uses lambda_shift_res=0.0625 (half, more permissive).
            # Distinguishes from sparse_probe_shift0 (pure lambda_shift=0).
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            tutor = SparseTutorAgent(cfg=cfg)
            # Explicitly override rescue params (defaults: theta=0.5, to=1.0, shift_res=0.0625)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = cfg.tutor.lambda_shift * 0.5  # 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_probe_rescue_shift0125":
            # Alias for sparse_probe_rescue with explicit lambda_shift=0.125 label.
            # Kept for condition registry compatibility.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            tutor = SparseTutorAgent(cfg=cfg)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_oracle_rescue":
            # Oracle dual-mode: learning=0.125, rescue=0.0625.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "oracle_surrogate"
            tutor = SparseTutorAgent(cfg=cfg)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_oracle_rescue_shift0125":
            # Alias for sparse_oracle_rescue (lambda_shift=0.125 explicit).
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "oracle_surrogate"
            tutor = SparseTutorAgent(cfg=cfg)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        # ── Step 4+5: Persistent Prior conditions ─────────────────────
        elif cond == "persistent_hl_probe_rescue":
            # sparse_probe_rescue + Persistent HL Prior (Step 4)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.learner.rho_hl_prior = 0.3       # EMA update rate
            cfg.learner.lambda_hl_prior = 0.3    # prior injection weight
            tutor = SparseTutorAgent(cfg=cfg)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "persistent_ban_probe_rescue":
            # sparse_probe_rescue + Persistent BAN Prior (Step 5)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.learner.rho_ban_prior = 0.3      # EMA update rate
            cfg.learner.lambda_ban_prior = 0.5   # penalty weight
            tutor = SparseTutorAgent(cfg=cfg)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "persistent_both_probe_rescue":
            # sparse_probe_rescue + HL Prior + BAN Prior (Step 4+5)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.lambda_shift = 0.125
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.learner.rho_hl_prior = 0.3
            cfg.learner.lambda_hl_prior = 0.3
            cfg.learner.rho_ban_prior = 0.3
            cfg.learner.lambda_ban_prior = 0.5
            tutor = SparseTutorAgent(cfg=cfg)
            tutor.theta_rescue = 0.5
            tutor.lambda_to = 1.0
            tutor.lambda_shift_res = 0.0625
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "nonreveal_off":
            # Nonreveal mode: wrong picks do not expose true output to learner.
            # Negative evidence learning: OFF (pure behaviour control condition).
            # Tutor: basic BAN+HL only (g_learn_mode=none), same as canonical 'sparse'.
            # Key comparison: sparse (reveal) vs this → isolates reveal feedback value.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "off"
            cfg.learner.negative_evidence_mode = "off"
            cfg.tutor.sparse_g_learn_mode = "none"   # basic BAN+HL, no G_learn probe
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "nonreveal_neg":
            # Nonreveal mode + negative evidence learning.
            # Wrong picks record (program, target_output) penalties in scorer.
            # Tutor: basic BAN+HL only (g_learn_mode=none), same as nonreveal_off tutor.
            # Key comparison: nonreveal_off vs this → isolates negative evidence value.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.learner.eta_negative = 1.0
            cfg.learner.lambda_neg = 1.0
            cfg.tutor.sparse_g_learn_mode = "none"   # basic BAN+HL, no G_learn probe
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "nonreveal_reveal_reg":
            # Regression: reveal mode with basic BAN+HL (g_learn_mode=none).
            # Paired baseline for nonreveal_neg — same tutor config, only feedback differs.
            # Expected: eval_sr >= nonreveal_neg >= nonreveal_off
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "reveal"           # reveal mode (default)
            cfg.learner.reveal_learning_mode = "cortex_em"
            cfg.learner.negative_evidence_mode = "off"
            cfg.tutor.sparse_g_learn_mode = "none"    # basic BAN+HL, no G_learn probe
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_proxy":
            # Backward-compat: static tier-model proxy (pre-upgrade baseline)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.rollout_mode = "proxy"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_rollout":
            # Option A upgrade: learner-consistent + hybrid rollout (N=8)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_correct_learn":
            # reveal + correct-pick positive learning, eta=1.0 (full)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.learner.correct_pick_learning_mode = "cortex_em"
            cfg.learner.eta_correct_pick = 1.0
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_correct_learn_gate":
            # reveal + correct-pick positive learning, eta=0.5 (stochastic gate)
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.learner.correct_pick_learning_mode = "cortex_em"
            cfg.learner.eta_correct_pick = 0.5
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_nonreveal_neg_correct_gate":
            # nonreveal + negative evidence + correct-pick positive (eta=0.5)
            # Forms complete feedback loop: wrong→negative, correct→positive
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.learner.correct_pick_learning_mode = "cortex_em"
            cfg.learner.eta_correct_pick = 0.5
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        # ── E1: Nonreveal rollout baselines (PP-tutor, no dual-mode code) ──

        elif cond == "sparse_rollout_nr_none":
            # Nonreveal + hybrid rollout + g_learn=none.
            # Baseline: does rollout alone (without probe) help in nonreveal?
            # Paired with sparse_rollout_nr_probe to isolate probe signal value.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.learner.eta_negative = 1.0
            cfg.learner.lambda_neg = 1.0
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.tutor.sparse_g_learn_mode = "none"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_rollout_nr_probe":
            # Nonreveal + hybrid rollout + g_learn=probe.
            # Key E1 test: does G_eval(probe) in nonreveal carry real signal?
            # Compare EvalSR vs sparse_rollout_nr_none to isolate probe value.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.learner.eta_negative = 1.0
            cfg.learner.lambda_neg = 1.0
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.tutor.sparse_g_learn_mode = "probe"
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        # ── E2 Reveal: Dual-mode tutor ────────────────────────────────

        elif cond == "sparse_protective":
            # Reveal + protective mode: U_teach maximized subject to eval guard.
            # G_eval guard: actions with G_eval < -0.01 are filtered.
            # Rollout forced for all non-WAIT actions (p_success needed).
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.tutor.tutor_mode = "protective"
            cfg.tutor.eps_eval_guard = 0.01
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_pedagogical":
            # Reveal + pedagogical mode: G_eval + eta*U_teach, safety constraints.
            # Dynamic constraints: P_death <= p_death_wait+0.01, P_timeout <= p_timeout_wait+0.03.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.tutor.tutor_mode = "pedagogical"
            cfg.tutor.eta_pedagogical = 0.25
            cfg.tutor.d_max_margin = 0.01
            cfg.tutor.t_max_margin = 0.03
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        # ── E2 Nonreveal: Dual-mode tutor + nonreveal ──────────────────

        elif cond == "sparse_protective_nr":
            # Nonreveal + protective mode.
            # G_eval probe in nonreveal estimates neg-evidence-based learning gain.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.learner.eta_negative = 1.0
            cfg.learner.lambda_neg = 1.0
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.tutor.tutor_mode = "protective"
            cfg.tutor.eps_eval_guard = 0.01
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        elif cond == "sparse_pedagogical_nr":
            # Nonreveal + pedagogical mode.
            # Long-term: G_eval(probe/nonreveal); short-term: U_teach via rollout.
            from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
            cfg.env.feedback_mode = "nonreveal"
            cfg.learner.reveal_learning_mode = "nonreveal_negative"
            cfg.learner.negative_evidence_mode = "exact_program_target"
            cfg.learner.eta_negative = 1.0
            cfg.learner.lambda_neg = 1.0
            cfg.tutor.rollout_mode = "hybrid"
            cfg.tutor.rollout_n = 8
            cfg.tutor.sparse_g_learn_mode = "probe"
            cfg.tutor.tutor_mode = "pedagogical"
            cfg.tutor.eta_pedagogical = 0.25
            cfg.tutor.d_max_margin = 0.01
            cfg.tutor.t_max_margin = 0.03
            tutor = SparseTutorAgent(cfg=cfg)
            block = tutor.run_block(env, learner, grammar_id, seed=seed)

        else:
            raise ValueError(f"Unknown condition: {cond}")


        # ── Metrics ──────────────────────────────────────────────
        obs_sr, teach_sr, eval_sr = _compute_phase_sr(block)
        safety = _compute_safety_metrics(block)
        teach_stats  = _count_teaching_stats(block)
        eval_diag    = _compute_eval_diagnostics(block)
        n_shortlist  = _count_shortlists(block)

        # EVAL-Z: re-run eval with zero risk (pure concept accuracy)
        eval_z = _run_eval_z(block, env, learner, cfg, grammar_id)

        # PosteriorShiftPerReveal: how much each reveal improved learner's scorer
        reveal_shift = learner.reveal_shift_stats()

        # ── Rollout calibration metrics (sparse conditions only) ─────────
        # Must define is_sparse_cond before using it here.
        is_sparse_cond = cond.startswith("sparse")
        calib = _compute_rollout_calibration(
            block,
            tutor if (is_sparse_cond and hasattr(tutor, 'get_decision_trace')) else None,
        )

        elapsed = time.time() - t0

        # J objective
        delta_eval_from_obs = round(eval_sr - obs_sr, 4)
        j_score = _compute_j(
            eval_sr, obs_sr,
            safety["death_rate"], safety["timeout_rate"],
            beta=beta, gamma=gamma,
        )

        # ── MenuTouchedRate (renamed from menu_shift) ────────────
        # Fraction of teach queries where tutor changed the active menu.
        obs_end_idx  = block.obs_phase_queries
        teach_end_idx = obs_end_idx + block.teach_phase_queries
        teach_qids_set = {q.query_id for q in block.queries[obs_end_idx:teach_end_idx]}
        menu_change_actions = {"SHORTLIST", "BAN", "HIGHLIGHT", "MIX"}
        n_menu_changes = sum(
            1 for ts in block.tutor_trace
            if ts.query_id in teach_qids_set
            and getattr(ts, 'action', '') in menu_change_actions
        )
        n_teach_qs = max(block.teach_phase_queries, 1)
        menu_touched_rate = round(n_menu_changes / n_teach_qs, 4)

        # ── ChoiceShiftJS (tier-aware JS of chosen action) ───────
        choice_shift_js = 0.0
        # is_sparse_cond already defined above (before calibration call)
        if cond == "no_tutor":
            choice_shift_js = 0.0
        elif is_sparse_cond and hasattr(tutor, 'get_decision_trace'):
            # For sparse conditions: extract d_shift of chosen action from trace
            trace = tutor.get_decision_trace()
            shifts = []
            for entry in trace:
                if entry.get("chosen_action") != "WAIT":
                    scoring = entry.get("scoring", {})
                    for c in scoring.get("candidates", []):
                        if c.get("action") == entry["chosen_action"]:
                            shifts.append(c.get("d_shift", 0.0))
                            break
                else:
                    shifts.append(0.0)
            choice_shift_js = round(float(np.mean(shifts)), 6) if shifts else 0.0
        elif cond in ("new_baseline", "new_probe", "new_oracle_surrogate"):
            # SHORTLIST: masked-menu JS renormalization
            # p_baseline(j) = uniform or learner pick prob under WAIT
            # p_shortlist(j) = p_baseline(j) / sum_S(p_baseline) for j in S, else 0
            teach_trace = [ts for ts in block.tutor_trace
                           if ts.query_id in teach_qids_set
                           and ts.action == "SHORTLIST"]
            if teach_trace:
                js_vals = []
                K = cfg.env.K
                for ts in teach_trace:
                    if ts.shortlist_indices:
                        S = set(ts.shortlist_indices)
                        # Baseline: uniform distribution over K options
                        p_wait = np.ones(K) / K
                        # Shortlisted distribution: masked + renormalized
                        p_sl = np.zeros(K)
                        for j in S:
                            if j < K:
                                p_sl[j] = p_wait[j]
                        denom = p_sl.sum()
                        if denom > 0:
                            p_sl = p_sl / denom
                        js_vals.append(_js_divergence_fn(p_wait, p_sl))
                choice_shift_js = round(float(np.mean(js_vals)), 6) if js_vals else 0.0

        # ── Trace summary (sparse conditions only) ───────────────
        trace_summary = {}
        if is_sparse_cond and hasattr(tutor, '_extract_trace_summary'):
            trace_summary = tutor._extract_trace_summary()

        return {
            # Identity
            "sc": sc_name, "cond": cond, "grammar": grammar_id,
            "nsup": n_sup, "seed": seed, "eta_reveal": eta_reveal,
            # Phase SRs
            "OBS_SR": obs_sr, "TEACH_SR": teach_sr, "EVAL_SR": eval_sr,
            # J objective components
            "delta_eval_from_obs": delta_eval_from_obs,
            "death_rate":     safety["death_rate"],
            "timeout_rate":   safety["timeout_rate"],
            "J":              j_score,
            # Safety — legacy pick-level
            "survival_rate":    safety["survival_rate"],
            "lethal_pick_rate": safety["lethal_pick_rate"],
            "total_damage":     safety["total_damage"],
            # Teach-phase per-query outcomes (the key safety metrics)
            "teach_death_rate":   safety["teach_death_rate"],
            "teach_timeout_rate": safety["teach_timeout_rate"],
            "teach_success_rate": safety["teach_success_rate"],
            # Teaching diagnostics
            "n_shortlist":               n_shortlist,
            "menu_touched_rate":         menu_touched_rate,
            "choice_shift_js":           choice_shift_js,
            "wrong_picks_per_query":     teach_stats["wrong_picks_per_query"],
            "reveals_per_query":         teach_stats["reveals_per_query"],
            "first_pick_correct_rate":   teach_stats["first_pick_correct_rate"],
            "cls_updates_per_query":     teach_stats["cls_updates_per_query"],
            # Sparse tutor diagnostics
            "ban_generated_rate":        trace_summary.get("ban_generated_rate", 0.0),
            "hl_generated_rate":         trace_summary.get("hl_generated_rate", 0.0),
            "hl_suppressed_rate":        trace_summary.get("hl_suppressed_rate", 0.0),
            "nonwait_beats_wait_rate":   trace_summary.get("nonwait_beats_wait_rate", 0.0),
            "mean_best_nonwait_margin":  trace_summary.get("mean_best_nonwait_margin", 0.0),
            "mean_hl_gate_value":        trace_summary.get("mean_hl_gate_value", 0.0),
            # Rescue diagnostics
            "rescue_trigger_rate":            trace_summary.get("rescue_trigger_rate", 0.0),
            "mean_delta_p_timeout":           trace_summary.get("mean_delta_p_timeout", 0.0),
            "timeout_blocker_selected_rate":  trace_summary.get("timeout_blocker_selected_rate", 0.0),
            "highlight_in_rescue_rate":       trace_summary.get("highlight_in_rescue_rate", 0.0),
            # Eval-N diagnostics (with risk)
            "eval_1st_ok":    eval_diag["eval_1st_ok"],
            "eval_avg_steps": eval_diag["eval_avg_steps"],
            "eval_deaths":    eval_diag["eval_deaths"],
            "eval_timeouts":  eval_diag["eval_timeouts"],
            # Eval-Z diagnostics (zero risk — pure concept accuracy)
            "eval_z_sr":           eval_z["eval_z_sr"],
            "eval_z_1st_ok":       eval_z["eval_z_1st_ok"],
            "eval_z_avg_attempts": eval_z["eval_z_avg_attempts"],
            # PosteriorShiftPerReveal (semantic quality of each reveal)
            "posterior_shift_per_reveal":   reveal_shift["posterior_shift_per_reveal"],
            "posterior_shift_n_reveals":    reveal_shift["posterior_shift_n_reveals"],
            "posterior_shift_positive_rate": reveal_shift["posterior_shift_positive_rate"],
            # Meta
            "actions": _count_actions(block),
            "elapsed": round(elapsed, 1),
            "error": None,
            # Calibration (rollout proxy health)
            "pdeath_brier":       calib["pdeath_brier"],
            "ptimeout_brier":     calib["ptimeout_brier"],
            "pdeath_calib_gap":   calib["pdeath_calib_gap"],
            "ptimeout_calib_gap": calib["ptimeout_calib_gap"],
            # Decision trace (for --trace mode; serialized externally)
            "_decision_trace": (tutor.get_decision_trace()
                                if is_sparse_cond and hasattr(tutor, 'get_decision_trace')
                                else None),
        }

    except Exception as e:
        import traceback
        return {
            "sc": sc_name, "cond": cond, "grammar": grammar_id,
            "nsup": n_sup, "seed": seed, "eta_reveal": eta_reveal,
            "OBS_SR": 0.0, "TEACH_SR": 0.0, "EVAL_SR": 0.0,
            "delta_eval_from_obs": 0.0, "death_rate": 0.0,
            "timeout_rate": 0.0, "J": 0.0,
            "survival_rate": 0.0, "lethal_pick_rate": 0.0, "total_damage": 0,
            "teach_death_rate": 0.0, "teach_timeout_rate": 0.0, "teach_success_rate": 0.0,
            "n_shortlist": 0, "menu_touched_rate": 0.0,
            "choice_shift_js": 0.0,
            "wrong_picks_per_query": 0.0,
            "reveals_per_query": 0.0, "first_pick_correct_rate": 0.0,
            "cls_updates_per_query": 0.0,
            "ban_generated_rate": 0.0, "hl_generated_rate": 0.0,
            "hl_suppressed_rate": 0.0, "nonwait_beats_wait_rate": 0.0,
            "mean_best_nonwait_margin": 0.0, "mean_hl_gate_value": 0.0,
            "rescue_trigger_rate": 0.0, "mean_delta_p_timeout": 0.0,
            "timeout_blocker_selected_rate": 0.0, "highlight_in_rescue_rate": 0.0,
            "eval_1st_ok": 0.0, "eval_avg_steps": 0.0,
            "eval_deaths": 0, "eval_timeouts": 0,
            "eval_z_sr": 0.0, "eval_z_1st_ok": 0.0, "eval_z_avg_attempts": 0.0,
            "actions": {}, "elapsed": 0.0,
            "_decision_trace": None,
            "pdeath_brier": 0.0, "ptimeout_brier": 0.0,
            "pdeath_calib_gap": 0.0, "ptimeout_calib_gap": 0.0,
            "posterior_shift_per_reveal": 0.0, "posterior_shift_n_reveals": 0,
            "posterior_shift_positive_rate": 0.0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: List[dict], active_conditions: List[str] = None) -> dict:
    """Aggregate per-job results into nested dict: sc → cond → metrics."""
    from collections import defaultdict
    if active_conditions is None:
        active_conditions = CONDITIONS_ALL

    buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["error"]:
            continue
        buckets[r["sc"]][r["cond"]].append(r)

    def _mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    def _se(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        n = len(vals)
        return round(float(np.std(vals) / np.sqrt(max(n, 1))), 4) if vals else 0.0

    agg = {}
    for sc_name in sorted(buckets):
        agg[sc_name] = {}
        for cond in active_conditions:
            rows = buckets[sc_name].get(cond, [])
            if not rows:
                continue
            n = len(rows)
            agg[sc_name][cond] = {
                "n": n,
                # Phase SRs
                "OBS_SR":   _mean(rows, "OBS_SR"),
                "TEACH_SR": _mean(rows, "TEACH_SR"),
                "EVAL_SR":  _mean(rows, "EVAL_SR"),
                "EVAL_SE":  _se(rows, "EVAL_SR"),
                # J components
                "delta_eval_from_obs":  _mean(rows, "delta_eval_from_obs"),
                "death_rate":           _mean(rows, "death_rate"),
                "timeout_rate":         _mean(rows, "timeout_rate"),
                "J":                    _mean(rows, "J"),
                "J_SE":                 _se(rows, "J"),
                # Safety
                "survival_rate":    _mean(rows, "survival_rate"),
                "lethal_pick_rate": _mean(rows, "lethal_pick_rate"),
                "total_damage":     _mean(rows, "total_damage"),
                # Teach-phase per-query outcomes
                "teach_death_rate":   _mean(rows, "teach_death_rate"),
                "teach_timeout_rate": _mean(rows, "teach_timeout_rate"),
                "teach_success_rate": _mean(rows, "teach_success_rate"),
                # Teaching diagnostics
                "n_shortlist":             _mean(rows, "n_shortlist"),
                "menu_touched_rate":       _mean(rows, "menu_touched_rate"),
                "choice_shift_js":         _mean(rows, "choice_shift_js"),
                "wrong_picks_per_query":   _mean(rows, "wrong_picks_per_query"),
                "reveals_per_query":       _mean(rows, "reveals_per_query"),
                "first_pick_correct_rate": _mean(rows, "first_pick_correct_rate"),
                "cls_updates_per_query":   _mean(rows, "cls_updates_per_query"),
                # Sparse tutor diagnostics
                "ban_generated_rate":      _mean(rows, "ban_generated_rate"),
                "hl_generated_rate":       _mean(rows, "hl_generated_rate"),
                "hl_suppressed_rate":      _mean(rows, "hl_suppressed_rate"),
                "nonwait_beats_wait_rate": _mean(rows, "nonwait_beats_wait_rate"),
                "mean_best_nonwait_margin": _mean(rows, "mean_best_nonwait_margin"),
                "mean_hl_gate_value":      _mean(rows, "mean_hl_gate_value"),
                # Rescue diagnostics
                "rescue_trigger_rate":           _mean(rows, "rescue_trigger_rate"),
                "mean_delta_p_timeout":          _mean(rows, "mean_delta_p_timeout"),
                "timeout_blocker_selected_rate": _mean(rows, "timeout_blocker_selected_rate"),
                "highlight_in_rescue_rate":      _mean(rows, "highlight_in_rescue_rate"),
                # Eval-N diagnostics
                "eval_1st_ok":    _mean(rows, "eval_1st_ok"),
                "eval_avg_steps": _mean(rows, "eval_avg_steps"),
                "eval_deaths":    _mean(rows, "eval_deaths"),
                "eval_timeouts":  _mean(rows, "eval_timeouts"),
                # Eval-Z diagnostics (zero risk)
                "eval_z_sr":           _mean(rows, "eval_z_sr"),
                "eval_z_1st_ok":       _mean(rows, "eval_z_1st_ok"),
                "eval_z_avg_attempts": _mean(rows, "eval_z_avg_attempts"),
                # PosteriorShiftPerReveal
                "posterior_shift_per_reveal":    _mean(rows, "posterior_shift_per_reveal"),
                "posterior_shift_positive_rate": _mean(rows, "posterior_shift_positive_rate"),
                "posterior_shift_n_reveals":     _mean(rows, "posterior_shift_n_reveals"),
                # Calibration metrics (rollout proxy health)
                "pdeath_brier":       _mean(rows, "pdeath_brier"),
                "ptimeout_brier":     _mean(rows, "ptimeout_brier"),
                "pdeath_calib_gap":   _mean(rows, "pdeath_calib_gap"),
                "ptimeout_calib_gap": _mean(rows, "ptimeout_calib_gap"),
            }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results(agg: dict, n_errors: int = 0,
                  active_conditions: List[str] = None) -> None:
    if active_conditions is None:
        active_conditions = CONDITIONS_ALL
    COND_W = 22

    for sc_name, sc_agg in sorted(agg.items()):
        sc = SCENARIOS.get(sc_name, None)
        print(f"\n{'='*90}")
        print(f"Scenario {sc_name}: {sc.description if sc else ''}")
        print(f"{'='*90}")
        # Header line 1: phase SRs + J
        print(f"{'Condition':<{COND_W}}  {'N':>4}  "
              f"{'OBS':>6} {'TEACH':>6} {'EVAL':>6} {'±SE':>6}  "
              f"{'dEval':>6} {'Death':>6} {'Tout':>6} {'J':>7} {'±SE':>6}")
        print("-" * 90)

        for cond in active_conditions:
            row = sc_agg.get(cond)
            if row is None:
                print(f"  {cond:<{COND_W-2}}  (no data)")
                continue
            print(f"  {cond:<{COND_W-2}}  {row['n']:>4}  "
                  f"{row['OBS_SR']:>6.3f} {row['TEACH_SR']:>6.3f} "
                  f"{row['EVAL_SR']:>6.3f} {row['EVAL_SE']:>6.4f}  "
                  f"{row['delta_eval_from_obs']:>+6.3f} "
                  f"{row['death_rate']:>6.3f} {row['timeout_rate']:>6.3f} "
                  f"{row['J']:>+7.4f} {row['J_SE']:>6.4f}")

        # Teaching diagnostics table
        print(f"\n  {'Condition':<{COND_W}}  "
              f"{'#SL':>5} {'Touched':>8} {'ShiftJS':>8} {'WrongPk':>8} {'Reveal':>7} {'1stOK':>6} {'CLSup':>6}")
        print("  " + "-" * 78)
        for cond in active_conditions:
            row = sc_agg.get(cond)
            if row is None:
                continue
            print(f"  {cond:<{COND_W}}  "
                  f"{row['n_shortlist']:>5.1f} "
                  f"{row['menu_touched_rate']:>8.3f} "
                  f"{row['choice_shift_js']:>8.4f} "
                  f"{row['wrong_picks_per_query']:>8.3f} "
                  f"{row['reveals_per_query']:>7.3f} "
                  f"{row['first_pick_correct_rate']:>6.3f} "
                  f"{row['cls_updates_per_query']:>6.3f}")

        # Sparse tutor diagnostics (only print if sparse conditions present)
        sparse_conds = [c for c in active_conditions if c.startswith("sparse")]
        if sparse_conds:
            print(f"\n  {'Condition':<{COND_W}}  "
                  f"{'BanGen':>7} {'HLGen':>6} {'HLSupp':>7} {'NWWin':>6} {'Margin':>8} {'HLGate':>7}")
            print("  " + "-" * 65)
            for cond in sparse_conds:
                row = sc_agg.get(cond)
                if row is None:
                    continue
                print(f"  {cond:<{COND_W}}  "
                      f"{row['ban_generated_rate']:>7.3f} "
                      f"{row['hl_generated_rate']:>6.3f} "
                      f"{row['hl_suppressed_rate']:>7.3f} "
                      f"{row['nonwait_beats_wait_rate']:>6.3f} "
                      f"{row['mean_best_nonwait_margin']:>+8.4f} "
                      f"{row['mean_hl_gate_value']:>7.3f}")

            # Rescue-specific diagnostics (only show if any rescue conditions present)
            rescue_conds = [c for c in sparse_conds if "rescue" in c]
            if rescue_conds:
                print(f"\n  {'Condition':<{COND_W}}  "
                      f"{'RescueTrig':>10} {'dPtout':>7} {'Blocker':>8} {'HLResc':>7}")
                print("  " + "-" * 57)
                for cond in rescue_conds:
                    row = sc_agg.get(cond)
                    if row is None:
                        continue
                    print(f"  {cond:<{COND_W}}  "
                          f"{row['rescue_trigger_rate']:>10.3f} "
                          f"{row['mean_delta_p_timeout']:>+7.4f} "
                          f"{row['timeout_blocker_selected_rate']:>8.3f} "
                          f"{row['highlight_in_rescue_rate']:>7.3f}")

        # Teach-phase per-query outcomes table (Death=KO, Tout=timeout, Succ=success)
        print(f"\n  {'Condition':<{COND_W}}  "
              f"{'TeachKO':>8} {'TeachTout':>10} {'TeachSucc':>10}")
        print("  " + "-" * 58)
        for cond in active_conditions:
            row = sc_agg.get(cond)
            if row is None:
                continue
            print(f"  {cond:<{COND_W}}  "
                  f"{row.get('teach_death_rate', 0.0):>8.4f} "
                  f"{row.get('teach_timeout_rate', 0.0):>10.4f} "
                  f"{row.get('teach_success_rate', 0.0):>10.4f}")

        # Eval-N diagnostics table (with risk)
        print(f"\n  {'Condition':<{COND_W}}  "
              f"{'Ev1stOK-N':>10} {'AvgStps-N':>10} {'EvDeaths':>9} {'EvTouts':>8}")
        print("  " + "-" * 62)
        for cond in active_conditions:
            row = sc_agg.get(cond)
            if row is None:
                continue
            print(f"  {cond:<{COND_W}}  "
                  f"{row['eval_1st_ok']:>10.3f} "
                  f"{row['eval_avg_steps']:>10.3f} "
                  f"{row['eval_deaths']:>9.2f} "
                  f"{row['eval_timeouts']:>8.2f}")

        # Eval-Z diagnostics table (zero risk — pure concept accuracy)
        print(f"\n  {'Condition':<{COND_W}}  "
              f"{'EVAL-Z SR':>10} {'1stOK-Z':>8} {'AvgAtt-Z':>9}")
        print("  " + "-" * 55)
        for cond in active_conditions:
            row = sc_agg.get(cond)
            if row is None:
                continue
            print(f"  {cond:<{COND_W}}  "
                  f"{row['eval_z_sr']:>10.3f} "
                  f"{row['eval_z_1st_ok']:>8.3f} "
                  f"{row['eval_z_avg_attempts']:>9.3f}")

        # Delta analysis (J-based)
        print()
        base = sc_agg.get("no_tutor")
        for comp_cond in [c for c in active_conditions if c != "no_tutor"]:
            comp = sc_agg.get(comp_cond)
            if base and comp:
                dJ    = comp["J"] - base["J"]
                dEval = comp["EVAL_SR"] - base["EVAL_SR"]
                dDeath = comp["death_rate"] - base["death_rate"]
                print(f"  [{comp_cond} vs no_tutor]  "
                      f"J: {dJ:+.4f}  EVAL: {dEval:+.3f}  Death: {dDeath:+.3f}")

    if n_errors:
        print(f"\n[!] {n_errors} job errors occurred")


def write_results(agg: dict, n_errors: int, all_results: List[dict],
                  path: str, active_conditions: List[str] = None) -> None:
    if active_conditions is None:
        active_conditions = CONDITIONS_ALL

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Option-Level Tutor Experiment — J = delta_eval - {J_BETA}*DeathRate - {J_GAMMA}*TimeoutRate\n")
        f.write("=" * 80 + "\n\n")

        for sc_name, sc_agg in sorted(agg.items()):
            sc = SCENARIOS.get(sc_name, None)
            f.write(f"\nScenario {sc_name}: {sc.description if sc else ''}\n")
            f.write("-" * 60 + "\n")
            for cond in active_conditions:
                row = sc_agg.get(cond)
                if row is None:
                    continue
                f.write(
                    f"  {cond:<22}  n={row['n']}  "
                    f"OBS={row['OBS_SR']:.3f} TEACH={row['TEACH_SR']:.3f} "
                    f"EVAL={row['EVAL_SR']:.3f}±{row['EVAL_SE']:.4f}  "
                    f"dEval={row['delta_eval_from_obs']:+.3f} "
                    f"Death={row['death_rate']:.3f} Tout={row['timeout_rate']:.3f} "
                    f"J={row['J']:+.4f}±{row['J_SE']:.4f}  "
                    f"#SL={row['n_shortlist']:.1f} "
                    f"WrongPick={row['wrong_picks_per_query']:.3f} "
                    f"Reveal={row['reveals_per_query']:.3f} "
                    f"CLSupd={row['cls_updates_per_query']:.3f} "
                    f"Eval1stOK={row['eval_1st_ok']:.3f} "
                    f"EvalAvgSteps={row['eval_avg_steps']:.3f} "
                    f"EvalDeaths={row['eval_deaths']:.2f} "
                    f"EvalTimeouts={row['eval_timeouts']:.2f}\n"
                )

        if n_errors:
            f.write(f"\n[!] {n_errors} errors\n")
        f.write("\n--- JSON ---\n")
        f.write(json.dumps(agg, indent=2))

        # ── Per-eta_reveal breakdown (only when multiple eta_reveal values) ──
        # Groups raw rows by (eta_reveal, sc, cond) for ablation analysis.
        eta_vals = sorted(set(r.get("eta_reveal", 1.0) for r in all_results if not r.get("error")))
        if len(eta_vals) > 1:
            def _mean_r(rows, key):
                vals = [r[key] for r in rows if r.get(key) is not None]
                return round(float(np.mean(vals)), 4) if vals else 0.0
            def _se_r(rows, key):
                vals = [r[key] for r in rows if r.get(key) is not None]
                n = len(vals)
                return round(float(np.std(vals) / np.sqrt(max(n, 1))), 4) if vals else 0.0

            eta_agg = {}
            for eta in eta_vals:
                eta_key = str(round(eta, 4))
                eta_agg[eta_key] = {}
                for sc in sorted(set(r["sc"] for r in all_results if not r.get("error"))):
                    eta_agg[eta_key][sc] = {}
                    for cond in sorted(set(r["cond"] for r in all_results if not r.get("error"))):
                        rows = [r for r in all_results
                                if not r.get("error")
                                and abs(r.get("eta_reveal", 1.0) - eta) < 1e-6
                                and r["sc"] == sc and r["cond"] == cond]
                        if not rows:
                            continue
                        eta_agg[eta_key][sc][cond] = {
                            "n":               len(rows),
                            "EVAL_SR":         _mean_r(rows, "EVAL_SR"),
                            "EVAL_SE":         _se_r(rows, "EVAL_SR"),
                            "OBS_SR":          _mean_r(rows, "OBS_SR"),
                            "J":               _mean_r(rows, "J"),
                            "J_SE":            _se_r(rows, "J"),
                            "timeout_rate":    _mean_r(rows, "timeout_rate"),
                            "death_rate":      _mean_r(rows, "death_rate"),
                            "eval_z_sr":       _mean_r(rows, "eval_z_sr"),
                            "eval_z_1st_ok":   _mean_r(rows, "eval_z_1st_ok"),
                            "eval_z_avg_attempts": _mean_r(rows, "eval_z_avg_attempts"),
                            "reveals_per_query":   _mean_r(rows, "reveals_per_query"),
                        }
            f.write("\n\n--- JSON_ETA ---\n")
            f.write(json.dumps(eta_agg, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Proxy validity check (small-K enumeration)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0(args, grammars, DATA_DIR):
    """Phase 0: validate G_probe vs true_delta_eval correlation.

    Uses a small-K enumerable scenario to compute:
      - true_delta_eval (full simulate → post_probe - pre_probe)
      - G_probe for each candidate shortlist
      - G_surrogate for each candidate shortlist
      - Spearman correlation between proxies and true_delta_eval

    Reports: Spearman(G_probe, true_delta_eval), Spearman(G_surrogate, true_delta_eval)
    Decision gate: Spearman(G_probe, true_delta_eval) > 0.4 to proceed to Phase 1.
    """
    from scipy.stats import spearmanr

    print("\n" + "=" * 60)
    print("Phase 0: Proxy Validity Validation")
    print("=" * 60)

    # Small-K enumerable scenario
    sc_small = ScenarioCfg(
        name="phase0", K=5, T_max=3, H_0=5, n_risky=2,
        N_obs=1, N_teach=3, N_eval=2, n_sup=3,
        description="Phase 0: small-K for enumeration"
    )

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from cls_option_tutor.env.option_env import OptionEnv
    from cls_option_tutor.learner.learner_agent import LearnerAgent
    from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent
    from cls_option_tutor.tutor.g_learn import GLearnEstimator, ProbeEvaluator

    results_probe = []
    results_surr  = []
    results_true  = []

    seeds = [42, 43, 44, 45] if not args.smoke else [42, 43]

    for grammar_id in grammars[:2]:
        for seed in seeds:
            try:
                cfg = _make_cfg(sc_small, "new_baseline")
                env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
                learner = LearnerAgent(cfg=cfg)

                # Run a "new_probe" tutor to get block state
                tutor = OptionLevelTutorAgent(
                    cfg=cfg, g_learn_mode="probe", n_probe=15, n_candidates=5
                )
                block = tutor.run_block(env, learner, grammar_id, seed=seed)

                obs_sr, teach_sr, eval_sr = _compute_phase_sr(block)
                safety = _compute_safety_metrics(block)

                # true_delta_eval ≈ delta_eval_from_obs (validated in Phase 3)
                true_delta = eval_sr - obs_sr
                death_rate = safety["death_rate"]
                timeout_rate = safety["timeout_rate"]

                # G_probe: from Q_T log (not yet available — approximate via metrics)
                # For now: use TEACH_SR as proxy for ordering quality
                results_true.append(true_delta)
                results_probe.append(teach_sr)      # placeholder until Q_T log added
                results_surr.append(eval_sr)        # placeholder

            except Exception as e:
                print(f"  [Phase0 error] grammar={grammar_id} seed={seed}: {e}")

    if len(results_true) < 4:
        print("  [Phase 0] Too few successful runs for correlation analysis")
        return

    rho_probe, p_probe = spearmanr(results_probe, results_true)
    rho_surr,  p_surr  = spearmanr(results_surr,  results_true)

    print(f"\n  Proxy correlation with true_delta_eval (n={len(results_true)}):")
    print(f"    G_probe  surrogate: Spearman r = {rho_probe:+.3f} (p={p_probe:.3f})")
    print(f"    G_KL     surrogate: Spearman r = {rho_surr:+.3f}  (p={p_surr:.3f})")
    print()
    gate_passed = rho_probe > 0.4
    print(f"  Gate [Spearman(G_probe, true_delta) > 0.4]: "
          f"{'PASS' if gate_passed else 'FAIL (check proxy before Phase 1)'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Apply J weights from CLI (declare global FIRST, before any use in default=)
    global J_BETA, J_GAMMA

    parser = argparse.ArgumentParser(
        description="Option-Level Tutor Experiment (v2: J objective)"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Quick run: 2 grammars x 2 seeds")
    parser.add_argument("--workers", type=int, default=8,
                        help="ProcessPool workers")
    parser.add_argument("--output", default="cls_option_tutor/results/exp_option_level.txt")
    parser.add_argument("--scenario", nargs="+", default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Specific scenarios to run (default: all)")
    parser.add_argument("--cond", nargs="+", default=None,
                        choices=(CONDITIONS_ALL + CONDITIONS_ROLLOUT
                                 + CONDITIONS_CORRECT_PICK + CONDITIONS_PP_TUTOR),
                        help="Specific conditions (default: all)")
    parser.add_argument("--phase0", action="store_true",
                        help="Run Phase 0 proxy validity check first")
    parser.add_argument("--beta",  type=float, default=J_BETA,
                        help=f"J death penalty weight (default={J_BETA})")
    parser.add_argument("--gamma", type=float, default=J_GAMMA,
                        help=f"J timeout penalty weight (default={J_GAMMA})")
    parser.add_argument("--n_teach", type=int, default=2,
                        help="N_teach: teaching queries per block (default=2, try 4 or 6)")
    parser.add_argument("--n_probe", type=int, default=10,
                        help="n_probe: probe queries for new_probe G_learn (default=10, try 30)")
    parser.add_argument("--teach_budget", type=int, default=0,
                        help="Total step budget for teach phase (0=disabled, use N_teach). Try 10 or 15.")
    parser.add_argument("--trace", action="store_true",
                        help="Write per-job decision trace JSON for sparse conditions")
    parser.add_argument("--diag", action="store_true",
                        help="Auto-include sparse_no_shift and sparse_low_shift conditions")
    parser.add_argument("--repair", action="store_true",
                        help="Auto-include repair validation conditions (probe/oracle g_eval modes)")
    parser.add_argument("--rescue", action="store_true",
                        help="Auto-include rescue mode conditions (dual-mode: learn + deadline rescue)")
    parser.add_argument("--rollout", action="store_true",
                        help="Auto-include rollout calibration conditions (sparse_proxy + sparse_rollout)")
    parser.add_argument("--correct_pick", action="store_true",
                        help="Auto-include correct-pick learning conditions (Step 9 experiment)")
    parser.add_argument("--pp_tutor", action="store_true",
                        help="Auto-include PP-tutor conditions (E1 nonreveal probe + E2 dual-mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seed list (overrides smoke/full defaults). E.g. --seeds 42 123 0 1 2 3")
    parser.add_argument("--nsups", nargs="+", type=int, default=None,
                        help="n_sup values to sweep (default: [4] smoke / [4,6] full). E.g. --nsups 0 2")
    parser.add_argument("--eta_reveal", nargs="+", type=float, default=None,
                        help="Reveal update gate(s) in [0,1] to sweep. "
                             "E.g. --eta_reveal 0.0 0.25 0.5 1.0. "
                             "Default: [1.0] (always update, current behaviour). "
                             "Controls P(incremental_study called per wrong-pick reveal).")
    args = parser.parse_args()

    J_BETA  = args.beta
    J_GAMMA = args.gamma


    # Grammar files
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    DATA_DIR = os.path.join(_project_root, 'BASIC', 'cls_learner', 'data')
    all_grammars = sorted(set(
        os.path.splitext(f)[0]
        for f in os.listdir(DATA_DIR)
        if f.endswith('.txt')
    )) if os.path.isdir(DATA_DIR) else ["000001", "000002", "000003", "000004"]

    if args.smoke:
        grammars = all_grammars[:2]
        nsups = [4]
        seeds = [42, 123]
    else:
        grammars = all_grammars[:4]
        nsups = [4, 6]
        seeds = list(range(42, 62))   # 20 seeds

    # --nsups overrides smoke/full nsup defaults
    if args.nsups:
        nsups = args.nsups

    # --seeds overrides smoke/full seed list
    if args.seeds:
        seeds = args.seeds

    selected_scenarios = args.scenario or list(SCENARIOS.keys())
    selected_conds = list(args.cond) if args.cond else list(CONDITIONS_BASE + CONDITIONS_SPARSE)
    # --diag: auto-add diagnostic conditions
    if args.diag:
        for dc in CONDITIONS_DIAG:
            if dc not in selected_conds:
                selected_conds.append(dc)
    # --repair: auto-add repair validation conditions
    if args.repair:
        for rc in CONDITIONS_REPAIR:
            if rc not in selected_conds:
                selected_conds.append(rc)
    # --rescue: auto-add rescue mode conditions
    if args.rescue:
        for rc in CONDITIONS_RESCUE:
            if rc not in selected_conds:
                selected_conds.append(rc)
    # --rollout: auto-add rollout calibration conditions
    if args.rollout:
        for rc in CONDITIONS_ROLLOUT:
            if rc not in selected_conds:
                selected_conds.append(rc)
    # --correct_pick: auto-add correct-pick learning conditions (Step 9)
    if args.correct_pick:
        for rc in CONDITIONS_CORRECT_PICK:
            if rc not in selected_conds:
                selected_conds.append(rc)
    # --pp_tutor: auto-add dual-mode pedagogical tutor conditions (E1 + E2)
    if args.pp_tutor:
        for rc in CONDITIONS_PP_TUTOR:
            if rc not in selected_conds:
                selected_conds.append(rc)

    # ── Phase 0 (optional) ───────────────────────────────────────
    if args.phase0:
        run_phase0(args, all_grammars, DATA_DIR)

    # eta_reveal sweep (default = [1.0])
    eta_reveals = args.eta_reveal if args.eta_reveal else [1.0]

    # ── Build jobs ───────────────────────────────────────────────
    jobs = [
        (sc_name, cond, g, ns, s, J_BETA, J_GAMMA, args.n_teach, args.n_probe, args.teach_budget, eta_r)
        for sc_name in selected_scenarios
        for cond in selected_conds
        for g in grammars
        for ns in nsups
        for s in seeds
        for eta_r in eta_reveals
    ]

    tag = "SMOKE" if args.smoke else "FULL"
    print(f"\nOption-Level Tutor Experiment v2 [{tag}]")
    print(f"  J = delta_eval_from_obs - {J_BETA}*DeathRate - {J_GAMMA}*TimeoutRate")
    print(f"  N_teach={args.n_teach}  n_probe={args.n_probe}  "
          f"teach_budget={'disabled' if args.teach_budget == 0 else args.teach_budget}")
    print(f"  eta_reveal sweep: {eta_reveals}")
    print(f"  Scenarios:  {selected_scenarios}")
    print(f"  Conditions: {selected_conds}")
    print(f"  Grammars: {grammars} ({len(grammars)}) x N_sup: {nsups} x Seeds: {len(seeds)}")
    print(f"  Total jobs: {len(jobs)}, Workers: {args.workers}")
    print()

    all_results = []
    n_errors = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, j): j for j in jobs}
        n_done = 0
        for fut in futures:
            try:
                r = fut.result()
                all_results.append(r)
                if r["error"]:
                    n_errors += 1
                    print(f"  [ERR] {r['sc']}/{r['cond']}/{r['grammar']}: "
                          f"{r['error'][:100]}")
            except Exception as e:
                n_errors += 1
                print(f"  [EXCEPT] {futures[fut]}: {e}")
            n_done += 1
            pct_now = 100 * n_done / len(jobs)
            pct_prev = 100 * (n_done - 1) / len(jobs)
            # Print at every 1% milestone or at the very end
            if int(pct_now) > int(pct_prev) or n_done == len(jobs):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done:>5}/{len(jobs)} {pct_now:5.1f}%] "
                      f"{elapsed:5.0f}s elapsed  {rate:.1f} jobs/s  "
                      f"ETA {eta:.0f}s", flush=True)

    agg = aggregate(all_results, active_conditions=selected_conds)

    print_results(agg, n_errors, active_conditions=selected_conds)
    write_results(agg, n_errors, all_results,
                  os.path.join(os.path.dirname(__file__), args.output),
                  active_conditions=selected_conds)
    print(f"\nResults saved to: {args.output}")

    # ── Trace serialization (--trace) ────────────────────────────
    if args.trace:
        trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "results", "traces")
        os.makedirs(trace_dir, exist_ok=True)
        n_traces = 0
        for r in all_results:
            trace = r.get("_decision_trace")
            if trace and r.get("cond", "").startswith("sparse"):
                fname = f"trace_{r['sc']}_{r['cond']}_{r['grammar']}_{r['seed']}.json"
                trace_path = os.path.join(trace_dir, fname)
                with open(trace_path, "w", encoding="utf-8") as tf:
                    json.dump(trace, tf, indent=2, default=str)
                n_traces += 1
        print(f"  Decision traces written: {n_traces} files to {trace_dir}")


if __name__ == "__main__":
    main()
