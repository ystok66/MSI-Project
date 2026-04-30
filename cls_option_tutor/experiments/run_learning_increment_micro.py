"""
run_learning_increment_micro.py â€” E6.4 Risk-Valid + Experience-Semantic Micro Benchmark.

Phase 6.4 additions over E6.2b:
  - self_correct vs then_answer protocol split
  - Phase-specific damage (ObsDmg, TeachDmg, ScriptedDmg, IncidentDmg, EvalDmg)
  - Local probe (LocalPreSR, LocalPostSR, DeltaLocalSR)
  - Semantic margin + correct rank diagnostics
  - Diagnostic vs random wrong selection comparison
  - Bootstrap CI + ImproveRate

Usage:
    python -m cls_option_tutor.experiments.run_learning_increment_micro --smoke
    python -m cls_option_tutor.experiments.run_learning_increment_micro --workers 16
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.direct_answer_tutor import DirectAnswerTutor
from cls_option_tutor.tutor.scripted_protocols import ScriptedProtocolRunner
from cls_option_tutor.eval.autonomous_probe import run_autonomous_probe
from cls_option_tutor.eval.learning_increment_metrics import (
    compute_learning_increment,
)
from cls_option_tutor.eval.local_probe import (
    run_local_probe, compute_local_learning,
)
from cls_option_tutor.experiments.condition_overrides import (
    apply_condition_overrides as _condition_overrides_impl,
    extract_scripted_protocol_name,
    resolve_condition_alias,
)
from cls_option_tutor.experiments.metrics_extractors import (
    compute_6e_metrics as _compute_6e_metrics_impl,
    compute_6fg_metrics as _compute_6fg_metrics_impl,
)
from cls_option_tutor.experiments.reporting import (
    bootstrap_ci as _bootstrap_ci_impl,
    print_row as _print_row_impl,
    write_summary as _write_summary_impl,
)
from cls_option_tutor.experiments.runner_io import (
    build_jobs as _build_jobs_impl,
    write_rows_csv as _write_rows_csv_impl,
)

# â”€â”€ Default sweep parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')
)

# Phase 6.4 core conditions
DEFAULT_CONDITIONS = [
    "direct_answer",
    "script_direct_correct",
    # then_answer: wrong â†’ SHORTLIST â†’ learner acts
    "script_wrong1_then_answer_safe",
    "script_wrong1_then_answer_bounded",
    "script_wrong1_then_answer_high",
    # self_correct: wrong â†’ force correct (no SHORTLIST)
    "script_wrong1_self_correct_safe",
    "script_wrong1_self_correct_bounded",
    "script_wrong1_self_correct_high",
    "script_wrong2_self_correct_safe",
    "script_wrong2_self_correct_safe_bounded",
    # diagnostic vs random
    "script_wrong1_self_correct_diagnostic_safe",
    "script_wrong1_self_correct_random_safe",
    # baseline
    "no_tutor_reveal",
]

SMOKE_CONDITIONS = [
    "direct_answer",
    "script_wrong1_then_answer_safe",
    "script_wrong1_self_correct_safe",
    "script_wrong1_self_correct_diagnostic_safe",
    "no_tutor_reveal",
]

DEFAULT_RHO_VALUES = [0.3, 1.0]
DEFAULT_N_SUP = [4]
DEFAULT_SEEDS = [42, 123, 7]
SMOKE_SEEDS = [42]

N_PROBE_GLOBAL = 20   # Phase 6.4: increased from 15
N_PROBE_LOCAL = 8     # Phase 6.4: new local probe


# â”€â”€ Config factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def make_cfg(
    n_sup: int = 4,
    rho_assist: float = 1.0,
    generator_mode: str = "v2_overlap",
    tutor_lg_mode: str = "off",
    highlight_mode: str = "diagnostic",
) -> FullConfig:
    """Build baseline config for micro benchmark."""
    cfg = FullConfig()
    cfg.learner.use_cls = True
    cfg.learner.n_sup = n_sup
    cfg.learner.n_em = 1
    cfg.learner.use_hpc = False
    cfg.learner.rho_assist = rho_assist
    cfg.learner.correct_pick_learning_mode = "cortex_em"
    cfg.env.K = 6
    cfg.env.T_max = 3
    cfg.env.N_obs = 1
    cfg.env.N_teach = 6    # Phase 6.4: increased from 3
    cfg.env.N_eval = 2
    cfg.env.M_queries = 9  # 1 obs + 6 teach + 2 eval
    cfg.env.n_risky = 2
    cfg.env.generator_mode = generator_mode  # Phase 6E
    cfg.env.highlight_mode = highlight_mode  # Phase 6F
    cfg.tutor.rollout_mode = "proxy"
    cfg.tutor.tutor_lg_mode = tutor_lg_mode  # Phase 6G
    return cfg


# â”€â”€ Run one condition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_one_job(job: Tuple) -> dict:
    """Top-level picklable wrapper for ProcessPoolExecutor."""
    task_id, seed, condition, n_sup, rho_assist, job_idx, total, generator_mode, tutor_lg_mode, highlight_mode = job
    try:
        cfg = make_cfg(n_sup=n_sup, rho_assist=rho_assist,
                       generator_mode=generator_mode, tutor_lg_mode=tutor_lg_mode,
                       highlight_mode=highlight_mode)
        row = run_one(task_id, seed, condition, cfg)
        row["_job_idx"] = job_idx
        return row
    except Exception as e:
        import traceback
        return {
            "task_id": task_id, "seed": seed,
            "condition": condition, "rho_assist": rho_assist,
            "n_sup": n_sup, "ERROR": str(e),
            "_job_idx": job_idx,
        }


# â”€â”€ Phase 6G: condition-name â†’ config mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _apply_condition_overrides(cfg, condition: str):
    """Thin wrapper around the extracted condition-override helper."""
    return _condition_overrides_impl(cfg, condition)


def _compute_6fg_metrics(block) -> dict:
    """Thin wrapper around extracted block-level tutor metrics."""
    return _compute_6fg_metrics_impl(block)


# â”€â”€ Phase 6H.6: condition alias mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_one(task_id, seed, condition, cfg):
    """Run one (task, seed, condition) and return full row dict."""
    t0 = time.time()

    # Phase 6H.6: resolve condition alias (e.g. short name â†’ scripted protocol)
    raw_condition = condition
    condition = resolve_condition_alias(condition)

    # P0: apply generic condition overrides before constructing env/learner so
    # scripted/no_tutor conditions also inherit runtime tags like tmax/er/cpoff.
    cfg = _apply_condition_overrides(copy.deepcopy(cfg), condition)

    # Phase 6H.5: scale rho_H by highlight_strength before constructing learner
    hl_strength = getattr(cfg.env, 'highlight_strength', 1.0)
    if hl_strength != 1.0:
        import copy as _copy
        cfg = _copy.deepcopy(cfg)
        # rho_H controls exp(rho_H) attention boost â†’ multiply the exponent
        # For 2x: rho_H = rho_H + log(2), for 4x: rho_H = rho_H + log(4)
        import math as _math
        boost_delta = _math.log(hl_strength)
        cfg.learner.rho_H = cfg.learner.rho_H + boost_delta
        cfg.tutor.tutor_rho_H = cfg.tutor.tutor_rho_H + boost_delta

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    learner = LearnerAgent(cfg=cfg, seed=seed)

    # Initialize learner with scorer/danger_head before any probing
    support, _, grammar = env.adapter.load_task(task_id)
    init_block = env.reset_block(task_id, seed=seed)
    learner.init_block(init_block, grammar, support)

    # â”€â”€ GLOBAL PRE-PROBE â”€â”€
    probe_seed = hash((task_id, seed, "phase6_probe")) % (2**31)
    pre_probe = run_autonomous_probe(
        learner, env, task_id, probe_seed=probe_seed, cfg=cfg,
        n_probe=N_PROBE_GLOBAL,
        freeze_semantic=True, freeze_risk=True, freeze_memory=True,
    )

    # â”€â”€ LOCAL PRE-PROBE â”€â”€
    local_seed = hash((task_id, seed, "local_probe")) % (2**31)
    try:
        local_pre = run_local_probe(
            learner, env, task_id, cfg=cfg,
            probe_seed=local_seed, n_local=N_PROBE_LOCAL,
        )
    except Exception:
        local_pre = None

    # ── TEACH PHASE ──
    scripted_result = None

    if condition == "direct_answer":
        teach_cfg = cfg
        effective_cfg = teach_cfg
        da_tutor = DirectAnswerTutor(cfg=teach_cfg)
        block = da_tutor.run_block(
            OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
            learner,
            task_id,
            seed=seed,
        )
    elif condition.startswith("script_") or condition.startswith("no_tutor_"):
        protocol = extract_scripted_protocol_name(condition)
        if protocol == "no_tutor_nonreveal_neg":
            teach_cfg = copy.deepcopy(cfg)
            teach_cfg.env.feedback_mode = "nonreveal"
            teach_cfg.learner.reveal_learning_mode = "nonreveal_negative"
            teach_cfg.learner.negative_evidence_mode = "exact_program_target"
        else:
            teach_cfg = cfg
        effective_cfg = teach_cfg
        runner = ScriptedProtocolRunner(cfg=teach_cfg, protocol=protocol)
        scripted_result = runner.run_block(
            OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
            learner,
            task_id,
            seed=seed,
        )
        block = scripted_result.block
    else:
        # Generic overrides have already been applied before learner construction.
        teach_cfg = cfg
        effective_cfg = teach_cfg
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
        tutor = SparseTutorAgent(cfg=teach_cfg)
        block = tutor.run_block(
            OptionEnv(cfg=teach_cfg, data_dir=DATA_DIR),
            learner,
            task_id,
            seed=seed,
        )

    # -- Phase 6H.6 / 6I-A: Causal Audit (batch from decision-time snapshots) --
    audit_summary = None
    audit_candidates = getattr(block, '_audit_candidates', [])
    if audit_candidates:
        from cls_option_tutor.tutor.causal_audit import (
            audit_post_reveal_action_effects, AuditSummary,
        )
        audit_summary = AuditSummary()
        for ac in audit_candidates:
            results = audit_post_reveal_action_effects(
                learner=ac["learner_snapshot"],
                qs=ac["qs_snapshot"],
                active=ac["active"],
                diag_labels=ac.get("labels"),
                highlight_cells_diagnostic=ac.get("hl_cells_diagnostic"),
                highlight_cells_fixed=ac.get("hl_cells_fixed"),
                highlight_cells_counterfactual=ac.get("hl_cells_counterfactual"),
                highlight_strength=getattr(effective_cfg.env, 'highlight_strength', 1.0),
            )
            for r in results:
                audit_summary.results_by_action.setdefault(
                    r.action_name, []).append(r)
            # 6I-A: store per-state result list for CATE aggregation
            audit_summary.per_state_results.append(results)
            audit_summary.n_states += 1

    # â”€â”€ GLOBAL POST-PROBE â”€â”€
    env_post = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    post_probe = run_autonomous_probe(
        learner, env_post, task_id, probe_seed=probe_seed, cfg=cfg,
        n_probe=N_PROBE_GLOBAL,
        freeze_semantic=True, freeze_risk=True, freeze_memory=True,
    )

    # â”€â”€ LOCAL POST-PROBE â”€â”€
    try:
        local_post = run_local_probe(
            learner, env_post, task_id, cfg=cfg,
            probe_seed=local_seed, n_local=N_PROBE_LOCAL,
        )
    except Exception:
        local_post = None

    # â”€â”€ Compute teach metrics â”€â”€
    obs_q = block.obs_phase_queries
    teach_q = block.teach_phase_queries
    teach_queries = block.queries[obs_q: obs_q + teach_q]
    eval_queries = block.queries[obs_q + teach_q:]
    n_teach = len(teach_queries)
    teach_sr = sum(1 for q in teach_queries if q.success) / max(n_teach, 1)

    # â”€â”€ Phase-specific damage â”€â”€
    obs_damage = 0
    teach_damage = 0
    eval_damage = 0
    for ls in block.learner_trace:
        dmg = ls.damage if ls.damage is not None else 0
        qid = ls.query_id if hasattr(ls, 'query_id') else None
        # Determine which phase this step belongs to
        if qid is not None:
            if qid < obs_q:
                obs_damage += dmg
            elif qid < obs_q + teach_q:
                teach_damage += dmg
            else:
                eval_damage += dmg
        else:
            teach_damage += dmg  # default to teach

    # Scripted vs incidental teach damage
    scripted_damage = 0
    if scripted_result is not None:
        for idx in scripted_result.forced_step_indices:
            if idx < len(block.learner_trace):
                d = block.learner_trace[idx].damage
                scripted_damage += (d if d is not None else 0)
    incidental_damage = teach_damage - scripted_damage

    total_damage = obs_damage + teach_damage + eval_damage
    death_count = sum(1 for q in teach_queries if q.hp <= 0 and not q.success)
    timeout_count = sum(
        1 for q in teach_queries if q.hp > 0 and not q.success and not q.skipped
    )
    death_rate = death_count / max(n_teach, 1)
    timeout_rate = timeout_count / max(n_teach, 1)
    n_interventions = sum(1 for ts in block.tutor_trace if ts.action != "WAIT")

    # â”€â”€ Scripted counters â”€â”€
    observed = _extract_observed_wrong_reveal_stats(block, teach_queries)
    wr_count = observed["wrong_reveal_count"]
    safe_wr = observed["safe_wrong_count"]
    risky_wr = observed["risky_wrong_count"]
    corr_count = observed["correct_pick_count"]
    da_count = sum(1 for ts in block.tutor_trace if ts.action == "SHORTLIST")
    skip_count = sum(1 for q in teach_queries if q.skipped)
    wr_risk = observed["wr_risk"]
    corr_after_wr = observed["correct_after_wrong"]
    death_before_corr = observed["death_before_correct"]
    protocol_wr_risk = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    protocol_corr_after_wr = 0
    protocol_death_before_corr = 0
    script_violations = 0
    then_ans_count = 0
    self_corr_count = 0

    if scripted_result is not None:
        protocol_wr_risk[0] = scripted_result.wrong_reveal_risk0
        protocol_wr_risk[1] = scripted_result.wrong_reveal_risk1
        protocol_wr_risk[2] = scripted_result.wrong_reveal_risk2
        protocol_wr_risk[3] = scripted_result.wrong_reveal_risk3
        protocol_wr_risk[4] = scripted_result.wrong_reveal_risk4
        protocol_corr_after_wr = scripted_result.correct_after_wrong_count
        protocol_death_before_corr = scripted_result.death_before_correct_count
        script_violations = scripted_result.script_violation_count
        then_ans_count = scripted_result.then_answer_count
        self_corr_count = scripted_result.self_correct_count

    # â”€â”€ Semantic update counters â”€â”€
    sem = getattr(learner, '_sem_counters', {})
    src = getattr(learner, '_src_counters', {})

    # â”€â”€ Learning increment â”€â”€
    li = compute_learning_increment(
        pre_probe=pre_probe,
        post_probe=post_probe,
        teach_sr=teach_sr,
        damage_sum=float(total_damage),
        death_rate=death_rate,
        timeout_rate=timeout_rate,
        n_interventions=n_interventions,
        wrong_reveal_count=wr_count,
        correct_pick_count=corr_count,
    )

    # â”€â”€ Local learning â”€â”€
    local_ll = None
    if local_pre is not None and local_post is not None:
        local_ll = compute_local_learning(local_pre, local_post)

    # â”€â”€ Derived phase-specific LPD â”€â”€
    lpd_teach = li.delta_probe_sr / (1 + teach_damage)
    lpd_script = li.delta_probe_sr / (1 + scripted_damage)

    elapsed = time.time() - t0

    # â”€â”€ Phase 6H.6: strict quota diagnostics from teach queries â”€â”€
    strict_attempt_sum = 0
    strict_satisfied_count = 0
    strict_fallback_count = 0
    strict_n_queries = 0
    for qs in teach_queries:
        qi = getattr(qs, '_quota_info', None)
        if qi and qi.get('strict_mode', False):
            strict_n_queries += 1
            strict_attempt_sum += qi.get('strict_attempt_count', 1)
            if qi.get('strict_quota_satisfied', False):
                strict_satisfied_count += 1
            if qi.get('strict_fallback', False):
                strict_fallback_count += 1

    productive_allow_planning = bool(effective_cfg.tutor.productive_allow_planning)
    productive_allow_mode = (
        effective_cfg.tutor.productive_allow_mode
        if productive_allow_planning else "N/A"
    )
    refresh_cap_potentially_binding = bool(
        effective_cfg.env.enforce_max_refreshes
        and effective_cfg.env.max_refreshes < max(effective_cfg.env.T_max - 1, 0)
    )
    productive_allow_diag_failures = int(
        getattr(block, "_productive_allow_diagnostic_failures", 0)
    )

    row = {
        "task_id": task_id,
        "seed": seed,
        "condition": raw_condition,
        "ConditionEffective": condition,
        "rho_assist": cfg.learner.rho_assist,
        "n_sup": cfg.learner.n_sup,
        "feedback_mode": cfg.env.feedback_mode,
        # Global probe
        "PreProbeSR": pre_probe.sr,
        "PostProbeSR": post_probe.sr,
        "DeltaProbeSR": li.delta_probe_sr,
        "PreFirstOK": pre_probe.first_ok,
        "PostFirstOK": post_probe.first_ok,
        "DeltaFirstOK": li.delta_first_ok,
        # Teach
        "TeachSR": teach_sr,
        # Phase-specific damage
        "ObsDamage": obs_damage,
        "TeachDamage": teach_damage,
        "ScriptedDamage": scripted_damage,
        "IncidentalDamage": incidental_damage,
        "EvalDamage": eval_damage,
        "TotalDamage": total_damage,
        "DeathRate": death_rate,
        "TimeoutRate": timeout_rate,
        # Efficiency
        "LPD_global": li.learning_per_damage,
        "LPD_teach": lpd_teach,
        "LPD_script": lpd_script,
        "LPI": li.learning_per_intervention,
        "AssistGap": li.assist_gap,
        # Scripted counts
        "WrongRevealCount": wr_count,
        "SafeWrongCount": safe_wr,
        "RiskyWrongCount": risky_wr,
        "CorrectPickCount": corr_count,
        "DirectAnswerCount": da_count,
        "InterventionCount": n_interventions,
        "SkippedQueryCount": skip_count,
        # Per-risk-class wrong reveals observed from actual learner picks
        "WR_Risk0": wr_risk[0],
        "WR_Risk1": wr_risk[1],
        "WR_Risk2": wr_risk[2],
        "WR_Risk3": wr_risk[3],
        "WR_Risk4": wr_risk[4],
        "CorrectAfterWrong": corr_after_wr,
        "DeathBeforeCorrect": death_before_corr,
        # Protocol/scripted-only semantics kept separate for clarity
        "Protocol_WR_Risk0": protocol_wr_risk[0],
        "Protocol_WR_Risk1": protocol_wr_risk[1],
        "Protocol_WR_Risk2": protocol_wr_risk[2],
        "Protocol_WR_Risk3": protocol_wr_risk[3],
        "Protocol_WR_Risk4": protocol_wr_risk[4],
        "Protocol_CorrectAfterWrong": protocol_corr_after_wr,
        "Protocol_DeathBeforeCorrect": protocol_death_before_corr,
        "ScriptViolations": script_violations,
        # Phase 6.4: experience semantics
        "ThenAnswerCount": then_ans_count,
        "SelfCorrectCount": self_corr_count,
        "Protocol_ThenAnswerCount": then_ans_count,
        "Protocol_SelfCorrectCount": self_corr_count,
        # Semantic update counters (coarse)
        "SemWR_att": sem.get("wrong_reveal_attempted", 0),
        "SemWR_app": sem.get("wrong_reveal_applied", 0),
        "SemCU_att": sem.get("correct_unassisted_attempted", 0),
        "SemCU_app": sem.get("correct_unassisted_applied", 0),
        "SemCA_att": sem.get("correct_assisted_attempted", 0),
        "SemCA_app": sem.get("correct_assisted_applied", 0),
        "SemDA_att": sem.get("direct_answer_attempted", 0),
        "SemDA_app": sem.get("direct_answer_applied", 0),
        # Phase 6.5: event-source counters
        "SrcWR_scr_att": src.get("wr_scripted_att", 0),
        "SrcWR_scr_app": src.get("wr_scripted_app", 0),
        "SrcWR_inc_att": src.get("wr_incidental_att", 0),
        "SrcWR_inc_app": src.get("wr_incidental_app", 0),
        "SrcCU_SC_att": src.get("cu_scripted_self_correct_att", 0),
        "SrcCU_SC_app": src.get("cu_scripted_self_correct_app", 0),
        "SrcCU_DC_att": src.get("cu_scripted_direct_correct_att", 0),
        "SrcCU_DC_app": src.get("cu_scripted_direct_correct_app", 0),
        "SrcCU_inc_att": src.get("cu_incidental_att", 0),
        "SrcCU_inc_app": src.get("cu_incidental_app", 0),
        "SrcDA_DA_att": src.get("da_direct_answer_att", 0),
        "SrcDA_DA_app": src.get("da_direct_answer_app", 0),
        "SrcDA_TA_att": src.get("da_then_answer_att", 0),
        "SrcDA_TA_app": src.get("da_then_answer_app", 0),
        "SrcDA_inc_att": src.get("da_incidental_shortlist_att", 0),
        "SrcDA_inc_app": src.get("da_incidental_shortlist_app", 0),
        # Local probe
        "LocalPreSR": local_pre.sr if local_pre else "",
        "LocalPostSR": local_post.sr if local_post else "",
        "DeltaLocalSR": local_ll.delta_local_sr if local_ll else "",
        "DeltaLocalFirstOK": local_ll.delta_local_first_ok if local_ll else "",
        "DeltaSemanticMargin": local_ll.delta_semantic_margin if local_ll else "",
        "DeltaPolicyMargin": local_ll.delta_policy_margin if local_ll else "",
        "DeltaSemanticRank": local_ll.delta_semantic_rank if local_ll else "",
        "DeltaPolicyRank": local_ll.delta_policy_rank if local_ll else "",
        "DeltaCorrectProb": local_ll.delta_correct_prob if local_ll else "",
        "PreSemanticMargin": local_pre.avg_semantic_margin if local_pre else "",
        "PostSemanticMargin": local_post.avg_semantic_margin if local_post else "",
        "PrePolicyMargin": local_pre.avg_policy_margin if local_pre else "",
        "PostPolicyMargin": local_post.avg_policy_margin if local_post else "",
        "PreCorrectProb": local_pre.avg_correct_pick_prob if local_pre else "",
        "PostCorrectProb": local_post.avg_correct_pick_prob if local_post else "",
        # Phase 6E: generator diagnostics â€” use effective_cfg (actual teach config)
        "GeneratorMode": effective_cfg.env.generator_mode,
        "TMax": effective_cfg.env.T_max,
        "MaxRefreshes": effective_cfg.env.max_refreshes,
        "TutorLGMode": effective_cfg.tutor.tutor_lg_mode,
        "HighlightMode": effective_cfg.env.highlight_mode,
        "HighlightStrength": getattr(effective_cfg.env, 'highlight_strength', 1.0),
        "EtaReveal": effective_cfg.learner.eta_reveal,
        "CorrectPickLearningMode": effective_cfg.learner.correct_pick_learning_mode,
        "PedagogicalFeedbackMode": getattr(effective_cfg.learner, 'pedagogical_feedback_mode', 'raw'),
        "EnforceMaxRefreshes": effective_cfg.env.enforce_max_refreshes,
        "RefreshCapPotentiallyBinding": refresh_cap_potentially_binding,
        "MaxHighlightCells": getattr(effective_cfg.tutor, 'max_highlight_cells', 2),
        "DiagnosticQuotaStrict": getattr(effective_cfg.env, 'diagnostic_quota_strict', False),
        "ProtectSafeDiagHardGuard": getattr(effective_cfg.tutor, 'protect_safe_diag_hard_guard', False),
        "PostRevealValueMode": getattr(effective_cfg.tutor, 'postreveal_value_mode', 'legacy'),
        "UsePostRevealConsolidationValue": getattr(
            effective_cfg.tutor, 'use_postreveal_consolidation_value', False
        ),
        "MixTargetMode": getattr(effective_cfg.tutor, 'mix_target_mode', 'current'),
        "PostRevealInfoWeight": getattr(effective_cfg.tutor, 'postreveal_info_weight', 0.0),
        "UseBayesianPostRevealValue": getattr(effective_cfg.tutor, 'use_bayesian_postreveal_value', False),
        "JointMixReplayGate": getattr(effective_cfg.tutor, 'joint_mix_replay_gate', False),
        "DirectMixSelector": getattr(effective_cfg.tutor, 'direct_mix_selector', False),
        "ProductiveAllowPlanning": productive_allow_planning,
        "ProductiveAllowMode": productive_allow_mode,
        "ProductiveAllowDiagnosticFailureCount": productive_allow_diag_failures,
        **_compute_6e_metrics(block),
        # Phase 6F/6G/6H.5: tutor decision metrics
        **_compute_6fg_metrics(block),
        # Phase 6H.6: strict quota diagnostics
        "StrictAttemptMean": strict_attempt_sum / max(strict_n_queries, 1) if strict_n_queries > 0 else "",
        "StrictQuotaSatisfiedRate": strict_satisfied_count / max(strict_n_queries, 1) if strict_n_queries > 0 else "",
        "StrictFallbackRate": strict_fallback_count / max(strict_n_queries, 1) if strict_n_queries > 0 else "",
        # Phase 6H.6: causal audit
        **(audit_summary.to_dict() if audit_summary else {"AuditNStates": 0}),
        "ElapsedSec": round(elapsed, 2),
    }
    return row


def _compute_6e_metrics(block) -> dict:
    """Thin wrapper around extracted block-level diagnostic metrics."""
    return _compute_6e_metrics_impl(block)


def _find_option_for_pick(qs, pick_index):
    if qs is None or pick_index is None:
        return None
    for opt in getattr(qs, "menu", []) or []:
        if getattr(opt, "index", None) == pick_index:
            return opt
    return None


def _extract_observed_wrong_reveal_stats(block, teach_queries):
    """Observed wrong-reveal stats from actual learner behavior.

    Unlike scripted protocol counters, these are defined for every condition.
    A "wrong reveal" here means a learner `pick` that is incorrect during the
    teaching phase; risk classes come from the picked option itself.
    """
    wr_risk = {k: 0 for k in range(5)}
    wr_count = 0
    safe_wr = 0
    risky_wr = 0
    corr_count = 0
    corr_after_wr = 0
    death_before_corr = 0

    teach_qids = {qs.query_id for qs in teach_queries}
    trace_by_qid = {}
    for ls in getattr(block, "learner_trace", []) or []:
        qid = getattr(ls, "query_id", None)
        if qid in teach_qids:
            trace_by_qid.setdefault(qid, []).append(ls)

    for qs in teach_queries:
        qtrace = sorted(
            trace_by_qid.get(qs.query_id, []),
            key=lambda ls: (getattr(ls, "round_t", -1), getattr(ls, "action", "")),
        )
        saw_wrong = False
        saw_correct = False
        for ls in qtrace:
            if getattr(ls, "action", None) != "pick":
                continue
            if bool(getattr(ls, "correct", False)):
                corr_count += 1
                saw_correct = True
                continue
            wr_count += 1
            saw_wrong = True
            opt = _find_option_for_pick(qs, getattr(ls, "pick_index", None))
            risk_class = int(getattr(opt, "risk_class", 0) if opt is not None else 0)
            risk_class = max(0, min(4, risk_class))
            wr_risk[risk_class] += 1
            if risk_class == 0:
                safe_wr += 1
            else:
                risky_wr += 1
        if saw_wrong and qs.success:
            corr_after_wr += 1
        if saw_wrong and (not qs.success) and getattr(qs, "hp", 1) <= 0:
            death_before_corr += 1

    return {
        "wrong_reveal_count": wr_count,
        "safe_wrong_count": safe_wr,
        "risky_wrong_count": risky_wr,
        "correct_pick_count": corr_count,
        "wr_risk": wr_risk,
        "correct_after_wrong": corr_after_wr,
        "death_before_correct": death_before_corr,
    }


def _print_row(row, total):
    """Thin wrapper around extracted row-printing helper."""
    return _print_row_impl(row, total)


def _bootstrap_ci(values, n_boot=2000, alpha=0.05):
    """Thin wrapper around extracted bootstrap helper."""
    return _bootstrap_ci_impl(values, n_boot=n_boot, alpha=alpha)


def _write_summary(rows, out_dir, ts_str, conditions, rho_values):
    """Thin wrapper around extracted summary writer."""
    del rho_values  # summary groups by values present in rows
    return _write_summary_impl(
        rows,
        out_dir,
        ts_str,
        conditions,
        n_probe_global=N_PROBE_GLOBAL,
        n_probe_local=N_PROBE_LOCAL,
    )


# â”€â”€ Main sweep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="E6.4+ Experience-Semantic Micro Benchmark")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rho", nargs="+", type=float, default=None)
    parser.add_argument("--n-sup", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    # Phase 6E/6F/6G
    parser.add_argument("--generator", type=str, default="v2_overlap",
                        choices=["v2_overlap", "diagnostic_quota",
                                 "diagnostic_quota_no_bounded",
                                 "diagnostic_quota_no_high_lure",
                                 "diagnostic_quota_allow_heavy",
                                 "diagnostic_quota_mixed_prod_harm_heavy",
                                 "diagnostic_quota_protect_critical_heavy",
                                 "diagnostic_quota_boring_mastery_heavy"])
    parser.add_argument("--lg-mode", type=str, default="off",
                        choices=["off", "diagnostic", "safety_only", "learning_only", "self_correct"])
    parser.add_argument("--highlight-mode", type=str, default="diagnostic",
                        choices=["diagnostic", "fixed", "none"])
    args = parser.parse_args()

    if args.smoke:
        task_ids = ["000001", "000002"]
        seeds = SMOKE_SEEDS
        conditions = SMOKE_CONDITIONS
        rho_values = [1.0]
        n_sup_values = [4]
    else:
        available = sorted(
            f.replace(".txt", "")
            for f in os.listdir(DATA_DIR)
            if f.endswith(".txt")
        )
        task_ids = args.tasks or available[:20]
        seeds = args.seeds or DEFAULT_SEEDS
        conditions = args.conditions or DEFAULT_CONDITIONS
        rho_values = args.rho or DEFAULT_RHO_VALUES
        n_sup_values = args.n_sup or DEFAULT_N_SUP

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'e6_micro')
    os.makedirs(out_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(out_dir, f"e64_micro_{ts_str}.csv")

    jobs = _build_jobs_impl(
        task_ids,
        seeds,
        conditions,
        n_sup_values,
        rho_values,
        generator_mode=args.generator,
        lg_mode=args.lg_mode,
        highlight_mode=args.highlight_mode,
    )
    total = len(jobs)

    workers = min(args.workers, total)
    print(f"E6.4 Experience-Semantic Micro Benchmark: {total} runs | {workers} worker(s)")
    print(f"  Tasks:       {len(task_ids)}")
    print(f"  Seeds:       {seeds}")
    print(f"  Conditions:  {conditions}")
    print(f"  rho_assist:  {rho_values}")
    print(f"  n_sup:       {n_sup_values}")
    print(f"  N_PROBE_G:   {N_PROBE_GLOBAL}")
    print(f"  N_PROBE_L:   {N_PROBE_LOCAL}")
    print(f"  N_teach:     6")
    print(f"  Output:      {out_path}")
    print()

    rows: List[dict] = []
    t_start = time.time()

    if workers <= 1:
        for job in jobs:
            row = _run_one_job(job)
            rows.append(row)
            _print_row_impl(row, total)
    else:
        print(f"Launching {workers} processes...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one_job, job): job for job in jobs}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _print_row_impl(row, total)

    rows.sort(key=lambda r: r.get("_job_idx", 0))

    fieldnames = _write_rows_csv_impl(rows, out_path)
    if fieldnames is None:
        print("No successful runs!")
        return

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in rows if "ERROR" not in r)
    n_err = len(rows) - n_ok
    print(f"\nDone. {n_ok} ok, {n_err} errors. {elapsed:.1f}s total.")
    print(f"CSV: {out_path}")

    _write_summary_impl(
        rows,
        out_dir,
        ts_str,
        conditions,
        n_probe_global=N_PROBE_GLOBAL,
        n_probe_local=N_PROBE_LOCAL,
    )

if __name__ == "__main__":
    main()

