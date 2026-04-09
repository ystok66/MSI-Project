"""
run_diagnostic.py — V2 diagnostic with learner levels.

Learner levels:
  L0: n_sup=0  (no prior examples — raw CLS prior)
  L1: n_sup=1  (1 support example)
  L2: n_sup=3  (3 support examples)
  L3: n_sup=5  (5 support examples — current default)

Usage:
    python cls_option_tutor/run_diagnostic.py --all --grammar 000001 --seed 1
    python cls_option_tutor/run_diagnostic.py --condition E-B4_full_tutor --level 3 --seed 1
    python cls_option_tutor/run_diagnostic.py --sweep-levels --grammar 000001 --seed 1
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cls_option_tutor.config import FullConfig, TutorConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.state import BlockState, ProfileState
from cls_option_tutor.env.interventions import get_active_menu
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.learner.cls_adapter import create_scorer
from cls_option_tutor.learner.semantic_scorer import DeterministicSemanticScorer
from cls_option_tutor.learner.policy import LearnerPolicy
from cls_option_tutor.tutor.tutor_agent import TutorAgent
from cls_option_tutor.tutor.counterfactual import CounterfactualScorer
from cls_option_tutor.tutor.tutor_policy import TutorPolicy
from cls_option_tutor.diagnostic_logger import DiagnosticLogger

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'BASIC', 'cls_learner', 'data')

# Learner levels: n_sup examples given to CLS
LEARNER_LEVELS = {0: 0, 1: 1, 2: 3, 3: 5}

CONDITIONS = {
    "E-B1_no_tutor": {"tutor_enabled": False, "use_cls": True},
    "E-B1_oracle":   {"tutor_enabled": False, "use_cls": False},
    "E-B3_highlight": {
        "tutor_enabled": True, "use_cls": True,
        "tutor_overrides": {"c_hint": 100.0, "c_skip": 100.0},
    },
    "E-B4_full_tutor": {"tutor_enabled": True, "use_cls": True},
}


def run_diagnostic_block(
    condition: str,
    grammar_id: str,
    seed: int,
    learner_level: int = 3,
) -> str:
    """Run one block with full diagnostic logging."""
    cond_cfg = CONDITIONS.get(condition, CONDITIONS["E-B4_full_tutor"])

    cfg = FullConfig()
    use_cls = cond_cfg.get("use_cls", True)
    cfg.learner.use_cls = use_cls

    # Set learner level via n_sup
    n_sup = LEARNER_LEVELS.get(learner_level, 5)
    cfg.learner.n_sup = n_sup

    if "tutor_overrides" in cond_cfg:
        for k, v in cond_cfg["tutor_overrides"].items():
            setattr(cfg.tutor, k, v)

    # Setup
    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    block = env.reset_block(grammar_id, seed=seed)
    support, _, grammar = env.adapter.load_task(grammar_id)

    # Create learner
    learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=use_cls)
    learner.init_block(block, grammar, support)

    # Oracle scorer for comparison
    oracle_scorer = DeterministicSemanticScorer(grammar, cfg.learner.tau_sem)

    # Logger
    tag = f"{condition}_L{learner_level}"
    logger = DiagnosticLogger(tag, grammar_id, seed)
    logger.set_oracle_scorer(oracle_scorer)

    # Tutor (if enabled)
    tutor_enabled = cond_cfg.get("tutor_enabled", False)
    tutor = None
    cf_scorer = None
    if tutor_enabled:
        tutor = TutorAgent(cfg=cfg)
        tutor.init_block(block, grammar, support)
        cf_scorer = CounterfactualScorer(cfg.tutor)

    cls_active = (learner._scorer.is_cls_active
                  if hasattr(learner._scorer, 'is_cls_active') else 'N/A')
    print(f"Running {tag} g={grammar_id} s={seed} n_sup={n_sup} cls={cls_active}")

    max_steps = len(block.queries) * 20
    step_count = 0

    while not block.done and step_count < max_steps:
        step_count += 1
        qs = block.current_query
        if qs is None or qs.done:
            break

        # ── Tutor acts ──
        if tutor_enabled and tutor:
            active = get_active_menu(qs)
            if active:
                target = qs.target_output
                K = len(active)
                t_sem = np.array([
                    oracle_scorer.score_option(target, o.text) for o in active])
                t_danger = np.zeros(K)
                t_ko = np.zeros(K)
                if tutor._danger_head:
                    for i, o in enumerate(active):
                        mu, u = tutor._danger_head.predict(o.danger_vec)
                        t_danger[i] = mu
                        t_ko[i] = tutor._danger_head.predict_ko_prob(
                            o.danger_vec, qs.hp)

                profile = getattr(block, 'profile_state', ProfileState())
                beta = 4.0
                sc = profile.semantic_competence
                shifted = sc * t_sem - t_danger
                shifted = shifted - np.max(shifted)
                t_p_pick = np.exp(beta * shifted)
                t_p_pick = t_p_pick / (t_p_pick.sum() + 1e-10)
                t_e_dmg = float(t_p_pick @ t_danger)

                candidates = cf_scorer.score_all(
                    qs, profile, oracle_scorer, tutor._danger_head)

                best = candidates[0]
                action = best.action
                kwargs = {}
                if best.ban_index is not None:
                    kwargs["ban_index"] = best.ban_index
                if best.hint_index is not None:
                    kwargs["hint_index"] = best.hint_index
                if best.highlight_cells is not None:
                    kwargs["cells"] = best.highlight_cells

                phase = ("observation" if block.in_observation_phase
                         else "teaching" if block.in_teaching_phase
                         else "evaluation")

                logger.log_tutor_step(
                    qs, candidates, action, kwargs,
                    profile, t_sem, t_danger, t_p_pick, t_e_dmg,
                    phase=phase)

            tutor.act(block, env)
        else:
            env.tutor_act(block, "WAIT")

        if qs.done:
            continue

        # ── Learner acts ──
        policy_out = learner.act(block, env)

        if policy_out is not None:
            active_opts = get_active_menu(qs)
            logger.log_learner_step(qs, policy_out, learner.policy, active_opts)

            if block.learner_trace:
                last_step = block.learner_trace[-1]
                picked_opt = None
                if last_step.action == "pick" and last_step.pick_index is not None:
                    for o in qs.menu:
                        if o.index == last_step.pick_index:
                            picked_opt = o
                            break
                logger.log_outcome(qs, last_step, picked_opt)

        if tutor_enabled and tutor:
            tutor.observe_learner_outcome(block)

    # ── Query summaries ──
    for qs in block.queries:
        logger.log_query_summary(qs, learner._scorer)

    # ── Save ──
    out_dir = os.path.join(os.path.dirname(__file__), 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"diag_{tag}_g{grammar_id}_s{seed}.json")
    logger.log.save(out_path)

    # Print summary
    metrics = OptionEnv.get_block_metrics(block)
    print(f"  SR={metrics['solve_rate']:.2f} DMG={metrics['total_damage']} "
          f"Rounds={metrics['total_rounds']}")
    print(f"  Steps logged: L={len(logger.log.learner_steps)} "
          f"T={len(logger.log.tutor_steps)} O={len(logger.log.outcomes)}")
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="E-B4_full_tutor",
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--grammar", default="000001")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--level", type=int, default=3,
                        choices=[0, 1, 2, 3],
                        help="Learner level (0=no examples, 3=5 examples)")
    parser.add_argument("--all", action="store_true",
                        help="Run all conditions at current level")
    parser.add_argument("--sweep-levels", action="store_true",
                        help="Run all levels x conditions")
    args = parser.parse_args()

    if args.sweep_levels:
        for level in [0, 1, 2, 3]:
            for cond in CONDITIONS:
                run_diagnostic_block(cond, args.grammar, args.seed,
                                     learner_level=level)
                print()
    elif args.all:
        for cond in CONDITIONS:
            run_diagnostic_block(cond, args.grammar, args.seed,
                                 learner_level=args.level)
            print()
    else:
        run_diagnostic_block(args.condition, args.grammar, args.seed,
                             learner_level=args.level)


if __name__ == "__main__":
    main()
