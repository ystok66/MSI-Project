"""
tutor_agent.py — Autonomous tutor agent for the CLS Option Tutor.

Orchestrates the full tutor lifecycle:
    1. Observation phase → WAIT + collect learner trace
    2. Profile inference from observed behavior
    3. Teaching phase → counterfactual scoring → intervention selection
    4. Outcome tracking for evaluation

P0 eval-aware mode:
    Tutor maintains a shadow CLS-based pedagogical simulator
    initialized from the same grammar/support and updated only
    with externally observable evidence. Used for action ranking
    on expected probe/eval gain.

§12 ANTI-ORACLE constraint: tutor never accesses option.is_correct.
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np

from ..config import FullConfig, TutorConfig
from ..interfaces import TutorStep
from ..env.state import BlockState
from ..env.option_env import OptionEnv
from ..learner.semantic_scorer import DeterministicSemanticScorer
from ..learner.danger_head import DangerHead, create_danger_head
from ..learner.cls_adapter import create_scorer
from ..learner.learner_agent import LearnerAgent
from .profile_inference import ProfileInference
from .tutor_policy import TutorPolicy


class TutorAgent:
    """Autonomous tutor that intervenes pedagogically.

    Lifecycle:
        agent = TutorAgent(cfg)
        block = agent.run_block(env, learner, task_id, seed)
    """

    def __init__(self, cfg: Optional[FullConfig] = None):
        self.cfg = cfg or FullConfig()
        self.profile_inference = ProfileInference(self.cfg.tutor)
        self.tutor_policy = TutorPolicy(self.cfg.tutor)
        self._scorer: Optional[DeterministicSemanticScorer] = None
        self._danger_head: Optional[DangerHead] = None
        # P0 eval-aware components
        self._shadow_learner = None
        self._probe_evaluator = None
        self._grammar = None

    def init_block(self, block: BlockState, grammar, support) -> None:
        """Initialize tutor for a new block."""
        self._scorer = create_scorer(grammar, support, use_cls=False,
                                     tau_sem=self.cfg.learner.tau_sem)
        self._danger_head = create_danger_head(self.cfg.env.danger_dim)
        self._grammar = grammar

        # P0: Initialize eval-aware components
        if self.cfg.tutor.tutor_scorer_mode == "eval_aware":
            from ..eval.probe_evaluator import ProbeEvaluator
            from .shadow_learner import ShadowLearner

            # Probe evaluator: fixed held-out queries
            existing_queries = None
            if hasattr(block, 'queries') and block.queries:
                from ..interfaces import Example
                existing_queries = [
                    Example(words=list(q.true_program),
                            output=list(q.target_output))
                    for q in block.queries
                    if q.target_output
                ]

            self._probe_evaluator = ProbeEvaluator(
                grammar,
                n_probes=self.cfg.tutor.n_probe,
                seed=self.cfg.tutor.probe_seed,
                existing=existing_queries,
                ood_ratio=self.cfg.tutor.probe_ood_ratio,
            )
            # Store accuracy mode flag so shadow_learner knows which method to call
            self._probe_use_accuracy = self.cfg.tutor.probe_use_accuracy


            # Shadow learner: tutor-side approximate simulator
            self._shadow_learner = ShadowLearner(
                grammar=grammar,
                support=support if support else [],
                n_sup=self.cfg.learner.n_sup,
                n_em=self.cfg.learner.n_em,
                use_hpc=self.cfg.learner.use_hpc,
                tau_sem=self.cfg.learner.tau_sem,
                use_accuracy=self.cfg.tutor.probe_use_accuracy,
                rollout_horizon=self.cfg.tutor.shadow_rollout_horizon,
                rollout_gamma=self.cfg.tutor.shadow_rollout_gamma,
            )
        else:
            self._shadow_learner = None
            self._probe_evaluator = None

    def act(self, block: BlockState, env: OptionEnv,
            learner_agent=None) -> TutorStep:
        """Execute one tutor turn on the current query.

        Observation phase: WAIT (collect data)
        Teaching phase: score interventions → act
        L0 RSA mode: uses select_action_l0() with live learner access
        """
        qs = block.current_query
        if qs is None or qs.done or block.done:
            return env.tutor_act(block, "WAIT")

        # Check if transitioning from observation to teaching
        if (block.current_query_idx == block.obs_phase_queries
                and not hasattr(block, '_profile_inferred')):
            self._infer_profile(block)
            block._profile_inferred = True

        # ── L0 speaker path (RSA mode, single-thread only) ──
        if (self.cfg.rsa.use_l0_tutor
                and learner_agent is not None
                and block.in_teaching_phase):
            action, kwargs = self.tutor_policy.select_action_l0(
                block, learner_agent, self.cfg.rsa)
        else:
            # ── Legacy/eval-aware path ──
            learner_state = self._get_latest_learner_state(block)
            access_mode = self.cfg.tutor.tutor_access_mode

            action, kwargs = self.tutor_policy.select_action(
                block, self._scorer, self._danger_head,
                learner_state=learner_state,
                access_mode=access_mode,
                shadow_learner=self._shadow_learner,
                probe_evaluator=self._probe_evaluator,
            )

        # Extract calibration data before passing to env (env doesn't accept _q_probe*)
        q_probe_chosen    = kwargs.pop("_q_probe",    None)
        q_probe_z_chosen  = kwargs.pop("_q_probe_z",  None)
        probe_std_chosen  = kwargs.pop("_probe_std",  None)

        # Execute via env
        step = env.tutor_act(block, action, **kwargs)

        # Attach eval-aware diagnostics to TutorStep for calibration logging
        if q_probe_chosen is not None:
            if step.q_scores is None:
                step.q_scores = {}
            step.q_scores["q_probe"] = float(q_probe_chosen)
            if q_probe_z_chosen is not None:
                step.q_scores["q_probe_z"] = float(q_probe_z_chosen)
                step.q_scores["probe_std"]  = float(probe_std_chosen) if probe_std_chosen is not None else 0.0

        return step


    def _get_latest_learner_state(self, block: BlockState):
        """Extract the latest PolicyStateSnapshot from block (for cheat modes)."""
        if not hasattr(block, '_policy_snapshots') or not block._policy_snapshots:
            return None
        return block._policy_snapshots[-1]

    def _infer_profile(self, block: BlockState) -> None:
        """Infer learner profile from observation phase trace.

        Called once at the transition from observation -> teaching.
        Uses PolicyStateSnapshots for RSA-style inverse planning.
        """
        # Filter learner trace to observation-phase queries only
        obs_trace = [
            s for s in block.learner_trace
            if s.query_id < block.obs_phase_queries
        ]

        # Get PolicyStateSnapshots from observation phase
        obs_snapshots = None
        if hasattr(block, '_policy_snapshots'):
            obs_snapshots = [
                s for s in block._policy_snapshots
                if s.query_id < block.obs_phase_queries
            ]

        if obs_trace:
            posterior = self.profile_inference.infer(
                obs_trace, snapshots=obs_snapshots)
            block.profile_state = posterior.map_profile

    def observe_learner_outcome(self, block: BlockState) -> None:
        """Update tutor's danger model from learner outcomes.

        Called after each learner action to keep the tutor's
        danger estimates in sync with observed data.

        P0 eval-aware: also feeds observed reveals to shadow learner.
        """
        if not block.learner_trace:
            return

        last = block.learner_trace[-1]
        if last.action == "pick" and last.correct is False:
            # Find the corresponding reveal event
            qs_idx = last.query_id
            if qs_idx < len(block.queries):
                qs = block.queries[qs_idx]
                if qs.reveal_history:
                    rev = qs.reveal_history[-1]
                    if self._danger_head is not None:
                        self._danger_head.update(rev.danger_vec, rev.damage)

                    # P0: Feed reveal to shadow learner (externally observable)
                    if self._shadow_learner is not None:
                        from ..interfaces import Example
                        reveal_ex = Example(
                            words=list(rev.option_text),
                            output=list(rev.revealed_output),
                        )
                        self._shadow_learner.observe_reveal(reveal_ex)

    def run_block(
        self,
        env: OptionEnv,
        learner: LearnerAgent,
        task_id: str,
        seed: int = 42,
        synthesize: bool = False,
    ) -> BlockState:
        """Run a full block with tutor + learner interaction.

        Args:
            synthesize: if True, use grammar-synthesized novel queries
        Returns the completed BlockState with full traces.
        """
        block = env.reset_block(task_id, seed=seed, synthesize=synthesize)
        support, _, grammar = env.adapter.load_task(task_id)

        self.init_block(block, grammar, support)
        learner.init_block(block, grammar, support)

        max_steps = len(block.queries) * 20  # safety guard
        steps = 0
        while not block.done and steps < max_steps:
            steps += 1
            qs = block.current_query
            if qs is None or qs.done:
                break

            # Tutor acts first (pass learner for L0 mode)
            self.act(block, env, learner_agent=learner)

            if qs.done:  # SKIP ended the query
                continue

            # Learner acts
            learner.act(block, env)

            # Tutor observes outcome
            self.observe_learner_outcome(block)

        # Force done if safety guard hit
        if not block.done:
            block.done = True

        return block

