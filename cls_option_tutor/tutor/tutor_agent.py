"""
tutor_agent.py — Autonomous tutor agent for the CLS Option Tutor.

Orchestrates the full tutor lifecycle:
    1. Observation phase → WAIT + collect learner trace
    2. Profile inference from observed behavior
    3. Teaching phase → counterfactual scoring → intervention selection
    4. Outcome tracking for evaluation

§12 ANTI-ORACLE constraint: tutor never accesses option.is_correct.
"""
from __future__ import annotations
from typing import Optional, Tuple
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

    def init_block(self, block: BlockState, grammar, support) -> None:
        """Initialize tutor for a new block."""
        self._scorer = create_scorer(grammar, support, use_cls=False,
                                     tau_sem=self.cfg.learner.tau_sem)
        self._danger_head = create_danger_head(self.cfg.env.danger_dim)

    def act(self, block: BlockState, env: OptionEnv) -> TutorStep:
        """Execute one tutor turn on the current query.

        Observation phase: WAIT (collect data)
        Teaching phase: score interventions → act
        """
        qs = block.current_query
        if qs is None or qs.done or block.done:
            return env.tutor_act(block, "WAIT")

        # Check if transitioning from observation to teaching
        if (block.current_query_idx == block.obs_phase_queries
                and not hasattr(block, '_profile_inferred')):
            self._infer_profile(block)
            block._profile_inferred = True

        # Select action
        action, kwargs = self.tutor_policy.select_action(
            block, self._scorer, self._danger_head)

        # Execute via env
        step = env.tutor_act(block, action, **kwargs)

        return step

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

            # Tutor acts first
            self.act(block, env)

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
