"""
learner_agent.py — Autonomous learner that plays the option tutor environment.

Ties together: semantic scorer + danger head + attention + memory + policy.
Interfaces with OptionEnv for end-to-end block execution.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from ..config import FullConfig, LearnerConfig
from ..interfaces import LearnerStep, PolicyStateSnapshot
from ..env.state import BlockState, QueryState
from ..env.option_env import OptionEnv
from ..env.interventions import get_active_menu
from .semantic_scorer import DeterministicSemanticScorer
from .cls_adapter import create_scorer
from .attention_model import AttentionModel
from .episodic_memory import EpisodicMemory
from .policy import LearnerPolicy, PolicyOutput


class LearnerAgent:
    """Autonomous learner agent for the option tutor environment.

    Lifecycle:
        agent = LearnerAgent(cfg)
        block = env.reset_block(task_id, seed)
        agent.init_block(block, grammar, support)
        while not block.done:
            policy_out = agent.act(block, env)
    """

    def __init__(self, cfg: Optional[FullConfig] = None,
                 seed: int = 42,
                 use_cls: bool = False):
        self.cfg = cfg or FullConfig()
        self.rng = np.random.default_rng(seed)
        # use_cls can be set via constructor OR config
        if use_cls:
            self.cfg.learner.use_cls = True
        self.policy = LearnerPolicy(self.cfg.learner)
        self._scorer = None

    def init_block(self, block: BlockState, grammar, support) -> None:
        """Initialize learner for a new block.

        With use_cls=True: creates CLSSemanticPosterior, calls study(support).
        With use_cls=False: creates DeterministicSemanticScorer (oracle).
        """
        lcfg = self.cfg.learner
        self._scorer = create_scorer(
            grammar, support,
            use_cls=lcfg.use_cls,
            n_sup=lcfg.n_sup,
            n_em=lcfg.n_em,
            use_hpc=lcfg.use_hpc,
            tau_sem=lcfg.tau_sem,
        )
        self.policy.init_for_block(self._scorer, m=self.cfg.env.danger_dim)
        self._teaching_examples = []  # Phase 3: accumulated reveals
        self._eval_frozen = False     # Phase 4: CLS freeze flag

    def act(self, block: BlockState, env: OptionEnv) -> Optional[PolicyOutput]:
        """Execute one learner turn on the current query.

        1. Initialize attention if new query
        2. Read any tutor interventions (highlight → attention)
        3. Compute policy
        4. Execute action via env
        5. Observe outcome (if wrong pick)

        Returns PolicyOutput, or None if block is done.
        """
        qs = block.current_query
        if qs is None or qs.done or block.done:
            return None

        # Initialize attention for new query (track by query_id)
        if self.policy.attention is None or self.policy.attention.L != len(qs.target_output):
            self.policy.init_for_query(len(qs.target_output))

        # Read highlight from tutor (only apply once per new highlight set)
        if qs.highlighted_cells and self.policy.attention is not None:
            if self.policy.attention.highlighted_cells != qs.highlighted_cells:
                self.policy.attention.apply_highlight(qs.highlighted_cells)

        # V2: Process RISK_HINT — update hazard head with weak labels
        if qs.risk_hint_history:
            for hint_evt in qs.risk_hint_history:
                opt = None
                for o in qs.menu:
                    if o.index == hint_evt.option_index:
                        opt = o
                        break
                if opt is not None:
                    self.policy.observe_risk_hint(
                        opt.danger_vec, eta=hint_evt.eta)

        # Compute policy
        policy_out = self.policy.compute_policy(qs, self.rng)

        # Record PolicyStateSnapshot for inverse planning (R2)
        active = get_active_menu(qs)
        snap = PolicyStateSnapshot(
            query_id=qs.query_id,
            step_idx=qs.rounds_used,
            target_output=list(qs.target_output),
            option_texts=[list(o.text) for o in active],
            option_danger_vecs=[o.danger_vec.copy() for o in active],
            option_indices=[o.index for o in active],
            active_bans=list(qs.banned_indices),
            active_highlights=qs.highlighted_cells,
            active_risk_hints=list(qs.risk_hints),
            hp_before=qs.hp,
            attempt_idx=qs.rounds_used,
            refresh_count=qs.refreshes_used,
            max_refreshes=qs.max_refreshes,
            hazard_posterior_mean=(
                self.policy.danger_head.hazard.w_mean.copy()
                if self.policy.danger_head is not None else None),
            severity_posterior_mean=(
                self.policy.danger_head.severity.w_mean.copy()
                if self.policy.danger_head is not None else None),
            danger_posterior_mean=(
                self.policy.danger_head.w_mean.copy()
                if self.policy.danger_head is not None else None),
            danger_posterior_cov=None,
            option_risk_classes=[o.risk_class for o in active],
            attention_weights=(
                self.policy.attention.weights.copy()
                if self.policy.attention is not None else None),
            semantic_scores=policy_out.semantic_scores.copy(),
            learner_action=policy_out.action,
            learner_pick_index=policy_out.pick_index,
        )
        if not hasattr(block, '_policy_snapshots'):
            block._policy_snapshots = []
        block._policy_snapshots.append(snap)

        # Execute
        step = env.learner_act(
            block, policy_out.action,
            pick_index=policy_out.pick_index,
        )

        # Observe outcome
        if step.action == "pick" and step.correct is False:
            # Find the reveal event
            if qs.reveal_history:
                last_reveal = qs.reveal_history[-1]
                self.policy.observe_outcome(
                    last_reveal.danger_vec,
                    last_reveal.damage,
                    reveal_event=last_reveal,
                )

                # Phase 3 (Teaching): learn from revealed (program, output)
                if block.in_teaching_phase and self._is_cls_scorer():
                    # The revealed output shows what program X actually renders to
                    from ..interfaces import Example
                    new_example = Example(
                        words=list(last_reveal.option_text),
                        output=list(last_reveal.revealed_output),
                    )
                    self._teaching_examples.append(new_example)
                    self._scorer.incremental_study([new_example])

        if step.action == "refresh":
            self.policy.on_refresh()

        # Phase transition: teach → eval → freeze CLS
        if (block.in_evaluation_phase
                and not self._eval_frozen
                and self._is_cls_scorer()):
            self._scorer.freeze()
            self._eval_frozen = True

        return policy_out

    def _is_cls_scorer(self) -> bool:
        """Check if current scorer supports incremental learning."""
        return (self._scorer is not None
                and hasattr(self._scorer, 'incremental_study'))

    def run_block(self, env: OptionEnv, task_id: str,
                  seed: int = 42, tutor_action: str = "WAIT",
                  synthesize: bool = False,
                  ) -> BlockState:
        """Run a full block autonomously (learner only, tutor=WAIT).

        Args:
            synthesize: if True, use grammar-synthesized novel queries
        Returns the completed BlockState.
        """
        block = env.reset_block(task_id, seed=seed, synthesize=synthesize)

        # Get grammar and support from the adapter
        support, _, grammar = env.adapter.load_task(task_id)
        self.init_block(block, grammar, support)

        while not block.done:
            qs = block.current_query
            if qs is None or qs.done:
                break

            # Tutor always WAITs in baseline mode
            env.tutor_act(block, tutor_action)
            if qs.done:  # SKIP case
                continue

            self.act(block, env)

        return block
