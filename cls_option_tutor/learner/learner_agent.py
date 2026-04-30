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
from .cls_adapter import create_scorer, NegativeMemory
from .attention_model import AttentionModel
from .episodic_memory import EpisodicMemory
from .policy import LearnerPolicy, PolicyOutput
from .rsa_listener import RSAListener


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
        # Pre-initialize attributes that may be read before init_block
        self._negative_memory = None
        self._teaching_examples = []
        self._reveal_shifts = []
        self._eval_frozen = False
        self._persistent_hl = None
        self._persistent_ban = None
        self._highlight_counts = None
        self._rsa_listener = None
        self._last_ban_trace_len = 0
        self._last_risk_hint_count_by_query = {}
        self._sem_counters = {
            "wrong_reveal_attempted": 0, "wrong_reveal_applied": 0,
            "correct_unassisted_attempted": 0, "correct_unassisted_applied": 0,
            "correct_assisted_attempted": 0, "correct_assisted_applied": 0,
            "direct_answer_attempted": 0, "direct_answer_applied": 0,
        }
        self._src_counters = {
            "wr_scripted_att": 0, "wr_scripted_app": 0,
            "wr_incidental_att": 0, "wr_incidental_app": 0,
            "cu_scripted_self_correct_att": 0, "cu_scripted_self_correct_app": 0,
            "cu_scripted_direct_correct_att": 0, "cu_scripted_direct_correct_app": 0,
            "cu_incidental_att": 0, "cu_incidental_app": 0,
            "da_direct_answer_att": 0, "da_direct_answer_app": 0,
            "da_then_answer_att": 0, "da_then_answer_app": 0,
            "da_incidental_shortlist_att": 0, "da_incidental_shortlist_app": 0,
        }

    def init_block(self, block: BlockState, grammar, support) -> None:
        """Initialize learner for a new block.

        With use_cls=True: creates CLSSemanticPosterior, calls study(support).
        With use_cls=False: creates DeterministicSemanticScorer (oracle).
        """
        lcfg = self.cfg.learner
        rcfg = self.cfg.rsa
        self._scorer = create_scorer(
            grammar, support,
            use_cls=lcfg.use_cls,
            n_sup=lcfg.n_sup,
            n_em=lcfg.n_em,
            use_hpc=lcfg.use_hpc,
            tau_sem=lcfg.tau_sem,
            lambda_neg=lcfg.lambda_neg,
        )
        self.policy.init_for_block(self._scorer, m=self.cfg.env.danger_dim)
        self._teaching_examples = []  # Phase 3: accumulated reveals
        self._eval_frozen = False     # Phase 4: CLS freeze flag
        # Root-cause modes
        self._negative_memory = (
            NegativeMemory(alpha_neg=lcfg.alpha_neg)
            if lcfg.reveal_learning_mode == "negative_memory" else None
        )
        # Clear any residual negative evidence from a previous block (safety)
        if hasattr(self._scorer, 'clear_negative_evidence'):
            self._scorer.clear_negative_evidence()
        self._highlight_counts = None  # block-level for persistent_prior (legacy)
        # PosteriorShiftPerReveal tracking
        self._reveal_shifts: list = []  # shift in score_option per reveal

        # ── Phase 6.3: Semantic update instrumentation ──────────────────
        self._sem_counters = {
            "wrong_reveal_attempted": 0,
            "wrong_reveal_applied": 0,
            "correct_unassisted_attempted": 0,
            "correct_unassisted_applied": 0,
            "correct_assisted_attempted": 0,
            "correct_assisted_applied": 0,
            "direct_answer_attempted": 0,
            "direct_answer_applied": 0,
        }

        # ── Phase 6.5: Event-source counters ──────────────────────────────
        self._src_counters = {
            # Wrong reveals
            "wr_scripted_att": 0, "wr_scripted_app": 0,
            "wr_incidental_att": 0, "wr_incidental_app": 0,
            # Correct unassisted
            "cu_scripted_self_correct_att": 0, "cu_scripted_self_correct_app": 0,
            "cu_scripted_direct_correct_att": 0, "cu_scripted_direct_correct_app": 0,
            "cu_incidental_att": 0, "cu_incidental_app": 0,
            # Direct answer
            "da_direct_answer_att": 0, "da_direct_answer_app": 0,
            "da_then_answer_att": 0, "da_then_answer_app": 0,
            "da_incidental_shortlist_att": 0, "da_incidental_shortlist_app": 0,
        }

        # ── Step 4: Persistent Highlight Prior (EMA over cell positions) ──
        # m_t: EMA attention prior vector, shape (L_max,)
        # Updated each time tutor issues HIGHLIGHT on this block.
        # Injected at query init: w_ℓ ∝ 1 + λ_hl * m_t[ℓ]
        # None = no HIGHLIGHT seen yet; treated as uniform (no bias).
        self._persistent_hl: "np.ndarray | None" = None

        # ── Step 5: Persistent Ban Prior (EMA over danger feature space) ──
        # n_t: EMA penalty vector, shape (danger_dim,)
        # Updated each time tutor issues BAN; applied as utility penalty.
        # None = no BAN seen yet; no penalty.
        self._persistent_ban: "np.ndarray | None" = None
        self._last_ban_trace_len = 0
        self._last_risk_hint_count_by_query = {}

        # ── RSA listener (new) ──
        # Instantiate per-block so state never leaks across blocks.
        if rcfg.use_rsa:
            self._rsa_listener = RSAListener(
                omega_hl=rcfg.omega_hl,
                lambda_ctx=rcfg.lambda_ctx,
                omega_ban=rcfg.omega_ban,
            )
            # Initialize block-level meta attention prior (uniform)
            # Use a generous L_max=16 to cover all possible query lengths
            L_max = max(self.cfg.env.danger_dim, 8)
            if self.policy.attention is None:
                # Create a temporary AttentionModel to hold the meta prior
                # Will be replaced on first query, but meta_prior persists via learner_agent
                self._meta_prior_init_L = L_max
            self._rsa_meta_prior_init_done = False  # init on first query
            self._rsa_last_processed_trace_len = 0  # idempotency guard
        else:
            self._rsa_listener = None

    def prepare_probe_block(self, block: 'BlockState') -> None:
        """Attach an already-trained learner to a fresh probe block without
        reinitializing scorer, danger head, memory, or persistent priors.

        Reset only query-local transient state:
          - attention (will be re-created per-query in act())
          - RSA trace pointer

        This preserves all learned state (scorer, danger_head, memory,
        persistent HL/BAN priors, negative memory, sem_counters).
        """
        # Reset per-query attention — will be recreated in act()
        self.policy.attention = None

        # Reset RSA trace pointer so it doesn't try to process old tutor trace
        if hasattr(self, '_rsa_last_processed_trace_len'):
            self._rsa_last_processed_trace_len = 0
        self._last_ban_trace_len = 0
        self._last_risk_hint_count_by_query = {}

        # Reset _teaching_examples for the probe episode
        # (these accumulate per-block and are not retained-learned state)
        self._teaching_examples = []

        # Reset per-query highlight counts (legacy)
        self._highlight_counts = None

        # RSA meta prior: leave initialized if already done
        # _persistent_hl, _persistent_ban: KEEP (these are learned)
        # _scorer, policy.danger_head, policy.memory: KEEP
        # _negative_memory: KEEP
        # _sem_counters, _src_counters: don't reset (probe should not update them)
        # _reveal_shifts: reset for probe episode
        self._reveal_shifts = []

    def get_policy_snapshot_for_query(
        self, qs: 'QueryState',
        rng_seed: int = 0,
    ) -> 'PolicyOutput':
        """Compute policy outputs without taking an action and without
        mutating any persistent state.

        Reuses the same path as act() up to compute_policy(),
        including attention, danger head, episodic memory, etc.

        Phase 6H.6: Also processes qs.highlighted_cells through attention
        (matching the highlight-reading block in act()) so that the causal
        audit's HIGHLIGHT intervention actually affects the distribution.

        Returns PolicyOutput with utilities, probs, semantic_scores, etc.
        """
        from ..env.interventions import get_active_menu
        # Ensure attention is initialized for this query length
        if self.policy.attention is None or self.policy.attention.L != len(qs.target_output):
            L = len(qs.target_output)
            self.policy.init_for_query(L)

        # Process highlighted cells through attention (mirrors act() lines 273-323)
        if qs.highlighted_cells and self.policy.attention is not None:
            if self.policy.attention.highlighted_cells != qs.highlighted_cells:
                if self._rsa_listener is not None:
                    rcfg = self.cfg.rsa
                    self.policy.attention.apply_rsa_highlight(
                        qs.highlighted_cells,
                        rho=rcfg.rho_attn,
                        gamma=rcfg.gamma_attn,
                    )
                else:
                    self.policy.attention.apply_highlight(qs.highlighted_cells)

        # Compute negative memory penalties (same path as act())
        neg_penalties = None
        if self._negative_memory is not None:
            active = get_active_menu(qs)
            neg_penalties = np.array([
                self._negative_memory.penalty(opt.text)
                for opt in active
            ])

        # Compute policy (read-only — rng is throwaway)
        snapshot_rng = np.random.default_rng(rng_seed)
        return self.policy.compute_policy(
            qs, snapshot_rng, neg_penalties=neg_penalties)

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
            L = len(qs.target_output)
            lcfg = self.cfg.learner

            # ── Step 4: Persistent HL Prior (new, highest priority) ──────
            # If rho_hl_prior > 0, use the block-level EMA vector as prior.
            # This overrides the legacy persistent_prior mode for attention init.
            if lcfg.rho_hl_prior > 0.0 and self._persistent_hl is not None:
                self.policy.init_for_query(L)
                self.policy.attention.init_for_query(
                    L,
                    prior_counts=self._persistent_hl,
                    eta_attn=lcfg.lambda_hl_prior)
            elif lcfg.attention_init_mode == "persistent_prior" and self._highlight_counts is not None:
                # Legacy persistent_prior mode (integer counts)
                self.policy.init_for_query(L)
                self.policy.attention.init_for_query(
                    L, prior_counts=self._highlight_counts,
                    eta_attn=lcfg.eta_attn)
            else:
                self.policy.init_for_query(L)

            # RSA: Initialize meta prior on first query of the block
            if self._rsa_listener is not None and not self._rsa_meta_prior_init_done:
                L_meta = max(L, getattr(self, '_meta_prior_init_L', 8))
                self.policy.attention.init_meta_prior(L_meta)
                self._rsa_meta_prior_init_done = True

        # Read highlight from tutor (only apply once per new highlight set)
        if qs.highlighted_cells and self.policy.attention is not None:
            if self.policy.attention.highlighted_cells != qs.highlighted_cells:
                if self._rsa_listener is not None:
                    # RSA path: apply_rsa_highlight updates meta prior + blends attention
                    rcfg = self.cfg.rsa
                    self.policy.attention.apply_rsa_highlight(
                        qs.highlighted_cells,
                        rho=rcfg.rho_attn,
                        gamma=rcfg.gamma_attn,
                    )
                else:
                    # Legacy path: plain attention boost
                    self.policy.attention.apply_highlight(qs.highlighted_cells)

                # ── Step 4: Update Persistent HL Prior (EMA) ─────────────
                # m_{t+1} = (1-ρ_hl)*m_t + ρ_hl * φ_hl(H_t)
                # φ_hl(H_t) = one-hot over highlighted cells (∑=1)
                lcfg = self.cfg.learner
                if lcfg.rho_hl_prior > 0.0:
                    L_cur = len(qs.target_output)
                    L_max = max(L_cur, 16)  # generous bound for future queries
                    # Initialize or grow _persistent_hl if needed
                    if self._persistent_hl is None:
                        self._persistent_hl = np.zeros(L_max)
                    elif len(self._persistent_hl) < L_max:
                        new = np.zeros(L_max)
                        new[:len(self._persistent_hl)] = self._persistent_hl
                        self._persistent_hl = new
                    # Build φ_hl: uniform over highlighted cells, zero elsewhere
                    phi_hl = np.zeros(len(self._persistent_hl))
                    valid_cells = [c for c in qs.highlighted_cells
                                   if 0 <= c < len(phi_hl)]
                    if valid_cells:
                        for c in valid_cells:
                            phi_hl[c] = 1.0 / len(valid_cells)
                    # EMA update
                    rho = lcfg.rho_hl_prior
                    self._persistent_hl = (1.0 - rho) * self._persistent_hl + rho * phi_hl

                # Record for legacy persistent_prior mode
                if self.cfg.learner.attention_init_mode == "persistent_prior":
                    if self._highlight_counts is None:
                        self._highlight_counts = np.zeros(8)  # L_max
                    L_max = max(8, len(qs.target_output))
                    if len(self._highlight_counts) < L_max:
                        new = np.zeros(L_max)
                        new[:len(self._highlight_counts)] = self._highlight_counts
                        self._highlight_counts = new
                    for c in qs.highlighted_cells:
                        if 0 <= c < len(self._highlight_counts):
                            self._highlight_counts[c] += 1

        # V2: Process RISK_HINT — update hazard head with weak labels
        if qs.risk_hint_history:
            last_hint_count = self._last_risk_hint_count_by_query.get(qs.query_id, 0)
            for hint_evt in qs.risk_hint_history[last_hint_count:]:
                opt = None
                for o in qs.menu:
                    if o.index == hint_evt.option_index:
                        opt = o
                        break
                if opt is not None:
                    self.policy.observe_risk_hint(
                        opt.danger_vec, eta=hint_evt.eta)
            self._last_risk_hint_count_by_query[qs.query_id] = len(qs.risk_hint_history)

        # Compute negative memory penalties (if active)
        neg_penalties = None
        if self._negative_memory is not None:
            active = get_active_menu(qs)
            neg_penalties = np.array([
                self._negative_memory.penalty(opt.text)
                for opt in active
            ])

        # ── Step 5: Persistent Ban Prior update + penalty ──────────────
        lcfg_ban = self.cfg.learner
        if lcfg_ban.rho_ban_prior > 0.0 and block.tutor_trace:
            # Only process newly appended BAN events while the originating
            # query menu is still available, otherwise ban_index cannot be
            # safely resolved against the current query's menu.
            m_ban = self.cfg.env.danger_dim
            last_ban_processed = getattr(self, '_last_ban_trace_len', 0)
            for ts in block.tutor_trace[last_ban_processed:]:
                if (getattr(ts, 'action', '') == 'BAN'
                        and ts.ban_index is not None
                        and ts.query_id == qs.query_id):
                    # Look up the banned option
                    banned_opt = next(
                        (o for o in qs.menu if o.index == ts.ban_index), None)
                    if banned_opt is not None and len(banned_opt.danger_vec) == m_ban:
                        dv = np.array(banned_opt.danger_vec, dtype=float)
                        # Initialize or update _persistent_ban
                        if self._persistent_ban is None:
                            self._persistent_ban = np.zeros(m_ban)
                        rho_ban = lcfg_ban.rho_ban_prior
                        self._persistent_ban = (
                            (1.0 - rho_ban) * self._persistent_ban
                            + rho_ban * dv / (np.linalg.norm(dv) + 1e-8)
                        )
            self._last_ban_trace_len = len(block.tutor_trace)

        # Apply persistent ban penalty to active options
        if (lcfg_ban.lambda_ban_prior > 0.0
                and self._persistent_ban is not None):
            active_now = get_active_menu(qs)
            ban_penalties = np.array([
                -lcfg_ban.lambda_ban_prior
                * float(np.dot(self._persistent_ban,
                               np.array(opt.danger_vec, dtype=float)
                               / (np.linalg.norm(opt.danger_vec) + 1e-8)))
                for opt in active_now
            ])
            if neg_penalties is None:
                neg_penalties = ban_penalties
            else:
                neg_penalties = neg_penalties + ban_penalties

        # ── RSA L1 inference ──
        # Read the most recent tutor action and compute posterior updates.
        # This runs AFTER attention is already initialized/updated (above),
        # so RSA semantic_log_bias is *additional* to the attention boost.
        semantic_log_bias = None
        risk_logit_shift = None
        rcfg = self.cfg.rsa

        if self._rsa_listener is not None and block.tutor_trace:
            trace_len = len(block.tutor_trace)
            last_processed = getattr(self, '_rsa_last_processed_trace_len', 0)
            if trace_len > last_processed:
                last_step = block.tutor_trace[-1]
                if last_step.query_id == qs.query_id:
                    last_action = last_step.action

                    # Build rendered outputs for active options (for HIGHLIGHT)
                    active_menu = get_active_menu(qs)
                    rendered_outputs = [
                        (self._scorer.predict_output(opt.text)
                         if self._scorer else None)
                        for opt in active_menu
                    ]
                    active_texts = [list(opt.text) for opt in active_menu]

                    # Determine action args
                    action_arg = None
                    action_cells = None
                    if last_action == "BAN" and last_step.ban_index is not None:
                        banned_full_idx = last_step.ban_index
                        for ai, opt in enumerate(active_menu):
                            if opt.index == banned_full_idx:
                                action_arg = ai
                                break
                    elif last_action == "HIGHLIGHT" and last_step.highlight_cells:
                        action_cells = last_step.highlight_cells

                    # ── Compute semantic gate ──────────────────────────────
                    # Gate input: q_t^(0) = base decision posterior (pre-RSA)
                    # Built from CLS + risk + unc — all channels, no RSA bias.
                    # Must NOT use raw sem_scores (those ignore risk/unc).
                    sem_gate = 1.0
                    if rcfg.use_sem_gate and last_action == "HIGHLIGHT":
                        K_active = len(active_menu)
                        if K_active > 0:
                            # Compute base scores for each active option
                            weights = (self.policy.attention.weights
                                       if self.policy.attention is not None
                                       else np.ones(len(qs.target_output)) / len(qs.target_output))
                            base_sem = np.array([
                                self._scorer.score_option(qs.target_output, opt.text,
                                                          attention_weights=weights)
                                for opt in active_menu
                            ])
                            base_danger = np.zeros(K_active)
                            base_unc = np.zeros(K_active)
                            if self.policy.danger_head is not None:
                                for i, opt in enumerate(active_menu):
                                    mu, u = self.policy.danger_head.predict(opt.danger_vec)
                                    base_danger[i] = mu
                                    base_unc[i] = u

                            # q_t^(0) = softmax(β * U_base) where
                            # U_base = α_sem*S_CLS - α_risk*μ_d - α_unc*u_d
                            cfg_l = self.cfg.learner
                            U_base = (cfg_l.alpha_sem * base_sem
                                      - cfg_l.alpha_risk * base_danger
                                      - cfg_l.alpha_unc * base_unc)
                            shifted = U_base - np.max(U_base)
                            q_prior = np.exp(cfg_l.beta_L * shifted)
                            q_prior = q_prior / (q_prior.sum() + 1e-10)

                            sem_gate = self._rsa_listener.compute_sem_gate(
                                q_prior,
                                gate_type=rcfg.sem_gate_type,
                                gate_lo=rcfg.sem_gate_lo,
                                gate_hi=rcfg.sem_gate_hi,
                            )
                    # ──────────────────────────────────────────────────────

                    rsa_update = self._rsa_listener.observe_tutor_action(
                        action=last_action,
                        target_output=list(qs.target_output),
                        active_texts=active_texts,
                        rendered_outputs=rendered_outputs,
                        action_arg=action_arg,
                        action_cells=action_cells,
                        sem_gate=sem_gate,
                    )

                    # PASS: abort current query (explicit tutor abort)
                    if rsa_update.pass_abort:
                        self._rsa_last_processed_trace_len = trace_len
                        return None

                    # BAN cross-query teach: update danger_head persistently
                    # CANONICAL DEFAULT: ban_teaches_risk=False (no eval gain per Exp F)
                    if (last_action == "BAN"
                            and rcfg.ban_teaches_risk
                            and last_step.ban_index is not None
                            and block.in_teaching_phase):
                        banned_full_idx = last_step.ban_index
                        for opt in qs.menu:
                            if opt.index == banned_full_idx:
                                self.policy.danger_head.update_from_ban(
                                    opt.danger_vec, omega_ban=rcfg.omega_ban)
                                break

                    semantic_log_bias = rsa_update.semantic_log_bias
                    risk_logit_shift = rsa_update.risk_logit_shift

                self._rsa_last_processed_trace_len = trace_len

        # Compute policy (RSA or legacy)
        policy_out = self.policy.compute_policy(
            qs, self.rng,
            neg_penalties=neg_penalties,
            semantic_log_bias=semantic_log_bias,
            risk_logit_shift=risk_logit_shift,
        )

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
            pick_probs=policy_out.probs.copy(),
            learner_action=policy_out.action,
            learner_pick_index=policy_out.pick_index,
        )
        if not hasattr(block, '_policy_snapshots'):
            block._policy_snapshots = []
        block._policy_snapshots.append(snap)

        # Execute
        # IMPORTANT: capture teaching phase BEFORE env.learner_act() because
        # a correct pick will advance current_query_idx, changing in_teaching_phase
        was_in_teaching = block.in_teaching_phase
        step = env.learner_act(
            block, policy_out.action,
            pick_index=policy_out.pick_index,
        )
        feedback_meta = self._new_feedback_meta()

        # Observe outcome
        if step.action == "pick" and step.correct is False:
            feedback_meta = self._new_feedback_meta(
                raw_feedback_kind="wrong_reveal",
                feedback_category=self._classify_wrong_feedback(
                    qs, qs.reveal_history[-1] if qs.reveal_history else None
                ),
            )
            # ── Wrong pick ─────────────────────────────────────────────
            # Find the reveal event
            if qs.reveal_history:
                last_reveal = qs.reveal_history[-1]
                self.policy.observe_outcome(
                    last_reveal.danger_vec,
                    last_reveal.damage,
                    reveal_event=last_reveal,
                )

                # Phase 3 (Teaching): handle reveal / wrong-pick feedback.
                # For cortex_em: need CLS scorer. For off/negative_memory/nonreveal_negative: always handle.
                reveal_mode = self.cfg.learner.reveal_learning_mode
                feedback_mode = self.cfg.env.feedback_mode
                nonreveal_active = (
                    feedback_mode == "nonreveal"
                    and self.cfg.learner.negative_evidence_mode == "exact_program_target"
                )
                should_handle = (
                    was_in_teaching
                    and (self._is_cls_scorer()
                         or reveal_mode not in ("cortex_em",)
                         or nonreveal_active)
                )
                if should_handle:
                    from ..interfaces import Example
                    new_example = Example(
                        words=list(last_reveal.option_text),
                        output=list(last_reveal.revealed_output),
                    )
                    self._teaching_examples.append(new_example)
                    # Pass qs so _handle_reveal can access target_output
                    # in nonreveal_negative mode without using revealed output.
                    feedback_meta = self._handle_reveal(new_example, qs=qs)

        elif step.action == "pick" and step.correct is True:
            feedback_meta = self._new_feedback_meta(
                raw_feedback_kind="correct_pick",
                feedback_category=(
                    "correct_after_feedback"
                    if (qs.reveal_history or qs.after_highlight_grace_round or qs.assist_level != "none")
                    else "correct_incidental"
                ),
            )
            # ── Correct pick: optional positive reinforcement ───────────
            # Learner confirmed (j*.text, target_output) — a complete positive example.
            # Only triggers if correct_pick_learning_mode != "off" and in teaching phase.
            if (was_in_teaching
                    and self._is_cls_scorer()
                    and self.cfg.learner.correct_pick_learning_mode != "off"):
                feedback_meta = self._handle_correct_pick(qs)

        if step.action == "refresh":
            self.policy.on_refresh()
            feedback_meta = self._new_feedback_meta(
                raw_feedback_kind="refresh",
                feedback_category="refresh",
            )

        self._apply_feedback_meta(qs, step, feedback_meta)

        # Phase transition: teach -> eval -> freeze CLS
        if (block.in_evaluation_phase
                and not self._eval_frozen
                and self._is_cls_scorer()):
            self._scorer.freeze()
            self._eval_frozen = True

        return policy_out

    def observe_forced_step(self, block: BlockState, step, qs: 'QueryState' = None) -> None:
        """Observe the outcome of a forced pick (from scripted protocol).

        This must be called AFTER env.force_learner_pick() to ensure
        learner semantic updates (CLS scorer, danger head, memory, counters)
        are properly triggered. Without this, forced picks bypass learner.act()
        and no learning occurs.

        IMPORTANT: qs must be passed explicitly because env.force_learner_pick()
        advances block.current_query_idx on correct pick (via _advance_query),
        so block.current_query would point to the next query by the time we run.

        The code paths here mirror the observe-outcome block in act().
        """
        if qs is None:
            qs = block.current_query
        if qs is None:
            return

        # Determine if query was in teaching phase based on its index
        # (can't use block.in_teaching_phase because current_query_idx may have advanced)
        qi = None
        for idx, q in enumerate(block.queries):
            if q is qs:
                qi = idx
                break
        if qi is None:
            qi = qs.query_id
        teach_start = block.obs_phase_queries
        teach_end = block.obs_phase_queries + block.teach_phase_queries
        was_in_teaching = teach_start <= qi < teach_end

        if step.action == "pick" and step.correct is False:
            feedback_meta = self._new_feedback_meta(
                raw_feedback_kind="wrong_reveal",
                feedback_category=self._classify_wrong_feedback(
                    qs, qs.reveal_history[-1] if qs.reveal_history else None
                ),
            )
            # Wrong pick — same path as act()
            if qs.reveal_history:
                last_reveal = qs.reveal_history[-1]
                self.policy.observe_outcome(
                    last_reveal.danger_vec,
                    last_reveal.damage,
                    reveal_event=last_reveal,
                )

                reveal_mode = self.cfg.learner.reveal_learning_mode
                feedback_mode = self.cfg.env.feedback_mode
                nonreveal_active = (
                    feedback_mode == "nonreveal"
                    and self.cfg.learner.negative_evidence_mode == "exact_program_target"
                )
                should_handle = (
                    was_in_teaching
                    and (self._is_cls_scorer()
                         or reveal_mode not in ("cortex_em",)
                         or nonreveal_active)
                )
                if should_handle:
                    from ..interfaces import Example
                    new_example = Example(
                        words=list(last_reveal.option_text),
                        output=list(last_reveal.revealed_output),
                    )
                    self._teaching_examples.append(new_example)
                    feedback_meta = self._handle_reveal(new_example, qs=qs)
            self._apply_feedback_meta(qs, step, feedback_meta)

        elif step.action == "pick" and step.correct is True:
            feedback_meta = self._new_feedback_meta(
                raw_feedback_kind="correct_pick",
                feedback_category=(
                    "correct_after_feedback"
                    if (qs.reveal_history or qs.after_highlight_grace_round or qs.assist_level != "none")
                    else "correct_incidental"
                ),
            )
            # Correct pick — same path as act()
            if (was_in_teaching
                    and self._is_cls_scorer()
                    and self.cfg.learner.correct_pick_learning_mode != "off"):
                feedback_meta = self._handle_correct_pick(qs)
            self._apply_feedback_meta(qs, step, feedback_meta)


    def _is_cls_scorer(self) -> bool:
        """Check if current scorer supports incremental learning."""
        return (self._scorer is not None
                and hasattr(self._scorer, 'incremental_study'))

    def _new_feedback_meta(
        self,
        *,
        raw_feedback_kind: str = "none",
        feedback_category: str = "none",
    ) -> dict:
        return {
            "raw_feedback_kind": raw_feedback_kind,
            "feedback_category": feedback_category,
            "semantic_credit": 0.0,
            "semantic_credit_type": "none",
            "semantic_credit_reason": "none",
            "semantic_update_attempted": False,
            "semantic_update_applied": False,
            "contrastive_ticket_consumed": False,
            "positive_ticket_consumed": False,
        }

    def _apply_feedback_meta(self, qs, step, meta: dict) -> None:
        """Attach pedagogical-credit annotations to learner trace + query state."""
        if step is not None:
            step.raw_feedback_kind = str(meta.get("raw_feedback_kind", "none"))
            step.feedback_category = str(meta.get("feedback_category", "none"))
            step.semantic_credit = float(meta.get("semantic_credit", 0.0))
            step.semantic_credit_type = str(meta.get("semantic_credit_type", "none"))
            step.semantic_credit_reason = str(meta.get("semantic_credit_reason", "none"))
            step.semantic_update_attempted = bool(meta.get("semantic_update_attempted", False))
            step.semantic_update_applied = bool(meta.get("semantic_update_applied", False))
            step.contrastive_ticket_consumed = bool(meta.get("contrastive_ticket_consumed", False))
            step.positive_ticket_consumed = bool(meta.get("positive_ticket_consumed", False))

        if qs is not None:
            qs.last_semantic_credit = float(meta.get("semantic_credit", 0.0))
            qs.last_feedback_credit_type = str(meta.get("semantic_credit_type", "none"))
            qs.last_feedback_credit_reason = str(meta.get("semantic_credit_reason", "none"))

    def _previous_reveal_option_index(self, qs) -> "int | None":
        history = getattr(qs, "reveal_history", None) or []
        if len(history) >= 2:
            return getattr(history[-2], "option_index", None)
        return None

    def _classify_wrong_feedback(self, qs, reveal_event) -> str:
        if qs is None or reveal_event is None:
            return "other_wrong"
        option_index = getattr(reveal_event, "option_index", None)
        prev_index = self._previous_reveal_option_index(qs)
        if prev_index is not None and option_index == prev_index:
            return "same_wrong"
        labels = getattr(qs, "option_diag_labels", {}) or {}
        label = labels.get(option_index, "")
        if label == "safe_diagnostic_wrong":
            return "safe_diag"
        if label == "bounded_diagnostic_wrong":
            return "bounded_diag"
        if label == "high_risk_lure":
            return "high_risk"
        if label in ("safe_far", "safe_random_wrong", "risky_far"):
            return "far_wrong"
        return "other_wrong"

    def _compute_wrong_semantic_credit(self, qs, reveal_event) -> dict:
        category = self._classify_wrong_feedback(qs, reveal_event)
        meta = self._new_feedback_meta(
            raw_feedback_kind="wrong_reveal",
            feedback_category=category,
        )
        meta["semantic_credit_type"] = "contrastive"

        feedback_mode = getattr(self.cfg.learner, "pedagogical_feedback_mode", "raw")
        if feedback_mode != "budgeted_v1":
            meta["semantic_credit"] = 1.0
            meta["semantic_credit_reason"] = "raw_mode"
            return meta

        if qs is None:
            meta["semantic_credit_reason"] = "missing_query_state"
            return meta
        if getattr(qs, "contrastive_update_used", False):
            meta["semantic_credit_reason"] = "ticket_spent"
            return meta

        info = 0.0
        if category == "safe_diag":
            info = 1.0
        elif category == "bounded_diag":
            info = float(getattr(self.cfg.learner, "bounded_reveal_credit", 0.5))

        risk_class = float(getattr(reveal_event, "risk_class", 0.0) if reveal_event is not None else 0.0)
        h0 = float(max(1, getattr(self.cfg.env, "H_0", 1)))
        safety = max(0.0, 1.0 - risk_class / h0)
        novelty = 0.0 if category == "same_wrong" else 1.0
        credit = float(info * safety * novelty)

        meta["semantic_credit"] = credit
        if credit > 0.0:
            meta["semantic_credit_reason"] = category
        elif category == "same_wrong":
            meta["semantic_credit_reason"] = "same_wrong"
        elif category == "high_risk":
            meta["semantic_credit_reason"] = "high_risk"
        elif category == "far_wrong":
            meta["semantic_credit_reason"] = "far_wrong"
        elif category == "other_wrong":
            meta["semantic_credit_reason"] = "other_wrong"
        else:
            meta["semantic_credit_reason"] = "zero_credit"
        return meta

    def _compute_correct_semantic_credit(self, qs) -> dict:
        assist_level = getattr(qs, "assist_level", "none") if qs is not None else "none"
        src = getattr(qs, "learning_event_source", "incidental") if qs is not None else "incidental"
        had_reveal = bool(getattr(qs, "reveal_history", None))
        after_grace = bool(getattr(qs, "after_highlight_grace_round", False)) if qs is not None else False
        pedagogical_context = (
            had_reveal
            or after_grace
            or assist_level in ("highlight", "ban", "mix", "direct_answer", "shortlist")
            or src.startswith("scripted")
        )
        category = "correct_after_feedback" if pedagogical_context else "correct_incidental"
        meta = self._new_feedback_meta(
            raw_feedback_kind="correct_pick",
            feedback_category=category,
        )
        meta["semantic_credit_type"] = "positive"

        feedback_mode = getattr(self.cfg.learner, "pedagogical_feedback_mode", "raw")
        if feedback_mode != "budgeted_v1":
            meta["semantic_credit"] = 1.0
            meta["semantic_credit_reason"] = "raw_mode"
            return meta

        if qs is None:
            meta["semantic_credit_reason"] = "missing_query_state"
            return meta
        if getattr(qs, "positive_update_used", False):
            meta["semantic_credit_reason"] = "ticket_spent"
            return meta

        if after_grace:
            meta["semantic_credit"] = 1.0
            meta["semantic_credit_reason"] = "after_grace"
        elif assist_level in ("highlight", "mix"):
            meta["semantic_credit"] = 1.0
            meta["semantic_credit_reason"] = "after_cue"
        elif had_reveal or assist_level == "ban":
            meta["semantic_credit"] = 1.0
            meta["semantic_credit_reason"] = "after_reveal"
        elif pedagogical_context:
            meta["semantic_credit"] = 1.0
            meta["semantic_credit_reason"] = "pedagogical_correct"
        else:
            meta["semantic_credit"] = float(getattr(self.cfg.learner, "incidental_correct_credit", 0.5))
            meta["semantic_credit_reason"] = "incidental_correct"
        return meta

    def _handle_correct_pick(self, qs) -> dict:
        """CLS positive reinforcement on correct pick."""
        lcfg = self.cfg.learner
        mode = lcfg.correct_pick_learning_mode
        eta = lcfg.eta_correct_pick
        n_em_ov = lcfg.correct_pick_n_em_override
        meta = self._compute_correct_semantic_credit(qs)

        if mode == "off":
            return meta
        credit = float(meta.get("semantic_credit", 0.0))
        if credit <= 0.0:
            return meta

        assist_lv = getattr(qs, 'assist_level', 'none')
        if assist_lv in ('direct_answer', 'shortlist'):
            sem_key = "direct_answer"
        elif assist_lv in ('highlight', 'ban', 'mix'):
            sem_key = "correct_assisted"
        else:
            sem_key = "correct_unassisted"
        self._sem_counters[f"{sem_key}_attempted"] += 1

        src = getattr(qs, 'learning_event_source', 'incidental')
        if sem_key == "correct_unassisted":
            if src == "scripted_self_correct":
                self._src_counters["cu_scripted_self_correct_att"] += 1
            elif src == "scripted_direct_correct":
                self._src_counters["cu_scripted_direct_correct_att"] += 1
            else:
                self._src_counters["cu_incidental_att"] += 1
        elif sem_key == "direct_answer":
            if src == "scripted_then_answer":
                self._src_counters["da_then_answer_att"] += 1
            elif src == "scripted_direct_answer":
                self._src_counters["da_direct_answer_att"] += 1
            else:
                self._src_counters["da_incidental_shortlist_att"] += 1

        omega = self._compute_assist_omega(qs)
        effective_eta = eta * omega * credit
        meta["semantic_update_attempted"] = True
        if getattr(self.cfg.learner, "pedagogical_feedback_mode", "raw") == "budgeted_v1":
            qs.positive_update_used = True
            meta["positive_ticket_consumed"] = True
        if effective_eta < 1.0 and self.rng.random() >= effective_eta:
            return meta
        self._sem_counters[f"{sem_key}_applied"] += 1
        meta["semantic_update_applied"] = True

        if sem_key == "correct_unassisted":
            if src == "scripted_self_correct":
                self._src_counters["cu_scripted_self_correct_app"] += 1
            elif src == "scripted_direct_correct":
                self._src_counters["cu_scripted_direct_correct_app"] += 1
            else:
                self._src_counters["cu_incidental_app"] += 1
        elif sem_key == "direct_answer":
            if src == "scripted_then_answer":
                self._src_counters["da_then_answer_app"] += 1
            elif src == "scripted_direct_answer":
                self._src_counters["da_direct_answer_app"] += 1
            else:
                self._src_counters["da_incidental_shortlist_app"] += 1

        correct_text = self._get_correct_pick_text(qs)
        if correct_text is None:
            return meta

        from ..interfaces import Example
        pos_example = Example(
            words=correct_text,
            output=list(qs.target_output),
        )

        if mode == "cortex_em":
            score_before = 0.0
            try:
                score_before = self._scorer.score_option(
                    list(qs.target_output), correct_text)
            except Exception:
                pass

            self._scorer.incremental_study([pos_example], n_em_override=n_em_ov)

            try:
                score_after = self._scorer.score_option(
                    list(qs.target_output), correct_text)
                self._reveal_shifts.append(float(score_after - score_before))
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown correct_pick_learning_mode: {mode}")
        return meta

    def _get_correct_pick_text(self, qs) -> "list | None":
        """Get the text of the correct option from qs.

        Priority: qs.true_program (set at QueryState creation, always accurate).
        Fallback: scan qs.menu for is_correct option.
        Returns None if not found.
        """
        # true_program is the ground-truth program (set at QueryState init)
        if hasattr(qs, 'true_program') and qs.true_program:
            return list(qs.true_program)
        # Fallback: scan menu for is_correct option
        for opt in qs.menu:
            if opt.is_correct:
                return list(opt.text)
        return None

    def _compute_assist_omega(self, qs) -> float:
        """Compute assist discount weight ω = rho_assist ** rank.

        Only gates SEMANTIC updates.  Risk/damage updates are NOT gated.
        """
        from ..interfaces_assist import ASSIST_RANK
        rho = self.cfg.learner.rho_assist
        if rho >= 1.0:
            return 1.0  # no discount
        level = getattr(qs, 'assist_level', 'none')
        rank = ASSIST_RANK.get(level, 0)
        return rho ** rank


    def _handle_reveal(self, example, qs=None) -> dict:
        """Route reveal / wrong-pick feedback through reveal_learning_mode.

        NOTE: This function is now a 'wrong-feedback handler', not just a
        reveal handler. In nonreveal mode, example.output is the env-generated
        RevealEvent output and MUST NOT be consumed as semantic supervision.
        Only example.words (the program text) and qs.target_output are safe.

        Args:
            example: Example(words=wrong_program, output=revealed_output)
                     In nonreveal mode, .output is untrusted — do not use.
            qs: QueryState for current query. Required for nonreveal_negative mode
                to access qs.target_output without it being the revealed output.

        eta_reveal controls the probability that a reveal actually triggers a
        scorer update (incremental_study). This is the primary knob to separate
        "tutor helped learner learn" vs "learner self-learned from reveals":
            eta_reveal = 1.0  → always update (default, current behaviour)
            eta_reveal = 0.5  → 50% chance of update per reveal
            eta_reveal = 0.0  → reveal never updates scorer (no self-learning)
        """
        mode = self.cfg.learner.reveal_learning_mode
        eta = self.cfg.learner.eta_reveal  # in [0, 1]
        feedback_mode = self.cfg.env.feedback_mode  # "reveal" | "nonreveal"
        reveal_event = qs.reveal_history[-1] if (qs is not None and getattr(qs, "reveal_history", None)) else None
        meta = self._compute_wrong_semantic_credit(qs, reveal_event)

        # ── Nonreveal check: guard against accidental output consumption ──
        # If env is in nonreveal mode, example.output must never be used for
        # semantic update. Route to nonreveal_negative branch or off.
        if feedback_mode == "nonreveal" and mode not in ("off", "nonreveal_negative"):
            # Safety override: treat as nonreveal_negative if configured,
            # otherwise silently skip (never consume revealed output).
            neg_mode = self.cfg.learner.negative_evidence_mode
            if neg_mode == "exact_program_target" and qs is not None:
                mode = "nonreveal_negative"
            else:
                return meta  # nonreveal + no negative evidence = off

        if mode == "cortex_em":
            # Measure score BEFORE update (PosteriorShiftPerReveal)
            score_before = 0.0
            probe_target = (
                list(qs.target_output)
                if (qs is not None and qs.target_output)
                else list(example.output)
            )
            if hasattr(self._scorer, 'score_option') and probe_target:
                try:
                    score_before = self._scorer.score_option(
                        probe_target, list(example.words))
                except Exception:
                    score_before = 0.0

            credit = float(meta.get("semantic_credit", 0.0))
            if credit <= 0.0:
                return meta
            # Stochastic gate: only update scorer with probability eta_reveal
            # then apply assist discount (semantic only, not risk)
            updated = False
            omega = self._compute_assist_omega(qs) if qs is not None else 1.0
            effective_eta = eta * omega * credit
            # Instrumentation: wrong_reveal is attempted before gate
            if hasattr(self, '_sem_counters'):
                self._sem_counters["wrong_reveal_attempted"] += 1
            # Phase 6.5: event-source wrong-reveal counter
            wr_src = getattr(qs, 'learning_event_source', 'incidental') if qs else 'incidental'
            if hasattr(self, '_src_counters'):
                if wr_src.startswith('scripted'):
                    self._src_counters["wr_scripted_att"] += 1
                else:
                    self._src_counters["wr_incidental_att"] += 1
            meta["semantic_update_attempted"] = True
            if qs is not None and getattr(self.cfg.learner, "pedagogical_feedback_mode", "raw") == "budgeted_v1":
                qs.contrastive_update_used = True
                meta["contrastive_ticket_consumed"] = True
            if effective_eta >= 1.0 or self.rng.random() < effective_eta:
                self._scorer.incremental_study([example])
                updated = True
                meta["semantic_update_applied"] = True
                if hasattr(self, '_sem_counters'):
                    self._sem_counters["wrong_reveal_applied"] += 1
                if hasattr(self, '_src_counters'):
                    if wr_src.startswith('scripted'):
                        self._src_counters["wr_scripted_app"] += 1
                    else:
                        self._src_counters["wr_incidental_app"] += 1

            # Measure score AFTER update and record shift
            if updated and probe_target:
                try:
                    score_after = self._scorer.score_option(
                        probe_target, list(example.words))
                    self._reveal_shifts.append(float(score_after - score_before))
                except Exception:
                    pass
        elif mode == "off":
            # Do nothing — cortex stays untouched
            pass
        elif mode == "negative_memory":
            # Store in negative memory (penalty applied during scoring)
            credit = float(meta.get("semantic_credit", 0.0))
            if credit <= 0.0:
                return meta
            omega = self._compute_assist_omega(qs) if qs is not None else 1.0
            effective_eta = eta * omega * credit
            meta["semantic_update_attempted"] = True
            if qs is not None and getattr(self.cfg.learner, "pedagogical_feedback_mode", "raw") == "budgeted_v1":
                qs.contrastive_update_used = True
                meta["contrastive_ticket_consumed"] = True
            if self._negative_memory is not None and (effective_eta >= 1.0 or self.rng.random() < effective_eta):
                self._negative_memory.add(example.words)
                meta["semantic_update_applied"] = True
        elif mode == "nonreveal_negative":
            # Nonreveal mode: do NOT consume revealed output.
            # Record (program, target_output) as negative evidence in scorer.
            # The scorer will penalise this option on future queries with the
            # same target_output. qs.target_output is the query's Y*, which is
            # safe to use (it's visible to the learner as the task target).
            assert feedback_mode == "nonreveal", (
                "nonreveal_negative mode requires feedback_mode='nonreveal'"
            )
            credit = float(meta.get("semantic_credit", 0.0))
            if credit <= 0.0:
                return meta
            meta["semantic_update_attempted"] = True
            if qs is not None and getattr(self.cfg.learner, "pedagogical_feedback_mode", "raw") == "budgeted_v1":
                qs.contrastive_update_used = True
                meta["contrastive_ticket_consumed"] = True
            if qs is not None and hasattr(self._scorer, 'add_negative_evidence'):
                eta_neg = self.cfg.learner.eta_negative
                if eta_neg is None:
                    eta_neg = self.cfg.learner.eta_reveal
                omega = self._compute_assist_omega(qs) if qs is not None else 1.0
                effective_eta = eta_neg * omega * credit
                if effective_eta >= 1.0 or self.rng.random() < effective_eta:
                    self._scorer.add_negative_evidence(
                        words=list(example.words),
                        target_output=list(qs.target_output),
                        weight=effective_eta,
                    )
                    meta["semantic_update_applied"] = True
        else:
            raise ValueError(f"Unknown reveal_learning_mode: {mode}")
        return meta

    def reveal_shift_stats(self) -> dict:
        """Return PosteriorShiftPerReveal statistics for the current block."""
        shifts = self._reveal_shifts
        if not shifts:
            return {
                "posterior_shift_per_reveal": 0.0,
                "posterior_shift_n_reveals": 0,
                "posterior_shift_positive_rate": 0.0,
            }
        shifts_arr = np.array(shifts)
        return {
            "posterior_shift_per_reveal":   float(np.mean(shifts_arr)),
            "posterior_shift_n_reveals":    len(shifts),
            "posterior_shift_positive_rate": float(np.mean(shifts_arr > 0)),
        }

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
