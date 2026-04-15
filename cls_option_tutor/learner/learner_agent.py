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
            for hint_evt in qs.risk_hint_history:
                opt = None
                for o in qs.menu:
                    if o.index == hint_evt.option_index:
                        opt = o
                        break
                if opt is not None:
                    self.policy.observe_risk_hint(
                        opt.danger_vec, eta=hint_evt.eta)

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
            # Detect new BAN events in tutor trace (cross-query persistent signal)
            # We process all BAN actions in the trace to update n_t.
            # Note: uses qs.menu (full menu) to look up danger_vec.
            m_ban = self.cfg.env.danger_dim
            for ts in block.tutor_trace:
                if (getattr(ts, 'action', '') == 'BAN'
                        and ts.ban_index is not None):
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
                    block.in_teaching_phase
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
                    self._handle_reveal(new_example, qs=qs)

        elif step.action == "pick" and step.correct is True:
            # ── Correct pick: optional positive reinforcement ───────────
            # Learner confirmed (j*.text, target_output) — a complete positive example.
            # Only triggers if correct_pick_learning_mode != "off" and in teaching phase.
            if (block.in_teaching_phase
                    and self._is_cls_scorer()
                    and self.cfg.learner.correct_pick_learning_mode != "off"):
                self._handle_correct_pick(qs)

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

    def _handle_correct_pick(self, qs) -> None:
        """CLS positive reinforcement on correct pick.

        Uses (j*.text, target_output) as a lightweight positive supervision signal.
        Only called when:
          - block.in_teaching_phase
          - _is_cls_scorer() is True
          - correct_pick_learning_mode != "off"

        Uses n_em_override=correct_pick_n_em_override (default 1) for lighter
        EM than full wrong-reveal restudy, to reduce overfitting risk.

        Stochastic gate: updates with probability eta_correct_pick.
        Score shift is recorded in _reveal_shifts for PosteriorShift tracking.
        """
        lcfg = self.cfg.learner
        mode = lcfg.correct_pick_learning_mode
        eta = lcfg.eta_correct_pick
        n_em_ov = lcfg.correct_pick_n_em_override

        if mode == "off":
            return

        # Stochastic gate
        if eta < 1.0 and self.rng.random() >= eta:
            return

        # Get correct option text
        correct_text = self._get_correct_pick_text(qs)
        if correct_text is None:
            return

        from ..interfaces import Example
        pos_example = Example(
            words=correct_text,
            output=list(qs.target_output),
        )

        if mode == "cortex_em":
            # Measure score BEFORE (PosteriorShiftPerReveal tracking)
            score_before = 0.0
            try:
                score_before = self._scorer.score_option(
                    list(qs.target_output), correct_text)
            except Exception:
                pass

            # Lightweight EM: n_em_override=1 (not full restudy strength)
            self._scorer.incremental_study([pos_example], n_em_override=n_em_ov)

            # Record shift (positive expected — j* score should increase)
            try:
                score_after = self._scorer.score_option(
                    list(qs.target_output), correct_text)
                self._reveal_shifts.append(float(score_after - score_before))
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown correct_pick_learning_mode: {mode}")

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

    def _handle_reveal(self, example, qs=None) -> None:
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
                return  # nonreveal + no negative evidence = off

        if mode == "cortex_em":
            # Measure score BEFORE update (PosteriorShiftPerReveal)
            score_before = 0.0
            if hasattr(self._scorer, 'score_option') and example.output:
                try:
                    score_before = self._scorer.score_option(
                        list(example.output), list(example.words))
                except Exception:
                    score_before = 0.0

            # Stochastic gate: only update scorer with probability eta_reveal
            updated = False
            if eta >= 1.0 or self.rng.random() < eta:
                self._scorer.incremental_study([example])
                updated = True

            # Measure score AFTER update and record shift
            if updated and example.output:
                try:
                    score_after = self._scorer.score_option(
                        list(example.output), list(example.words))
                    self._reveal_shifts.append(float(score_after - score_before))
                except Exception:
                    pass
        elif mode == "off":
            # Do nothing — cortex stays untouched
            pass
        elif mode == "negative_memory":
            # Store in negative memory (penalty applied during scoring)
            if self._negative_memory is not None:
                self._negative_memory.add(example.words)
        elif mode == "nonreveal_negative":
            # Nonreveal mode: do NOT consume revealed output.
            # Record (program, target_output) as negative evidence in scorer.
            # The scorer will penalise this option on future queries with the
            # same target_output. qs.target_output is the query's Y*, which is
            # safe to use (it's visible to the learner as the task target).
            assert feedback_mode == "nonreveal", (
                "nonreveal_negative mode requires feedback_mode='nonreveal'"
            )
            if qs is not None and hasattr(self._scorer, 'add_negative_evidence'):
                eta_neg = self.cfg.learner.eta_negative
                if eta_neg is None:
                    eta_neg = self.cfg.learner.eta_reveal
                if eta_neg >= 1.0 or self.rng.random() < eta_neg:
                    self._scorer.add_negative_evidence(
                        words=list(example.words),
                        target_output=list(qs.target_output),
                        weight=eta_neg,
                    )
        else:
            raise ValueError(f"Unknown reveal_learning_mode: {mode}")

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
