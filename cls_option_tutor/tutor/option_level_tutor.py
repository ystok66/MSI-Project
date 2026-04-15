"""
option_level_tutor.py — Option-Level Strong Tutor with Q_T objective.

Objective (v2):
    J = delta_EvalAcc - beta * DeathRate - gamma * TimeoutRate

Q_T decision function (per teaching step):
    Q_T(h_t, a) = lambda * G_learn_hat(h_t, a)
               - beta  * P_death_hat(h_t, a)
               - gamma * P_timeout_hat(h_t, a)

G_learn modes:
  "none"             : confusion-first shortlist (original baseline)
  "probe"            : ProbeEvaluator (deepcopy, method A)
  "oracle_surrogate" : OracleDistanceSurrogate (Dirichlet KL + L2 emit, method B)

Protocol (per teaching query step):
  1. Compute tau_t = remaining rounds in this query
  2. Identify lethal options: L_t = {j ∈ A_t : risk(j) >= HP_t}
  3. Build safe candidates: A_safe = A_t \ L_t
  4. If g_learn_mode="none": confusion-first shortlist (original)
     Else: enumerate {WAIT} U {top-K candidates}, select argmax Q_T
  5. Emit SHORTLIST(S) or WAIT

Invariants (verified by tests/test_option_level.py):
  - j* ∈ S                        (correct answer always reachable)
  - |S| = tau_t                    (completable within remaining rounds)
  - no lethal options in S         (safety guaranteed by construction)
  - final choice ∈ S               (enforced by get_active_menu in env)
  - final choice ∉ banned_indices  (enforced by env._do_pick)

Access mode: ORACLE BASELINE
  - Reads learner's live policy state (pick_probs, danger_head, scorer)
  - Reads opt.is_correct directly
  - This is the STRONG TUTOR upper bound; not realistic in deployment
"""
from __future__ import annotations
from typing import List, Optional, Set, Tuple
import numpy as np

from ..config import FullConfig
from ..env.state import BlockState, QueryState
from ..env.option_env import OptionEnv
from ..env.interventions import get_active_menu
from ..interfaces import TutorStep, Option
from ..learner.learner_agent import LearnerAgent


class OptionLevelTutorAgent:
    """Strong oracle tutor with Q_T objective and G_learn proxy.

    Args:
        cfg: FullConfig. Uses env.T_max, env.H_0, env.N_obs/N_teach/N_eval.
        shortlist_mode: distractor selection strategy ("confusion" | "random" | "easiest")
        lethal_threshold: fraction of HP for lethal classification (default=1.0)
        g_learn_mode: G_learn estimation mode:
            "none"             - original confusion-first (no Q_T scoring)
            "probe"            - ProbeEvaluator (Method A, deepcopy isolation)
            "oracle_surrogate" - OracleDistanceSurrogate (Method B, KL+L2)
        lambda_learn: weight for G_learn_hat in Q_T (default 1.0)
        beta: weight for P_death_hat in Q_T (default 0.5)
        gamma: weight for P_timeout_hat in Q_T (default 0.2)
        n_probe: number of probe queries for mode="probe" (default 20)
        n_candidates: number of candidate shortlists to compare in Q_T (default 3)
    """

    def __init__(
        self,
        cfg: Optional[FullConfig] = None,
        shortlist_mode: str = "confusion",
        lethal_threshold: float = 1.0,
        g_learn_mode: str = "none",
        lambda_learn: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.2,
        n_probe: int = 20,
        n_candidates: int = 3,
    ):
        self.cfg = cfg or FullConfig()
        self.shortlist_mode = shortlist_mode
        self.lethal_threshold = lethal_threshold
        self.g_learn_mode = g_learn_mode
        self.lambda_learn = lambda_learn
        self.beta = beta
        self.gamma = gamma
        self.n_probe = n_probe
        self.n_candidates = n_candidates

        # G_learn estimator (initialized per-block in init_block)
        from .g_learn import GLearnEstimator
        self._g_learn_estimator = GLearnEstimator(
            mode=g_learn_mode,
            n_probe=n_probe,
            seed=42,
        )
        self._probe_queries: List = []   # set in init_block

    # ── Main entry ───────────────────────────────────────────────

    def act(
        self,
        block: BlockState,
        env: OptionEnv,
        learner_agent: Optional[LearnerAgent] = None,
    ) -> TutorStep:
        """Execute one tutor turn.

        Observation / Evaluation phase: always WAIT.
        Teaching phase: Safety + Shortlist protocol.
        """
        qs = block.current_query
        if qs is None or qs.done or block.done:
            return env.tutor_act(block, "WAIT")

        if block.in_observation_phase or block.in_evaluation_phase:
            return env.tutor_act(block, "WAIT")

        # Already shortlisted this query → WAIT (one shortlist per query)
        if qs.shortlisted_indices is not None:
            return env.tutor_act(block, "WAIT")

        return self._act_teaching(block, env, learner_agent)

    def _act_teaching(
        self,
        block: BlockState,
        env: OptionEnv,
        learner_agent: Optional[LearnerAgent],
    ) -> TutorStep:
        """Teaching-phase decision.

        When g_learn_mode='none': original confusion-first shortlist.
        Otherwise: enumerate candidate shortlists + WAIT, select argmax Q_T.
        """
        qs = block.current_query
        active = get_active_menu(qs)
        K = len(active)
        if K == 0:
            return env.tutor_act(block, "WAIT")

        tau_t = self._compute_tau_t(qs)
        if tau_t == 0:
            return env.tutor_act(block, "WAIT")

        # Identify lethal options
        lethal_indices = self._get_lethal_indices(qs, active, learner_agent)
        safe_active = [o for o in active if o.index not in lethal_indices]
        K_safe = len(safe_active)

        # Scenario D: ample time and no safety issues → don't intervene
        if K_safe <= tau_t and len(lethal_indices) == 0:
            return env.tutor_act(block, "WAIT")

        if self.g_learn_mode == "none":
            # Original confusion-first shortlist (no Q_T)
            shortlist = self._select_shortlist(qs, safe_active, tau_t, learner_agent)
        else:
            # Q_T-maximizing shortlist selection
            shortlist = self._select_shortlist_by_qt(
                block, qs, safe_active, tau_t, learner_agent
            )

        if shortlist is None:
            return env.tutor_act(block, "WAIT")

        return env.tutor_act(block, "SHORTLIST", shortlist_indices=shortlist)

    # ── Core computations ────────────────────────────────────────

    def _compute_tau_t(self, qs: QueryState) -> int:
        """Remaining rounds in this query.

        tau_t = max_rounds - rounds_used
        Shortlist size will be min(tau_t, K_safe).
        """
        return max(0, qs.max_rounds - qs.rounds_used)

    def _get_lethal_indices(
        self,
        qs: QueryState,
        active: List[Option],
        learner_agent: Optional[LearnerAgent],
    ) -> Set[int]:
        """Return set of option indices where expected damage >= threshold * HP_t.

        Uses learner's DangerHead if available (oracle mode).
        Falls back to risk_class heuristic otherwise.

        Lethal threshold: risk >= HP_t * lethal_threshold
        Default lethal_threshold=1.0 → only options that guarantee KO.
        """
        hp = qs.hp
        lethal = set()
        threshold = hp * self.lethal_threshold

        for opt in active:
            if learner_agent is not None and learner_agent.policy.danger_head is not None:
                # Oracle: use learner's actual danger head prediction
                mu, _ = learner_agent.policy.danger_head.predict(opt.danger_vec)
                if mu >= threshold:
                    lethal.add(opt.index)
            else:
                # Fallback: use risk_class directly
                if opt.risk_class >= threshold:
                    lethal.add(opt.index)
        return lethal

    def _select_shortlist(
        self,
        qs: QueryState,
        safe_active: List[Option],
        tau_t: int,
        learner_agent: Optional[LearnerAgent],
    ) -> Optional[List[int]]:
        """Select shortlist indices S ⊆ safe_active with j* ∈ S, |S| = min(tau_t, |safe_active|).

        Invariants guaranteed:
          - j* ∈ S                    (found in safe_active by is_correct)
          - |S| = target_size         (= min(tau_t, K_safe))
          - no lethal options in S    (safe_active is already filtered)

        Returns None if correct option is not in safe_active (should not happen
        unless j* is lethal with HP=1, which means the query is unsolvable).
        """
        if not safe_active:
            return None

        # Find correct option in safe_active
        correct_opts = [o for o in safe_active if o.is_correct]
        if not correct_opts:
            # j* is lethal (or missing) — shortlist impossible
            return None
        j_star = correct_opts[0]

        non_correct = [o for o in safe_active if not o.is_correct]
        K_safe = len(safe_active)
        target_size = min(tau_t, K_safe)

        # If safe_active already <= tau_t, include everything
        if K_safe <= tau_t:
            return [o.index for o in safe_active]

        # Need to select target_size - 1 distractors
        n_distractors = target_size - 1
        if n_distractors == 0:
            return [j_star.index]

        distractors = self._score_distractors(
            qs, non_correct, n_distractors, learner_agent
        )
        return [j_star.index] + [o.index for o in distractors]

    def _score_distractors(
        self,
        qs: QueryState,
        non_correct: List[Option],
        n: int,
        learner_agent: Optional[LearnerAgent],
    ) -> List[Option]:
        """Select n distractors according to shortlist_mode.

        Modes:
          "confusion" : highest pick_prob (most confusing for learner)
          "easiest"   : lowest pick_prob (easiest to eliminate)
          "random"    : random selection (no learner model needed)
        """
        if n <= 0 or not non_correct:
            return []
        if n >= len(non_correct):
            return non_correct

        if self.shortlist_mode == "random" or learner_agent is None:
            # No model available — random selection
            rng = np.random.default_rng()
            idx = rng.choice(len(non_correct), size=n, replace=False)
            return [non_correct[i] for i in sorted(idx)]

        # Compute learner pick probabilities for non-correct options
        pick_probs = self._get_pick_probs(qs, non_correct, learner_agent)

        if self.shortlist_mode == "easiest":
            # Lowest pick_probs = least confusing
            order = np.argsort(pick_probs)
        else:
            # Default: "confusion" — highest pick_probs = most confusing
            order = np.argsort(pick_probs)[::-1]

        return [non_correct[i] for i in order[:n]]

    def _get_pick_probs(
        self,
        qs: QueryState,
        options: List[Option],
        learner_agent: LearnerAgent,
    ) -> np.ndarray:
        """Estimate learner's pick probability for each option.

        Oracle mode: use learner's actual CLS scorer + DangerHead.
        """
        policy = learner_agent.policy
        scorer = learner_agent._scorer
        K = len(options)
        if K == 0:
            return np.array([])

        # Attention weights (use current if available)
        if policy.attention is not None:
            attn = policy.attention.weights
        else:
            L = len(qs.target_output)
            attn = np.ones(L) / max(L, 1)

        # Semantic scores
        sem = np.zeros(K)
        for i, opt in enumerate(options):
            if scorer is not None:
                sem[i] = scorer.score_option(
                    qs.target_output, opt.text, attention_weights=attn
                )

        # Danger predictions
        danger = np.zeros(K)
        unc = np.zeros(K)
        if policy.danger_head is not None:
            for i, opt in enumerate(options):
                mu, u = policy.danger_head.predict(opt.danger_vec)
                danger[i] = mu
                unc[i] = u

        # Utility = alpha_sem * S_sem - alpha_risk * mu - alpha_unc * u
        lcfg = learner_agent.cfg.learner
        U = (lcfg.alpha_sem * sem
             - lcfg.alpha_risk * danger
             - lcfg.alpha_unc * unc)

        shifted = U - np.max(U)
        exp_u = np.exp(lcfg.beta_L * shifted)
        pick_probs = exp_u / (exp_u.sum() + 1e-10)
        return pick_probs

    # ── Q_T computation (new) ────────────────────────────────────

    def _compute_q_t(
        self,
        block,
        qs,
        action: str,
        shortlist: Optional[List[int]],
        learner_agent,
    ) -> float:
        """Q_T(h_t, a) = lambda * G_hat - beta * P_death - gamma * P_timeout."""
        # ── G_learn_hat ───────────────────────────────────────────
        if action == "WAIT" or not shortlist:
            g_learn = 0.0
        else:
            scorer = getattr(learner_agent, '_scorer', None)
            g_learn = self._g_learn_estimator.estimate(
                scorer, qs, shortlist, learner_agent,
                probe_queries=self._probe_queries or None,
            )

        # ── P_death_hat ───────────────────────────────────────────
        p_death = 0.0
        if shortlist:
            menu_by_idx = {opt.index: opt for opt in qs.menu}
            hp = qs.hp
            all_probs = self._get_pick_probs(
                qs,
                [menu_by_idx[i] for i in shortlist if i in menu_by_idx],
                learner_agent,
            )
            for prob, idx in zip(all_probs, shortlist):
                opt = menu_by_idx.get(idx)
                if opt is not None:
                    is_lethal = (opt.risk_class >= hp * self.lethal_threshold)
                    p_death += float(prob) * float(is_lethal)

        # ── P_timeout_hat ─────────────────────────────────────────
        p_timeout = self.estimate_p_timeout(qs, action, shortlist, learner_agent)

        q = (self.lambda_learn * g_learn
             - self.beta  * p_death
             - self.gamma * p_timeout)
        return float(q)

    def estimate_p_timeout(
        self,
        qs,
        action: str,
        shortlist: Optional[List[int]],
        learner_agent,
    ) -> float:
        """Estimate P(query ends by timeout | action).

        SHORTLIST: returns 0.0 (|S|=tau_t guarantees completion).
        WAIT:      uses learner pick_prob rollout.
                   P(timeout) = P(j* not picked in tau_t draws)
                              ≈ (1 - p_j*)^tau_t
        """
        if action == "SHORTLIST" and shortlist is not None:
            # Invariant: |S| <= tau_t and j* in S → no timeout possible
            tau_t = self._compute_tau_t(qs)
            if len(shortlist) <= tau_t:
                return 0.0
            # Overfull shortlist (shouldn't happen): pessimistic
            return 0.2

        # WAIT: compute via pick_prob of j* over active menu
        tau_t = self._compute_tau_t(qs)
        if tau_t <= 0:
            return 1.0

        try:
            active = get_active_menu(qs)
            if not active:
                return 1.0
            probs = self._get_pick_probs(qs, active, learner_agent)
            correct_idx_in_active = next(
                (i for i, o in enumerate(active) if o.is_correct), None
            )
            if correct_idx_in_active is None:
                return 1.0
            p_j_star = float(probs[correct_idx_in_active])
            # Geometric: P(success in tau_t tries) = 1 - (1 - p)^tau_t
            p_success = 1.0 - (1.0 - p_j_star) ** tau_t
            return float(max(0.0, 1.0 - p_success))
        except Exception:
            # Fallback: crude tau/K heuristic
            K = len(qs.menu)
            return float(max(0.0, 1.0 - tau_t / max(K, 1)))

    def _select_shortlist_by_qt(
        self,
        block,
        qs,
        safe_active: List,
        tau_t: int,
        learner_agent,
    ) -> Optional[List[int]]:
        """Select shortlist via Q_T maximization.

        Compares WAIT vs. top-n_candidates confusion-first shortlists.
        Returns None (→ WAIT) if WAIT has higher Q_T.
        """
        # Candidate 0: WAIT
        q_wait = self._compute_q_t(block, qs, "WAIT", None, learner_agent)
        best_q = q_wait
        best_shortlist: Optional[List[int]] = None

        # Generate candidates: confusion-first shortlists of varying size
        candidates = self._generate_qt_candidates(
            qs, safe_active, tau_t, learner_agent
        )

        for sl in candidates:
            q = self._compute_q_t(block, qs, "SHORTLIST", sl, learner_agent)
            if q > best_q:
                best_q = q
                best_shortlist = sl

        return best_shortlist   # None → tutor will WAIT

    def _generate_qt_candidates(
        self,
        qs,
        safe_active: List,
        tau_t: int,
        learner_agent,
    ) -> List[List[int]]:
        """Generate up to n_candidates shortlist options for Q_T comparison.

        Strategy: build confusion-first shortlists of decreasing size
        (tau_t, tau_t-1, tau_t-2 distractors) to give Q_T a range of
        'tight vs loose' shortlists to compare.
        """
        candidates = []
        correct_opts = [o for o in safe_active if o.is_correct]
        if not correct_opts:
            return candidates
        j_star = correct_opts[0]
        non_correct = [o for o in safe_active if not o.is_correct]

        # Score all distractors by confusion (pick_prob descending)
        if len(non_correct) > 0 and learner_agent is not None:
            try:
                all_probs = self._get_pick_probs(qs, non_correct, learner_agent)
                order = np.argsort(all_probs)[::-1]
                sorted_distractors = [non_correct[i] for i in order]
            except Exception:
                sorted_distractors = non_correct
        else:
            sorted_distractors = non_correct

        K_safe = len(safe_active)
        target = min(tau_t, K_safe)

        # Generate candidates: tau_t distractors, tau_t-1, ...
        for n_d in range(min(target - 1, len(sorted_distractors)),
                         max(-1, min(target - 1, len(sorted_distractors))
                             - self.n_candidates), -1):
            if n_d < 0:
                break
            chosen = [j_star.index] + [sorted_distractors[i].index
                                        for i in range(n_d)]
            candidates.append(chosen)

        return candidates

    # ── Block runner ─────────────────────────────────────────────

    def init_block(self, block, grammar, support) -> None:
        """Initialize per-block state including G_learn estimator."""
        # Build probe queries from support for mode="probe"
        if self.g_learn_mode == "probe" and support:
            from ..interfaces import Example
            # Use support items converted to probe query format
            probe_queries = [
                {
                    'target_output': list(ex.output),
                    'menu': [],   # empty menu for support-based probe
                }
                for ex in support
            ]
            self._probe_queries = probe_queries
        else:
            self._probe_queries = []

        # Initialize G_learn estimator
        self._g_learn_estimator.init_block(
            support=support or [],
            grammar=grammar,
            cfg=getattr(self.cfg, '_cls_cfg', None),
            probe_queries=self._probe_queries,
        )

    def run_block(
        self,
        env: OptionEnv,
        learner: LearnerAgent,
        task_id: str,
        seed: int = 42,
    ) -> BlockState:
        """Run a full block with option-level tutor + learner interaction.

        Mirrors TutorAgent.run_block() interface for drop-in comparison.
        """
        block = env.reset_block(task_id, seed=seed)
        support, _, grammar = env.adapter.load_task(task_id)

        self.init_block(block, grammar, support)
        learner.init_block(block, grammar, support)

        max_steps = len(block.queries) * 20
        steps = 0
        while not block.done and steps < max_steps:
            steps += 1
            qs = block.current_query
            if qs is None or qs.done:
                break

            # Tutor acts first
            self.act(block, env, learner_agent=learner)

            if qs.done:  # SHORTLIST doesn't end query; SKIP would
                continue

            # Learner acts
            learner.act(block, env)

        if not block.done:
            block.done = True

        return block

    # ── Diagnostics ──────────────────────────────────────────────

    def get_block_summary(self, block: BlockState) -> dict:
        """Compute shortlist-specific diagnostics for a completed block."""
        shortlist_steps = [
            t for t in block.tutor_trace if t.action == "SHORTLIST"
        ]
        n_shortlist = len(shortlist_steps)
        avg_size = (
            sum(len(t.shortlist_indices) for t in shortlist_steps) / n_shortlist
            if n_shortlist > 0 else 0.0
        )
        return {
            "n_shortlist": n_shortlist,
            "n_wait": sum(1 for t in block.tutor_trace if t.action == "WAIT"),
            "avg_shortlist_size": round(avg_size, 2),
            "total_correct": block.total_correct,
            "solve_rate": block.total_correct / max(len(block.queries), 1),
            "total_damage": block.total_damage,
        }
