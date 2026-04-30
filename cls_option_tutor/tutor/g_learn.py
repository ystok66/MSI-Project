"""
g_learn.py — G_learn proxy estimators for Q_T tutor decision-making.

Unified objective:
    J = delta_EvalAcc - beta * DeathRate - gamma * TimeoutRate

G_learn estimation methods:
    A. ProbeEvaluator ("probe"):
       G_probe(S) = ProbeAcc(theta_L_after_S) - ProbeAcc(theta_L_current)
       Uses deepcopy to guarantee scorer state isolation (no side-effects).

    B. OracleDistanceSurrogate ("oracle_surrogate"):
       G_surrogate(S) = D_total(theta* || theta_L) - D_total(theta* || theta_L_after_S)
       D_total = Dirichlet-KL(role) + L2-distance(emit mean)
       NOTE: This is NOT full NIG KL — emit part uses mean L2 as a tractable surrogate.

State isolation guarantee (critical):
    ProbeEvaluator uses copy.deepcopy(scorer) before any simulation.
    The original scorer is NEVER modified during G_learn estimation.
    test_probe_restore_exact_state verifies this invariant.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
import numpy as np
from scipy.special import gammaln, digamma

if TYPE_CHECKING:
    from ..env.state import BlockState, QueryState
    from ..interfaces import Example, Option
    from ..learner.cls_adapter import CLSSemanticPosterior
    from ..learner.learner_agent import LearnerAgent

# ── ROLES constant (mirrors ns_learner.ns_primitives) ──────────────────────
ROLES = ['EMIT', 'REPEAT', 'SWAP_INFIX', 'CONCAT_INFIX', 'OVER_INFIX']


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _dirichlet_kl(alpha_p: np.ndarray, alpha_q: np.ndarray) -> float:
    """KL(Dir(alpha_p) || Dir(alpha_q)) — exact, closed form.

    D_KL = ln B(alpha_q) - ln B(alpha_p)
           + sum_r (alpha_p_r - alpha_q_r) * [psi(alpha_p_r) - psi(sum alpha_p)]

    where B(alpha) = prod Gamma(alpha_r) / Gamma(sum alpha_r)
    """
    alpha_p = np.maximum(alpha_p, 1e-30)
    alpha_q = np.maximum(alpha_q, 1e-30)

    sum_p = alpha_p.sum()
    sum_q = alpha_q.sum()

    # ln B(alpha) = sum ln Gamma(alpha_r) - ln Gamma(sum alpha)
    ln_B_p = gammaln(alpha_p).sum() - gammaln(sum_p)
    ln_B_q = gammaln(alpha_q).sum() - gammaln(sum_q)

    kl = (ln_B_q - ln_B_p
          + np.sum((alpha_p - alpha_q) * (digamma(alpha_p) - digamma(sum_p))))
    return float(max(0.0, kl))


def _take_scorer_snapshot(scorer) -> Optional[Dict]:
    """Snapshot CLS scorer state for exact equality testing.

    Returns dict covering:
      - _support_history lengths and word content (hash)
      - cortex.library: per-word role_counts, repeat_counts, emit_stats, color_counts
      - _studied flag
    """
    if scorer is None or not hasattr(scorer, '_agent') or scorer._agent is None:
        return None

    library = scorer._agent.cortex.library
    snapshot = {
        '_support_n': len(getattr(scorer, '_support_history', [])),
        '_studied': getattr(scorer, '_studied', False),
        'library': {}
    }
    for word, concept in library.items():
        snapshot['library'][word] = {
            'role_counts':   dict(concept.role_counts),
            'repeat_counts': dict(concept.repeat_counts),
            'emit_stats': {
                'sum_w':   float(concept.emit_stats['sum_w']),
                'sum_wx':  concept.emit_stats['sum_wx'].copy(),
                'sum_wx2': concept.emit_stats['sum_wx2'].copy(),
            },
            'color_counts': dict(concept.color_counts),
        }
    return snapshot


def assert_scorer_state_equal(scorer, snapshot: Dict, tol: float = 1e-9) -> None:
    """Assert scorer state matches snapshot exactly (within tol).

    Raises AssertionError with detailed diff if mismatch detected.
    Used by test_probe_restore_exact_state.
    """
    if snapshot is None:
        return

    assert hasattr(scorer, '_agent') and scorer._agent is not None, \
        "Scorer lost its _agent during probe (state pollution!)"

    library = scorer._agent.cortex.library
    snap_lib = snapshot['library']

    for word, s in snap_lib.items():
        assert word in library, f"Word '{word}' disappeared from library after probe"
        c = library[word]

        # role_counts
        for r, v in s['role_counts'].items():
            got = c.role_counts.get(r, 0.0)
            assert abs(got - v) < tol, \
                f"role_counts[{word}][{r}] changed: {v:.6g} → {got:.6g}"

        # repeat_counts
        for k, v in s['repeat_counts'].items():
            got = c.repeat_counts.get(k, 0.0)
            assert abs(got - v) < tol, \
                f"repeat_counts[{word}][{k}] changed: {v:.6g} → {got:.6g}"

        # emit_stats
        es = s['emit_stats']
        assert abs(c.emit_stats['sum_w'] - es['sum_w']) < tol, \
            f"emit_stats.sum_w[{word}] changed"
        assert np.allclose(c.emit_stats['sum_wx'],  es['sum_wx'],  atol=tol), \
            f"emit_stats.sum_wx[{word}] changed"
        assert np.allclose(c.emit_stats['sum_wx2'], es['sum_wx2'], atol=tol), \
            f"emit_stats.sum_wx2[{word}] changed"

        # color_counts
        for col, v in s['color_counts'].items():
            got = c.color_counts.get(col, 0.0)
            assert abs(got - v) < tol, \
                f"color_counts[{word}][{col}] changed: {v:.6g} → {got:.6g}"

    assert len(scorer._support_history) == snapshot['_support_n'], \
        f"_support_history length changed: {snapshot['_support_n']} → {len(scorer._support_history)}"


# ══════════════════════════════════════════════════════════════════════════════
# Shared Utility: probabilistic reveal simulation
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_expected_reveals(
    qs,
    shortlist_indices: list,
    learner_agent,
    threshold: float = 0.1,
    p_a: Optional[np.ndarray] = None,
) -> list:
    """Compute the expected set of reveals from a shortlist interaction.

    For each non-correct option d in S, estimates the probability that d
    gets picked before j* (using the learner's current policy), and returns
    an Example-like dict for each option with p_reveal > threshold.

    When p_a is provided (tier-aware distribution from SparseTutorAgent),
    it is used DIRECTLY instead of recomputing pick probs internally.
    This is the correct path for Bayes Gate sparse interventions where
    BAN/HIGHLIGHT are tier-based reorderings rather than logit perturbations.

    Formula (sequential pick approximation):
        p_reveal(d) = p_d / (p_d + p_j*)   for each non-correct d in S

    Args:
        qs:                QueryState for the current query.
        shortlist_indices: proposed shortlist option indices (used to select
                           the relevant subset when p_a is None).
        learner_agent:     LearnerAgent (oracle access to policy + scorer).
        threshold:         minimum p_reveal to include a reveal (default 0.1).
        p_a:               Optional tier-aware probability distribution over
                           the FULL active menu (same order as qs.menu).
                           When provided, skip internal pick-prob computation.

    Returns:
        List of Example objects for options with p_reveal > threshold.
        Returns [] if no correct option in shortlist or policy unavailable.
    """
    from ..interfaces import Example

    menu_by_idx = {opt.index: opt for opt in qs.menu}
    shortlist_opts = [menu_by_idx[i] for i in shortlist_indices if i in menu_by_idx]

    correct_in_S     = [o for o in shortlist_opts if o.is_correct]
    non_correct_in_S = [o for o in shortlist_opts if not o.is_correct]

    if not correct_in_S or not non_correct_in_S:
        return []

    # ── Path A: use caller-provided tier-aware p_a ─────────────────
    if p_a is not None and len(p_a) > 0:
        # p_a is indexed over the FULL active menu in menu order
        # Build index map: option.index → position in qs.menu
        idx_to_pos = {opt.index: i for i, opt in enumerate(qs.menu)}

        j_star = correct_in_S[0]
        p_j_star = float(p_a[idx_to_pos[j_star.index]]) if j_star.index in idx_to_pos else 1e-6

        reveals = []
        for d in non_correct_in_S:
            p_d = float(p_a[idx_to_pos[d.index]]) if d.index in idx_to_pos else 0.0
            p_reveal = p_d / (p_d + p_j_star + 1e-10)
            if p_reveal > threshold:
                rendered = d.rendered_output or []
                reveals.append(
                    Example(words=list(d.text), output=list(rendered))
                )
        return reveals

    # ── Path B: internal pick-prob computation (legacy / shortlist path) ──
    policy = getattr(learner_agent, 'policy', None)
    scorer = getattr(learner_agent, '_scorer', None)
    if policy is None or scorer is None:
        # Fallback: no policy info → treat all as 50% reveal
        if threshold <= 0.5:
            return [
                Example(words=list(o.text), output=list(o.rendered_output))
                for o in non_correct_in_S
            ]
        return []

    # ── Compute pick probs over shortlist ──────────────────────────
    if policy.attention is not None:
        attn = policy.attention.weights
    else:
        L = len(qs.target_output)
        attn = np.ones(L) / max(L, 1)

    K = len(shortlist_opts)
    sem   = np.zeros(K)
    danger = np.zeros(K)
    unc    = np.zeros(K)

    for i, opt in enumerate(shortlist_opts):
        if scorer is not None:
            sem[i] = scorer.score_option(
                qs.target_output, opt.text, attention_weights=attn
            )
        if policy.danger_head is not None:
            mu, u = policy.danger_head.predict(opt.danger_vec)
            danger[i] = mu
            unc[i] = u

    lcfg = learner_agent.cfg.learner
    U = (lcfg.alpha_sem * sem
         - lcfg.alpha_risk * danger
         - lcfg.alpha_unc * unc)
    shifted    = U - np.max(U)
    exp_u      = np.exp(lcfg.beta_L * shifted)
    probs      = exp_u / (exp_u.sum() + 1e-10)
    prob_by_idx = {opt.index: float(p)
                   for opt, p in zip(shortlist_opts, probs)}

    # ── Filter by p_reveal ─────────────────────────────────────────
    j_star    = correct_in_S[0]
    p_j_star  = prob_by_idx.get(j_star.index, 1e-6)

    reveals = []
    for d in non_correct_in_S:
        p_d = prob_by_idx.get(d.index, 0.0)
        p_reveal = p_d / (p_d + p_j_star + 1e-10)
        if p_reveal > threshold:
            reveals.append(
                Example(words=list(d.text), output=list(d.rendered_output))
            )
    return reveals


class ProbeEvaluator:
    """G_learn via shadow probe evaluation (Method A).

    G_probe(S) = ProbeAcc(theta_L_after_S) - ProbeAcc(theta_L_current)

    Isolation guarantee:
      All simulation happens on copy.deepcopy(scorer).
      The original scorer is guaranteed unmodified after estimate().

    Args:
        n_probe: number of hold-out probe queries to evaluate on
        seed: RNG seed for probe query sampling
    """

    def __init__(self, n_probe: int = 20, seed: int = 99):
        self.n_probe = n_probe
        self.rng = np.random.default_rng(seed)
        self._probe_queries: Optional[List] = None   # set externally per block

    def set_probe_queries(self, probe_queries: List) -> None:
        """Set the fixed probe query set for this block."""
        self._probe_queries = list(probe_queries)

    def probe_accuracy(self, scorer, probe_queries: List) -> float:
        """Score learner accuracy on probe_queries using current scorer.

        NEVER calls incremental_study — read-only access to scorer.score_option().
        """
        if not probe_queries or scorer is None:
            return 0.0

        correct = 0
        for q in probe_queries:
            menu = q.get('menu', [])
            target = q.get('target_output', [])
            if not menu or not target:
                continue
            scores = [
                scorer.score_option(target, opt.get('text', []))
                for opt in menu
            ]
            best_idx = int(np.argmax(scores))
            if menu[best_idx].get('is_correct', False):
                correct += 1

        return correct / max(len(probe_queries), 1)

    def simulate_expected_reveals(
        self,
        qs,
        shortlist_indices: List[int],
        learner_agent,
        p_a: Optional[np.ndarray] = None,
    ) -> List:
        """Compute expected reveals from shortlist experience.

        Delegates to module-level _simulate_expected_reveals() to share
        the same probabilistic filtering logic with OracleDistanceSurrogate.
        """
        return _simulate_expected_reveals(
            qs, shortlist_indices, learner_agent, threshold=0.1, p_a=p_a
        )

    def _get_pick_probs_for_opts(self, qs, opts, learner_agent) -> np.ndarray:
        """Recompute pick probs for a specific option list."""
        policy = learner_agent.policy
        scorer = learner_agent._scorer
        K = len(opts)
        if K == 0:
            return np.array([])

        if policy.attention is not None:
            attn = policy.attention.weights
        else:
            L = len(qs.target_output)
            attn = np.ones(L) / max(L, 1)

        sem = np.zeros(K)
        for i, opt in enumerate(opts):
            if scorer is not None:
                sem[i] = scorer.score_option(
                    qs.target_output, opt.text,
                    attention_weights=attn
                )

        danger = np.zeros(K)
        unc = np.zeros(K)
        if policy.danger_head is not None:
            for i, opt in enumerate(opts):
                mu, u = policy.danger_head.predict(opt.danger_vec)
                danger[i] = mu
                unc[i] = u

        lcfg = learner_agent.cfg.learner
        U = (lcfg.alpha_sem * sem
             - lcfg.alpha_risk * danger
             - lcfg.alpha_unc * unc)

        shifted = U - np.max(U)
        exp_u = np.exp(lcfg.beta_L * shifted)
        return exp_u / (exp_u.sum() + 1e-10)

    def estimate(
        self,
        scorer,
        qs,
        shortlist_indices: List[int],
        learner_agent,
        probe_queries: Optional[List] = None,
        p_a: Optional[np.ndarray] = None,
        feedback_mode: str = "reveal",
    ) -> float:
        """Estimate G_learn for the proposed shortlist.

        Args:
            scorer: CLSSemanticPosterior (NEVER modified by this method)
            qs: current QueryState
            shortlist_indices: proposed shortlist
            learner_agent: for pick_prob computation
            probe_queries: override probe set (uses self._probe_queries if None)
            p_a: Optional tier-aware probability distribution (from SparseTutorAgent).
                 When provided, bypasses internal pick-prob recomputation.
            feedback_mode: "reveal" (default) or "nonreveal".
                 Determines whether the simulated update uses study() (reveal)
                 or add_negative_evidence() (nonreveal) on the scorer copy.

        Returns:
            G_probe = ProbeAcc_after - ProbeAcc_before  (float, can be negative)
        """
        if probe_queries is None:
            probe_queries = self._probe_queries or []
        if not probe_queries:
            return 0.0

        # Step 1: snapshot for integrity check
        snapshot = _take_scorer_snapshot(scorer)

        # Step 2: acc_before (original scorer, read-only)
        acc_before = self.probe_accuracy(scorer, probe_queries)

        # Step 3: simulate expected wrong picks (same probabilistic filter)
        try:
            simulated_reveals = self.simulate_expected_reveals(
                qs, shortlist_indices, learner_agent, p_a=p_a
            )
        except TypeError:
            # Backward-compatible test/mocking path: older call sites and unit
            # tests patch simulate_expected_reveals(qs, shortlist, learner)
            # without the optional p_a keyword.
            simulated_reveals = self.simulate_expected_reveals(
                qs, shortlist_indices, learner_agent
            )
        if not simulated_reveals:
            # No expected wrong picks -> G_learn = 0
            assert_scorer_state_equal(scorer, snapshot)   # integrity check
            return 0.0

        # Step 4: deepcopy isolation
        scorer_sim = copy.deepcopy(scorer)

        if feedback_mode == "reveal":
            # Reveal path: simulate CLS re-study with observed true outputs.
            new_history = list(scorer_sim._support_history) + simulated_reveals
            scorer_sim.study(
                new_history,
                n_em=getattr(scorer_sim, '_n_em', 2),
                use_hpc=getattr(scorer_sim, '_use_hpc', True),
            )
        else:
            # Nonreveal path: simulate negative evidence accumulation.
            # Example.output is the env-generated revealed output and MUST NOT
            # be consumed here. Only .words and qs.target_output are safe.
            if hasattr(scorer_sim, 'add_negative_evidence'):
                target_output = list(qs.target_output) if qs is not None else []
                lcfg = getattr(getattr(learner_agent, 'cfg', None), 'learner', None)
                eta_neg = getattr(lcfg, 'eta_negative', 1.0) if lcfg else 1.0
                for rev in simulated_reveals:
                    scorer_sim.add_negative_evidence(
                        words=list(rev.words),
                        target_output=target_output,
                        weight=eta_neg,
                    )
            # scorer_sim._support_history is NOT grown in nonreveal mode

        # Step 5: acc_after on simulated scorer
        acc_after = self.probe_accuracy(scorer_sim, probe_queries)

        # Step 6: assert original scorer is untouched
        assert_scorer_state_equal(scorer, snapshot)

        return float(acc_after - acc_before)


# ══════════════════════════════════════════════════════════════════════════════
# Method B: OracleDistanceSurrogate
# ══════════════════════════════════════════════════════════════════════════════

class OracleDistanceSurrogate:
    """G_learn via oracle parameter distance (Method B).

    G_surrogate(S) = D_total(theta* || theta_L) - D_total(theta* || theta_L_after_S)

    D_total = sum_w [D_KL_Dirichlet_role(w) + d_L2_emit(w)]

    IMPORTANT: This is NOT full NIG KL.
      - Role:  Exact Dirichlet KL (using scipy gammaln + digamma)
      - Emit:  L2 distance between posterior means (surrogate, not NIG KL)
    Use ProbeEvaluator (Method A) as the ground-truth proxy.
    This is a cheap alternative for when probe is too expensive.

    Args:
        oracle_agent: CLSAgent trained on full correct support (set via init_oracle)
        missing_word_penalty: penalty per oracle-vocab word absent from learner
    """

    def __init__(self, missing_word_penalty: float = 5.0):
        self.missing_word_penalty = missing_word_penalty
        self._oracle_agent = None
        self._alpha_prior = None

    def init_oracle(self, support: List, grammar, cfg) -> None:
        """Train oracle CLSAgent on the full correct support set.

        MUST be called at the start of each block before estimate().
        """
        try:
            import sys, os
            _basic = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC'))
            if _basic not in sys.path:
                sys.path.insert(0, _basic)

            from cls_learner.agent import CLSAgent
            from cls_learner.interfaces import Example as CLSExample

            self._oracle_agent = CLSAgent(cfg)
            self._oracle_agent.reset_episode()

            cls_support = [CLSExample(words=ex.words, output=ex.output)
                           for ex in support]
            self._oracle_agent.study(cls_support, verbose=False)
            self._alpha_prior = self._oracle_agent.priors.alpha

        except Exception as e:
            self._oracle_agent = None
            self._alpha_prior = None

    def _concept_distance(self, oracle_concept, learner_concept) -> float:
        """Distance for one word: Dirichlet KL (role) + L2 mean (emit)."""
        if self._alpha_prior is None:
            return 0.0

        total = 0.0

        # ── Role: exact Dirichlet KL ──────────────────────────────
        alpha_o = np.array([
            self._alpha_prior.get(r, 1.0) + oracle_concept.role_counts.get(r, 0.0)
            for r in ROLES
        ])
        alpha_l = np.array([
            self._alpha_prior.get(r, 1.0) + learner_concept.role_counts.get(r, 0.0)
            for r in ROLES
        ])
        total += _dirichlet_kl(alpha_o, alpha_l)

        # ── Emit: L2 mean distance (surrogate, not true NIG KL) ───
        sw_o = oracle_concept.emit_stats['sum_w']
        sw_l = learner_concept.emit_stats['sum_w']
        if sw_o > 1e-10:
            mu_o = oracle_concept.emit_stats['sum_wx'] / sw_o
            if sw_l > 1e-10:
                mu_l = learner_concept.emit_stats['sum_wx'] / sw_l
            else:
                mu_l = np.zeros_like(mu_o)
            total += float(np.sum((mu_o - mu_l) ** 2))

        return total

    def total_distance(self, learner_agent) -> float:
        """Compute D_total(theta* || theta_L) over shared vocabulary."""
        if self._oracle_agent is None:
            return 0.0

        scorer = learner_agent._scorer
        if not hasattr(scorer, '_agent') or scorer._agent is None:
            return float('inf')

        oracle_lib  = self._oracle_agent.cortex.library
        learner_lib = scorer._agent.cortex.library

        shared = set(oracle_lib.keys()) & set(learner_lib.keys())
        missing = set(oracle_lib.keys()) - set(learner_lib.keys())

        d_total = sum(
            self._concept_distance(oracle_lib[w], learner_lib[w])
            for w in shared
        )
        d_total += len(missing) * self.missing_word_penalty
        return float(d_total)

    def estimate(
        self,
        scorer,
        qs,
        shortlist_indices: List[int],
        learner_agent,
        probe_queries=None,   # unused (kept for interface compatibility)
        feedback_mode: str = "reveal",
    ) -> float:
        """Estimate G_surrogate(S) = D_before - D_after.

        Uses _simulate_expected_reveals() with pick-probability filtering
        (same as ProbeEvaluator) to avoid the 100%-reveal over-estimation bug.
        Only reveals that learner is likely (p_reveal > 0.1) to encounter
        are included in the simulated scorer update.

        When feedback_mode="nonreveal": OracleDistanceSurrogate measures cortex
        library parameters which are unaffected by negative evidence (no study()
        call in nonreveal mode). Returns 0.0 to avoid a misleading positive
        estimate. Use ProbeEvaluator for nonreveal G_learn estimation.
        """
        if self._oracle_agent is None:
            return 0.0

        # Guard: oracle_surrogate + nonreveal not yet supported
        if feedback_mode == "nonreveal":
            return 0.0

        # D_before
        d_before = self.total_distance(learner_agent)

        # Simulate reveals (probabilistic filter, same as ProbeEvaluator)
        simulated_reveals = _simulate_expected_reveals(
            qs, shortlist_indices, learner_agent, threshold=0.1
        )
        if not simulated_reveals:
            return 0.0

        # Deepcopy isolation
        scorer_sim = copy.deepcopy(scorer)
        new_history = list(scorer_sim._support_history) + simulated_reveals
        scorer_sim.study(
            new_history,
            n_em=getattr(scorer_sim, '_n_em', 2),
            use_hpc=getattr(scorer_sim, '_use_hpc', True),
        )

        # Temporarily swap for distance computation
        learner_sim = copy.copy(learner_agent)   # shallow copy
        learner_sim._scorer = scorer_sim

        d_after = self.total_distance(learner_sim)

        return float(d_before - d_after)


# ══════════════════════════════════════════════════════════════════════════════
# Unified GLearnEstimator
# ══════════════════════════════════════════════════════════════════════════════

class GLearnEstimator:
    """Unified G_learn interface.

    Args:
        mode: "none" | "probe" | "oracle_surrogate"
        n_probe: probe query count (mode="probe" only)
        seed: RNG seed
    """

    def __init__(
        self,
        mode: str = "none",
        n_probe: int = 20,
        seed: int = 99,
    ):
        assert mode in ("none", "probe", "oracle_surrogate"), \
            f"Unknown mode: {mode}"
        self.mode = mode
        self._probe_eval: Optional[ProbeEvaluator] = None
        self._oracle_surrogate: Optional[OracleDistanceSurrogate] = None

        if mode == "probe":
            self._probe_eval = ProbeEvaluator(n_probe=n_probe, seed=seed)
        elif mode == "oracle_surrogate":
            self._oracle_surrogate = OracleDistanceSurrogate()

    def init_block(self, support: List, grammar, cfg,
                   probe_queries: Optional[List] = None) -> None:
        """Initialize per-block state.

        probe_queries: for mode="probe", the fixed hold-out query list.
        support + grammar + cfg: for mode="oracle_surrogate" to train oracle.
        """
        if self.mode == "probe" and self._probe_eval is not None:
            self._probe_eval.set_probe_queries(probe_queries or [])

        if self.mode == "oracle_surrogate" and self._oracle_surrogate is not None:
            self._oracle_surrogate.init_oracle(support, grammar, cfg)

    def estimate(
        self,
        scorer,
        qs,
        shortlist_indices: List[int],
        learner_agent,
        probe_queries: Optional[List] = None,
        p_a: Optional[np.ndarray] = None,
        feedback_mode: str = "reveal",
    ) -> float:
        """Estimate G_learn for the proposed shortlist.

        Args:
            p_a: Optional tier-aware probability distribution (from SparseTutorAgent).
                 When provided, forwarded to ProbeEvaluator to bypass internal
                 pick-prob computation.
            feedback_mode: "reveal" (default) or "nonreveal".
                 Passed through to the underlying backend estimator.

        Returns 0.0 for mode="none".
        Returns G_probe for mode="probe".
        Returns G_surrogate for mode="oracle_surrogate" (or 0.0 if nonreveal).
        """
        if self.mode == "none" or not shortlist_indices:
            return 0.0

        if self.mode == "probe" and self._probe_eval is not None:
            return self._probe_eval.estimate(
                scorer, qs, shortlist_indices, learner_agent, probe_queries,
                p_a=p_a,
                feedback_mode=feedback_mode,
            )

        if self.mode == "oracle_surrogate" and self._oracle_surrogate is not None:
            return self._oracle_surrogate.estimate(
                scorer, qs, shortlist_indices, learner_agent,
                feedback_mode=feedback_mode,
            )

        return 0.0


    def get_scorer_snapshot(self, scorer) -> Optional[Dict]:
        """Return current scorer state snapshot (for external integrity checks)."""
        return _take_scorer_snapshot(scorer)

    def assert_scorer_unmodified(self, scorer, snapshot: Optional[Dict]) -> None:
        """Assert scorer state matches snapshot (raises AssertionError if not)."""
        if snapshot is not None:
            assert_scorer_state_equal(scorer, snapshot)
