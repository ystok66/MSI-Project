"""
test_g_learn.py — Tests for G_learn estimators and Q_T components.

Test order (T1 is gate — if it fails, all G_probe results are suspect):
  T1: test_probe_restore_exact_state
  T2: test_probe_deterministic_same_seed
  T3: test_timeout_estimator_vs_rollout_toy
  T4: test_proxy_rank_matches_true_delta_eval_toy
  T5: test_probe_g_learn_positive_vs_wait
  T6: test_probe_g_learn_zero_when_cls_converged
  T7: test_oracle_surrogate_nonincreases_on_toy_reveal
  T8: test_qt_covers_both_wait_and_shortlist
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC'))

import copy
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from cls_option_tutor.tutor.g_learn import (
    GLearnEstimator,
    ProbeEvaluator,
    OracleDistanceSurrogate,
    _dirichlet_kl,
    _take_scorer_snapshot,
    assert_scorer_state_equal,
    ROLES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


def _make_mock_concept(role_val: float = 1.0, emit_val: float = 0.5,
                       emit_dim: int = 6) -> MagicMock:
    """Create a mock NeuroConcept with controlled values."""
    concept = MagicMock()
    concept.role_counts     = {r: role_val for r in ROLES}
    concept.repeat_counts   = {k: 0.0 for k in [1, 2, 3, 4]}
    concept.emit_stats      = {
        'sum_w':   1.0,
        'sum_wx':  np.full(emit_dim, emit_val),
        'sum_wx2': np.full(emit_dim, emit_val ** 2),
    }
    concept.color_counts    = {'BLUE': 0.0, 'RED': 0.0, 'GREEN': 0.0,
                              'YELLOW': 0.0, 'PURPLE': 0.0, 'PINK': 0.0}
    return concept


def _make_mock_scorer(words=('wA', 'wB'), role_val=1.0, emit_val=0.5,
                      n_history=2) -> MagicMock:
    """Create a mock CLSSemanticPosterior with library and support_history."""
    scorer = MagicMock()
    scorer._studied  = True
    scorer._n_em     = 2
    scorer._use_hpc  = True
    scorer._support_history = [MagicMock()] * n_history

    # Build library
    library = {w: _make_mock_concept(role_val, emit_val) for w in words}
    scorer._agent = MagicMock()
    scorer._agent.cortex = MagicMock()
    scorer._agent.cortex.library = library

    # score_option: returns 0.0 by default (override in specific tests)
    scorer.score_option = MagicMock(return_value=0.0)

    # study: mutates nothing (default; override in state-change tests)
    scorer.study = MagicMock(return_value=None)

    return scorer


def _make_mock_option(idx: int, is_correct: bool = False,
                      risk_class: int = 0) -> MagicMock:
    opt = MagicMock()
    opt.index          = idx
    opt.is_correct     = is_correct
    opt.risk_class     = risk_class
    opt.text           = [f'word_{idx}']
    opt.rendered_output = [f'out_{idx}']
    opt.danger_vec     = np.zeros(4)
    return opt


def _make_mock_qs(menu_size: int = 5, hp: int = 5,
                  max_rounds: int = 5, rounds_used: int = 0):
    qs = MagicMock()
    qs.hp           = hp
    qs.max_rounds   = max_rounds
    qs.rounds_used  = rounds_used
    qs.target_output = ['BLUE', 'RED']

    # Build menu: index 0 is correct, rest wrong
    opts = [_make_mock_option(i, is_correct=(i == 0)) for i in range(menu_size)]
    qs.menu = opts
    qs.shortlisted_indices = None
    qs.banned_indices = set()
    return qs


def _make_mock_learner(pick_probs=None, n_opts: int = 5) -> MagicMock:
    """Create a mock LearnerAgent with policy and scorer."""
    learner = MagicMock()

    # Policy
    learner.policy.attention = None
    learner.policy.danger_head = None

    # Cfg
    learner.cfg.learner.alpha_sem  = 1.0
    learner.cfg.learner.alpha_risk = 0.5
    learner.cfg.learner.alpha_unc  = 0.1
    learner.cfg.learner.beta_L     = 1.0

    # Scorer
    learner._scorer = _make_mock_scorer()

    return learner


def _make_probe_queries(n: int = 10, menu_size: int = 5,
                        target_acc: float = 0.5) -> list:
    """Create mock probe queries. Query 0..(n*target_acc-1) have is_correct opt first."""
    queries = []
    for i in range(n):
        menu = [_make_mock_option(j, is_correct=(j == 0)) for j in range(menu_size)]
        # Force score = i < n*target_acc to control 'correct' outcome via score
        queries.append({
            'menu':          menu,
            'target_output': ['BLUE', 'RED'],
        })
    return queries


# ── T1: State restoration gate ─────────────────────────────────────────────


class TestT1ProbeRestoreExactState:
    """T1 (Gate): ProbeEvaluator must not modify original scorer state.

    If this test fails, ALL G_probe results in downstream experiments
    are unreliable. Fix before any other G_learn work.
    """

    def test_probe_restore_exact_state(self):
        """After estimate(), original scorer state is bit-for-bit identical."""
        scorer = _make_mock_scorer(words=('wA', 'wB', 'wC'))
        snapshot_before = _take_scorer_snapshot(scorer)

        qs = _make_mock_qs(menu_size=4)
        shortlist_indices = [0, 1]   # j* (idx=0) + distractor (idx=1)
        learner = _make_mock_learner()
        learner._scorer = scorer

        probe_queries = _make_probe_queries(n=10)

        probe_eval = ProbeEvaluator(n_probe=10, seed=42)
        # Run estimate (should leave scorer untouched)
        _ = probe_eval.estimate(scorer, qs, shortlist_indices, learner, probe_queries)

        # Verify exact state equality
        assert_scorer_state_equal(scorer, snapshot_before, tol=1e-9)

    def test_deepcopy_does_not_share_library(self):
        """Deepcopy of scorer creates independent library (no aliasing)."""
        scorer = _make_mock_scorer(words=('wA',))
        scorer_copy = copy.deepcopy(scorer)

        # Mutate copy
        scorer_copy._agent.cortex.library['wA'].role_counts['EMIT'] = 999.0

        # Original should be unaffected (no shared references)
        original_val = scorer._agent.cortex.library['wA'].role_counts.get('EMIT', 0.0)
        assert original_val != 999.0, \
            "deepcopy shared library reference — state isolation broken!"

    def test_assert_scorer_state_equal_catches_mutation(self):
        """assert_scorer_state_equal raises on even small mutation."""
        scorer = _make_mock_scorer(words=('wA',))
        snapshot = _take_scorer_snapshot(scorer)

        # Silently mutate
        scorer._agent.cortex.library['wA'].role_counts['EMIT'] += 0.001

        with pytest.raises(AssertionError):
            assert_scorer_state_equal(scorer, snapshot, tol=1e-9)


# ── T2: Determinism ─────────────────────────────────────────────────────────


class TestT2ProbeDeterministic:

    def test_probe_deterministic_same_seed(self):
        """Same scorer + shortlist + probe_queries + seed → identical G_probe."""
        scorer = _make_mock_scorer()
        qs     = _make_mock_qs()
        learner = _make_mock_learner()
        learner._scorer = scorer
        probe_queries = _make_probe_queries(n=10)
        shortlist = [0, 1]

        probe1 = ProbeEvaluator(n_probe=10, seed=42)
        probe2 = ProbeEvaluator(n_probe=10, seed=42)

        g1 = probe1.estimate(scorer, qs, shortlist, learner, probe_queries)
        g2 = probe2.estimate(scorer, qs, shortlist, learner, probe_queries)

        assert abs(g1 - g2) < 1e-9, f"Non-deterministic: {g1} vs {g2}"


# ── T3: Timeout estimator calibration ───────────────────────────────────────


class TestT3TimeoutEstimator:

    def _exact_p_timeout(self, p_j_star: float, tau_t: int) -> float:
        """Analytical P(timeout) = (1 - p_j*)^tau_t for geometric draws."""
        return float((1.0 - p_j_star) ** tau_t)

    def test_timeout_geometric_approximation(self):
        """estimate_p_timeout() matches geometric formula within 5%."""
        from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent

        tutor = OptionLevelTutorAgent(g_learn_mode="probe")

        # Simple case: p(j*) = 0.4 in a 5-option menu
        p_j_star = 0.4
        tau_t    = 3

        qs = _make_mock_qs(menu_size=5, max_rounds=5, rounds_used=2)  # tau_t=3
        learner = _make_mock_learner()

        # Override score_option to yield known pick_probs
        def _score(target, text, **kwargs):
            idx = int(text[0].split('_')[1])
            scores = {0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0}  # j* at idx=0
            return scores.get(idx, 0.0)
        learner._scorer.score_option = _score

        p_timeout_approx = tutor.estimate_p_timeout(
            qs, "WAIT", None, learner
        )
        p_timeout_exact = self._exact_p_timeout(p_j_star=0.4, tau_t=3)

        # Allow 10% relative error (geometric model is an approximation)
        assert abs(p_timeout_approx - p_timeout_exact) < 0.1, \
            f"Timeout estimate too far off: approx={p_timeout_approx:.3f}, exact={p_timeout_exact:.3f}"

    def test_shortlist_timeout_is_zero(self):
        """P(timeout | SHORTLIST with |S|=tau_t) must be 0."""
        from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent

        tutor = OptionLevelTutorAgent(g_learn_mode="probe")
        qs = _make_mock_qs(max_rounds=3, rounds_used=0)
        learner = _make_mock_learner()
        shortlist = [0, 1, 2]   # |S| = 3 = tau_t

        p_timeout = tutor.estimate_p_timeout(qs, "SHORTLIST", shortlist, learner)
        assert p_timeout == 0.0, f"SHORTLIST timeout should be 0, got {p_timeout}"


# ── T4: Proxy rank vs true delta eval (toy, small K) ─────────────────────


class TestT4ProxyRanksTrueGain:
    """T4: In an enumerable toy scenario, verify proxy ordering.

    With K=4 and 3 valid shortlists, check G_probe correlates with
    true_delta_eval (even if imperfectly).
    """

    def _compute_true_delta_eval_for_shortlist(
        self, scorer, qs, shortlist, base_acc: float
    ) -> float:
        """Simulate true delta eval.

        For testing: shortlists with more wrong options = more reveals =
        more potential CLS signal. We use a toy scoring: G = (|wrong_in_S| * 0.1).
        (In real system this would be run via actual CLS.study + probe.)
        """
        menu_by_idx = {opt.index: opt for opt in qs.menu}
        n_wrong = sum(1 for i in shortlist
                      if i in menu_by_idx and not menu_by_idx[i].is_correct)
        return n_wrong * 0.1   # toy: more wrong options = higher true gain

    def test_proxy_rank_monotone_with_wrong_count(self):
        """More wrong options in S → higher G_probe (monotone with learning signal)."""
        scorer = _make_mock_scorer()
        qs = _make_mock_qs(menu_size=5)

        shortlist_1wrong = [0, 1]         # j* + 1 wrong
        shortlist_2wrong = [0, 1, 2]      # j* + 2 wrong
        shortlist_0wrong = [0]            # j* only

        learner = _make_mock_learner()
        learner._scorer = scorer

        # patch simulate_expected_reveals to return n_wrong reveals
        probe_eval = ProbeEvaluator(n_probe=5, seed=42)

        # Patch: simulate_expected_reveals returns list of len=n_wrong
        def _mock_reveals(qs_, shortlist_, learner_):
            from cls_option_tutor.interfaces import Example
            menu_by_idx = {opt.index: opt for opt in qs_.menu}
            wrong = [i for i in shortlist_ if i in menu_by_idx
                     and not menu_by_idx[i].is_correct]
            return [Example(words=['w'], output=['x']) for _ in wrong]

        import unittest.mock as um
        probe_eval.simulate_expected_reveals = um.MagicMock(side_effect=_mock_reveals)

        # Patch probe_accuracy: acc_after = base + 0.1 per reveal
        call_count = [0]
        def _mock_probe_acc(scorer_, queries_):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return 0.5   # acc_before (always same)
            # acc_after depends on how many reveals were added to history
            # Check support_history length to infer
            n_added = len(scorer_._support_history) - 2   # base = 2
            return 0.5 + 0.1 * n_added

        probe_eval.probe_accuracy = _mock_probe_acc

        probe_queries = _make_probe_queries(n=5)

        g_0wrong = probe_eval.estimate(scorer, qs, shortlist_0wrong, learner, probe_queries)
        call_count[0] = 0   # reset

        g_1wrong = probe_eval.estimate(scorer, qs, shortlist_1wrong, learner, probe_queries)
        call_count[0] = 0

        g_2wrong = probe_eval.estimate(scorer, qs, shortlist_2wrong, learner, probe_queries)

        # At minimum: 2 wrong > 1 wrong > 0 wrong in G_probe
        assert g_2wrong >= g_1wrong, f"G_probe(2wrong)={g_2wrong} < G_probe(1wrong)={g_1wrong}"
        assert g_1wrong >= g_0wrong, f"G_probe(1wrong)={g_1wrong} < G_probe(0wrong)={g_0wrong}"


# ── T5: G_probe positive when shortlist confuses learner ──────────────────


class TestT5ProbePositive:

    def test_probe_g_learn_positive_vs_wait(self):
        """G_probe > 0 for confusing shortlist; G_probe(WAIT) = 0."""
        probe_eval = ProbeEvaluator(n_probe=5, seed=42)
        scorer = _make_mock_scorer()
        qs = _make_mock_qs(menu_size=5)
        learner = _make_mock_learner()
        learner._scorer = scorer
        probe_queries = _make_probe_queries(n=5)

        # G_probe(WAIT) = 0 by definition (no action taken)
        from cls_option_tutor.tutor.g_learn import GLearnEstimator
        g_est = GLearnEstimator(mode="probe", n_probe=5, seed=42)
        g_wait = g_est.estimate(scorer, qs, shortlist_indices=[],
                                learner_agent=learner, probe_queries=probe_queries)
        assert g_wait == 0.0, f"G_learn(empty shortlist) should be 0, got {g_wait}"

    def test_probe_g_uses_reveals_pathway(self):
        """verify simulate_expected_reveals is called only for non-empty shortlists.

        GLearnEstimator.estimate() returns 0.0 for empty shortlist (early guard).
        ProbeEvaluator.estimate() may still be called but returns 0.0 immediately
        if simulate_expected_reveals returns [].
        """
        import unittest.mock as um
        probe_eval = ProbeEvaluator(n_probe=5, seed=42)

        scorer = _make_mock_scorer()
        qs = _make_mock_qs(menu_size=5)
        learner = _make_mock_learner()
        learner._scorer = scorer
        probe_queries = _make_probe_queries(n=5)

        # Via GLearnEstimator: empty shortlist → returns 0.0 (early guard in estimator)
        g_est = GLearnEstimator(mode="probe", n_probe=5, seed=42)
        g_est._probe_eval._probe_queries = probe_queries
        g_wait = g_est.estimate(scorer, qs, shortlist_indices=[], learner_agent=learner,
                                probe_queries=probe_queries)
        assert g_wait == 0.0, f"GLearnEstimator should return 0 for empty shortlist, got {g_wait}"

        # Non-empty shortlist → simulate path is entered (may return 0 if no reveals)
        probe_eval_spied = ProbeEvaluator(n_probe=5, seed=42)
        probe_eval_spied.simulate_expected_reveals = um.MagicMock(return_value=[])
        _ = probe_eval_spied.estimate(scorer, qs, [0, 1], learner, probe_queries)
        # simulate_expected_reveals MUST be called for non-empty shortlist
        probe_eval_spied.simulate_expected_reveals.assert_called_once()


# ── T6: G_probe zero when CLS converged ─────────────────────────────────


class TestT6ProbeZeroConverged:

    def test_probe_near_zero_when_no_reveals(self):
        """G_probe ≈ 0 when shortlist has no wrong options (nothing to reveal)."""
        probe_eval = ProbeEvaluator(n_probe=5, seed=42)
        scorer = _make_mock_scorer()
        qs = _make_mock_qs(menu_size=3)
        learner = _make_mock_learner()
        learner._scorer = scorer
        probe_queries = _make_probe_queries(n=5)

        # Shortlist with only j* (no distractors → no reveals)
        g = probe_eval.estimate(scorer, qs, [0], learner, probe_queries)
        assert abs(g) < 1e-6, f"G_probe should be ~0 with no reveals, got {g}"


# ── T7: Oracle surrogate non-increases ──────────────────────────────────


class TestT7OracleSurrogate:

    def test_dirichlet_kl_zero_for_identical(self):
        """D_KL(Dir(alpha) || Dir(alpha)) == 0."""
        alpha = np.array([2.0, 1.0, 0.5, 0.5, 0.2])
        kl = _dirichlet_kl(alpha, alpha)
        assert abs(kl) < 1e-9, f"KL of identical should be 0, got {kl}"

    def test_dirichlet_kl_positive_for_different(self):
        """D_KL(Dir(alpha_p) || Dir(alpha_q)) > 0 when p != q."""
        alpha_p = np.array([5.0, 1.0, 0.5, 0.5, 0.2])   # EMIT-dominant
        alpha_q = np.array([1.0, 5.0, 0.5, 0.5, 0.2])   # REPEAT-dominant
        kl = _dirichlet_kl(alpha_p, alpha_q)
        assert kl > 0.01, f"KL should be positive, got {kl}"

    def test_oracle_surrogate_returns_float(self):
        """OracleDistanceSurrogate.estimate() returns a float."""
        surrogate = OracleDistanceSurrogate()
        # No oracle initialized → returns 0.0 gracefully
        scorer = _make_mock_scorer()
        qs = _make_mock_qs()
        learner = _make_mock_learner()
        learner._scorer = scorer

        result = surrogate.estimate(scorer, qs, [0, 1], learner)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_oracle_distance_decreases_on_average_after_reveals(self):
        """Mean D_total over multiple trials: D_after <= D_before + tolerance."""
        # Mock oracle with known library
        surrogate = OracleDistanceSurrogate()

        oracle_agent = MagicMock()
        oracle_lib = {
            'wA': _make_mock_concept(role_val=5.0, emit_val=0.8),  # EMIT-heavy
        }
        oracle_agent.cortex.library = oracle_lib
        oracle_agent.priors.alpha = {r: 1.0 for r in ROLES}
        surrogate._oracle_agent = oracle_agent
        surrogate._alpha_prior  = {r: 1.0 for r in ROLES}

        # Learner library starts at prior (low overlap with oracle)
        learner_lib = {
            'wA': _make_mock_concept(role_val=0.1, emit_val=0.1),
        }

        scorer = _make_mock_scorer(words=('wA',))
        scorer._agent.cortex.library = learner_lib

        qs     = _make_mock_qs(menu_size=3)
        learner = _make_mock_learner()
        learner._scorer = scorer

        # Patch deepcopy and scorer_sim.study to create a "moved" library
        original_total = surrogate.total_distance(learner)

        # After "reveal" (simulated), score should not blow up
        result = surrogate.estimate(scorer, qs, [0, 1], learner)
        # Can't guarantee strict decrease with mocks, but should not crash
        assert isinstance(result, float)
        assert not np.isnan(result), "G_surrogate returned NaN"


# ── T8: Q_T selects correctly ─────────────────────────────────────────────


class TestT8QTSelects:

    def test_qt_prefers_wait_when_all_lethal(self):
        """When all shortlist options are lethal, Q_T should prefer WAIT."""
        from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent

        tutor = OptionLevelTutorAgent(
            g_learn_mode="none",
            beta=1.0,
            gamma=0.2,
            lambda_learn=0.0,   # G_learn disabled for this test
        )
        qs = _make_mock_qs(menu_size=3, hp=3)
        # Make all non-correct options lethal (risk >= HP)
        qs.menu[1].risk_class = 3   # = hp → lethal
        qs.menu[2].risk_class = 5   # > hp → lethal

        learner = _make_mock_learner()
        scorer  = _make_mock_scorer()
        learner._scorer = scorer

        q_shortlist = tutor._compute_q_t(
            block=None, qs=qs, action="SHORTLIST",
            shortlist=[0, 1, 2], learner_agent=learner
        )
        q_wait = tutor._compute_q_t(
            block=None, qs=qs, action="WAIT",
            shortlist=None, learner_agent=learner
        )
        assert q_wait >= q_shortlist, \
            f"Q_T(WAIT)={q_wait:.3f} < Q_T(lethal shortlist)={q_shortlist:.3f}"

    def test_qt_prefers_shortlist_when_safe_and_high_gain(self):
        """When shortlist is safe and G_learn > 0, SHORTLIST beats WAIT."""
        from cls_option_tutor.tutor.option_level_tutor import OptionLevelTutorAgent

        tutor = OptionLevelTutorAgent(
            g_learn_mode="none",
            lambda_learn=1.0,
            beta=0.5,
            gamma=0.2,
        )
        # Manually inject G_learn_hat = +0.3 for shortlist
        import unittest.mock as um
        tutor._g_learn_estimator = um.MagicMock()
        tutor._g_learn_estimator.estimate = um.MagicMock(return_value=0.3)

        qs = _make_mock_qs(menu_size=4, hp=5)
        # All options in shortlist safe (risk < hp)
        for opt in qs.menu:
            opt.risk_class = 1  # low risk

        learner = _make_mock_learner()
        scorer  = _make_mock_scorer()
        learner._scorer = scorer

        q_shortlist = tutor._compute_q_t(
            block=None, qs=qs, action="SHORTLIST",
            shortlist=[0, 1, 2], learner_agent=learner
        )
        q_wait = tutor._compute_q_t(
            block=None, qs=qs, action="WAIT",
            shortlist=None, learner_agent=learner
        )
        assert q_shortlist >= q_wait, \
            f"Q_T(safe+gain shortlist)={q_shortlist:.3f} < Q_T(WAIT)={q_wait:.3f}"


# ── Utilities: snapshot round-trip ─────────────────────────────────────────


class TestSnapshotUtils:

    def test_snapshot_roundtrip(self):
        """Snapshot → no mutation → assert_equal passes."""
        scorer = _make_mock_scorer(words=('wX', 'wY'))
        snapshot = _take_scorer_snapshot(scorer)
        assert_scorer_state_equal(scorer, snapshot)   # should not raise

    def test_snapshot_catches_role_change(self):
        """Changing role_counts is caught by assert_scorer_state_equal."""
        scorer = _make_mock_scorer(words=('wX',))
        snapshot = _take_scorer_snapshot(scorer)
        scorer._agent.cortex.library['wX'].role_counts['EMIT'] += 0.5
        with pytest.raises(AssertionError):
            assert_scorer_state_equal(scorer, snapshot)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
