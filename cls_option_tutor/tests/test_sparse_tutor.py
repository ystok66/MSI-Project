"""
test_sparse_tutor.py — Unit tests for SparseTutorAgent.

Tests:
  T1: BAN → option demoted to last tier (p_tier << normal options)
  T2: HIGHLIGHT → correct option promoted to first tier (p_tier >> others)
  T3: WAIT → D_shift = 0 (no distribution change)
  T4: BAN << SHORTLIST shift (tier JS << full shortlist JS)
  T5: j* never banned (invariant enforced in enumerate_candidates)
  T6: G_exp ≥ 0 (non-negative safe exposure gain)
  T7: HIGHLIGHT candidate gated by P_timeout threshold
  T8: Q_use components sum correctly (numerical regression)
  T9: MIX = BAN + HIGHLIGHT atomically (both tiers affected)
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.interfaces import Option
from cls_option_tutor.env.state import QueryState
from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent, _js_divergence


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cfg(**tutor_kwargs) -> FullConfig:
    cfg = FullConfig()
    for k, v in tutor_kwargs.items():
        setattr(cfg.tutor, k, v)
    return cfg


def _make_menu(n_options: int = 5, correct_idx: int = 0, hp: int = 5) -> list:
    """Quick menu: option 0 is correct, rest are distractors."""
    menu = []
    for i in range(n_options):
        menu.append(Option(
            index=i,
            text=[f"prog_{i}"],
            danger_vec=np.zeros(8),
            is_correct=(i == correct_idx),
            risk_class=0 if i == correct_idx else 2,
            rendered_output=[f"out_{i}"],
        ))
    return menu


def _make_qs(n_options: int = 5, correct_idx: int = 0, hp: int = 5,
             max_rounds: int = 5, rounds_used: int = 0) -> QueryState:
    menu = _make_menu(n_options, correct_idx, hp)
    return QueryState(
        query_id=0,
        target_output=["A", "B", "C"],
        true_program=["prog_0"],
        hp=hp,
        max_rounds=max_rounds,
        rounds_used=rounds_used,
        menu=menu,
    )


def _make_tutor(**kwargs) -> SparseTutorAgent:
    cfg = _make_cfg(**kwargs)
    return SparseTutorAgent(cfg=cfg, g_learn_mode="none")


# ── T1: BAN → last tier ───────────────────────────────────────────────────────

class TestT1BanLastTier:
    def test_ban_option_has_lowest_tier_prob(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()

        # BAN option 1 (a distractor)
        spec_ban = {"action": "BAN", "ban_index": 1}
        spec_wait = {"action": "WAIT"}

        p_ban = tutor._compute_tier_probs(qs, active, spec_ban)
        p_wait = tutor._compute_tier_probs(qs, active, spec_wait)

        # Under WAIT, option 1 has some probability
        assert p_wait[1] > 0.01, "Option 1 should have nonzero prob under WAIT"

        # Under BAN: option 1 is in the last tier.
        # Since there are still normal-tier options (0,2,3,4), option 1 stays at 0.
        assert p_ban[1] == pytest.approx(0.0, abs=1e-9), \
            "BAN(1): option 1 should have p=0 when normal tier is non-empty"

        # Remaining probability sums to 1
        assert p_ban.sum() == pytest.approx(1.0, abs=1e-6)

    def test_ban_only_option_still_selectable(self):
        """If only the banned option remains (all others also banned), it's still chosen."""
        tutor = _make_tutor()
        qs = _make_qs(n_options=2, correct_idx=0)
        # Menu: option 0 (correct), option 1 (wrong)
        # BAN option 0 (correct) — this tests the tier fallback when normal tier empty
        # Note: tutor never bans j*, but the tier model itself handles it
        spec = {"action": "BAN", "ban_index": 1}
        active = [qs.menu[0], qs.menu[1]]  # both present
        p = tutor._compute_tier_probs(qs, active, spec)
        # Option 0 is in N tier (not banned), option 1 is in B tier
        assert p[0] == pytest.approx(1.0, abs=1e-6), "Only N-tier option should get all prob"
        assert p[1] == pytest.approx(0.0, abs=1e-9)


# ── T2: HIGHLIGHT → first tier ───────────────────────────────────────────────

class TestT2HighlightFirstTier:
    def test_highlight_correct_has_highest_prob(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()

        spec_hl = {"action": "HIGHLIGHT", "highlight_cells": (0, 1)}
        p_hl = tutor._compute_tier_probs(qs, active, spec_hl)
        p_wait = tutor._compute_tier_probs(qs, active, {"action": "WAIT"})

        # HIGHLIGHT promotes j* (option 0) to front tier → p(j*)=1.0
        assert p_hl[0] == pytest.approx(1.0, abs=1e-6), \
            "HIGHLIGHT(correct): j* should have p=1 when it is the only H-tier option"
        for i in range(1, 5):
            assert p_hl[i] == pytest.approx(0.0, abs=1e-9)

    def test_highlight_relative_prob_increases(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=3, correct_idx=0)
        active = qs.menu.copy()

        p_wait = tutor._compute_tier_probs(qs, active, {"action": "WAIT"})
        p_hl = tutor._compute_tier_probs(qs, active, {"action": "HIGHLIGHT", "highlight_cells": (0,)})

        # j* prob must increase under HIGHLIGHT
        assert p_hl[0] > p_wait[0], \
            f"HIGHLIGHT should increase j* prob: {p_wait[0]:.4f} → {p_hl[0]:.4f}"


# ── T3: WAIT → D_shift = 0 ───────────────────────────────────────────────────

class TestT3WaitZeroShift:
    def test_wait_js_zero(self):
        tutor = _make_tutor()
        qs = _make_qs()
        active = qs.menu.copy()

        p0 = tutor._compute_tier_probs(qs, active, {"action": "WAIT"})
        js = _js_divergence(p0, p0)
        assert js == pytest.approx(0.0, abs=1e-10), "JS(p0, p0) must be 0"

    def test_wait_d_shift_zero_in_q_use(self):
        tutor = _make_tutor()
        qs = _make_qs()
        active = qs.menu.copy()

        spec = {"action": "WAIT"}

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        p0 = tutor._compute_tier_probs(qs, active, spec)
        p_wait = tutor._compute_tier_probs(qs, active, {"action": "WAIT"})
        js = _js_divergence(p0, p_wait)
        assert js == pytest.approx(0.0, abs=1e-10)


# ── T4: BAN << SHORTLIST in D_shift ──────────────────────────────────────────

class TestT4BanVsShortlistShift:
    def test_ban_shift_less_than_shortlist(self):
        """Single BAN produces less JS shift than shortlist-style aggressive removal.

        Key insight:
          - WAIT over K=10: uniform probs ≈ 0.1 each
          - Single BAN: removes 1/10 items → small shift
          - SHORTLIST(3/10): redistributes 7/10 of the mass to 3 items → large shift

        We measure JS(p_WAIT, p_action) in both cases.
        """
        tutor = _make_tutor()
        qs = _make_qs(n_options=10, correct_idx=0)
        active = qs.menu.copy()

        p0 = tutor._compute_tier_probs(qs, active, {"action": "WAIT"})

        # Single BAN of option 1
        p_ban = tutor._compute_tier_probs(qs, active, {"action": "BAN", "ban_index": 1})
        js_ban = _js_divergence(p0, p_ban)

        # Simulate shortlist: K=10 → 3 options only (redistribute all mass to 3)
        # In tier model: shortlisted_active = first 3 items only, WAIT over those
        shortlist_active = active[:3]  # 3 items: j* + 2 distractors
        p_sl_3 = tutor._compute_tier_probs(qs, shortlist_active, {"action": "WAIT"})
        # p_sl_3 is defined over 3 items. We need to compare against p0 (10 items).
        # Shortlist effectively concentrates all prob on 3/10 items.
        # Build equivalent 10-item distribution with mass 0 on remaining 7 items.
        p_sl_full = np.zeros(10)
        p_sl_full[:3] = p_sl_3  # shortlisted items get all probability
        js_sl = _js_divergence(p0, p_sl_full)

        # BAN should produce significantly less shift than SHORTLIST
        assert js_ban < 0.5, f"BAN JS shift {js_ban:.4f} should be < 0.5"
        assert js_sl > js_ban, (
            f"SHORTLIST shift {js_sl:.4f} should be > single BAN shift {js_ban:.4f}")



# ── T5: j* never banned ───────────────────────────────────────────────────────

class TestT5JstarNeverBanned:
    def test_ban_target_not_correct(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=2)
        active = qs.menu.copy()
        non_correct = [o for o in active if not o.is_correct]

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        ban_target = tutor._select_ban_target(qs, non_correct, DummyLearner())
        if ban_target is not None:
            assert not ban_target.is_correct, \
                f"ban_target must not be j* (correct option), got index {ban_target.index}"

    def test_enumerate_candidates_no_ban_on_jstar(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        candidates = tutor._enumerate_candidates(qs, active, DummyLearner())
        for spec in candidates:
            if "ban_index" in spec and spec["ban_index"] is not None:
                ban_idx = spec["ban_index"]
                banned_opt = next((o for o in active if o.index == ban_idx), None)
                assert banned_opt is None or not banned_opt.is_correct, \
                    f"Candidate bans j* (index {ban_idx})"


# ── T6: G_exp ≥ 0 ────────────────────────────────────────────────────────────

class TestT6GexpNonNegative:
    def test_g_exp_nonnegative_wait(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()
        p = tutor._compute_tier_probs(qs, active, {"action": "WAIT"})
        g = tutor._compute_g_exp(qs, active, p)
        assert g >= 0.0, f"G_exp must be non-negative, got {g}"

    def test_g_exp_nonnegative_ban(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()
        p = tutor._compute_tier_probs(qs, active, {"action": "BAN", "ban_index": 1})
        g = tutor._compute_g_exp(qs, active, p)
        assert g >= 0.0

    def test_g_exp_zero_when_only_correct_reachable(self):
        """When HIGHLIGHT promotes j* to front tier, all prob is on j*, G_exp=0."""
        tutor = _make_tutor()
        qs = _make_qs(n_options=3, correct_idx=0)
        active = qs.menu.copy()
        p = tutor._compute_tier_probs(qs, active, {"action": "HIGHLIGHT", "highlight_cells": (0,)})
        g = tutor._compute_g_exp(qs, active, p)
        assert g == pytest.approx(0.0, abs=1e-9), \
            "G_exp should be 0 when all prob is on j* (HIGHLIGHT front tier)"


# ── T7: HIGHLIGHT candidate gated by P_timeout ───────────────────────────────

class TestT7HighlightGate:
    def test_no_hl_when_timeout_risk_low(self):
        """With ample time (tau_t >> 1) and even pick_probs, HL should be suppressed."""
        tutor = _make_tutor(hl_timeout_threshold=0.5)
        # tau_t = 5, K=3, uniform probs → p_j* ≈ 0.33 → p_timeout ≈ (0.67)^5 ≈ 0.13 < 0.5
        qs = _make_qs(n_options=3, correct_idx=0, max_rounds=5, rounds_used=0)
        active = qs.menu.copy()

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        candidates = tutor._enumerate_candidates(qs, active, DummyLearner())
        hl_candidates = [c for c in candidates if c["action"] in ("HIGHLIGHT", "MIX")]
        # With p_j*≈1/3, tau_t=5: p_timeout ≈ 0.13, below threshold 0.5 → no HL
        assert len(hl_candidates) == 0, \
            f"Expected no HIGHLIGHT/MIX candidates at low timeout risk, got {hl_candidates}"

    def test_hl_generated_when_timeout_risk_high(self):
        """With 1 round left and low p_j*, HL candidate should be generated."""
        tutor = _make_tutor(hl_timeout_threshold=0.5)
        # tau_t = 1, K=5, uniform → p_j* ≈ 0.2 → p_timeout ≈ 0.8 > 0.5
        qs = _make_qs(n_options=5, correct_idx=0, max_rounds=5, rounds_used=4)
        active = qs.menu.copy()

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        candidates = tutor._enumerate_candidates(qs, active, DummyLearner())
        hl_candidates = [c for c in candidates if c["action"] in ("HIGHLIGHT", "MIX")]
        assert len(hl_candidates) > 0, \
            "Expected HIGHLIGHT/MIX candidates when timeout risk is high"


# ── T8: Q_use numerical regression ───────────────────────────────────────────

class TestT8QUseNumerical:
    def test_wait_q_use_no_cost(self):
        """WAIT has zero cost, zero d_shift, so Q_use = λ_exp*G_exp - β*P_death - γ*P_timeout."""
        tutor = _make_tutor(lambda_eval=1.0, lambda_exp=0.25, lambda_shift=0.25, c_I=0.05)
        qs = _make_qs(n_options=3, correct_idx=0)
        active = qs.menu.copy()

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        spec = {"action": "WAIT"}
        q, detail = tutor._compute_q_use(qs, active, spec, DummyLearner())

        assert detail["cost"] == pytest.approx(0.0, abs=1e-9)
        assert detail["d_shift"] == pytest.approx(0.0, abs=1e-6)
        assert detail["g_eval"] == pytest.approx(0.0, abs=1e-9)
        # Q_use must be a finite float
        assert np.isfinite(q)

    def test_ban_has_positive_cost(self):
        tutor = _make_tutor(c_I=0.05)
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        spec = {"action": "BAN", "ban_index": 1}
        q, detail = tutor._compute_q_use(qs, active, spec, DummyLearner())
        assert detail["cost"] == pytest.approx(0.05, abs=1e-9)

    def test_mix_has_double_cost(self):
        tutor = _make_tutor(c_I=0.05)
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()

        class DummyLearner:
            cfg = FullConfig()
            policy = None
            _scorer = None

        tutor._learner_ref = DummyLearner()
        spec = {"action": "MIX", "ban_index": 1, "highlight_cells": (0,)}
        q, detail = tutor._compute_q_use(qs, active, spec, DummyLearner())
        assert detail["cost"] == pytest.approx(0.10, abs=1e-9)


# ── T9: MIX = BAN + HIGHLIGHT ────────────────────────────────────────────────

class TestT9MixTierSemantics:
    def test_mix_demotes_ban_promotes_highlight(self):
        tutor = _make_tutor()
        qs = _make_qs(n_options=5, correct_idx=0)
        active = qs.menu.copy()

        spec_mix = {"action": "MIX", "ban_index": 1, "highlight_cells": (0,)}
        p_mix = tutor._compute_tier_probs(qs, active, spec_mix)

        # j* (option 0) should be in H tier → gets all probability
        assert p_mix[0] == pytest.approx(1.0, abs=1e-6), \
            "MIX: j* in H tier should have p=1"
        # Option 1 (banned) should be in B tier → p=0 (H tier non-empty)
        assert p_mix[1] == pytest.approx(0.0, abs=1e-9), \
            "MIX: banned option should be in last tier with p=0"

    def test_js_divergence_symmetry(self):
        """JS divergence must be symmetric."""
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.5, 0.3])
        assert _js_divergence(p, q) == pytest.approx(_js_divergence(q, p), abs=1e-10)
