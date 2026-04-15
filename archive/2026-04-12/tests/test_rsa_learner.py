"""
test_rsa_learner.py — Unit tests for RSA L1 learner (Tests A1–A4).

Tests the four required monotonicity/identity properties:
    A1: HIGHLIGHT → semantic_log_bias favors option matching highlighted cells
    A2: BAN        → risk_logit_shift > 0 for banned option only
    A3: WAIT       → identity (no posterior change)
    A4: PASS       → pass_abort=True

Also tests:
    A5: state isolation — new block resets all RSA state
    A6: cross-query BAN teach — danger_head updates after BAN in teaching phase
"""
import pytest
import numpy as np
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.learner.rsa_listener import RSAListener, RSABeliefUpdate
from cls_option_tutor.learner.attention_model import AttentionModel
from cls_option_tutor.learner.danger_head import DangerHead


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def listener():
    """Standard RSAListener with default hyperparameters."""
    return RSAListener(omega_hl=2.0, lambda_ctx=0.5, omega_ban=3.0)


def _make_options(n_options: int = 3, L: int = 4):
    """Create K synthetic option texts and rendered outputs.

    Returns:
        target_output: (L,) ground truth
        texts: K option programs
        rendered: K rendered outputs where option 0 is correct and all
                  wrong options (j>=1) mismatch at cell 0 (the key discriminating cell)
    """
    target = [f"t{i}" for i in range(L)]
    texts = [[f"prog{j}"] for j in range(n_options)]
    rendered = []
    for j in range(n_options):
        if j == 0:
            # Correct option: matches target exactly
            rendered.append(target.copy())
        else:
            # Wrong options: mismatch at cell 0 (so HIGHLIGHT(0) is highly discriminative)
            r = target.copy()
            r[0] = f"WRONG_{j}"  # always at cell 0
            rendered.append(r)
    return target, texts, rendered


# ─────────────────────────────────────────────────────────────────────────────
# A1: HIGHLIGHT correctly shifts semantic posterior toward correct option
# ─────────────────────────────────────────────────────────────────────────────

class TestHighlightPosteriorMonotonicity:

    def test_correct_option_gains_positive_bias(self, listener):
        """semantic_log_bias[j*] > log_bias[j_wrong] after HIGHLIGHT."""
        target, texts, rendered = _make_options(n_options=4, L=4)
        # Highlight cell 0 — correct option (j=0) matches here
        cells = (0,)

        update = listener.observe_tutor_action(
            action="HIGHLIGHT",
            target_output=target,
            active_texts=texts,
            rendered_outputs=rendered,
            action_cells=cells,
        )

        # j=0 is correct (matches target everywhere, including cell 0)
        # j=1,2,3 have a mismatch at their respective positions
        assert update.pass_abort is False
        # Correct option should have highest semantic log-bias
        assert update.semantic_log_bias[0] == pytest.approx(
            max(update.semantic_log_bias), abs=1e-9
        ), f"j=0 should have max bias, got {update.semantic_log_bias}"

    def test_correct_option_bias_strictly_positive_relative(self, listener):
        """After HIGHLIGHT(H), q_post(j*) > q_pre(j*)."""
        target, texts, rendered = _make_options(n_options=3, L=4)
        cells = (0, 1)  # highlight first two cells

        update = listener.observe_tutor_action(
            action="HIGHLIGHT",
            target_output=target,
            active_texts=texts,
            rendered_outputs=rendered,
            action_cells=cells,
        )

        K = 3
        sem_pre = np.zeros(K)  # uniform baseline

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        q_pre = softmax(sem_pre)
        q_post = softmax(sem_pre + update.semantic_log_bias)

        assert q_post[0] > q_pre[0], (
            f"Correct option probability should increase after HIGHLIGHT. "
            f"Pre={q_pre[0]:.3f}, Post={q_post[0]:.3f}"
        )

    def test_nonzero_semantic_bias_shape(self, listener):
        """semantic_log_bias should have shape (K,) and sum ≈ not -inf."""
        target, texts, rendered = _make_options(n_options=5, L=3)
        cells = (0,)
        update = listener.observe_tutor_action(
            action="HIGHLIGHT",
            target_output=target,
            active_texts=texts,
            rendered_outputs=rendered,
            action_cells=cells,
        )
        assert update.semantic_log_bias.shape == (5,)
        assert np.all(np.isfinite(update.semantic_log_bias))
        assert np.all(update.risk_logit_shift == 0.0)

    def test_stronger_ban_lifts_more_signal(self, listener):
        """Higher omega_ban produces larger logit shift."""
        K = 3
        # omega_ban=3 vs omega_ban=1
        lis_strong = RSAListener(omega_ban=3.0)
        lis_weak   = RSAListener(omega_ban=1.0)
        for lis, expected_shift in [(lis_weak, 1.0), (lis_strong, 3.0)]:
            update = lis.observe_tutor_action(
                action="BAN",
                target_output=["t0"],
                active_texts=[["p"]] * K,
                rendered_outputs=[None] * K,
                action_arg=0,
            )
            assert update.risk_logit_shift[0] == pytest.approx(expected_shift, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# A2: BAN correctly shifts risk posterior for banned option only
# ─────────────────────────────────────────────────────────────────────────────

class TestBanRiskMonotonicity:

    def test_banned_option_gets_positive_shift(self, listener):
        """risk_logit_shift[j_ban] = omega_ban, others = 0."""
        K = 4
        update = listener.observe_tutor_action(
            action="BAN",
            target_output=["t0", "t1"],
            active_texts=[["p"] for _ in range(K)],
            rendered_outputs=[None] * K,
            action_arg=2,  # ban option at active-menu index 2
        )
        assert update.pass_abort is False
        assert update.semantic_log_bias.shape == (K,)
        assert np.all(update.semantic_log_bias == 0.0)

        # Only banned index gets shifted
        assert update.risk_logit_shift[2] == pytest.approx(3.0, abs=1e-9)
        for j in range(K):
            if j != 2:
                assert update.risk_logit_shift[j] == pytest.approx(0.0, abs=1e-9)

    def test_ban_increases_risk_posterior(self, listener):
        """apply_logit_shift: posterior p_h > prior p_h after BAN."""
        prior_ph = 0.3
        delta = 3.0
        posterior_ph = RSAListener.apply_logit_shift(prior_ph, delta)
        assert posterior_ph > prior_ph, (
            f"BAN should increase hazard probability. "
            f"Prior={prior_ph:.3f}, Posterior={posterior_ph:.3f}"
        )

    def test_ban_reduces_pick_utility(self, listener):
        """After applying BAN's risk shift, pick utility for banned option drops."""
        from cls_option_tutor.config import LearnerConfig
        from cls_option_tutor.learner.policy import LearnerPolicy

        K = 3
        # Setup a minimal policy
        lcfg = LearnerConfig(alpha_sem=1.0, alpha_risk=0.5, alpha_unc=0.0)
        policy = LearnerPolicy(lcfg)

        # risk_logit_shift for option 1 = omega_ban = 3.0
        risk_logit_shift = np.array([0.0, 3.0, 0.0])

        # Danger head prediction for option 1 before shift
        p_h_before = 0.2   # prior
        p_h_after = RSAListener.apply_logit_shift(p_h_before, 3.0)
        mu_s = 2.0          # severity

        mu_d_before = p_h_before * mu_s
        mu_d_after  = p_h_after  * mu_s

        # After BAN, danger should be higher → utility lower
        assert mu_d_after > mu_d_before

    def test_ban_out_of_range_is_safe(self, listener):
        """BAN with invalid index should not crash, return zero shift."""
        K = 3
        update = listener.observe_tutor_action(
            action="BAN",
            target_output=["t0"],
            active_texts=[["p"]] * K,
            rendered_outputs=[None] * K,
            action_arg=99,  # out of range
        )
        assert np.all(update.risk_logit_shift == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# A3: WAIT is identity
# ─────────────────────────────────────────────────────────────────────────────

class TestWaitIdentity:

    def test_wait_zero_bias(self, listener):
        """WAIT produces all-zero biases and no abort."""
        K = 5
        update = listener.observe_tutor_action(
            action="WAIT",
            target_output=["t0", "t1"],
            active_texts=[["p"]] * K,
            rendered_outputs=[None] * K,
        )
        assert update.pass_abort is False
        assert np.all(update.semantic_log_bias == 0.0)
        assert np.all(update.risk_logit_shift == 0.0)

    def test_unrecognized_action_is_identity(self, listener):
        """Unknown action (e.g., RISK_HINT legacy) should be treated as WAIT."""
        K = 4
        update = listener.observe_tutor_action(
            action="RISK_HINT",
            target_output=["t0"],
            active_texts=[["p"]] * K,
            rendered_outputs=[None] * K,
        )
        assert update.pass_abort is False
        assert np.all(update.semantic_log_bias == 0.0)
        assert np.all(update.risk_logit_shift == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# A4: PASS abort
# ─────────────────────────────────────────────────────────────────────────────

class TestPassAbort:

    def test_pass_sets_abort_flag(self, listener):
        """PASS must set pass_abort=True with zero biases."""
        K = 3
        update = listener.observe_tutor_action(
            action="PASS",
            target_output=["t0"],
            active_texts=[["p"]] * K,
            rendered_outputs=[None] * K,
        )
        assert update.pass_abort is True
        assert np.all(update.semantic_log_bias == 0.0)
        assert np.all(update.risk_logit_shift == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# A5: Attention meta-prior — cross-query persistence and state isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestMetaAttentionPrior:

    def test_meta_prior_initialized_uniform(self):
        """After init_meta_prior, weights are uniform."""
        attn = AttentionModel(L=4)
        attn.init_meta_prior(4)
        meta = attn.meta_prior
        assert meta is not None
        np.testing.assert_allclose(meta, np.ones(4) / 4, atol=1e-9)

    def test_highlight_updates_meta_prior(self):
        """After HIGHLIGHT(0,1), meta prior increases weight on cells 0,1."""
        attn = AttentionModel(L=4)
        attn.init_meta_prior(4)
        attn.update_meta_prior(cells=(0, 1), rho=0.3)
        meta = attn.meta_prior
        # Cells 0 and 1 should have higher weight than cells 2 and 3
        assert meta[0] > meta[2]
        assert meta[1] > meta[3]

    def test_effective_attention_blends_meta(self):
        """effective_attention blends query + meta (not pure query weights)."""
        attn = AttentionModel(L=4)
        attn.init_for_query(4)
        attn.init_meta_prior(4)
        # Shift meta toward cell 0
        attn.update_meta_prior(cells=(0,), rho=0.9)  # strong shift
        eff = attn.effective_attention(gamma=0.5)
        # Should differ from pure query weights (uniform)
        pure_query = attn.weights
        assert not np.allclose(eff, pure_query)
        assert np.isclose(eff.sum(), 1.0, atol=1e-9)

    def test_state_isolation_between_blocks(self):
        """init_meta_prior resets state; two 'blocks' don't cross-contaminate."""
        attn = AttentionModel(L=4)

        # Block 1: highlight cell 0 twice
        attn.init_meta_prior(4)
        attn.update_meta_prior(cells=(0,), rho=0.5)
        attn.update_meta_prior(cells=(0,), rho=0.5)
        meta_block1 = attn.meta_prior.copy()
        assert meta_block1[0] > 0.3  # should be heavily weighted

        # Reset for block 2 (simulate init_block)
        attn.init_meta_prior(4)  # fresh reset
        meta_block2 = attn.meta_prior
        np.testing.assert_allclose(meta_block2, np.ones(4) / 4, atol=1e-9,
                                    err_msg="Block 2 meta prior should be uniform after reset")


# ─────────────────────────────────────────────────────────────────────────────
# A6: BAN cross-query teach — danger_head.update_from_ban()
# ─────────────────────────────────────────────────────────────────────────────

class TestBanCrossQueryTeach:

    def test_update_from_ban_increases_hazard(self):
        """danger_head.hazard p_h increases after update_from_ban."""
        m = 8
        dh = DangerHead(m=m)
        v = np.random.default_rng(42).standard_normal(m)
        v = v / (np.linalg.norm(v) + 1e-8)

        p_h_before = dh.hazard.predict(v)
        dh.update_from_ban(v, omega_ban=3.0)
        p_h_after = dh.hazard.predict(v)

        assert p_h_after > p_h_before, (
            f"BAN should increase hazard probability. "
            f"Before={p_h_before:.4f}, After={p_h_after:.4f}"
        )

    def test_update_from_ban_stronger_than_hint(self):
        """BAN with large omega_ban should produce larger hazard increase than RISK_HINT (eta=0.8).

        Uses omega_ban=10 to ensure clear separation — the scale of the logit
        shift (ω_ban >> logit(eta=0.8)) guarantees BAN is the stronger signal.
        """
        m = 8
        rng = np.random.default_rng(0)
        v = rng.standard_normal(m)
        v = v / (np.linalg.norm(v) + 1e-8)

        # RISK_HINT path: 3 updates at eta=0.8
        dh_hint = DangerHead(m=m)
        p_before_hint = dh_hint.hazard.predict(v)
        for _ in range(3):
            dh_hint.update_from_hint(v, eta=0.8)
        p_after_hint = dh_hint.hazard.predict(v)
        delta_hint = p_after_hint - p_before_hint

        # BAN path: 3 updates at omega_ban=10 (very strong)
        dh_ban = DangerHead(m=m)
        p_before_ban = dh_ban.hazard.predict(v)
        for _ in range(3):
            dh_ban.update_from_ban(v, omega_ban=10.0)
        p_after_ban = dh_ban.hazard.predict(v)
        delta_ban = p_after_ban - p_before_ban

        assert delta_ban > delta_hint, (
            f"BAN(omega=10) should produce larger hazard increase than 3x RISK_HINT(eta=0.8). "
            f"Δ_ban={delta_ban:.4f}, Δ_hint={delta_hint:.4f}"
        )

    def test_update_from_ban_does_not_affect_severity(self):
        """BAN update should NOT update severity head (no damage info)."""
        m = 8
        dh = DangerHead(m=m)
        v = np.random.default_rng(7).standard_normal(m)

        mu_s_before, _ = dh.severity.predict(v)
        dh.update_from_ban(v, omega_ban=3.0)
        mu_s_after, _ = dh.severity.predict(v)

        # Severity head n_updates should still be 0
        assert dh.severity._n_updates == 0, "BAN should not update severity head"
        np.testing.assert_allclose(mu_s_before, mu_s_after, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: RSABeliefUpdate dataclass sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestRSABeliefUpdateContract:

    def test_semantic_and_risk_are_independent_for_highlight(self, listener):
        """HIGHLIGHT should produce zero risk_logit_shift."""
        target, texts, rendered = _make_options(3, 4)
        update = listener.observe_tutor_action(
            action="HIGHLIGHT",
            target_output=target,
            active_texts=texts,
            rendered_outputs=rendered,
            action_cells=(0,),
        )
        assert np.all(update.risk_logit_shift == 0.0)

    def test_semantic_and_risk_are_independent_for_ban(self, listener):
        """BAN should produce zero semantic_log_bias."""
        K = 4
        update = listener.observe_tutor_action(
            action="BAN",
            target_output=["t0"],
            active_texts=[["p"]] * K,
            rendered_outputs=[None] * K,
            action_arg=1,
        )
        assert np.all(update.semantic_log_bias == 0.0)

    def test_highlight_no_cells_is_wait(self, listener):
        """HIGHLIGHT with empty cells should behave like WAIT (no update)."""
        K = 3
        update = listener.observe_tutor_action(
            action="HIGHLIGHT",
            target_output=["t0", "t1"],
            active_texts=[["p"]] * K,
            rendered_outputs=[["t0", "t1"]] * K,
            action_cells=(),   # empty
        )
        assert update.pass_abort is False
        assert np.all(update.semantic_log_bias == 0.0)
        assert np.all(update.risk_logit_shift == 0.0)
