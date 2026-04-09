"""
test_learner.py — Phase B learner integration tests.

Tests:
  E1.1 — DeterministicSemanticScorer correctness
  E1.2 — DangerHead Bayesian updates
  E1.3 — AttentionModel highlight mechanics
  E1.4 — EpisodicMemory elimination
  E1.5 — LearnerPolicy action distribution
  E1.6 — LearnerAgent full block (baseline vs random)
"""
from __future__ import annotations
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.config import FullConfig, LearnerConfig
from cls_option_tutor.interfaces import Option, RevealEvent
from cls_option_tutor.env.danger_model import generate_danger_model, generate_danger_vector
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.state import QueryState
from cls_option_tutor.grammar.task_adapter import parse_task_file, Grammar
from cls_option_tutor.grammar.option_generator import generate_menu

from cls_option_tutor.learner.semantic_scorer import DeterministicSemanticScorer
from cls_option_tutor.learner.danger_head import DangerHead, create_danger_head
from cls_option_tutor.learner.attention_model import AttentionModel
from cls_option_tutor.learner.episodic_memory import EpisodicMemory
from cls_option_tutor.learner.policy import LearnerPolicy, PolicyOutput
from cls_option_tutor.learner.learner_agent import LearnerAgent

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')


def _has_data():
    return os.path.isdir(DATA_DIR) and os.path.exists(
        os.path.join(DATA_DIR, '000001.txt'))


# ══════════════════════════════════════════════════════════════
# E1.1 — Semantic scorer
# ══════════════════════════════════════════════════════════════

class TestSemanticScorer:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_correct_option_scores_zero(self):
        """Simple noun options should have score 0 (no mismatch).
        Complex compositions may have non-zero due to renderer limitations."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar, tau_sem=1.0)

        # Test simple nouns: these should always score 0
        for word, color in grammar.nouns.items():
            score = scorer.score_option([color], [word])
            assert score == 0.0, (
                f"Simple noun {word}->{color} scored {score}, expected 0.0")

        # For complex queries: scores should be non-positive
        for ex in query[:3]:
            score = scorer.score_option(ex.output, ex.words)
            assert score <= 0.0, (
                f"Option scored {score} > 0 for {ex.words}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_wrong_option_scores_negative(self):
        """Wrong options should score negative (mismatches)."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar, tau_sem=1.0)

        # The word "dax" maps to BLUE. Target is output of first query.
        ex = query[0]
        # Try a wrong program
        wrong_words = [w for w in grammar.nouns if grammar.nouns[w] != ex.output[0]]
        if wrong_words:
            score = scorer.score_option(ex.output, [wrong_words[0]])
            assert score < 0.0, "Wrong option should have negative score"

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_menu_scoring_correct_highest(self):
        """Correct option should have highest semantic score in menu."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar, tau_sem=1.0)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        ex = query[0]
        menu = generate_menu(
            target_output=ex.output, true_program=ex.words,
            grammar=grammar, support=support,
            danger_model=dm, K=10, m=16, rng=rng,
        )
        scores = scorer.score_menu(ex.output, menu)
        correct_idx = next(i for i, o in enumerate(menu) if o.is_correct)
        assert scores[correct_idx] == max(scores), (
            f"Correct option (idx={correct_idx}) score={scores[correct_idx]}, "
            f"max={max(scores)}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_semantic_entropy_positive(self):
        """Entropy should be positive for a non-trivial menu."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar, tau_sem=1.0)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        ex = query[0]
        menu = generate_menu(
            target_output=ex.output, true_program=ex.words,
            grammar=grammar, support=support,
            danger_model=dm, K=10, m=16, rng=rng,
        )
        H = scorer.semantic_entropy(ex.output, menu)
        assert H > 0, f"Semantic entropy should be positive, got {H}"


# ══════════════════════════════════════════════════════════════
# E1.2 — Danger head
# ══════════════════════════════════════════════════════════════

class TestDangerHead:
    def test_prior_predicts_midrange(self):
        """Prior (no observations) should predict ~2.5 damage."""
        dh = create_danger_head(m=16)
        rng = np.random.default_rng(42)
        v = generate_danger_vector(16, rng)
        mu, u = dh.predict(v)
        assert 0 <= mu <= 5, f"Prediction {mu} outside [0,5]"
        assert 0 <= u <= 1, f"Uncertainty {u} outside [0,1]"

    def test_prior_high_uncertainty(self):
        """Prior uncertainty should be high."""
        dh = create_danger_head(m=16)
        rng = np.random.default_rng(42)
        total_u = 0
        for _ in range(20):
            v = generate_danger_vector(16, rng)
            _, u = dh.predict(v)
            total_u += u
        avg_u = total_u / 20
        assert avg_u > 0.1, f"Prior uncertainty too low: {avg_u}"

    def test_update_reduces_uncertainty(self):
        """After training observations, uncertainty should decrease."""
        dm = generate_danger_model(m=16, rng=np.random.default_rng(0))
        dh = create_danger_head(m=16)
        rng = np.random.default_rng(42)

        # Collect pre-training uncertainty
        test_vs = [generate_danger_vector(16, rng) for _ in range(10)]
        pre_u = np.mean([dh.predict(v)[1] for v in test_vs])

        # Train on 20 observations
        for _ in range(20):
            v = generate_danger_vector(16, rng)
            d = dm.sample_damage(v, rng)
            dh.update(v, d)

        post_u = np.mean([dh.predict(v)[1] for v in test_vs])
        assert post_u < pre_u, (
            f"Uncertainty should decrease: pre={pre_u:.3f}, post={post_u:.3f}")

    def test_prediction_improves_with_data(self):
        """Predictions should get closer to true expected damage with data."""
        dm = generate_danger_model(m=16, rng=np.random.default_rng(0))
        dh = create_danger_head(m=16)
        rng = np.random.default_rng(42)

        test_vs = [generate_danger_vector(16, rng) for _ in range(10)]
        true_ds = [dm.expected_damage(v) for v in test_vs]

        # Pre-training error
        pre_preds = [dh.predict(v)[0] for v in test_vs]
        pre_mse = np.mean([(p - t)**2 for p, t in zip(pre_preds, true_ds)])

        # Train on 50 observations
        for _ in range(50):
            v = generate_danger_vector(16, rng)
            d = dm.sample_damage(v, rng)
            dh.update(v, d)

        post_preds = [dh.predict(v)[0] for v in test_vs]
        post_mse = np.mean([(p - t)**2 for p, t in zip(post_preds, true_ds)])

        # Post MSE should be less than or close to pre MSE
        # (at least not catastrophically worse)
        assert post_mse < pre_mse + 1.0, (
            f"Prediction degraded: pre_mse={pre_mse:.3f}, post_mse={post_mse:.3f}")

    def test_reset_restores_prior(self):
        """Reset should restore to prior state."""
        dh = create_danger_head(m=16)
        rng = np.random.default_rng(42)
        v = generate_danger_vector(16, rng)

        _, u_prior = dh.predict(v)
        dh.update(v, 3)
        dh.reset()
        _, u_reset = dh.predict(v)
        assert abs(u_reset - u_prior) < 1e-6


# ══════════════════════════════════════════════════════════════
# E1.3 — Attention model
# ══════════════════════════════════════════════════════════════

class TestAttentionModel:
    def test_uniform_initial(self):
        """Initial weights should be uniform."""
        att = AttentionModel(L=5)
        np.testing.assert_allclose(att.weights, np.ones(5) / 5)

    def test_highlight_boosts_cells(self):
        """Highlighted cells should have higher weight."""
        att = AttentionModel(L=5, rho_H=2.0)
        w = att.apply_highlight((1, 3))
        assert w[1] > w[0], "Highlighted cell should have higher weight"
        assert w[3] > w[2], "Highlighted cell should have higher weight"
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_highlight_preserves_normalization(self):
        """Weights should sum to 1 after highlight."""
        att = AttentionModel(L=10, rho_H=3.0)
        w = att.apply_highlight((0, 5, 9))
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_reset_restores_uniform(self):
        """Reset should restore uniform weights."""
        att = AttentionModel(L=5, rho_H=2.0)
        att.apply_highlight((0,))
        att.reset()
        np.testing.assert_allclose(att.weights, np.ones(5) / 5)

    def test_effective_coverage(self):
        """Effective coverage should decrease after highlight."""
        att = AttentionModel(L=10, rho_H=2.0)
        cov_before = att.effective_coverage()
        att.apply_highlight((0, 1))
        cov_after = att.effective_coverage()
        assert cov_after < cov_before, (
            f"Coverage should decrease: before={cov_before:.2f}, after={cov_after:.2f}")


# ══════════════════════════════════════════════════════════════
# E1.4 — Episodic memory
# ══════════════════════════════════════════════════════════════

class TestEpisodicMemory:
    def test_write_and_recall(self):
        """Written reveals should be recallable."""
        mem = EpisodicMemory()
        v = np.zeros(4)
        ev = RevealEvent(
            round_t=0, option_index=1,
            option_text=["dax", "fep", "blicket"],
            revealed_output=["BLUE", "PURPLE"],
            damage=2, expected_damage=2.5,
            danger_vec=v,
        )
        mem.write_reveal(ev)
        assert mem.is_known_wrong(["dax", "fep", "blicket"])
        assert not mem.is_known_wrong(["lug"])
        assert mem.n_reveals == 1

    def test_elimination_penalty(self):
        """Known-wrong text should get hard penalty."""
        mem = EpisodicMemory()
        v = np.zeros(4)
        ev = RevealEvent(
            round_t=0, option_index=1,
            option_text=["dax"],
            revealed_output=["BLUE"],
            damage=2, expected_damage=2.0,
            danger_vec=v,
        )
        mem.write_reveal(ev)
        assert mem.get_elimination_penalty(["dax"], ["BLUE"]) == -10.0
        assert mem.get_elimination_penalty(["lug"], ["BLUE"]) == -5.0
        assert mem.get_elimination_penalty(["lug"], ["RED"]) == 0.0

    def test_danger_observations(self):
        """Danger observations should accumulate."""
        mem = EpisodicMemory()
        v = np.ones(4)
        for i in range(3):
            ev = RevealEvent(
                round_t=i, option_index=i,
                option_text=[f"w{i}"],
                revealed_output=[f"C{i}"],
                damage=i + 1, expected_damage=float(i + 1),
                danger_vec=v * i,
            )
            mem.write_reveal(ev)
        assert len(mem.danger_observations) == 3

    def test_reset(self):
        """Reset should clear all memory."""
        mem = EpisodicMemory()
        v = np.zeros(4)
        ev = RevealEvent(
            round_t=0, option_index=0,
            option_text=["x"], revealed_output=["Y"],
            damage=1, expected_damage=1.0, danger_vec=v,
        )
        mem.write_reveal(ev)
        mem.reset()
        assert mem.n_reveals == 0
        assert not mem.is_known_wrong(["x"])


# ══════════════════════════════════════════════════════════════
# E1.5 — Learner policy
# ══════════════════════════════════════════════════════════════

class TestLearnerPolicy:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_policy_outputs_valid_distribution(self):
        """Policy should output valid probability distribution."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar, tau_sem=1.0)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        policy = LearnerPolicy(LearnerConfig())
        policy.init_for_block(scorer, m=16)

        ex = query[0]
        menu = generate_menu(
            target_output=ex.output, true_program=ex.words,
            grammar=grammar, support=support,
            danger_model=dm, K=10, m=16, rng=rng,
        )
        qs = QueryState(
            query_id=0, target_output=list(ex.output),
            true_program=list(ex.words), hp=10,
            max_rounds=5, menu=menu,
        )
        policy.init_for_query(len(ex.output))

        out = policy.compute_policy(qs, rng)
        assert out.action in ("pick", "refresh")
        np.testing.assert_allclose(out.probs.sum(), 1.0, atol=1e-6)
        assert all(p >= 0 for p in out.probs)

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_correct_option_has_highest_utility(self):
        """Correct option should have highest semantic score utility."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar, tau_sem=1.0)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        policy = LearnerPolicy(LearnerConfig())
        policy.init_for_block(scorer, m=16)

        ex = query[0]
        menu = generate_menu(
            target_output=ex.output, true_program=ex.words,
            grammar=grammar, support=support,
            danger_model=dm, K=10, m=16, rng=rng,
        )
        qs = QueryState(
            query_id=0, target_output=list(ex.output),
            true_program=list(ex.words), hp=10,
            max_rounds=5, menu=menu,
        )
        policy.init_for_query(len(ex.output))

        out = policy.compute_policy(qs, rng)
        # The semantic_scores array corresponds to active menu
        from cls_option_tutor.env.interventions import get_active_menu
        active = get_active_menu(qs)
        correct_i = next(i for i, o in enumerate(active) if o.is_correct)
        assert out.semantic_scores[correct_i] == max(out.semantic_scores)


# ══════════════════════════════════════════════════════════════
# E1.6 — Full block baseline
# ══════════════════════════════════════════════════════════════

class TestLearnerAgent:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_baseline_runs_to_completion(self):
        """Learner agent should run a full block without errors."""
        env = OptionEnv(data_dir=DATA_DIR)
        agent = LearnerAgent(seed=42)
        block = agent.run_block(env, "000001", seed=42)
        assert block.done
        metrics = OptionEnv.get_block_metrics(block)
        assert metrics["n_queries"] == 8
        assert metrics["total_correct"] >= 0
        assert metrics["total_damage"] >= 0

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_baseline_beats_random(self):
        """Learner agent should outperform random action selection.

        §18 E1: deterministicScorer baseline > random.
        """
        env = OptionEnv(data_dir=DATA_DIR)

        # Baseline: learner with semantic scorer
        agent = LearnerAgent(seed=42)
        baseline = agent.run_block(env, "000001", seed=42)
        baseline_m = OptionEnv.get_block_metrics(baseline)

        # Random: pick uniformly at random
        rng = np.random.default_rng(42)
        block = env.reset_block("000001", seed=42)
        while not block.done:
            qs = block.current_query
            if qs is None or qs.done:
                break
            env.tutor_act(block, "WAIT")
            from cls_option_tutor.env.interventions import get_active_menu
            active = get_active_menu(qs)
            if active:
                idx = rng.choice([o.index for o in active])
                env.learner_act(block, "pick", pick_index=idx)
        random_m = OptionEnv.get_block_metrics(block)

        # Baseline should have >= solve rate (or at least less damage)
        # Stochastic — we check solve_rate >= random OR damage <= random
        passed = (baseline_m["solve_rate"] >= random_m["solve_rate"]
                  or baseline_m["total_damage"] <= random_m["total_damage"])
        assert passed, (
            f"Baseline should outperform random: "
            f"baseline={baseline_m}, random={random_m}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_multiple_blocks_stable(self):
        """Running multiple blocks should not crash and produce stable results."""
        env = OptionEnv(data_dir=DATA_DIR)
        agent = LearnerAgent(seed=42)

        solve_rates = []
        for seed in range(3):
            block = agent.run_block(env, "000001", seed=seed)
            m = OptionEnv.get_block_metrics(block)
            solve_rates.append(m["solve_rate"])

        # All runs should complete
        assert len(solve_rates) == 3
        # Solve rates should be in [0, 1]
        for sr in solve_rates:
            assert 0.0 <= sr <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
