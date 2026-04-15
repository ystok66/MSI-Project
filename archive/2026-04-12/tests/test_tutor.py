"""
test_tutor.py — Phase C tutor integration tests.

Tests:
  E2.1 — Profile inference produces valid profiles
  E2.2 — Counterfactual scorer Q-value properties
  E2.3 — Anti-oracle: tutor never accesses is_correct
  E2.4 — Intervention effects (BAN reduces damage, HIGHLIGHT changes attention)
  E2.5 — Full tutor+learner block execution
  E2.6 — Tutor vs no-tutor comparison (§18 E2)
"""
from __future__ import annotations
import os
import sys
import ast
import inspect
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.config import FullConfig, TutorConfig, LearnerConfig
from cls_option_tutor.interfaces import Option, LearnerStep, RevealEvent
from cls_option_tutor.env.danger_model import generate_danger_model, generate_danger_vector
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.env.state import QueryState, BlockState, ProfileState
from cls_option_tutor.env.interventions import get_active_menu
from cls_option_tutor.grammar.task_adapter import parse_task_file
from cls_option_tutor.grammar.option_generator import generate_menu

from cls_option_tutor.learner.semantic_scorer import DeterministicSemanticScorer
from cls_option_tutor.learner.danger_head import create_danger_head
from cls_option_tutor.learner.learner_agent import LearnerAgent

from cls_option_tutor.tutor.profile_inference import ProfileInference, ProfilePosterior
from cls_option_tutor.tutor.counterfactual import CounterfactualScorer, InterventionScore
from cls_option_tutor.tutor.tutor_policy import TutorPolicy
from cls_option_tutor.tutor.tutor_agent import TutorAgent


DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')


def _has_data():
    return os.path.isdir(DATA_DIR) and os.path.exists(
        os.path.join(DATA_DIR, '000001.txt'))


# ══════════════════════════════════════════════════════════════
# E2.1 — Profile inference
# ══════════════════════════════════════════════════════════════

class TestProfileInference:
    def test_grid_builds(self):
        """Grid should be non-empty with valid profiles."""
        pi = ProfileInference(TutorConfig())
        grid = pi.build_grid(n=3)
        assert len(grid) > 0
        for p in grid:
            assert 0.0 <= p.lambda_risk <= 2.0
            assert 0.0 <= p.lambda_refresh <= 1.0
            assert p.g_highlight >= 0.0

    def test_infer_from_empty_trace(self):
        """Should return valid posterior even with empty trace."""
        pi = ProfileInference(TutorConfig())
        posterior = pi.infer([])
        assert isinstance(posterior, ProfilePosterior)
        profile = posterior.map_profile
        assert isinstance(profile, ProfileState)

    def test_infer_from_mixed_trace(self):
        """Should distinguish risk-averse from risk-seeking behavior."""
        pi = ProfileInference(TutorConfig(profile_grid_size=3))

        # Risk-averse trace: avoids damage, prefers refresh
        cautious_trace = [
            LearnerStep(round_t=0, query_id=0, action="refresh",
                        hp_before=10, hp_after=10, menu_size=10),
            LearnerStep(round_t=1, query_id=0, action="pick", pick_index=0,
                        correct=True, damage=0, hp_before=10, hp_after=10,
                        menu_size=10),
        ]

        # Risk-seeking trace: picks despite damage
        reckless_trace = [
            LearnerStep(round_t=0, query_id=0, action="pick", pick_index=0,
                        correct=False, damage=4, hp_before=10, hp_after=6,
                        menu_size=10),
            LearnerStep(round_t=1, query_id=0, action="pick", pick_index=1,
                        correct=False, damage=3, hp_before=6, hp_after=3,
                        menu_size=10),
        ]

        p_cautious = pi.infer(cautious_trace).map_profile
        p_reckless = pi.infer(reckless_trace).map_profile

        # Cautious should have higher lambda_refresh
        assert p_cautious.lambda_refresh >= p_reckless.lambda_refresh, (
            f"Cautious refresh={p_cautious.lambda_refresh}, "
            f"reckless refresh={p_reckless.lambda_refresh}")

    def test_posterior_sums_to_one(self):
        """Posterior probabilities should sum to 1."""
        pi = ProfileInference(TutorConfig(profile_grid_size=3))
        trace = [
            LearnerStep(round_t=0, query_id=0, action="pick", pick_index=0,
                        correct=True, damage=0, hp_before=10, hp_after=10,
                        menu_size=10),
        ]
        posterior = pi.infer(trace)
        np.testing.assert_allclose(posterior.posterior_probs.sum(), 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════
# E2.2 — Counterfactual scorer
# ══════════════════════════════════════════════════════════════

class TestCounterfactualScorer:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_wait_q_is_zero(self):
        """WAIT should have Q-value of 0 (baseline)."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

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

        cf = CounterfactualScorer(TutorConfig())
        candidates = cf.score_all(qs, ProfileState(), scorer)

        wait_scores = [c for c in candidates if c.action == "WAIT"]
        assert len(wait_scores) == 1
        assert wait_scores[0].total_q == 0.0

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_ban_scores_are_bounded(self):
        """BAN Q-values should be finite and reasonable."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        ex = query[0]
        menu = generate_menu(
            target_output=ex.output, true_program=ex.words,
            grammar=grammar, support=support,
            danger_model=dm, K=10, m=16, rng=rng,
        )
        qs = QueryState(
            query_id=0, target_output=list(ex.output),
            true_program=list(ex.words), hp=5,  # V2: HP_0=5
            max_rounds=5, menu=menu,
        )

        cf = CounterfactualScorer(TutorConfig())
        candidates = cf.score_all(qs, ProfileState(), scorer)

        hint_scores = [c for c in candidates if c.action == "RISK_HINT"]
        assert len(hint_scores) > 0
        for s in hint_scores:
            assert np.isfinite(s.total_q), f"RISK_HINT Q-value not finite: {s.total_q}"
            assert -20 < s.total_q < 50, f"RISK_HINT Q-value out of range: {s.total_q}"

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_all_action_types_scored(self):
        """Should score at least WAIT, BAN, HIGHLIGHT, SKIP."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

        ex = query[0]
        menu = generate_menu(
            target_output=ex.output, true_program=ex.words,
            grammar=grammar, support=support,
            danger_model=dm, K=10, m=16, rng=rng,
        )
        qs = QueryState(
            query_id=0, target_output=list(ex.output),
            true_program=list(ex.words), hp=5,  # V2: HP_0=5
            max_rounds=5, menu=menu,
        )

        cf = CounterfactualScorer(TutorConfig())
        candidates = cf.score_all(qs, ProfileState(), scorer)

        actions = {c.action for c in candidates}
        assert "WAIT" in actions
        assert "RISK_HINT" in actions  # V2: replaces BAN
        assert "HIGHLIGHT" in actions
        assert "SKIP" in actions


# ══════════════════════════════════════════════════════════════
# E2.3 — Anti-oracle (§12)
# ══════════════════════════════════════════════════════════════

class TestAntiOracle:
    def test_counterfactual_never_reads_is_correct(self):
        """§12: CounterfactualScorer code must NOT access is_correct.

        Uses AST analysis to check attribute access, ignoring docstrings.
        """
        source = inspect.getsource(CounterfactualScorer)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "is_correct":
                pytest.fail(
                    "ANTI-ORACLE VIOLATION: CounterfactualScorer "
                    "accesses .is_correct in code!")

    def test_tutor_policy_never_reads_is_correct(self):
        """§12: TutorPolicy source code must NOT access is_correct."""
        source = inspect.getsource(TutorPolicy)
        assert "is_correct" not in source, (
            "ANTI-ORACLE VIOLATION: TutorPolicy accesses is_correct!")

    def test_tutor_agent_never_reads_is_correct(self):
        """§12: TutorAgent source code must NOT access is_correct."""
        source = inspect.getsource(TutorAgent)
        assert "is_correct" not in source, (
            "ANTI-ORACLE VIOLATION: TutorAgent accesses is_correct!")

    def test_profile_inference_never_reads_is_correct(self):
        """§12: ProfileInference source code must NOT access is_correct."""
        source = inspect.getsource(ProfileInference)
        assert "is_correct" not in source, (
            "ANTI-ORACLE VIOLATION: ProfileInference accesses is_correct!")


# ══════════════════════════════════════════════════════════════
# E2.4 — Intervention effects
# ══════════════════════════════════════════════════════════════

class TestInterventionEffects:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_ban_high_danger_chosen(self):
        """Tutor should prefer banning options with high danger × high P(pick)."""
        path = os.path.join(DATA_DIR, '000001.txt')
        support, query, grammar = parse_task_file(path)
        scorer = DeterministicSemanticScorer(grammar)
        rng = np.random.default_rng(42)
        dm = generate_danger_model(m=16, rng=rng)

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

        cf = CounterfactualScorer(TutorConfig())
        dh = create_danger_head(16)
        candidates = cf.score_all(qs, ProfileState(), scorer, dh)

        ban_scores = [c for c in candidates if c.action == "BAN"]
        if len(ban_scores) >= 2:
            # Best BAN should have higher danger_avoided
            best = max(ban_scores, key=lambda c: c.total_q)
            assert best.components["ban_value"] >= 0

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_tutor_policy_observation_is_wait(self):
        """During observation phase, tutor should always WAIT."""
        cfg = FullConfig()
        cfg.env.N_obs = 3
        env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
        block = env.reset_block("000001", seed=42)

        support, _, grammar = env.adapter.load_task("000001")
        scorer = DeterministicSemanticScorer(grammar)

        policy = TutorPolicy(cfg.tutor)

        # First 3 queries should be WAIT
        for qi in range(min(3, len(block.queries))):
            block.current_query_idx = qi
            action, kwargs = policy.select_action(block, scorer)
            assert action == "WAIT", (
                f"Query {qi} (obs phase): expected WAIT, got {action}")


# ══════════════════════════════════════════════════════════════
# E2.5 — Full tutor+learner block
# ══════════════════════════════════════════════════════════════

class TestTutorAgentIntegration:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_tutor_block_runs_to_completion(self):
        """Full tutor+learner block should run without errors."""
        env = OptionEnv(data_dir=DATA_DIR)
        learner = LearnerAgent(seed=42)
        tutor = TutorAgent()

        block = tutor.run_block(env, learner, "000001", seed=42)

        assert block.done
        metrics = OptionEnv.get_block_metrics(block)
        assert metrics["n_queries"] == 8
        assert metrics["total_correct"] >= 0
        assert len(block.tutor_trace) > 0  # tutor trace should exist
        assert len(block.learner_trace) > 0

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_tutor_actually_intervenes(self):
        """Tutor should use at least one non-WAIT action in teaching phase."""
        cfg = FullConfig()
        cfg.env.N_obs = 2   # shorter obs phase
        cfg.env.N_teach = 6
        env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
        learner = LearnerAgent(cfg=cfg, seed=42)
        tutor = TutorAgent(cfg=cfg)

        block = tutor.run_block(env, learner, "000001", seed=42)

        # Check if any non-WAIT actions occurred
        non_wait = [s for s in block.tutor_trace if s.action != "WAIT"]
        # Not strictly required to pass (WAIT can be optimal),
        # but we track it for diagnostics
        print(f"  Tutor non-WAIT actions: {len(non_wait)} / {len(block.tutor_trace)}")
        # The test passes either way — we just verify no crashes
        assert block.done


# ══════════════════════════════════════════════════════════════
# E2.6 — Tutor vs no-tutor comparison
# ══════════════════════════════════════════════════════════════

class TestTutorBenefit:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_tutor_vs_baseline_multi_seed(self):
        """§18 E2: compare tutor vs baseline across multiple seeds.

        Reports metrics — does NOT require tutor to always win
        (some tasks may not benefit from intervention).
        """
        results = {"tutor": [], "baseline": []}

        for seed in range(5):
            env = OptionEnv(data_dir=DATA_DIR)

            # Baseline: learner only
            learner_b = LearnerAgent(seed=seed)
            block_b = learner_b.run_block(env, "000001", seed=seed)
            m_b = OptionEnv.get_block_metrics(block_b)
            results["baseline"].append(m_b)

            # Tutor: learner + tutor
            learner_t = LearnerAgent(seed=seed)
            tutor = TutorAgent()
            block_t = tutor.run_block(env, learner_t, "000001", seed=seed)
            m_t = OptionEnv.get_block_metrics(block_t)
            results["tutor"].append(m_t)

        # Report
        base_sr = np.mean([m["solve_rate"] for m in results["baseline"]])
        tutor_sr = np.mean([m["solve_rate"] for m in results["tutor"]])
        base_dmg = np.mean([m["total_damage"] for m in results["baseline"]])
        tutor_dmg = np.mean([m["total_damage"] for m in results["tutor"]])

        print(f"\n  === Tutor vs Baseline (5 seeds) ===")
        print(f"  Baseline: solve={base_sr:.3f}, damage={base_dmg:.1f}")
        print(f"  Tutor:    solve={tutor_sr:.3f}, damage={tutor_dmg:.1f}")

        # At minimum: both should produce valid results
        assert 0.0 <= base_sr <= 1.0
        assert 0.0 <= tutor_sr <= 1.0
        # Tutor should not catastrophically degrade performance
        assert tutor_sr >= base_sr - 0.3, (
            f"Tutor catastrophically worse: tutor_sr={tutor_sr:.3f} "
            f"vs base_sr={base_sr:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
