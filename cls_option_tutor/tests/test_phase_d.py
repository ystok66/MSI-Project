"""
test_phase_d.py — Phase D robustness & generalization tests.

Tests:
  E3.1 — Query synthesizer generates valid, novel programs
  E3.2 — Synthesized blocks run without errors
  E3.3 — Danger head convergence
  E3.4 — Within-grammar multi-block benchmark
  E3.5 — Generalisation gap: synth queries vs file queries
"""
from __future__ import annotations
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.interfaces import Example
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.grammar.task_adapter import parse_task_file, TaskAdapter
from cls_option_tutor.grammar.query_synthesizer import synthesize_queries
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.tutor_agent import TutorAgent
from cls_option_tutor.eval.benchmark import (
    run_benchmark, run_danger_convergence, BenchmarkResult,
)


DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')


def _has_data():
    return os.path.isdir(DATA_DIR) and os.path.exists(
        os.path.join(DATA_DIR, '000001.txt'))


# ══════════════════════════════════════════════════════════════
# E3.1 — Query synthesizer
# ══════════════════════════════════════════════════════════════

class TestQuerySynthesizer:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_synthesizer_produces_valid_programs(self):
        """Synthesized queries should render to non-None outputs."""
        _, queries, grammar = parse_task_file(
            os.path.join(DATA_DIR, '000001.txt'))
        rng = np.random.default_rng(42)
        synth = synthesize_queries(
            grammar, n=20, max_depth=3, max_len=6,
            rng=rng, existing=queries)

        assert len(synth) > 0, "Synthesizer produced no queries"
        print(f"  Synthesized {len(synth)} queries")

        for ex in synth:
            rendered = TaskAdapter.render(ex.words, grammar)
            assert rendered is not None, (
                f"Synth query {ex.words} renders to None")
            assert rendered == ex.output, (
                f"Synth query {ex.words}: render={rendered} != output={ex.output}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_synthesizer_avoids_duplicates(self):
        """Synthesized queries should not duplicate existing queries."""
        _, queries, grammar = parse_task_file(
            os.path.join(DATA_DIR, '000001.txt'))
        rng = np.random.default_rng(42)
        existing_keys = {tuple(ex.words) for ex in queries}

        synth = synthesize_queries(
            grammar, n=20, max_depth=3, max_len=6,
            rng=rng, existing=queries)

        for ex in synth:
            assert tuple(ex.words) not in existing_keys, (
                f"Synthesized query {ex.words} duplicates existing")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_synthesizer_diversity(self):
        """Synthesized queries should have diverse output lengths."""
        _, queries, grammar = parse_task_file(
            os.path.join(DATA_DIR, '000001.txt'))
        rng = np.random.default_rng(42)
        synth = synthesize_queries(
            grammar, n=30, max_depth=3, max_len=6,
            rng=rng, existing=queries)

        if len(synth) >= 5:
            output_lens = set(len(ex.output) for ex in synth)
            assert len(output_lens) >= 2, (
                f"Insufficient diversity in output lengths: {output_lens}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_different_seeds_different_queries(self):
        """Different seeds should produce different query orderings."""
        _, queries, grammar = parse_task_file(
            os.path.join(DATA_DIR, '000001.txt'))

        s1 = synthesize_queries(grammar, n=10, rng=np.random.default_rng(0),
                                existing=queries)
        s2 = synthesize_queries(grammar, n=10, rng=np.random.default_rng(999),
                                existing=queries)

        keys1 = [tuple(ex.words) for ex in s1]
        keys2 = [tuple(ex.words) for ex in s2]
        # Order should differ (the same queries may exist but shuffled)
        assert keys1 != keys2, "Different seeds produced identical query order"


# ══════════════════════════════════════════════════════════════
# E3.2 — Synthesized blocks
# ══════════════════════════════════════════════════════════════

class TestSynthesizedBlocks:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_synth_block_runs_baseline(self):
        """Synthesized block should run baseline to completion."""
        env = OptionEnv(data_dir=DATA_DIR)
        agent = LearnerAgent(seed=42)
        block = agent.run_block(
            env, "000001", seed=42, synthesize=True)
        assert block.done
        m = OptionEnv.get_block_metrics(block)
        assert m["n_queries"] > 0
        print(f"  Synth baseline: {m['n_queries']}Q, "
              f"sr={m['solve_rate']:.3f}, dmg={m['total_damage']}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_synth_block_runs_tutor(self):
        """Synthesized block should run tutor+learner to completion."""
        env = OptionEnv(data_dir=DATA_DIR)
        learner = LearnerAgent(seed=42)
        tutor = TutorAgent()
        block = tutor.run_block(
            env, learner, "000001", seed=42, synthesize=True)
        assert block.done
        m = OptionEnv.get_block_metrics(block)
        print(f"  Synth tutor: {m['n_queries']}Q, "
              f"sr={m['solve_rate']:.3f}, dmg={m['total_damage']}")

    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_multiple_synth_blocks_stable(self):
        """Multiple synthesized blocks should produce stable results."""
        env = OptionEnv(data_dir=DATA_DIR)
        rates = []
        for s in range(5):
            agent = LearnerAgent(seed=s)
            block = agent.run_block(env, "000001", seed=s, synthesize=True)
            m = OptionEnv.get_block_metrics(block)
            rates.append(m["solve_rate"])

        assert len(rates) == 5
        mean_sr = np.mean(rates)
        print(f"  5 synth blocks: mean_sr={mean_sr:.3f}, "
              f"std={np.std(rates):.3f}, rates={rates}")
        # Should all be in [0, 1]
        for r in rates:
            assert 0.0 <= r <= 1.0


# ══════════════════════════════════════════════════════════════
# E3.3 — Danger convergence
# ══════════════════════════════════════════════════════════════

class TestDangerConvergence:
    def test_mse_decreases_over_observations(self):
        """Danger head MSE should decrease with more observations."""
        result = run_danger_convergence(
            data_dir=DATA_DIR if _has_data() else "")

        mse = result["mse"]
        assert mse[0] > mse[-1], (
            f"MSE should decrease: start={mse[0]:.3f}, end={mse[-1]:.3f}")
        print(f"  MSE: {mse[0]:.3f} -> {mse[-1]:.3f} "
              f"({(1 - mse[-1]/mse[0])*100:.0f}% reduction)")

    def test_uncertainty_decreases(self):
        """Uncertainty should decrease with observations."""
        result = run_danger_convergence(
            data_dir=DATA_DIR if _has_data() else "")

        unc = result["uncertainty"]
        assert unc[0] > unc[-1], (
            f"Uncertainty should decrease: start={unc[0]:.3f}, end={unc[-1]:.3f}")


# ══════════════════════════════════════════════════════════════
# E3.4 — Within-grammar multi-block benchmark
# ══════════════════════════════════════════════════════════════

class TestMultiBlockBenchmark:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_benchmark_runs(self):
        """Benchmark harness should run without errors."""
        result = run_benchmark(
            task_id="000001",
            n_blocks=2,
            seeds=[0, 1],
            data_dir=DATA_DIR,
            test_synthesized=True,
            test_file_queries=True,
            verbose=False,
        )

        # Should have: 2 modes × 2 seeds × 2 blocks × 2 conditions = 16 runs
        assert len(result.runs) > 0

        print(f"\n{result.summary_table()}")
        deltas = result.delta_summary()
        for k, v in deltas.items():
            print(f"  {k}: D_sr={v['delta_solve_rate']:+.3f}, "
                  f"D_dmg={v['delta_damage']:+.1f}")


# ══════════════════════════════════════════════════════════════
# E3.5 — Generalisation gap
# ══════════════════════════════════════════════════════════════

class TestGeneralisationGap:
    @pytest.mark.skipif(not _has_data(), reason="CLS data not found")
    def test_synth_sr_not_catastrophic(self):
        """Solve rate on synthesized queries should not catastrophically drop."""
        env = OptionEnv(data_dir=DATA_DIR)

        # File queries
        file_rates = []
        for s in range(3):
            agent = LearnerAgent(seed=s)
            block = agent.run_block(env, "000001", seed=s, synthesize=False)
            file_rates.append(OptionEnv.get_block_metrics(block)["solve_rate"])

        # Synth queries
        synth_rates = []
        for s in range(3):
            agent = LearnerAgent(seed=s)
            block = agent.run_block(env, "000001", seed=s, synthesize=True)
            synth_rates.append(OptionEnv.get_block_metrics(block)["solve_rate"])

        file_sr = np.mean(file_rates)
        synth_sr = np.mean(synth_rates)

        print(f"  File SR: {file_sr:.3f}, Synth SR: {synth_sr:.3f}, "
              f"Gap: {file_sr - synth_sr:+.3f}")

        # Synth should not be catastrophically worse
        assert synth_sr >= file_sr - 0.5, (
            f"Too large generalisation gap: file={file_sr:.3f}, synth={synth_sr:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
