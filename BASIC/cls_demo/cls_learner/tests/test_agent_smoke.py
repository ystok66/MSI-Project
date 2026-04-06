"""
test_agent_smoke.py — Smoke tests for CLSAgent on simple examples.

Verifies:
  1. CLSAgent can be instantiated with default config
  2. study() + predict() round-trip works
  3. HPC on/off produces results (not crashes)
  4. Stack and AST modes both work
  5. Blockwise Mahalanobis CA1 calibration runs
  6. evaluate_episode() returns valid results
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import unittest

from cls_learner.agent import CLSAgent
from cls_learner.config import CLSConfig
from cls_learner.interfaces import Example, Episode, MemBias
from cls_learner.layer2_hpc.hpc import EpisodeHPC
from cls_learner.layer2_hpc.ca1 import CA1Comparator


class TestCLSAgentSmoke(unittest.TestCase):
    """Smoke tests for CLSAgent."""

    def _make_simple_support(self):
        """Simple 3-word dataset: dax→BLUE, lug→RED, wif→GREEN."""
        return [
            Example(words=['dax'], output=['BLUE']),
            Example(words=['lug'], output=['RED']),
            Example(words=['wif'], output=['GREEN']),
            Example(words=['dax', 'lug'], output=['BLUE', 'RED']),
        ]

    # ── Instantiation ──────────────────────────────────────────

    def test_default_config(self):
        agent = CLSAgent()
        self.assertIsNotNone(agent.cortex)
        self.assertIsNotNone(agent.hpc)  # use_hpc=True by default
        self.assertIsNotNone(agent.control)

    def test_no_hpc(self):
        cfg = CLSConfig(use_hpc=False)
        agent = CLSAgent(cfg=cfg)
        self.assertIsNone(agent.hpc)

    # ── Study + Predict round-trip ─────────────────────────────

    def test_study_stack_no_hpc(self):
        cfg = CLSConfig(mode='stack', use_hpc=False, n_em=2)
        agent = CLSAgent(cfg=cfg)
        support = self._make_simple_support()
        agent.reset_episode()
        agent.study(support)

        pred = agent.predict(['dax'])
        self.assertIsInstance(pred, list)
        self.assertGreater(len(pred), 0)

    def test_study_stack_with_hpc(self):
        cfg = CLSConfig(mode='stack', use_hpc=True, n_em=2)
        agent = CLSAgent(cfg=cfg)
        support = self._make_simple_support()
        agent.reset_episode()
        agent.study(support)

        pred = agent.predict(['dax'])
        self.assertIsInstance(pred, list)

    def test_study_ast_no_hpc(self):
        cfg = CLSConfig(mode='ast', use_hpc=False, n_em=2)
        agent = CLSAgent(cfg=cfg)
        support = self._make_simple_support()
        agent.reset_episode()
        agent.study(support)

        pred = agent.predict(['dax'])
        self.assertIsInstance(pred, list)

    def test_study_ast_with_hpc(self):
        cfg = CLSConfig(mode='ast', use_hpc=True, n_em=2)
        agent = CLSAgent(cfg=cfg)
        support = self._make_simple_support()
        agent.reset_episode()
        agent.study(support)

        pred = agent.predict(['dax'])
        self.assertIsInstance(pred, list)

    # ── Accuracy check ─────────────────────────────────────────

    def test_learn_simple_nouns(self):
        """Agent should learn simple 1:1 noun mappings."""
        cfg = CLSConfig(mode='stack', use_hpc=False, n_em=3)
        agent = CLSAgent(cfg=cfg)
        support = self._make_simple_support()
        agent.reset_episode()
        agent.study(support)

        # Check dax → BLUE
        pred_dax = agent.predict(['dax'])
        self.assertEqual(pred_dax, ['BLUE'],
                         f"Expected ['BLUE'], got {pred_dax}")

        # Check lug → RED
        pred_lug = agent.predict(['lug'])
        self.assertEqual(pred_lug, ['RED'],
                         f"Expected ['RED'], got {pred_lug}")

    # ── evaluate_episode ───────────────────────────────────────

    def test_evaluate_episode(self):
        cfg = CLSConfig(mode='stack', use_hpc=True, n_em=2)
        agent = CLSAgent(cfg=cfg)

        episode = Episode(
            support=self._make_simple_support(),
            query=[
                Example(words=['dax'], output=['BLUE']),
                Example(words=['lug'], output=['RED']),
            ],
        )

        result = agent.evaluate_episode(episode)
        self.assertIn('accuracy', result)
        self.assertIn('correct', result)
        self.assertIn('total', result)
        self.assertEqual(result['total'], 2)

    # ── learn() compatibility ──────────────────────────────────

    def test_learn_compat(self):
        """Test learn() with dict format for backward compatibility."""
        cfg = CLSConfig(mode='stack', use_hpc=False, n_em=2)
        agent = CLSAgent(cfg=cfg)
        support_dicts = [
            {'input': ['dax'], 'output': ['BLUE']},
            {'input': ['lug'], 'output': ['RED']},
        ]
        agent.learn(support_dicts)
        pred = agent.predict(['dax'])
        self.assertIsInstance(pred, list)


class TestCA1Mahalanobis(unittest.TestCase):
    """Test blockwise Mahalanobis CA1."""

    def test_euclidean_fallback(self):
        """Without calibration, falls back to Euclidean."""
        ca1 = CA1Comparator(eps=1e-3)
        e1 = np.array([1.0, 0.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0, 0.0])
        delta = ca1.mismatch(e1, [e2], [1.0])
        # Euclidean: ||[1,-1,0,0]|| squared = 2
        self.assertAlmostEqual(delta, 2.0, places=5)

    def test_blockwise_calibration(self):
        """After calibration, uses blockwise Mahalanobis."""
        ca1 = CA1Comparator(eps=1e-3, mix_a=0.7)
        ca1.set_block_ranges([(0, 2), (2, 4)])

        # Residuals with different variances per block
        residuals = [
            np.array([0.1, 0.1, 1.0, 1.0]),
            np.array([-0.1, -0.1, -1.0, -1.0]),
            np.array([0.05, 0.05, 0.5, 0.5]),
            np.array([-0.05, -0.05, -0.5, -0.5]),
        ]
        features = residuals  # same for simplicity

        ca1.calibrate(residuals, features)
        self.assertTrue(ca1._calibrated)
        self.assertEqual(len(ca1._inv_var_blocks), 2)

        # Block 1 has small var → large inv_var → more sensitive
        # Block 2 has large var → small inv_var → less sensitive
        self.assertGreater(ca1._inv_var_blocks[0].mean(),
                          ca1._inv_var_blocks[1].mean())

    def test_gate_modes(self):
        """Gate should produce retrieve/mixed/explore modes."""
        ca1 = CA1Comparator(eps=1e-3)
        ca1.th_low = 0.3
        ca1.th_high = 0.7
        ca1.th = 0.5
        ca1.temp = 0.04

        lam, mode = ca1.gate(0.1)
        self.assertEqual(mode, 'retrieve')
        self.assertGreater(lam, 0.5)

        lam, mode = ca1.gate(1.0)
        self.assertEqual(mode, 'explore')
        self.assertLess(lam, 0.5)

        lam, mode = ca1.gate(0.5)
        self.assertEqual(mode, 'mixed')


class TestEpisodeHPC(unittest.TestCase):
    """Test EpisodeHPC lifecycle."""

    def test_lifecycle(self):
        """Full HPC lifecycle: reset → write → calibrate → get_bias → replay."""
        hpc = EpisodeHPC()
        hpc.reset()

        from cls_learner.interfaces import TraceSummary

        # Write examples
        ts1 = TraceSummary(
            per_word_role={'dax': 'EMIT'},
            per_word_color={'dax': 'BLUE'},
            trace_roles={'dax': {'EMIT': 1.0, 'REPEAT': 0.0}},
        )
        idx1 = hpc.write_example(['dax'], ['BLUE'], ts1)
        self.assertEqual(idx1, 0)

        ts2 = TraceSummary(
            per_word_role={'lug': 'EMIT'},
            per_word_color={'lug': 'RED'},
            trace_roles={'lug': {'EMIT': 1.0, 'REPEAT': 0.0}},
        )
        idx2 = hpc.write_example(['lug'], ['RED'], ts2)
        self.assertEqual(idx2, 1)

        # Calibrate
        hpc.calibrate_gate()

        # Get bias
        bias = hpc.get_bias(['dax'])
        self.assertIsInstance(bias, MemBias)
        self.assertIn(bias.mode, ('retrieve', 'mixed', 'explore'))

        # Replay
        replays = hpc.sample_replay(batch_size=2)
        self.assertLessEqual(len(replays), 2)


if __name__ == '__main__':
    unittest.main()
