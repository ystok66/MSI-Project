"""
test_phase6efg.py — Phase 6E/6F/6G integrity tests.

Tests:
  6E: Diagnostic menu generator quota satisfaction & label correctness
  6F: Diagnostic HIGHLIGHT cell selection
  6G: G_exp enrichment with diagnostic labels
"""
import os
import sys
import pytest
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.interfaces import Option, Example
from cls_option_tutor.env.state import QueryState
from cls_option_tutor.env.option_env import OptionEnv


DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'cls_learner', 'data')
)


def _make_env(generator_mode="v2_overlap"):
    cfg = FullConfig()
    cfg.env.K = 6
    cfg.env.generator_mode = generator_mode
    cfg.env.n_risky = 2
    return OptionEnv(cfg=cfg, data_dir=DATA_DIR), cfg


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6E Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDiagnosticGenerator:
    """Phase 6E: diagnostic menu generator tests."""

    def _collect_diag_counts(self, generator_mode, task_id="000001", seeds=range(42, 48)):
        env, _ = _make_env(generator_mode)
        counts = Counter()
        for seed in seeds:
            block = env.reset_block(task_id, seed=seed)
            for qs in block.queries:
                for opt in qs.menu:
                    if opt.is_correct:
                        continue
                    counts[qs.option_diag_labels.get(opt.index, "unknown")] += 1
        return counts

    def test_v2_overlap_attaches_sidecar_labels(self):
        """Even in v2_overlap mode, menus should have sidecar labels."""
        env, cfg = _make_env("v2_overlap")
        block = env.reset_block("000001", seed=42)
        qs = block.current_query
        assert qs is not None
        # Should have labels for all options
        assert len(qs.option_diag_labels) == len(qs.menu)
        assert len(qs.option_confound_types) == len(qs.menu)
        # Correct option should be labeled "correct"
        for opt in qs.menu:
            if opt.is_correct:
                assert qs.option_diag_labels[opt.index] == "correct"

    def test_diagnostic_quota_generates_labels(self):
        """diagnostic_quota mode should produce labeled menus."""
        env, cfg = _make_env("diagnostic_quota")
        block = env.reset_block("000001", seed=42)
        qs = block.current_query
        assert qs is not None
        assert len(qs.option_diag_labels) == len(qs.menu)
        # At least one option should be non-correct
        non_correct_labels = [
            qs.option_diag_labels[opt.index]
            for opt in qs.menu if not opt.is_correct
        ]
        assert len(non_correct_labels) > 0

    def test_safe_random_control_is_not_diagnostic(self):
        """SAFE_RANDOM_WRONG should have FAR_DISTRACTOR confound type."""
        from cls_option_tutor.grammar.confound_labels import (
            ConfoundType, DiagnosticRiskLabel
        )
        env, cfg = _make_env("diagnostic_quota")
        # Run multiple blocks to find a safe_random_wrong
        found = False
        for seed in range(42, 52):
            block = env.reset_block("000001", seed=seed)
            for qs in block.queries:
                for opt in qs.menu:
                    if qs.option_diag_labels.get(opt.index) == "safe_random_wrong":
                        ct = qs.option_confound_types.get(opt.index)
                        assert ct == "far_distractor", \
                            f"SAFE_RANDOM_WRONG should be FAR_DISTRACTOR, got {ct}"
                        found = True
                        break
                if found:
                    break
            if found:
                break
        # It's OK if no safe_random_wrong was generated — pool may be too small

    def test_high_risk_lure_has_high_risk_class(self):
        """HIGH_RISK_LURE options should have risk_class >= 3."""
        env, cfg = _make_env("diagnostic_quota")
        for seed in range(42, 52):
            block = env.reset_block("000001", seed=seed)
            for qs in block.queries:
                for opt in qs.menu:
                    if qs.option_diag_labels.get(opt.index) == "high_risk_lure":
                        assert opt.risk_class >= 3, \
                            f"HIGH_RISK_LURE should have risk_class >= 3, got {opt.risk_class}"

    def test_allow_heavy_generator_biases_toward_productive_labels(self):
        counts = self._collect_diag_counts("diagnostic_quota_allow_heavy")
        productive = counts["safe_diagnostic_wrong"] + counts["bounded_diagnostic_wrong"]
        harmful = counts["high_risk_lure"] + counts["risky_far"]
        assert productive > harmful

    def test_mixed_prod_harm_heavy_generator_contains_both_sides(self):
        counts = self._collect_diag_counts("diagnostic_quota_mixed_prod_harm_heavy")
        productive = counts["safe_diagnostic_wrong"] + counts["bounded_diagnostic_wrong"]
        harmful = counts["high_risk_lure"] + counts["risky_far"]
        assert productive > 0
        assert harmful > 0

    def test_protect_critical_heavy_generator_biases_toward_harmful_labels(self):
        counts = self._collect_diag_counts("diagnostic_quota_protect_critical_heavy")
        productive = counts["safe_diagnostic_wrong"] + counts["bounded_diagnostic_wrong"]
        harmful = counts["high_risk_lure"] + counts["risky_far"]
        assert harmful > productive

    def test_boring_mastery_heavy_generator_biases_toward_safe_far_controls(self):
        counts = self._collect_diag_counts("diagnostic_quota_boring_mastery_heavy")
        boring_like = counts["safe_far"] + counts["safe_random_wrong"]
        informative = (
            counts["safe_diagnostic_wrong"]
            + counts["bounded_diagnostic_wrong"]
            + counts["high_risk_lure"]
        )
        assert boring_like > informative

    def test_confound_labels_output_only(self):
        """Labels should be based on output comparison, not hidden correctness."""
        from cls_option_tutor.grammar.confound_labels import label_confound, ConfoundType
        target = ['R', 'G', 'B', 'Y']
        # Near output: 1 cell diff out of 4 → h=0.25
        ct = label_confound(['R', 'G', 'B', 'X'], target, is_correct=False)
        assert ct == ConfoundType.NEAR_OUTPUT
        # Far distractor: all differ
        ct = label_confound(['X', 'X', 'X', 'X'], target, is_correct=False)
        assert ct == ConfoundType.FAR_DISTRACTOR


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6F Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDiagnosticHighlight:
    """Phase 6F: diagnostic HIGHLIGHT cell selection."""

    def _make_qs_with_options(self, target, option_outputs, pick_probs):
        """Create a QS + active list + probs for testing."""
        qs = QueryState(
            query_id=0,
            target_output=list(target),
            true_program=["dummy"],
            hp=5,
        )
        active = []
        for i, out in enumerate(option_outputs):
            active.append(Option(
                index=i, text=["dummy"],
                danger_vec=np.zeros(16),
                is_correct=(i == 0),
                rendered_output=list(out),
            ))
        probs = np.array(pick_probs, dtype=np.float64)
        return qs, active, probs

    def test_prefers_high_disagreement_cells(self):
        """Diagnostic selector should prefer cells where ~50% of p-mass disagrees."""
        from cls_option_tutor.tutor.highlight_selection import (
            select_diagnostic_highlight_cells
        )
        # Target: R G B R
        # Opt 0 (correct): R G B R  (p=0.1)
        # Opt 1: R X B R  (p=0.45)  — disagrees at cell 1
        # Opt 2: R G B X  (p=0.45)  — disagrees at cell 3
        # m_1 = 0.45, m_3 = 0.45 → D_1 = D_3 = 0.2475
        # m_0 = 0, m_2 = 0 → D = 0
        qs, active, probs = self._make_qs_with_options(
            target=['R', 'G', 'B', 'R'],
            option_outputs=[
                ['R', 'G', 'B', 'R'],  # correct
                ['R', 'X', 'B', 'R'],  # wrong at cell 1
                ['R', 'G', 'B', 'X'],  # wrong at cell 3
            ],
            pick_probs=[0.1, 0.45, 0.45],
        )
        cells = select_diagnostic_highlight_cells(qs, active, probs, max_cells=2)
        assert cells is not None
        # Should pick cells 1 and 3 (where disagreement is highest)
        assert set(cells) == {1, 3}

    def test_differs_from_fixed_first_cells(self):
        """With a skewed menu, diagnostic cells should differ from (0, 1)."""
        from cls_option_tutor.tutor.highlight_selection import (
            select_diagnostic_highlight_cells
        )
        # Target: A B C D
        # Wrong options all disagree at cell 2 only
        qs, active, probs = self._make_qs_with_options(
            target=['A', 'B', 'C', 'D'],
            option_outputs=[
                ['A', 'B', 'C', 'D'],  # correct
                ['A', 'B', 'X', 'D'],  # wrong at cell 2
                ['A', 'B', 'Y', 'D'],  # wrong at cell 2
            ],
            pick_probs=[0.1, 0.45, 0.45],
        )
        cells = select_diagnostic_highlight_cells(qs, active, probs, max_cells=1)
        assert cells is not None
        # Should pick cell 2 (not cell 0 as the old stub would)
        assert cells == (2,)

    def test_uses_pick_distribution(self):
        """Different pick distributions should produce different cell selections."""
        from cls_option_tutor.tutor.highlight_selection import (
            select_diagnostic_highlight_cells
        )
        # Target: A B C
        # Opt 1: X B C (wrong at 0), Opt 2: A X C (wrong at 1)
        qs, active, probs1 = self._make_qs_with_options(
            target=['A', 'B', 'C'],
            option_outputs=[
                ['A', 'B', 'C'],
                ['X', 'B', 'C'],
                ['A', 'X', 'C'],
            ],
            pick_probs=[0.1, 0.8, 0.1],
        )
        _, _, probs2 = self._make_qs_with_options(
            target=['A', 'B', 'C'],
            option_outputs=[
                ['A', 'B', 'C'],
                ['X', 'B', 'C'],
                ['A', 'X', 'C'],
            ],
            pick_probs=[0.1, 0.1, 0.8],
        )
        cells1 = select_diagnostic_highlight_cells(qs, active, probs1, max_cells=1)
        cells2 = select_diagnostic_highlight_cells(qs, active, probs2, max_cells=1)
        # With p mostly on opt1 -> cell 0; with p mostly on opt2 -> cell 1
        assert cells1 == (0,)
        assert cells2 == (1,)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6G Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGExpDiagnostic:
    """Phase 6G: G_exp enrichment with diagnostic labels."""

    def _make_tutor_and_qs(self, lg_mode="off"):
        from cls_option_tutor.tutor.sparse_tutor import SparseTutorAgent
        cfg = FullConfig()
        cfg.tutor.tutor_lg_mode = lg_mode
        tutor = SparseTutorAgent(cfg=cfg)
        qs = QueryState(
            query_id=0,
            target_output=['R', 'G', 'B'],
            true_program=['foo'],
            hp=5,
        )
        return tutor, qs

    def test_diagnostic_mode_adds_bonus_for_diagnostic_wrongs(self):
        """In diagnostic mode, g_exp should be higher when diagnostic labels present."""
        tutor_off, qs = self._make_tutor_and_qs("off")
        tutor_diag, qs2 = self._make_tutor_and_qs("diagnostic")
        # Same qs but with labels
        qs2.option_diag_labels = {0: "correct", 1: "safe_diagnostic_wrong", 2: "safe_far"}

        active = [
            Option(index=0, text=["a"], danger_vec=np.zeros(16),
                   is_correct=True, rendered_output=['R', 'G', 'B']),
            Option(index=1, text=["b"], danger_vec=np.zeros(16),
                   is_correct=False, rendered_output=['R', 'X', 'B'], risk_class=0),
            Option(index=2, text=["c"], danger_vec=np.zeros(16),
                   is_correct=False, rendered_output=['X', 'X', 'X'], risk_class=0),
        ]
        probs = np.array([0.2, 0.5, 0.3])

        g_off = tutor_off._compute_g_exp(qs, active, probs)
        g_diag = tutor_diag._compute_g_exp(qs2, active, probs)

        # Diagnostic mode should give higher g_exp due to diagnostic bonus
        assert g_diag > g_off

    def test_safety_only_returns_zero(self):
        """In safety_only mode, g_exp should always be 0."""
        tutor, qs = self._make_tutor_and_qs("safety_only")
        active = [
            Option(index=0, text=["a"], danger_vec=np.zeros(16),
                   is_correct=True, rendered_output=['R'], risk_class=0),
            Option(index=1, text=["b"], danger_vec=np.zeros(16),
                   is_correct=False, rendered_output=['X'], risk_class=0),
        ]
        probs = np.array([0.5, 0.5])
        assert tutor._compute_g_exp(qs, active, probs) == 0.0

    def test_learning_only_includes_lethal_wrongs(self):
        """In learning_only mode, even lethal wrongs contribute to g_exp."""
        tutor_off, qs = self._make_tutor_and_qs("off")
        tutor_lo, qs2 = self._make_tutor_and_qs("learning_only")

        active = [
            Option(index=0, text=["a"], danger_vec=np.zeros(16),
                   is_correct=True, rendered_output=['R'], risk_class=0),
            Option(index=1, text=["b"], danger_vec=np.zeros(16),
                   is_correct=False, rendered_output=['X'], risk_class=5),  # lethal (>= hp=5)
        ]
        probs = np.array([0.3, 0.7])

        g_off = tutor_off._compute_g_exp(qs, active, probs)
        g_lo = tutor_lo._compute_g_exp(qs2, active, probs)

        # off mode: lethal is excluded → g_exp = 0
        assert g_off == 0.0
        # learning_only mode: lethal included → g_exp = 0.7
        assert g_lo > 0.0
