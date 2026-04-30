"""
highlight_selection.py — Phase 6F: Diagnostic HIGHLIGHT cell selector.

Replaces the stub in sparse_tutor._select_highlight_cells().
Uses inverse-predictor pick distribution to find cells that best
separate plausible hypotheses (maximize self-correction signal).

Core formula:
    m_l = Σ_j p_j · 𝟙[Y_j[l] ≠ Y*[l]]     (expected mistake mass)
    D_l = m_l · (1 - m_l)                     (diagnostic score)

D_l is maximized at m_l = 0.5 (cell where half of the probability mass
makes mistakes → most informative for self-correction).
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from ..env.state import QueryState
from ..interfaces import Option


def select_diagnostic_highlight_cells(
    qs: QueryState,
    active: List[Option],
    pick_probs: np.ndarray,
    max_cells: int,
) -> Optional[Tuple[int, ...]]:
    """Select output cells that maximize diagnostic self-correction signal.

    Args:
        qs: Current query state (has target_output)
        active: Active menu options
        pick_probs: (K_active,) array of learner pick probabilities
        max_cells: Maximum number of cells to highlight

    Returns:
        Tuple of cell indices, or None if no valid selection.
    """
    L = len(qs.target_output)
    if L == 0:
        return None

    target = qs.target_output
    K = len(active)

    if K == 0 or len(pick_probs) != K:
        return tuple(range(min(max_cells, L))) if L > 0 else None

    # ── Compute per-cell expected mistake mass ──
    m = np.zeros(L, dtype=np.float64)
    for i, opt in enumerate(active):
        if opt.is_correct:
            continue
        p_j = float(pick_probs[i])
        rendered = opt.rendered_output
        if rendered is None:
            continue
        for l in range(min(L, len(rendered))):
            if rendered[l] != target[l]:
                m[l] += p_j

    # ── Diagnostic score: D_l = m_l * (1 - m_l) ──
    D = m * (1.0 - m)

    # ── Select top-k cells ──
    # Fallback: if D is flat (all cells equally diagnostic), use m_l instead
    if np.std(D) < 0.01:
        # Use m_l directly — highlight cells with highest mistake mass
        scores = m
    else:
        scores = D

    n_select = min(max_cells, L)
    if n_select <= 0:
        return None

    # Get top-k indices
    top_indices = np.argsort(-scores)[:n_select]
    # Sort by position for consistent output
    top_indices = sorted(top_indices)

    return tuple(int(idx) for idx in top_indices) if top_indices else None


# ── Phase 6I-B: counterfactual P(correct) highlight selector ────────────

def select_counterfactual_highlight_cells(
    qs: QueryState,
    active: List[Option],
    learner,
    max_cells: int = 2,
    m_candidates: int = 4,
) -> Optional[Tuple[int, ...]]:
    """Select highlight cells that maximize ΔP(correct) vs no-highlight.

    Two-stage procedure:
      1. Nominate top-m candidate cells via D_l diagnostic score.
      2. Enumerate single cells, pairs, and fixed fallback.
         For each candidate, clone learner+qs, set highlighted_cells,
         call get_policy_snapshot_for_query(), measure P(correct).
      3. Return candidate with max ΔP. If max ΔP <= 0, return fixed fallback.

    Args:
        qs: Current query state (post-reveal)
        active: Active menu options
        learner: LearnerAgent with get_policy_snapshot_for_query()
        max_cells: Max cells per highlight action (default 2)
        m_candidates: Number of D_l candidate cells to consider (default 4)

    Returns:
        Tuple of cell indices, or None if no valid selection.
    """
    import copy
    import itertools

    L = len(qs.target_output) if qs.target_output else 0
    if L == 0:
        return None

    K = len(active)
    if K == 0:
        return tuple(range(min(max_cells, L)))

    # ── Stage 1: D_l nomination ──
    # Get baseline pick probs for D_l computation
    try:
        baseline_policy = learner.get_policy_snapshot_for_query(qs)
        baseline_probs = np.asarray(baseline_policy.probs, dtype=float)
        # Strip refresh slot if present
        if len(baseline_probs) == K + 1:
            p_correct_baseline = 0.0
            pick_probs = baseline_probs[:K]
        elif len(baseline_probs) == K:
            pick_probs = baseline_probs
        else:
            pick_probs = np.ones(K) / K
    except Exception:
        pick_probs = np.ones(K) / K

    # Compute D_l scores
    target = qs.target_output
    m_mass = np.zeros(L, dtype=np.float64)
    for i, opt in enumerate(active):
        if opt.is_correct or i >= len(pick_probs):
            continue
        p_j = float(pick_probs[i])
        rendered = opt.rendered_output
        if rendered is None:
            continue
        for l_idx in range(min(L, len(rendered))):
            if rendered[l_idx] != target[l_idx]:
                m_mass[l_idx] += p_j

    D_l = m_mass * (1.0 - m_mass)
    top_m_indices = list(np.argsort(-D_l)[:m_candidates])

    # ── Stage 2: enumerate candidates, measure P(correct) ──
    # Fixed fallback
    fixed_cells = tuple(range(min(max_cells, L)))

    # Build candidate cell sets: singles + combinations up to max_cells + fixed
    cell_candidates = []
    # Singles
    for idx in top_m_indices:
        cell_candidates.append((idx,))
    # Larger subsets
    max_r = min(max_cells, len(top_m_indices))
    for r in range(2, max_r + 1):
        for combo in itertools.combinations(top_m_indices, r):
            cell_candidates.append(combo)
    # Fixed fallback
    if fixed_cells not in cell_candidates:
        cell_candidates.append(fixed_cells)

    # Baseline P(correct) without highlight
    p_correct_no_hl = _eval_p_correct(qs, active, learner, highlight_cells=None)

    best_delta = -999.0
    best_cells = fixed_cells

    for cells in cell_candidates:
        p_correct_hl = _eval_p_correct(qs, active, learner, highlight_cells=tuple(cells))
        delta = p_correct_hl - p_correct_no_hl
        if delta > best_delta:
            best_delta = delta
            best_cells = tuple(cells)

    # If no positive delta, fall back to fixed
    if best_delta <= 0:
        return fixed_cells

    # Sort by position for consistent output
    return tuple(sorted(int(c) for c in best_cells))


def _eval_p_correct(
    qs: QueryState,
    active: List[Option],
    learner,
    highlight_cells: Optional[Tuple[int, ...]] = None,
) -> float:
    """Evaluate P(correct) for a given highlight cell configuration.

    Clones qs, sets highlighted_cells, and calls learner snapshot.
    Returns P(correct) pick probability (excluding refresh mass).
    """
    import copy

    qs_clone = copy.deepcopy(qs)
    if highlight_cells is not None:
        qs_clone.highlighted_cells = tuple(highlight_cells)
    else:
        qs_clone.highlighted_cells = ()

    try:
        policy_out = learner.get_policy_snapshot_for_query(qs_clone)
        raw = np.asarray(policy_out.probs, dtype=float)
        K = len(active)
        # Pick probs = raw[:K] (strip refresh if K+1)
        if len(raw) >= K:
            pick_probs = raw[:K]
        else:
            return 0.0
        for i, opt in enumerate(active):
            if opt.is_correct and i < len(pick_probs):
                return float(pick_probs[i])
    except Exception:
        pass
    return 0.0
