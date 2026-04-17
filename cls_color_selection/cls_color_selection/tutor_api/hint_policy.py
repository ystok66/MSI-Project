"""
hint_policy.py — Fine-grained hint policy (Phase 4.5 + Phase 5).

Phase 4.5 (P1-P3):
  P1: Continuous hint utility (Q_hint > 0 → give hint)
  P2: Targeted position selection (hint "most wrong & most uncertain")
  P3: Adaptive k (greedy marginal gain, stop when ΔQ ≤ 0)

Phase 5 additions:
  P5-1: Trace salience T_i in position scoring
  P5-2: High-entropy fallback (flat beam → fallback scoring)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .beam_analysis import (
    BeamQueryAnalysis, PositionAnalysis,
    analyze_beam, compute_p_succ_wait, compute_p_succ_with_hints,
)


# ── Policy config ─────────────────────────────────────────────

@dataclass
class HintPolicyConfig:
    """Hyperparameters for fine-grained hint policy."""
    # P1: continuous utility weights
    lambda_to: float = 1.5      # timeout prevention weight
    lambda_err: float = 1.0     # expected error weight
    lambda_unc: float = 0.5     # uncertainty (1-margin) weight
    lambda_H: float = 0.3       # beam entropy weight
    lambda_over: float = 0.8    # over-help penalty
    lambda_int: float = 0.2     # fixed intervention cost

    # P2: position scoring
    lambda_pos: float = 1.0     # position entropy weight
    lambda_impact: float = 1.5  # position impact weight

    # P5-1: trace salience
    lambda_trace: float = 0.8   # trace salience weight in position scoring

    # P3: adaptive k
    k_max: int = 4              # max positions to hint

    # Competence sigmoid
    comp_midpoint: float = 0.5  # sigmoid midpoint
    comp_steepness: float = 5.0 # sigmoid steepness

    # P5-2: flat-beam fallback
    tau_flat: float = 0.85      # normalized entropy threshold for fallback
    lambda_mask: float = 0.5    # base value for wrong positions in fallback

    # S2: learning-loss proxy (pollution penalty) — disabled by default
    # Phase 6v2 showed S2 is anti-productive at current teach scale.
    # Set >0 to activate (e.g. 2.0-3.0 for visible effect).
    lambda_pollution: float = 0.0


DEFAULT_HINT_CONFIG = HintPolicyConfig()


# ── P1: Continuous hint utility ───────────────────────────────

def compute_coarse_hint_utility(
    analysis: BeamQueryAnalysis,
    c_left: int,
    competence: float = 0.5,
    cfg: HintPolicyConfig = DEFAULT_HINT_CONFIG,
) -> float:
    """Q_hint^coarse: should tutor give ANY hint?

    Q = λ_to · (1 - P_succ_wait)
      + λ_err · E_wrong
      + λ_unc · (1 - margin)
      + λ_H · H_beam_norm
      - λ_over · σ(competence)
      - λ_int

    Returns: Q_hint - Q_wait (positive → hint is beneficial)
    """
    P_succ_wait = compute_p_succ_wait(analysis.p_exact, c_left)

    # Gains
    g_timeout = cfg.lambda_to * (1.0 - P_succ_wait)
    g_error = cfg.lambda_err * analysis.E_wrong
    g_uncertainty = cfg.lambda_unc * (1.0 - analysis.margin)
    g_entropy = cfg.lambda_H * analysis.H_beam_norm

    # Costs
    sigma_comp = 1.0 / (1.0 + np.exp(
        -cfg.comp_steepness * (competence - cfg.comp_midpoint)))
    c_over = cfg.lambda_over * sigma_comp
    c_int = cfg.lambda_int

    return g_timeout + g_error + g_uncertainty + g_entropy - c_over - c_int


# ── P2 + P5-1: Position scoring (with trace salience) ────────

def score_positions(
    analysis: BeamQueryAnalysis,
    wrong_mask: List[bool],
    c_left: int,
    trace_salience: Optional[np.ndarray] = None,
    cfg: HintPolicyConfig = DEFAULT_HINT_CONFIG,
    use_fallback: bool = False,
) -> List[Tuple[int, float, str]]:
    """Score wrong positions for hint priority.

    Phase 5 formula:
      score_i = 1[wrong] · (λ_H·H_i + λ_I·I_i + λ_T·T_i)

    If use_fallback=True (flat beam):
      score_i = 1[wrong] · (λ_H·H_i + λ_T·T_i + λ_mask)

    Returns: sorted list of (position_idx, score, correct_color)
    """
    if not analysis.positions:
        return []

    P_succ_wait = compute_p_succ_wait(analysis.p_exact, c_left)
    scored = []

    for pos in analysis.positions:
        if pos.idx >= len(wrong_mask):
            continue
        if wrong_mask[pos.idx]:
            continue  # Position is correct → don't hint

        if pos.gt_color is None:
            continue

        # Trace salience T_i
        T_i = 0.0
        if trace_salience is not None and pos.idx < len(trace_salience):
            T_i = float(trace_salience[pos.idx])

        if use_fallback:
            # P5-2: Flat-beam fallback — no impact (I_i unreliable)
            score = (cfg.lambda_pos * pos.H_i
                     + cfg.lambda_trace * T_i
                     + cfg.lambda_mask)
        else:
            # Normal mode: impact + uncertainty + trace
            P_succ_i = compute_p_succ_with_hints(
                analysis, [pos.idx], c_left)
            I_i = P_succ_i - P_succ_wait

            score = (cfg.lambda_pos * pos.H_i
                     + cfg.lambda_impact * I_i
                     + cfg.lambda_trace * T_i)

        scored.append((pos.idx, score, pos.gt_color))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── P3: Adaptive k ────────────────────────────────────────────

def select_hint_positions_greedy(
    analysis: BeamQueryAnalysis,
    wrong_mask: List[bool],
    c_left: int,
    competence: float = 0.5,
    trace_salience: Optional[np.ndarray] = None,
    cfg: HintPolicyConfig = DEFAULT_HINT_CONFIG,
    use_fallback: bool = False,
) -> List[Tuple[int, str]]:
    """Greedy marginal gain: select positions until ΔQ ≤ 0.

    Returns: [(position_idx, correct_color), ...] for selected hints
    """
    scored = score_positions(
        analysis, wrong_mask, c_left, trace_salience, cfg, use_fallback)
    if not scored:
        return []

    sigma_comp = 1.0 / (1.0 + np.exp(
        -cfg.comp_steepness * (competence - cfg.comp_midpoint)))

    P_succ_wait = compute_p_succ_wait(analysis.p_exact, c_left)

    selected: List[Tuple[int, str]] = []
    selected_indices: List[int] = []

    for pos_idx, pos_score, gt_color in scored:
        if len(selected) >= cfg.k_max:
            break

        if use_fallback:
            # In fallback mode, accept top positions more readily
            # Only check: score > 0 and not too many
            delta_Q = pos_score - cfg.lambda_over * sigma_comp
            if len(selected) == 0:
                delta_Q -= cfg.lambda_int
        else:
            # Normal mode: compute marginal gain
            candidate = selected_indices + [pos_idx]
            P_succ_new = compute_p_succ_with_hints(
                analysis, candidate, c_left)
            P_succ_old = (compute_p_succ_with_hints(
                analysis, selected_indices, c_left)
                if selected_indices else P_succ_wait)

            delta_succ = P_succ_new - P_succ_old

            # Include trace salience in marginal gain
            T_i = 0.0
            if trace_salience is not None and pos_idx < len(trace_salience):
                T_i = float(trace_salience[pos_idx])

            H_i = (analysis.positions[pos_idx].H_i
                    if pos_idx < len(analysis.positions) else 0.0)

            delta_Q = (cfg.lambda_to * delta_succ
                       + cfg.lambda_pos * H_i / max(len(analysis.positions), 1)
                       + cfg.lambda_trace * T_i / max(len(analysis.positions), 1)
                       - cfg.lambda_over * sigma_comp)

            if len(selected) == 0:
                delta_Q -= cfg.lambda_int

        if delta_Q > 0:
            selected.append((pos_idx, gt_color))
            selected_indices.append(pos_idx)
        else:
            break

    return selected


# ── S2: Learning-loss proxy ───────────────────────────────────

def compute_learning_loss(
    analysis: BeamQueryAnalysis,
    hint_positions: List[int],
    trace_salience: Optional[np.ndarray] = None,
) -> float:
    """L^pollution(S): learning loss from hinting a set of positions.

    L = (1/L) * Σ_{i ∈ S} p_wrong_i · (1 + T_i)

    Intuition: positions that are high-p_wrong AND high structural
    uncertainty are where learner should learn from mistakes.
    Hinting them shortcircuits learning.

    Returns: scalar loss in [0, ~2]
    """
    if not analysis.positions or not hint_positions:
        return 0.0

    L = len(analysis.positions)
    loss = 0.0
    for pos_idx in hint_positions:
        if pos_idx >= len(analysis.positions):
            continue
        pa = analysis.positions[pos_idx]
        T_i = 0.0
        if trace_salience is not None and pos_idx < len(trace_salience):
            T_i = float(trace_salience[pos_idx])
        loss += pa.p_wrong * (1.0 + T_i)

    return loss / max(L, 1)


# ── Full policy: decide and select (Phase 6 version) ──────────

def decide_hint(
    beam: list,
    gt: List[str],
    wrong_mask: List[bool],
    c_left: int,
    competence: float = 0.5,
    trace_salience: Optional[np.ndarray] = None,
    words: Optional[List[str]] = None,
    cfg: HintPolicyConfig = DEFAULT_HINT_CONFIG,
) -> Tuple[bool, List[Tuple[int, str]], Dict]:
    """Full P1+P2+P3+P5 hint decision.

    Args:
        beam: [(score, trace, Y_k), ...] from learner model
        gt: ground truth output
        wrong_mask: [True=correct, False=wrong] per position
        c_left: confirms remaining
        competence: learner competence estimate (0-1)
        trace_salience: T_i array from trace_analysis (optional, Phase 5)
        words: query words (for trace analysis, optional)
        cfg: policy config

    Returns:
        (should_hint, positions, diagnostics)
    """
    analysis = analyze_beam(beam, gt)

    # P1: Should we hint at all?
    Q_hint = compute_coarse_hint_utility(
        analysis, c_left, competence, cfg)

    diag = {
        'Q_hint': float(Q_hint),
        'p_exact': float(analysis.p_exact),
        'H_beam': float(analysis.H_beam),
        'H_beam_norm': float(analysis.H_beam_norm),
        'margin': float(analysis.margin),
        'E_wrong': float(analysis.E_wrong),
        'K': len(analysis.q_k),
        'has_trace_salience': trace_salience is not None,
    }

    if Q_hint <= 0:
        diag['decision'] = 'WAIT'
        diag['reason'] = 'Q_hint <= 0'
        return False, [], diag

    # P5-2: Detect flat beam → use fallback mode
    use_fallback = analysis.H_beam_norm > cfg.tau_flat
    diag['use_fallback'] = use_fallback

    # P2 + P3 + P5-1: Select positions (with trace salience if available)
    positions = select_hint_positions_greedy(
        analysis, wrong_mask, c_left, competence,
        trace_salience=trace_salience,
        cfg=cfg, use_fallback=use_fallback)

    if not positions:
        # P5-2 extra: if fallback still fails, try most-wrong positions
        if use_fallback:
            wrong_positions = []
            for pos in analysis.positions:
                if pos.idx < len(wrong_mask) and not wrong_mask[pos.idx]:
                    if pos.gt_color is not None:
                        wrong_positions.append(
                            (pos.idx, pos.p_wrong, pos.gt_color))
            wrong_positions.sort(key=lambda x: x[1], reverse=True)
            for pos_idx, _, gt_color in wrong_positions[:2]:
                positions.append((pos_idx, gt_color))
            diag['fallback_rescue'] = len(positions) > 0

    if not positions:
        diag['decision'] = 'WAIT'
        diag['reason'] = 'no valuable positions (even after fallback)'
        return False, [], diag

    # S2: Learning-loss penalty — deduct from Q_hint
    hint_indices = [p[0] for p in positions]
    L_poll = compute_learning_loss(analysis, hint_indices, trace_salience)
    Q_after_pollution = Q_hint - cfg.lambda_pollution * L_poll
    diag['L_pollution'] = float(L_poll)
    diag['Q_after_pollution'] = float(Q_after_pollution)

    # If pollution penalty pushes utility below 0, reconsider
    if Q_after_pollution <= 0 and not use_fallback:
        diag['decision'] = 'WAIT'
        diag['reason'] = 'Q_hint positive but pollution penalty too high'
        return False, [], diag

    diag['decision'] = 'HINT'
    diag['n_positions'] = len(positions)
    diag['hinted_positions'] = hint_indices

    # Per-position diagnostics
    pos_diags = []
    for pos_idx, _ in positions:
        pd = {'idx': pos_idx}
        if pos_idx < len(analysis.positions):
            pa = analysis.positions[pos_idx]
            pd['p_wrong'] = float(pa.p_wrong)
            pd['H_i'] = float(pa.H_i)
        if trace_salience is not None and pos_idx < len(trace_salience):
            pd['T_i'] = float(trace_salience[pos_idx])
        pos_diags.append(pd)
    diag['position_details'] = pos_diags

    return True, positions, diag
