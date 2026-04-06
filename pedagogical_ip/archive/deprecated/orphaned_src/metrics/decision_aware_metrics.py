"""Decision-aware metrics for pedagogical tutoring framework.

OAA  = Oracle Action Agreement (tutor action vs oracle)
TR   = Timing Regret (how late a warn arrives)
ECE  = Expected Calibration Error for posteriors
TransferSBCR = Safe Branch Choice Rate post-tutor
AutonomyGain = Perf(post_tutor) / Perf(pre_tutor)
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════
# Oracle Action Agreement
# ══════════════════════════════════════════

def oracle_action(d_commit: int, d_reveal: int, subtype: str = "") -> str:
    """Determine what oracle would do.
    
    WAIT if enough info will be revealed before commitment.
    WARN if agent must commit before info arrives.
    """
    if d_commit < d_reveal:
        return "WARN"
    elif d_commit == d_reveal:
        return "WARN"  # tie → err toward caution
    else:
        return "WAIT"


def compute_oaa(traces: list[dict]) -> float:
    """OAA = fraction of episodes where tutor matches oracle."""
    if not traces:
        return 0.0
    correct = 0
    for t in traces:
        oa = oracle_action(t.get("d_commit", 3), t.get("d_reveal", 2),
                           t.get("subtype", ""))
        tutor_action = "WARN" if t.get("warned", False) else "WAIT"
        if tutor_action == oa:
            correct += 1
    return correct / len(traces)


# ══════════════════════════════════════════
# Timing Regret
# ══════════════════════════════════════════

def compute_timing_regret(traces: list[dict]) -> float:
    """TR = avg delay on episodes where warn was needed.
    
    On episodes where oracle says WARN:
      TR = max(0, t_warn - t_oracle) / session_length
    We proxy this with d_reveal as "when oracle would warn" and
    episode_idx as "when tutor actually warned".
    """
    warn_eps = [t for t in traces
                if oracle_action(t.get("d_commit", 3), t.get("d_reveal", 2)) == "WARN"]
    if not warn_eps:
        return 0.0
    regrets = []
    for t in warn_eps:
        if t.get("warned", False):
            regrets.append(0.0)  # warned correctly
        else:
            # Missed warning: regret proportional to how dangerous
            regrets.append(1.0)
    return float(np.mean(regrets))


# ══════════════════════════════════════════
# Expected Calibration Error
# ══════════════════════════════════════════

def compute_ece(
    predicted_probs: list[float],
    actuals: list[float],
    n_bins: int = 5,
) -> float:
    """ECE = Σ_b (|B_b|/N) · |acc_b - conf_b|."""
    if not predicted_probs:
        return 0.0
    preds = np.array(predicted_probs)
    acts = np.array(actuals)
    n = len(preds)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (preds == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = float(np.mean(preds[mask]))
        avg_acc = float(np.mean(acts[mask]))
        ece += (mask.sum() / n) * abs(avg_acc - avg_conf)
    return float(ece)


# ══════════════════════════════════════════
# Transfer metrics
# ══════════════════════════════════════════

def compute_transfer_sbcr(post_tutor_traces: list[dict]) -> float:
    """SBCR on episodes without tutor intervention."""
    if not post_tutor_traces:
        return 0.0
    safe = sum(1 for t in post_tutor_traces if t.get("agent_safe", False))
    return safe / len(post_tutor_traces)


def compute_autonomy_gain(
    pre_tutor_sbcr: float,
    post_tutor_sbcr: float,
) -> float:
    """AutonomyGain = post / pre (>1 means tutor helped long-term)."""
    if pre_tutor_sbcr < 0.01:
        return post_tutor_sbcr / 0.01
    return post_tutor_sbcr / pre_tutor_sbcr


def compute_intervention_efficiency(
    sbcr: float,
    warn_rate: float,
) -> float:
    """IE = SBCR / max(WarnRate, 0.01) — higher = more efficient tutoring."""
    return sbcr / max(warn_rate, 0.01)
