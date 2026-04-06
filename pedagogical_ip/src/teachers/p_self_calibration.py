"""Lightweight calibration layer for three-outcome p_self.

Applies post-hoc calibration to raw (p_self, p_fail, p_undecided) outputs.
Three options:
  1. Isotonic calibration (non-parametric, data-driven)
  2. Beta calibration (2-parameter sigmoid transform)
  3. Identity pass-through (no calibration, for ablation)

Since we don't have enough data for fit-then-apply in Step 2,
we use a fixed parametric transform derived from Step 1 diagnostics:
  - Raw posterior p_self tends to UNDER-estimate (Brier=0.228 vs actual ~0.33)
  - Raw p_undecided tends to be substantial (~0.17)

The calibration layer can be updated with observed outcomes as data accumulates.

Shadow-only. Does NOT modify any frozen module.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


class CalibrationMode(Enum):
    """Calibration mode for p_self outputs."""
    NONE = "none"              # pass-through (ablation)
    FIXED_BETA = "fixed_beta"  # fixed parametric transform
    ONLINE = "online"          # accumulates data, recalibrates periodically


@dataclass
class PSelfCalibrator:
    """Lightweight calibration for three-outcome (p_self, p_fail, p_undecided).

    Fixed beta calibration:
      p̃ = σ(a · logit(p) + b)
    where σ is sigmoid and logit is log-odds.

    Online calibration:
      Accumulates (predicted, actual) pairs and recalibrates via isotonic
      regression every `recal_interval` observations.
    """
    mode: CalibrationMode = CalibrationMode.FIXED_BETA

    # Fixed beta parameters (from Step 1 diagnostics)
    # These slightly shrink extreme probabilities toward 0.5
    a_self: float = 0.85     # < 1 → compresses toward 0.5
    b_self: float = -0.1     # negative → slightly lowers p_self
    a_fail: float = 0.9
    b_fail: float = 0.1      # positive → slightly raises p_fail
    a_undecided: float = 0.8
    b_undecided: float = 0.0

    # Online calibration data
    _history: List = field(default_factory=list)
    recal_interval: int = 50
    _call_count: int = 0
    _isotonic_map: Optional[Dict] = None

    def calibrate(self, p_self: float, p_fail: float,
                  p_undecided: float = 0.0) -> Dict[str, float]:
        """Apply calibration to raw three-outcome probabilities.

        Returns calibrated (p̃_self, p̃_fail, p̃_undecided) that still sum to 1.
        """
        if self.mode == CalibrationMode.NONE:
            return {
                "p_self": p_self,
                "p_fail": p_fail,
                "p_undecided": p_undecided,
            }

        elif self.mode == CalibrationMode.FIXED_BETA:
            return self._fixed_beta(p_self, p_fail, p_undecided)

        elif self.mode == CalibrationMode.ONLINE:
            if self._isotonic_map is not None:
                return self._apply_isotonic(p_self, p_fail, p_undecided)
            else:
                return self._fixed_beta(p_self, p_fail, p_undecided)

        return {"p_self": p_self, "p_fail": p_fail, "p_undecided": p_undecided}

    def observe(self, p_self: float, p_fail: float, p_undecided: float,
                actual_outcome: str):
        """Record an observed outcome for online recalibration.

        actual_outcome: "self_discovery" | "failure" | "undecided"
        """
        self._history.append({
            "p_self": p_self, "p_fail": p_fail, "p_undecided": p_undecided,
            "outcome": actual_outcome,
        })
        self._call_count += 1
        if (self.mode == CalibrationMode.ONLINE and
                self._call_count % self.recal_interval == 0 and
                len(self._history) >= 20):
            self._fit_isotonic()

    def _fixed_beta(self, p_self: float, p_fail: float,
                    p_undecided: float) -> Dict[str, float]:
        """Apply fixed beta (sigmoid) calibration."""
        ps = self._beta_transform(p_self, self.a_self, self.b_self)
        pf = self._beta_transform(p_fail, self.a_fail, self.b_fail)
        pu = self._beta_transform(p_undecided, self.a_undecided, self.b_undecided)

        # Renormalize to sum to 1
        total = ps + pf + pu + 1e-10
        return {
            "p_self": round(float(ps / total), 4),
            "p_fail": round(float(pf / total), 4),
            "p_undecided": round(float(pu / total), 4),
        }

    @staticmethod
    def _beta_transform(p: float, a: float, b: float) -> float:
        """p̃ = σ(a · logit(p) + b)"""
        p_clip = max(min(p, 0.999), 0.001)
        logit = np.log(p_clip / (1.0 - p_clip))
        return float(1.0 / (1.0 + np.exp(-(a * logit + b))))

    def _fit_isotonic(self):
        """Fit isotonic regression from accumulated history."""
        # Simple bin-based isotonic: sort by predicted p_self, compute
        # empirical frequency in bins
        data = sorted(self._history, key=lambda x: x["p_self"])
        n = len(data)
        bin_size = max(n // 5, 5)
        iso_map = {}
        for start in range(0, n, bin_size):
            chunk = data[start:start + bin_size]
            pred_mean = np.mean([d["p_self"] for d in chunk])
            actual_mean = np.mean([1.0 if d["outcome"] == "self_discovery" else 0.0
                                    for d in chunk])
            iso_map[round(pred_mean, 2)] = round(actual_mean, 3)
        self._isotonic_map = iso_map

    def _apply_isotonic(self, p_self, p_fail, p_undecided):
        """Apply isotonic map (nearest-neighbor interpolation)."""
        if not self._isotonic_map:
            return self._fixed_beta(p_self, p_fail, p_undecided)

        keys = sorted(self._isotonic_map.keys())
        # Find nearest key
        dists = [abs(p_self - k) for k in keys]
        nearest = keys[int(np.argmin(dists))]
        cal_p_self = self._isotonic_map[nearest]

        # Scale p_fail and p_undecided proportionally
        remainder_raw = max(1.0 - p_self, 1e-10)
        remainder_cal = max(1.0 - cal_p_self, 0.0)
        ratio = remainder_cal / remainder_raw
        cal_p_fail = p_fail * ratio
        cal_p_undecided = p_undecided * ratio

        return {
            "p_self": round(float(cal_p_self), 4),
            "p_fail": round(float(cal_p_fail), 4),
            "p_undecided": round(float(cal_p_undecided), 4),
        }

    def reset(self):
        self._history = []
        self._call_count = 0
        self._isotonic_map = None
