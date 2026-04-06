"""Persistent Learner Profile — Manager.

In-memory primary path with optional JSON export for debug/reproducibility.
Core path never touches disk; artifact path is opt-in.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np

from .profile_state import ProfileState, SessionSummary


class ProfileManager:
    """Manages cross-session profiles for one or more learners.

    Primary storage is in-memory. JSON export is optional (for debug/analysis).

    Usage:
        pm = ProfileManager()
        pm.finalize_session("learner_0", profile)
        prev = pm.latest("learner_0")
        # ... bootstrap next session from prev ...
    """

    def __init__(self):
        self._profiles: Dict[str, List[ProfileState]] = {}

    # ═══ Core API ═══════════════════════════════════════

    def finalize_session(self, learner_id: str, profile: ProfileState):
        """Store completed session profile."""
        profile.learner_id = learner_id
        if learner_id not in self._profiles:
            self._profiles[learner_id] = []
        self._profiles[learner_id].append(profile.copy())

    def latest(self, learner_id: str) -> Optional[ProfileState]:
        """Get most recent profile for a learner. None if no history."""
        sessions = self._profiles.get(learner_id, [])
        return sessions[-1].copy() if sessions else None

    def session_count(self, learner_id: str) -> int:
        return len(self._profiles.get(learner_id, []))

    def all_sessions(self, learner_id: str) -> List[ProfileState]:
        return [p.copy() for p in self._profiles.get(learner_id, [])]

    def reset_learner(self, learner_id: str):
        """Clear all sessions for a learner."""
        self._profiles.pop(learner_id, None)

    def learner_ids(self) -> List[str]:
        return list(self._profiles.keys())

    # ═══ Aggregate Analytics ════════════════════════════

    def aggregate_history(self, learner_id: str) -> dict:
        """Compute aggregate statistics across all sessions.

        Returns summary dict suitable for curriculum need-scoring.
        """
        sessions = self._profiles.get(learner_id, [])
        if not sessions:
            return {"n_sessions": 0}

        all_probe_means = {}
        total_warn = 0
        total_wait = 0
        total_steps = 0

        for ps in sessions:
            h = ps.history
            total_warn += h.n_warn
            total_wait += h.n_wait
            total_steps += h.n_steps
            for p, v in h.probe_means.items():
                all_probe_means.setdefault(p, []).append(v)

        return {
            "n_sessions": len(sessions),
            "total_steps": total_steps,
            "overall_warn_rate": total_warn / max(total_warn + total_wait, 1),
            "probe_means": {p: float(np.mean(vs))
                           for p, vs in all_probe_means.items()},
            "latest_m_hat": dict(sessions[-1].m_hat_terminal),
            "latest_confidence": dict(sessions[-1].confidence),
        }

    def drift_metric(self, learner_id: str, window: int = 3) -> dict:
        """Compute inter-session state drift: |m̄_s - m̄_{s-1}|_W.

        Returns per-dimension drift and overall drift.
        """
        sessions = self._profiles.get(learner_id, [])
        if len(sessions) < 2:
            return {"overall": 0.0, "per_dim": {}}

        recent = sessions[-window:] if len(sessions) >= window else sessions
        dims = list(recent[0].m_hat_terminal.keys())
        drifts = {d: [] for d in dims}

        for i in range(1, len(recent)):
            prev = recent[i - 1].m_hat_terminal
            curr = recent[i].m_hat_terminal
            for d in dims:
                drifts[d].append(abs(curr.get(d, 0) - prev.get(d, 0)))

        per_dim = {d: float(np.mean(vs)) if vs else 0.0
                   for d, vs in drifts.items()}
        overall = float(np.mean(list(per_dim.values())))
        return {"overall": overall, "per_dim": per_dim}

    def calibration_trend(self, learner_id: str) -> List[dict]:
        """Return per-session calibration errors (|m̂_T - m_T| per dim)."""
        sessions = self._profiles.get(learner_id, [])
        return [dict(ps.history.calibration_error) for ps in sessions]

    def probe_weakness_summary(self, learner_id: str,
                               rho: float = 0.5) -> dict:
        """Compute EMA-smoothed per-probe weakness: z̄_{s,p}.

        z̄_{s,p} = (1-ρ) · z̄_{s-1,p} + ρ · z_{s,p}^obs

        Returns dict {probe_name: z̄_value}. Higher = more mastered.
        Curriculum need hook computes: [z*_p - z̄_p]+ for deficit.

        Args:
            learner_id: Learner identifier.
            rho: EMA decay. 0.5 = moderate smoothing. Higher = more recent.
        """
        from ..curriculum.lesson_library_v2 import PROBE_NAMES
        sessions = self._profiles.get(learner_id, [])
        if not sessions:
            return {p: 0.5 for p in PROBE_NAMES}  # default prior

        z_bar = {p: 0.5 for p in PROBE_NAMES}  # init at 0.5
        for ps in sessions:
            pm = ps.history.probe_means
            for p in PROBE_NAMES:
                if p in pm:
                    z_bar[p] = (1 - rho) * z_bar[p] + rho * pm[p]
        return {p: round(v, 6) for p, v in z_bar.items()}

    # ═══ Optional JSON Export ═══════════════════════════

    def save_profiles(self, path: Path):
        """Export all profiles to JSON (debug/reproducibility artifact)."""
        data = {}
        for lid, sessions in self._profiles.items():
            data[lid] = [ps.to_dict() for ps in sessions]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load_profiles(self, path: Path):
        """Load profiles from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for lid, sessions in data.items():
            self._profiles[lid] = [ProfileState.from_dict(s) for s in sessions]
