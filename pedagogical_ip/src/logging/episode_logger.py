"""
Episode logger — saves per-step JSONL and NPZ snapshots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeLogger:
    """Logs episode data to JSONL (events) and NPZ (belief snapshots)."""

    def __init__(self, log_dir: str, save_npz: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_npz = save_npz
        self._current_episode: int = -1
        self._jsonl_file = None

    def start_episode(self, episode_id: int) -> None:
        """Open a new JSONL file for this episode."""
        self._current_episode = episode_id
        filepath = self.log_dir / f"episode_{episode_id:04d}.jsonl"
        self._jsonl_file = open(filepath, "w", encoding="utf-8")

        if self.save_npz:
            npz_dir = self.log_dir / f"episode_{episode_id:04d}_npz"
            npz_dir.mkdir(exist_ok=True)

    def log_step(
        self,
        episode_id: int,
        step: int,
        robot_action: dict,
        agent_action: str,
        agent_pos_before: list,
        agent_pos_after: list,
        true_cost: float,
        true_risk: float,
        time_left: int,
        risk_budget_left: float,
        has_object: bool,
        has_shield: bool,
        terminated: bool,
        truncated: bool,
        reward: float,
        # Optional metrics
        epistemic_gain: float = 0.0,
        frustration_score: float = 0.0,
        teacher_info: dict | None = None,
        # Belief snapshots
        belief_snapshot: dict | None = None,
        true_cost_map: np.ndarray | None = None,
        true_risk_map: np.ndarray | None = None,
    ) -> None:
        """Log one step to JSONL and optionally save NPZ."""
        record: dict[str, Any] = {
            "episode_id": episode_id,
            "step": step,
            "robot_action": robot_action,
            "agent_action": agent_action,
            "agent_pos_before": agent_pos_before,
            "agent_pos_after": agent_pos_after,
            "true_cost": true_cost,
            "true_risk": true_risk,
            "time_left": time_left,
            "risk_budget_left": round(risk_budget_left, 4),
            "has_object": has_object,
            "has_shield": has_shield,
            "epistemic_gain": round(epistemic_gain, 6),
            "frustration_score": round(frustration_score, 4),
            "reward": round(reward, 4),
            "terminated": terminated,
            "truncated": truncated,
        }

        if teacher_info:
            # Flatten teacher prediction scores
            for k, v in teacher_info.items():
                if k == "scores":
                    record["teacher_scores"] = v
                elif isinstance(v, (int, float, str, bool)):
                    record[k] = v

        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(record) + "\n")
            self._jsonl_file.flush()

        # Save NPZ
        if self.save_npz and belief_snapshot is not None:
            npz_dir = self.log_dir / f"episode_{episode_id:04d}_npz"
            npz_path = npz_dir / f"step_{step:04d}.npz"
            arrays = dict(belief_snapshot)
            if true_cost_map is not None:
                arrays["true_cost_map"] = true_cost_map
            if true_risk_map is not None:
                arrays["true_risk_map"] = true_risk_map
            np.savez_compressed(str(npz_path), **arrays)

    def end_episode(self) -> None:
        """Close the JSONL file."""
        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None

    def __del__(self):
        if self._jsonl_file:
            self._jsonl_file.close()
