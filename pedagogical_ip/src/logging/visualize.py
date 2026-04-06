"""
Visualization utilities for belief maps and episode replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_belief_heatmap(
    belief_cost_mean: np.ndarray,
    belief_cost_var: np.ndarray,
    belief_risk_mean: np.ndarray,
    belief_risk_var: np.ndarray,
    true_cost: Optional[np.ndarray] = None,
    true_risk: Optional[np.ndarray] = None,
    agent_pos: Optional[tuple[int, int]] = None,
    step: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a 2×2 (or 2×3) heatmap grid of belief maps.

    Panels: cost_mean, cost_var, risk_mean, risk_var
    Optionally adds true_cost and true_risk columns.
    """
    has_true = true_cost is not None and true_risk is not None
    ncols = 3 if has_true else 2

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9))
    fig.suptitle(f"Belief Maps — Step {step}", fontsize=14, fontweight="bold")

    # Cost row
    _plot_panel(axes[0, 0], belief_cost_mean, "Belief Cost Mean",
                agent_pos, cmap="YlOrRd", vmin=0, vmax=10)
    _plot_panel(axes[0, 1], belief_cost_var, "Belief Cost Var",
                agent_pos, cmap="Blues", vmin=0, vmax=5)

    # Risk row
    _plot_panel(axes[1, 0], belief_risk_mean, "Belief Risk Mean",
                agent_pos, cmap="Reds", vmin=0, vmax=1)
    _plot_panel(axes[1, 1], belief_risk_var, "Belief Risk Var",
                agent_pos, cmap="Purples", vmin=0, vmax=0.5)

    if has_true:
        tc = np.where(np.isinf(true_cost), 10.0, true_cost)
        _plot_panel(axes[0, 2], tc, "True Cost",
                    agent_pos, cmap="YlOrRd", vmin=0, vmax=10)
        _plot_panel(axes[1, 2], true_risk, "True Risk",
                    agent_pos, cmap="Reds", vmin=0, vmax=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _plot_panel(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    agent_pos: Optional[tuple[int, int]],
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Helper to plot a single heatmap panel."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if agent_pos is not None:
        ax.plot(agent_pos[1], agent_pos[0], "ko", markersize=10, markeredgewidth=2)
        ax.plot(agent_pos[1], agent_pos[0], "w*", markersize=6)

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))


def save_episode_belief_sequence(
    npz_dir: str,
    output_dir: str,
    max_frames: int = 60,
) -> list[str]:
    """
    Load NPZ snapshots from an episode directory and save
    a sequence of belief heatmap PNGs.

    Returns list of saved PNG paths.
    """
    npz_path = Path(npz_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_path.glob("step_*.npz"))[:max_frames]
    saved = []

    for npz_file in npz_files:
        data = np.load(str(npz_file))
        step_num = int(npz_file.stem.split("_")[1])
        save_file = str(out_path / f"belief_step_{step_num:04d}.png")

        true_cost = data.get("true_cost_map")
        true_risk = data.get("true_risk_map")

        plot_belief_heatmap(
            belief_cost_mean=data["belief_cost_mean"],
            belief_cost_var=data["belief_cost_var"],
            belief_risk_mean=data["belief_risk_mean"],
            belief_risk_var=data["belief_risk_var"],
            true_cost=true_cost,
            true_risk=true_risk,
            step=step_num,
            save_path=save_file,
        )
        saved.append(save_file)

    return saved
