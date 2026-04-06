"""TIC-v2: Teaching-Internalization Corridor with mechanism-separating subtypes.

Additional subtypes:
  verified_warn     — warning verified by later reveal (↑τ, not ν)
  self_discovery_needed — WAIT enables self-discovery (↓ν)
  false_suppression_cost — lure branch is actually beneficial (↑γ_gen penalty)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .teaching_internalization_corridor import (
    TICEpisodeSpec, TICSessionSpec, SUBTYPE_PARAMS as BASE_PARAMS,
    generate_tic_scenario,
)

TIC_V2_SUBTYPES = [
    "temptation_repeat",
    "self_discovery_teach",
    "warn_rescue",
    "boundary_obs",
    "verified_warn",
    "self_discovery_needed",
    "false_suppression_cost",
]

SUBTYPE_V2_PARAMS = {
    **BASE_PARAMS,
    "verified_warn":          {"d_c": (4, 6), "d_r": (1, 2), "lure": (0.5, 0.8), "risk": (0.3, 0.5)},
    "self_discovery_needed":  {"d_c": (5, 8), "d_r": (1, 3), "lure": (0.3, 0.6), "risk": (0.2, 0.35)},
    "false_suppression_cost": {"d_c": (3, 5), "d_r": (2, 4), "lure": (0.5, 0.8), "risk": (0.05, 0.15)},
}


def generate_tic_v2_session(
    session_id: int,
    theta_true: str = "safe",
    n_tutor: int = 10,
    n_same: int = 4,
    n_shift: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> TICSessionSpec:
    if rng is None:
        rng = np.random.default_rng(session_id)

    eps = []
    idx = 0

    # Phase A: mix including new subtypes
    weights_a = [0.15, 0.10, 0.15, 0.10, 0.20, 0.15, 0.15]
    for _ in range(n_tutor):
        st = rng.choice(TIC_V2_SUBTYPES, p=weights_a)
        sp = SUBTYPE_V2_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="A",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    # Phase B: same structure
    for _ in range(n_same):
        st = rng.choice(TIC_V2_SUBTYPES, p=weights_a)
        sp = SUBTYPE_V2_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="B",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    # Phase C: shifted
    shift_subs = ["temptation_repeat", "self_discovery_needed", "false_suppression_cost"]
    for _ in range(n_shift):
        st = rng.choice(shift_subs)
        sp = SUBTYPE_V2_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="C",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0] + 0.1, min(sp["lure"][1] + 0.15, 1.0))), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1] + 0.05)), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    return TICSessionSpec(session_id, theta_true, eps)


# Reuse generate_tic_scenario from v1, add subtype metadata
def generate_tic_v2_scenario(ep: TICEpisodeSpec):
    gm, cfg, meta, sc = generate_tic_scenario(ep)
    # Tag scenario with v2 subtype info
    sc.is_verified_warn = (ep.subtype == "verified_warn")
    sc.is_self_discovery_needed = (ep.subtype == "self_discovery_needed")
    sc.is_false_suppression = (ep.subtype == "false_suppression_cost")
    sc.risky_branch_actually_good = (ep.subtype == "false_suppression_cost")
    return gm, cfg, meta, sc
