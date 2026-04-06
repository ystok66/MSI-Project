"""TIC-v4: 5-Phase Teaching-Internalization Corridor.

Phase A: Tutor present (10 ep)
Phase B: Autonomy transfer (4 ep, no tutor)
Phase C: Sparse valid advice (4 ep)
Phase D: Sparse invalid advice (4 ep)
Phase E: Beneficial novelty probe (4 ep)
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .teaching_internalization_corridor import (
    TICEpisodeSpec, TICSessionSpec, SUBTYPE_PARAMS as BASE_PARAMS,
    generate_tic_scenario,
)
from .teaching_internalization_corridor_v3 import SUBTYPE_V3_PARAMS

TIC_V4_SUBTYPES = [
    "temptation_repeat", "self_discovery_teach", "warn_rescue",
    "boundary_obs", "verified_warn", "self_discovery_needed",
    "false_suppression_cost", "sparse_valid_advice",
    "sparse_invalid_advice", "beneficial_novelty",
]

SUBTYPE_V4_PARAMS = {**BASE_PARAMS, **SUBTYPE_V3_PARAMS}


def generate_tic_v4_session(
    session_id: int, theta_true: str = "safe",
    n_tutor: int = 10, n_autonomy: int = 4,
    n_valid: int = 4, n_invalid: int = 4, n_novelty: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> TICSessionSpec:
    if rng is None:
        rng = np.random.default_rng(session_id)

    eps = []
    idx = 0
    wts = np.ones(len(TIC_V4_SUBTYPES)) / len(TIC_V4_SUBTYPES)

    def _make(st, phase, boost_lure=0.0, boost_risk=0.0):
        nonlocal idx
        sp = SUBTYPE_V4_PARAMS[st]
        e = TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase=phase,
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(
                sp["lure"][0] + boost_lure,
                min(sp["lure"][1] + boost_lure, 1.0))), 3),
            risk_level=round(float(rng.uniform(
                sp["risk"][0] + boost_risk,
                min(sp["risk"][1] + boost_risk, 1.0))), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        )
        idx += 1
        return e

    # Phase A
    for _ in range(n_tutor):
        eps.append(_make(rng.choice(TIC_V4_SUBTYPES, p=wts), "A"))

    # Phase B: autonomy
    auto_subs = ["temptation_repeat", "self_discovery_needed",
                 "false_suppression_cost", "beneficial_novelty"]
    for _ in range(n_autonomy):
        eps.append(_make(rng.choice(auto_subs), "B"))

    # Phase C: sparse valid
    for _ in range(n_valid):
        eps.append(_make("sparse_valid_advice", "C"))

    # Phase D: sparse invalid
    for _ in range(n_invalid):
        eps.append(_make("sparse_invalid_advice", "D"))

    # Phase E: beneficial novelty
    for _ in range(n_novelty):
        eps.append(_make("beneficial_novelty", "E"))

    return TICSessionSpec(session_id, theta_true, eps)


def generate_tic_v4_scenario(ep: TICEpisodeSpec):
    gm, cfg, meta, sc = generate_tic_scenario(ep)
    sc.is_verified_warn = (ep.subtype == "verified_warn")
    sc.is_self_discovery_needed = (ep.subtype == "self_discovery_needed")
    sc.is_false_suppression = (ep.subtype == "false_suppression_cost")
    sc.is_sparse_valid = (ep.subtype == "sparse_valid_advice")
    sc.is_sparse_invalid = (ep.subtype == "sparse_invalid_advice")
    sc.is_beneficial_novelty = (ep.subtype == "beneficial_novelty")
    sc.risky_branch_actually_good = (ep.subtype in ("false_suppression_cost", "beneficial_novelty"))
    sc.advice_valid = (ep.subtype == "sparse_valid_advice")
    sc.advice_invalid = (ep.subtype == "sparse_invalid_advice")
    return gm, cfg, meta, sc
