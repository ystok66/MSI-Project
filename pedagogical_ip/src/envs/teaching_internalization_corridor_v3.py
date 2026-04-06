"""TIC-v3: Teaching-Internalization Corridor with Dual Transfer Protocol.

Phase A: Tutor present (10 ep, 10 subtypes including new mechanism types)
Phase B: Autonomy transfer (4 ep, no tutor)
Phase C: Sparse-valid-advice transfer (4 ep, rare correct hints)
Phase D: Sparse-invalid-advice transfer (4 ep, occasional wrong hints)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .teaching_internalization_corridor import (
    TICEpisodeSpec, TICSessionSpec, SUBTYPE_PARAMS as BASE_PARAMS,
    generate_tic_scenario,
)

TIC_V3_SUBTYPES = [
    "temptation_repeat",
    "self_discovery_teach",
    "warn_rescue",
    "boundary_obs",
    "verified_warn",
    "self_discovery_needed",
    "false_suppression_cost",
    "sparse_valid_advice",
    "sparse_invalid_advice",
    "beneficial_novelty",
]

SUBTYPE_V3_PARAMS = {
    **BASE_PARAMS,
    "verified_warn":          {"d_c": (4, 6), "d_r": (1, 2), "lure": (0.5, 0.8), "risk": (0.3, 0.5)},
    "self_discovery_needed":  {"d_c": (5, 8), "d_r": (1, 3), "lure": (0.3, 0.6), "risk": (0.2, 0.35)},
    "false_suppression_cost": {"d_c": (3, 5), "d_r": (2, 4), "lure": (0.5, 0.8), "risk": (0.05, 0.15)},
    "sparse_valid_advice":    {"d_c": (4, 7), "d_r": (1, 3), "lure": (0.3, 0.5), "risk": (0.2, 0.4)},
    "sparse_invalid_advice":  {"d_c": (3, 5), "d_r": (2, 4), "lure": (0.4, 0.7), "risk": (0.1, 0.25)},
    "beneficial_novelty":     {"d_c": (4, 6), "d_r": (2, 4), "lure": (0.3, 0.6), "risk": (0.05, 0.12)},
}


def generate_tic_v3_session(
    session_id: int,
    theta_true: str = "safe",
    n_tutor: int = 10,
    n_autonomy: int = 4,
    n_sparse_valid: int = 4,
    n_sparse_invalid: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> TICSessionSpec:
    if rng is None:
        rng = np.random.default_rng(session_id)

    eps = []
    idx = 0

    # Phase A: tutor present with all subtypes
    phase_a_subs = TIC_V3_SUBTYPES
    wts = np.ones(len(phase_a_subs)) / len(phase_a_subs)
    for _ in range(n_tutor):
        st = rng.choice(phase_a_subs, p=wts)
        sp = SUBTYPE_V3_PARAMS[st]
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

    # Phase B: autonomy (no tutor, same structure)
    auto_subs = ["temptation_repeat", "self_discovery_needed", "false_suppression_cost",
                 "beneficial_novelty"]
    for _ in range(n_autonomy):
        st = rng.choice(auto_subs)
        sp = SUBTYPE_V3_PARAMS[st]
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

    # Phase C: sparse valid advice
    for _ in range(n_sparse_valid):
        st = "sparse_valid_advice"
        sp = SUBTYPE_V3_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="C",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    # Phase D: sparse invalid advice
    for _ in range(n_sparse_invalid):
        st = "sparse_invalid_advice"
        sp = SUBTYPE_V3_PARAMS[st]
        eps.append(TICEpisodeSpec(
            episode_idx=idx, theta_true=theta_true, subtype=st, phase="D",
            mirror=int(rng.integers(0, 2)),
            d_commit=int(rng.integers(sp["d_c"][0], sp["d_c"][1] + 1)),
            d_reveal=int(rng.integers(sp["d_r"][0], sp["d_r"][1] + 1)),
            lure_strength=round(float(rng.uniform(sp["lure"][0], sp["lure"][1])), 3),
            risk_level=round(float(rng.uniform(sp["risk"][0], sp["risk"][1])), 3),
            cue_layout_seed=int(rng.integers(0, 100000)),
        ))
        idx += 1

    return TICSessionSpec(session_id, theta_true, eps)


def generate_tic_v3_scenario(ep: TICEpisodeSpec):
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
