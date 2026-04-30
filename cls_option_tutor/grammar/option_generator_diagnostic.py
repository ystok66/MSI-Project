"""
option_generator_diagnostic.py — Phase 6E: Quota-based diagnostic menu generator.

Wraps option_generator_v2 with diagnostic confound labeling and quota sampling.
Does NOT replace the base generator — uses it as fallback when quota cannot be met.

Design:
  1. Generate candidate pool (reuse ProgramPool from v2)
  2. Label each candidate with confound_labels
  3. Sample from labeled buckets to meet diagnostic quota
  4. Fall back to generate_menu_v2 if quota infeasible
  5. Return (menu, quota_info) with sidecar labels
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from ..interfaces import Option
from ..env.danger_model import DangerModel, generate_danger_vector
from .option_generator_v2 import (
    ProgramPool, compute_cell_overlap, compute_error_mask,
    generate_menu_v2, _diverse_select,
)
from .confound_labels import (
    ConfoundType, DiagnosticRiskLabel,
    label_confound, label_diagnostic_risk,
)


# ── Quota definitions ─────────────────────────────────────────────────────────

# K -> list of (label, count) excluding correct
QUOTA_K6 = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 1),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 1),
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 1),
    ("SAFE_CONTROL", 1),   # SAFE_FAR or SAFE_RANDOM_WRONG
    ("FILL", 1),           # any remaining
]

QUOTA_K10 = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 1),
    ("SAFE_CONTROL", 2),   # SAFE_FAR or SAFE_RANDOM_WRONG
    ("FILL", 2),           # any remaining
]

# Ablation: no bounded diagnostic slots → replaced with safe control
QUOTA_K6_NO_BOUNDED = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 1),
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 1),
    ("SAFE_CONTROL", 2),
    ("FILL", 1),
]

QUOTA_K10_NO_BOUNDED = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 1),
    ("SAFE_CONTROL", 4),
    ("FILL", 2),
]

# Ablation: no high-risk lure slots → replaced with safe control
QUOTA_K6_NO_LURE = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 1),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 1),
    ("SAFE_CONTROL", 2),
    ("FILL", 1),
]

QUOTA_K10_NO_LURE = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 2),
    ("SAFE_CONTROL", 3),
    ("FILL", 2),
]

# Generator-shaped family-heavy menu priors used for Phase 6I.13+ benchmark
# slices. These shape the menu opportunity prior; they do not guarantee the
# final decision-time family after learner/tutor interaction.
QUOTA_K6_ALLOW_HEAVY = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 2),
    ("SAFE_CONTROL", 1),
]

QUOTA_K10_ALLOW_HEAVY = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 4),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 3),
    ("SAFE_CONTROL", 2),
]

QUOTA_K6_MIXED_PROD_HARM_HEAVY = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 1),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 1),
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 1),
    ("SAFE_CONTROL", 1),
    ("RISKY_FAR_FILL", 1),
]

QUOTA_K10_MIXED_PROD_HARM_HEAVY = [
    (DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 2),
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 2),
    ("SAFE_CONTROL", 1),
    ("RISKY_FAR_FILL", 2),
]

QUOTA_K6_PROTECT_CRITICAL_HEAVY = [
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 2),
    ("RISKY_FAR_FILL", 2),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 1),
]

QUOTA_K10_PROTECT_CRITICAL_HEAVY = [
    (DiagnosticRiskLabel.HIGH_RISK_LURE, 3),
    ("RISKY_FAR_FILL", 3),
    (DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG, 2),
    ("SAFE_CONTROL", 1),
]

QUOTA_K6_BORING_MASTERY_HEAVY = [
    ("SAFE_CONTROL", 3),
    ("SAFE_FAR_FILL", 2),
]

QUOTA_K10_BORING_MASTERY_HEAVY = [
    ("SAFE_CONTROL", 5),
    ("SAFE_FAR_FILL", 4),
]


def _get_quota(K: int, quota_mode: str = "diagnostic_quota"):
    """Return quota spec for given K and mode."""
    if quota_mode == "diagnostic_quota_no_bounded":
        return QUOTA_K6_NO_BOUNDED if K <= 6 else QUOTA_K10_NO_BOUNDED
    elif quota_mode == "diagnostic_quota_no_high_lure":
        return QUOTA_K6_NO_LURE if K <= 6 else QUOTA_K10_NO_LURE
    elif quota_mode == "diagnostic_quota_allow_heavy":
        return QUOTA_K6_ALLOW_HEAVY if K <= 6 else QUOTA_K10_ALLOW_HEAVY
    elif quota_mode == "diagnostic_quota_mixed_prod_harm_heavy":
        return (
            QUOTA_K6_MIXED_PROD_HARM_HEAVY
            if K <= 6 else QUOTA_K10_MIXED_PROD_HARM_HEAVY
        )
    elif quota_mode == "diagnostic_quota_protect_critical_heavy":
        return (
            QUOTA_K6_PROTECT_CRITICAL_HEAVY
            if K <= 6 else QUOTA_K10_PROTECT_CRITICAL_HEAVY
        )
    elif quota_mode == "diagnostic_quota_boring_mastery_heavy":
        return (
            QUOTA_K6_BORING_MASTERY_HEAVY
            if K <= 6 else QUOTA_K10_BORING_MASTERY_HEAVY
        )
    # Default: full diagnostic quota
    if K <= 6:
        return QUOTA_K6
    return QUOTA_K10


def _sample_risk_class_for_mode(
    quota_mode: str,
    rng: np.random.Generator,
    *,
    p_safe_default: float,
) -> int:
    """Sample distractor risk class with a family-specific prior."""
    if quota_mode == "diagnostic_quota_allow_heavy":
        if rng.random() < max(p_safe_default, 0.75):
            return 0
        return int(rng.choice([1, 2]))

    if quota_mode == "diagnostic_quota_mixed_prod_harm_heavy":
        if rng.random() < 0.40:
            return 0
        return int(rng.choice([1, 2, 3, 4]))

    if quota_mode == "diagnostic_quota_protect_critical_heavy":
        if rng.random() < 0.15:
            return 0
        return int(rng.choice([3, 4]))

    if quota_mode == "diagnostic_quota_boring_mastery_heavy":
        if rng.random() < 0.90:
            return 0
        return 1

    if rng.random() < p_safe_default:
        return 0
    return int(rng.choice([1, 2, 3, 4]))


def _selection_pool_for_special_label(
    *,
    label: str,
    buckets: Dict[str, List[dict]],
    labeled: List[dict],
    used_progs: set,
    rng: np.random.Generator,
) -> List[dict]:
    """Build a candidate pool for synthetic quota labels."""
    if label == "SAFE_CONTROL":
        pool = [
            c for c in buckets[DiagnosticRiskLabel.SAFE_FAR]
            if c["prog"] not in used_progs
        ]
        pool += [
            c for c in buckets[DiagnosticRiskLabel.SAFE_RANDOM_WRONG]
            if c["prog"] not in used_progs and c not in pool
        ]
        return pool

    if label == "SAFE_FAR_FILL":
        pool = [
            c for c in buckets[DiagnosticRiskLabel.SAFE_FAR]
            if c["prog"] not in used_progs
        ]
        pool += [
            c for c in buckets[DiagnosticRiskLabel.SAFE_RANDOM_WRONG]
            if c["prog"] not in used_progs and c not in pool
        ]
        return pool

    if label == "RISKY_FAR_FILL":
        pool = [
            c for c in buckets[DiagnosticRiskLabel.RISKY_FAR]
            if c["prog"] not in used_progs
        ]
        rng.shuffle(pool)
        return pool

    if label == "FILL":
        pool = [c for c in labeled if c["prog"] not in used_progs]
        rng.shuffle(pool)
        return pool

    return []


# ── Label a candidate ─────────────────────────────────────────────────────────

def _label_candidate(
    out: Tuple[str, ...],
    target: Tuple[str, ...],
    risk_class: int,
) -> Tuple[ConfoundType, DiagnosticRiskLabel]:
    """Label a single candidate (not the correct option)."""
    ct = label_confound(list(out), list(target), is_correct=False)
    dr = label_diagnostic_risk(ct, risk_class)
    return ct, dr


# ── Main diagnostic generator ────────────────────────────────────────────────

def generate_menu_diagnostic(
    target_output: List[str],
    true_program: List[str],
    pool: ProgramPool,
    danger_model: DangerModel,
    K: int = 6,
    m: int = 16,
    rng: Optional[np.random.Generator] = None,
    *,
    quota_mode: str = "diagnostic_quota",
) -> Tuple[List[Option], Dict]:
    """Generate a menu with diagnostic quota sampling.

    Returns:
        (menu, quota_info):
            menu: List[Option] with K options (1 correct + K-1 distractors)
            quota_info: dict with per-option labels and quota diagnostics
    """
    rng = rng or np.random.default_rng()
    target_tuple = tuple(target_output)
    L = len(target_output)
    n_need = K - 1

    quota_info: Dict = {
        "quota_mode": quota_mode,
        "quota_met": False,
        "quota_fail_reason": None,
        "fallback_used": False,
        "confound_types": {},      # {opt_index: ConfoundType.value}
        "diag_labels": {},         # {opt_index: DiagnosticRiskLabel.value}
        "bucket_counts": {},       # {label: available_count}
    }

    # ── 1. Get candidate pool ──
    candidates = pool.get_programs_by_output_length(L)
    true_key = tuple(true_program)
    available = [(prog, out) for prog, out in candidates
                 if out != target_tuple and prog != true_key]

    if len(available) < n_need:
        # Pool too small — fallback immediately
        quota_info["quota_fail_reason"] = f"pool_too_small ({len(available)} < {n_need})"
        quota_info["fallback_used"] = True
        menu = generate_menu_v2(
            target_output, true_program, pool, danger_model, K, m, rng,
        )
        _label_menu_post_hoc(menu, target_tuple, quota_info)
        return menu, quota_info

    # ── 2. Pre-assign risk classes to candidates ──
    # Use env-consistent risk distribution instead of hardcoded 0.6/K-4 heuristics.
    # danger_model.n_risky_default is the canonical env n_risky setting.
    n_risky = getattr(danger_model, 'n_risky_default', None)
    if n_risky is None:
        # Fallback: derive from K typical env ratio
        n_risky = max(1, K // 3)
    n_safe_distractor = max(0, (K - 1) - n_risky)  # K-1 distractors total
    p_safe = n_safe_distractor / max(K - 1, 1)      # env-consistent safe fraction

    cand_risk_classes = [
        _sample_risk_class_for_mode(quota_mode, rng, p_safe_default=p_safe)
        for _ in available
    ]

    # ── 3. Label all candidates ──
    labeled = []
    for i, (prog, out) in enumerate(available):
        rc = cand_risk_classes[i]
        ct, dr = _label_candidate(out, target_tuple, rc)
        labeled.append({
            "prog": prog,
            "out": out,
            "risk_class": rc,
            "confound_type": ct,
            "diag_label": dr,
            "overlap": compute_cell_overlap(out, target_tuple)[0] / max(L, 1),
            "error_mask": compute_error_mask(out, target_tuple),
        })

    # ── 4. Bucket candidates ──
    buckets: Dict[str, List[dict]] = {
        DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG: [],
        DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG: [],
        DiagnosticRiskLabel.HIGH_RISK_LURE: [],
        DiagnosticRiskLabel.SAFE_FAR: [],
        DiagnosticRiskLabel.SAFE_RANDOM_WRONG: [],
        DiagnosticRiskLabel.RISKY_FAR: [],
    }

    for item in labeled:
        dl = item["diag_label"]
        if dl in buckets:
            buckets[dl].append(item)
        # Phase 6H.7: SAFE_RANDOM_WRONG must be a true far negative control:
        # only SAFE_FAR items are real far negatives — do NOT include
        # diagnostic-adjacent safe items (SAFE_DIAGNOSTIC_WRONG, SAFE_RANDOM_WRONG
        # from overlap) in the SAFE_RANDOM bucket.
        # SAFE_FAR is added to SAFE_RANDOM only for fill/control slots.
        if dl == DiagnosticRiskLabel.SAFE_FAR:
            buckets[DiagnosticRiskLabel.SAFE_RANDOM_WRONG].append(item)

    # Shuffle each bucket
    for bk in buckets.values():
        rng.shuffle(bk)

    quota_info["bucket_counts"] = {k: len(v) for k, v in buckets.items()}

    # ── 5. Quota-based selection ──
    quota = _get_quota(K, quota_mode)
    selected: List[dict] = []
    used_progs = set()

    for label, count in quota:
        if isinstance(label, str) and label in {
            "SAFE_CONTROL",
            "SAFE_FAR_FILL",
            "RISKY_FAR_FILL",
            "FILL",
        }:
            pool_special = _selection_pool_for_special_label(
                label=label,
                buckets=buckets,
                labeled=labeled,
                used_progs=used_progs,
                rng=rng,
            )
            picks = pool_special[:count]
            if label == "SAFE_CONTROL":
                for p in picks:
                    if p["diag_label"] != DiagnosticRiskLabel.SAFE_FAR:
                        p["diag_label"] = DiagnosticRiskLabel.SAFE_RANDOM_WRONG
        else:
            pool_label = [c for c in buckets.get(label, [])
                          if c["prog"] not in used_progs]
            picks = pool_label[:count]

        for p in picks:
            selected.append(p)
            used_progs.add(p["prog"])

    # ── 6. Check quota satisfaction ──
    has_safe_diag = any(s["diag_label"] == DiagnosticRiskLabel.SAFE_DIAGNOSTIC_WRONG
                        for s in selected)
    has_bounded = any(s["diag_label"] == DiagnosticRiskLabel.BOUNDED_DIAGNOSTIC_WRONG
                      for s in selected)
    has_lure = any(s["diag_label"] == DiagnosticRiskLabel.HIGH_RISK_LURE
                   for s in selected)

    if len(selected) < n_need:
        # Not enough candidates — fill remainder from labeled pool
        remaining = [c for c in labeled if c["prog"] not in used_progs]
        rng.shuffle(remaining)
        for item in remaining:
            if len(selected) >= n_need:
                break
            selected.append(item)
            used_progs.add(item["prog"])

    if len(selected) < n_need:
        quota_info["quota_fail_reason"] = f"insufficient_candidates ({len(selected)} < {n_need})"
        quota_info["fallback_used"] = True
        menu = generate_menu_v2(
            target_output, true_program, pool, danger_model, K, m, rng,
        )
        _label_menu_post_hoc(menu, target_tuple, quota_info)
        return menu, quota_info

    quota_info["quota_met"] = has_safe_diag and has_bounded and has_lure

    # ── 7. Build menu ──
    correct_v = generate_danger_vector(m, rng)
    correct_option = Option(
        index=0,
        text=list(true_program),
        danger_vec=correct_v,
        is_correct=True,
        rendered_output=list(target_output),
    )

    distractors = []
    for i, sel in enumerate(selected[:n_need]):
        v = danger_model.sample_danger_vec(sel["risk_class"], rng)
        distractors.append(Option(
            index=i + 1,
            text=list(sel["prog"]),
            danger_vec=v,
            is_correct=False,
            rendered_output=list(sel["out"]),
            risk_class=sel["risk_class"],
        ))

    # Shuffle
    all_options = [correct_option] + distractors
    indices = list(range(len(all_options)))
    rng.shuffle(indices)
    result = []
    for new_idx, old_idx in enumerate(indices):
        opt = all_options[old_idx]
        result.append(Option(
            index=new_idx,
            text=opt.text,
            danger_vec=opt.danger_vec,
            is_correct=opt.is_correct,
            rendered_output=opt.rendered_output,
            risk_class=opt.risk_class,
        ))

    # Assign risk classes to correct option
    n_safe = max(0, K - 4)  # Use env default
    risk_classes = danger_model.assign_risk_classes(K, n_safe, rng)
    for i, opt in enumerate(result):
        if opt.is_correct:
            opt.risk_class = risk_classes[i]
            opt.danger_vec = danger_model.sample_danger_vec(opt.risk_class, rng)

    # ── 8. Build sidecar labels ──
    # Re-label the final menu using the output-based labeler
    _label_menu_post_hoc(result, target_tuple, quota_info)

    return result, quota_info


def _label_menu_post_hoc(
    menu: List[Option],
    target_tuple: Tuple[str, ...],
    quota_info: Dict,
) -> None:
    """Label an existing menu in place, storing labels in quota_info."""
    for opt in menu:
        out = tuple(opt.rendered_output) if opt.rendered_output else ()
        ct = label_confound(list(out), list(target_tuple), is_correct=opt.is_correct)
        dr = label_diagnostic_risk(ct, opt.risk_class)
        quota_info["confound_types"][opt.index] = ct.value
        quota_info["diag_labels"][opt.index] = dr.value
