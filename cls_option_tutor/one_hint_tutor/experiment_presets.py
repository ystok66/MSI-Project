from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Iterable, List


PRELEARN_PROFILES: Dict[str, tuple[int, int, int]] = {
    "4": (1, 2, 1),
    "6": (2, 2, 2),
    "8": (2, 4, 2),
    "12": (3, 6, 3),
}


def apply_named_presets(cfg) -> None:
    profile = str(getattr(cfg, "prelearn_profile", "custom") or "custom").strip().lower()
    if profile in PRELEARN_PROFILES:
        easy, medium, hard = PRELEARN_PROFILES[profile]
        cfg.n_pre_easy = int(easy)
        cfg.n_pre_medium = int(medium)
        cfg.n_pre_hard = int(hard)


def config_from_overrides(base_cfg, overrides: Dict[str, Any]):
    cfg = base_cfg.__class__(**{f.name: getattr(base_cfg, f.name) for f in fields(base_cfg)})
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown OneHintConfig field: {key}")
        if key == "hint_families":
            if isinstance(value, str):
                value = tuple(part.strip() for part in value.split(",") if part.strip())
            else:
                value = tuple(str(part).strip() for part in value if str(part).strip())
        setattr(cfg, key, value)
    apply_named_presets(cfg)
    return cfg


def parse_seed_spec(raw: Any) -> List[int]:
    if raw is None:
        return [0]
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        # Python-style end-exclusive range: "0:10" means seeds 0..9.
        return list(range(start, end, step))
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(text)]


def family_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(part).strip() for part in raw if str(part).strip()]


def shallow_field_dict(cfg) -> Dict[str, Any]:
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


def seed_bundle(seed: int, cfg) -> Dict[str, int]:
    base = int(seed)
    if not bool(getattr(cfg, "common_randomness", True)):
        return {
            "context": base,
            "learner": base,
            "prelearn": base,
            "obs": base,
            "teach": base,
            "eval": base,
            "plan": base,
            "baseline": base,
            "oracle": base,
        }
    return {
        "context": base + int(getattr(cfg, "seed_context_offset", 11)),
        "learner": base + int(getattr(cfg, "seed_learner_offset", 23)),
        "prelearn": base + int(getattr(cfg, "seed_prelearn_offset", 101)),
        "obs": base + int(getattr(cfg, "seed_obs_offset", 211)),
        "teach": base + int(getattr(cfg, "seed_teach_offset", 307)),
        "eval": base + int(getattr(cfg, "seed_eval_offset", 401)),
        "plan": base + int(getattr(cfg, "seed_plan_offset", 503)),
        "baseline": base + int(getattr(cfg, "seed_baseline_offset", 601)),
        "oracle": base + int(getattr(cfg, "seed_oracle_offset", 701)),
    }


def resolved_hint_count_budget(cfg) -> int:
    return max(0, int(getattr(cfg, "hint_count_budget", 1)))


def resolved_no_tutor_bonus_attempts(cfg) -> int:
    explicit = getattr(cfg, "no_tutor_bonus_attempts", None)
    if explicit is None:
        return resolved_hint_count_budget(cfg)
    return max(0, int(explicit))


def resolved_no_tutor_tplush_limit(cfg) -> int:
    return min(
        int(getattr(cfg, "teach_menu_size", 20)),
        int(getattr(cfg, "max_attempts_main", 6)) + resolved_no_tutor_bonus_attempts(cfg),
    )
