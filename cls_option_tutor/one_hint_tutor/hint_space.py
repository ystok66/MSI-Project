from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..grammar.task_adapter import TaskAdapter
from ..interfaces import Example
from .interfaces import HintCandidate, TaskContext, TeachCase


def _operator_names(context: TaskContext) -> set[str]:
    return {spec.name for spec in context.operator_specs}


def _classify_words_difficulty(words: Sequence[str], operator_names: Iterable[str]) -> str:
    op_names = set(operator_names)
    op_count = sum(1 for tok in words if tok in op_names)
    if op_count == 0 and len(words) <= 1:
        return "easy"
    if op_count >= 2 or len(words) >= 4:
        return "hard"
    return "medium"


def _hint_difficulty(context: TaskContext, words: Sequence[str], fallback: str) -> str:
    operators = _operator_names(context)
    if not operators:
        return fallback
    return _classify_words_difficulty(words, operators)


def _hint_metadata(kind: str, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    payload = {"family": kind}
    if extra:
        payload.update(extra)
    return payload


def _make_hint(
    context: TaskContext,
    words: Sequence[str],
    output: Sequence[str],
    kind: str,
    fallback_difficulty: str,
    source_index: Optional[int] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> HintCandidate:
    return HintCandidate(
        example=Example(words=list(words), output=list(output)),
        difficulty=_hint_difficulty(context, words, fallback_difficulty),
        kind=kind,
        source_index=source_index,
        metadata=_hint_metadata(kind, metadata),
    )


def _dedupe_candidates(candidates: Iterable[HintCandidate]) -> List[HintCandidate]:
    deduped: List[HintCandidate] = []
    seen: set[Tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate.example.words)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _target_overlap_metadata(
    words: Sequence[str],
    teach_words: Sequence[str],
    operator_names: Iterable[str],
    source: str,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    op_names = set(operator_names)
    teach_ops = {tok for tok in teach_words if tok in op_names}
    teach_atoms = {tok for tok in teach_words if tok not in op_names}
    words_ops = {tok for tok in words if tok in op_names}
    words_atoms = {tok for tok in words if tok not in op_names}
    operator_overlap = len(teach_ops & words_ops)
    atom_overlap = len(teach_atoms & words_atoms)
    quality_score = (
        4.0 * float(operator_overlap)
        + 1.5 * float(atom_overlap)
        - 0.25 * abs(len(words) - len(teach_words))
    )
    payload: Dict[str, object] = {
        "source": source,
        "operator_overlap": int(operator_overlap),
        "atom_overlap": int(atom_overlap),
        "quality_score": float(quality_score),
    }
    if extra:
        payload.update(extra)
    return payload


def build_free_hint_candidates(
    context: TaskContext,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    candidates: List[HintCandidate] = []
    limits = {
        "easy": int(cfg.free_hint_pool_easy),
        "medium": int(cfg.free_hint_pool_medium),
        "hard": int(cfg.free_hint_pool_hard),
    }
    for diff, limit in limits.items():
        if limit <= 0:
            continue
        pool = list(context.example_pools.get(diff, []))
        rng.shuffle(pool)
        for ex in pool[:limit]:
            candidates.append(
                _make_hint(
                    context=context,
                    words=ex.words,
                    output=ex.output,
                    kind="free",
                    fallback_difficulty=diff,
                )
            )
    return candidates


def build_menu_hint_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    allow_correct_hint: bool = False,
    wrong_limit: Optional[int] = None,
    correct_limit: Optional[int] = None,
) -> List[HintCandidate]:
    candidates: List[HintCandidate] = []
    wrong_budget = None if wrong_limit is None else max(0, int(wrong_limit))
    correct_budget = None if correct_limit is None else max(0, int(correct_limit))
    for opt in teach_case.menu:
        if opt.is_correct:
            if not allow_correct_hint or correct_budget == 0:
                continue
            kind = "menu_correct_ceiling"
            if correct_budget is not None:
                correct_budget -= 1
        else:
            if wrong_budget == 0:
                continue
            kind = "menu_wrong"
            if wrong_budget is not None:
                wrong_budget -= 1
        candidates.append(
            _make_hint(
                context=context,
                words=opt.text,
                output=(opt.rendered_output or []),
                kind=kind,
                fallback_difficulty=teach_case.difficulty,
                source_index=opt.index,
                metadata={"menu_based": True},
            )
        )
    return candidates


def _instantiate_operator_words(pattern, left_words, right_words=None) -> List[str]:
    words: List[str] = []
    for token in pattern:
        if token in ("x1", "u1"):
            words.extend(list(left_words))
        elif token in ("x2", "u2"):
            if right_words is None:
                return []
            words.extend(list(right_words))
        else:
            words.append(token)
    return words


def _render_hint_words(context: TaskContext, words: Sequence[str]) -> Optional[List[str]]:
    rendered = TaskAdapter.render(list(words), context.grammar)
    if not rendered:
        return None
    return list(rendered)


def _atom_examples(context: TaskContext) -> List[Example]:
    operators = _operator_names(context)
    pool = [
        ex
        for ex in (
            list(context.example_pools.get("easy", []))
            + list(context.example_pools.get("medium", []))
        )
        if all(tok not in operators for tok in ex.words)
    ]
    if not pool:
        pool = list(context.example_pools.get("easy", [])) or list(context.example_pools.get("medium", []))
    return list(pool)


def build_operator_probe_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    spec_by_name = {spec.name: spec for spec in context.operator_specs}
    teach_specs = []
    seen_specs = set()
    for tok in teach_case.example.words:
        spec = spec_by_name.get(tok)
        if spec is None or spec.name in seen_specs:
            continue
        teach_specs.append(spec)
        seen_specs.add(spec.name)
    if not teach_specs:
        ranked = sorted(context.operator_specs, key=lambda spec: (int(spec.score), spec.name), reverse=True)
        teach_specs = ranked[:1]

    atom_pool = _atom_examples(context)
    rng.shuffle(atom_pool)
    atom_pool = atom_pool[: max(4, int(getattr(cfg, "operator_probe_limit", 6)))]

    candidates: List[HintCandidate] = []
    seen = set()
    for spec in teach_specs:
        if spec.arity == 1:
            for ex in atom_pool:
                words = _instantiate_operator_words(spec.pattern, ex.words)
                if not words or len(words) > 10:
                    continue
                key = tuple(words)
                if key in seen:
                    continue
                rendered = _render_hint_words(context, words)
                if rendered is None:
                    continue
                seen.add(key)
                candidates.append(
                    _make_hint(
                        context=context,
                        words=words,
                        output=rendered,
                        kind="operator_probe",
                        fallback_difficulty=teach_case.difficulty,
                        metadata={"operator": spec.name, "arity": int(spec.arity)},
                    )
                )
        elif spec.arity == 2:
            for left in atom_pool:
                for right in atom_pool:
                    if tuple(left.words) == tuple(right.words):
                        continue
                    words = _instantiate_operator_words(spec.pattern, left.words, right.words)
                    if not words or len(words) > 10:
                        continue
                    key = tuple(words)
                    if key in seen:
                        continue
                    rendered = _render_hint_words(context, words)
                    if rendered is None:
                        continue
                    seen.add(key)
                    candidates.append(
                        _make_hint(
                            context=context,
                            words=words,
                            output=rendered,
                            kind="operator_probe",
                            fallback_difficulty=teach_case.difficulty,
                            metadata={"operator": spec.name, "arity": int(spec.arity)},
                        )
                    )
    rng.shuffle(candidates)
    return candidates[: max(0, int(getattr(cfg, "operator_probe_limit", 6)))]


def _contiguous_neighborhood_candidates(
    context: TaskContext,
    teach_case: TeachCase,
) -> List[HintCandidate]:
    teach_words = list(teach_case.example.words)
    operator_names = _operator_names(context)
    rendered_candidates: List[HintCandidate] = []
    for span_len in range(len(teach_words) - 1, 0, -1):
        for start in range(0, len(teach_words) - span_len + 1):
            words = teach_words[start : start + span_len]
            if tuple(words) == tuple(teach_words):
                continue
            rendered = _render_hint_words(context, words)
            if rendered is None:
                continue
            rendered_candidates.append(
                _make_hint(
                    context=context,
                    words=words,
                    output=rendered,
                    kind="target_neighborhood",
                    fallback_difficulty=teach_case.difficulty,
                    metadata=_target_overlap_metadata(
                        words=words,
                        teach_words=teach_words,
                        operator_names=operator_names,
                        source="contiguous_subexpr",
                        extra={"span_len": int(span_len), "start": int(start)},
                    ),
                )
            )
    return rendered_candidates


def _replacement_neighborhood_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    operators = _operator_names(context)
    atom_pool = _atom_examples(context)
    rng.shuffle(atom_pool)
    replacement_budget = max(1, int(getattr(cfg, "target_neighborhood_atom_replacements", 4)))
    atom_words = []
    seen_atoms = set()
    for ex in atom_pool:
        if len(ex.words) != 1:
            continue
        token = ex.words[0]
        if token in seen_atoms:
            continue
        seen_atoms.add(token)
        atom_words.append(token)
        if len(atom_words) >= replacement_budget:
            break

    teach_words = list(teach_case.example.words)
    content_positions = [idx for idx, tok in enumerate(teach_words) if tok not in operators]
    candidates: List[HintCandidate] = []
    for pos in content_positions:
        original = teach_words[pos]
        for token in atom_words:
            if token == original:
                continue
            words = list(teach_words)
            words[pos] = token
            rendered = _render_hint_words(context, words)
            if rendered is None:
                continue
            candidates.append(
                _make_hint(
                    context=context,
                    words=words,
                    output=rendered,
                    kind="target_neighborhood",
                    fallback_difficulty=teach_case.difficulty,
                    metadata=_target_overlap_metadata(
                        words=words,
                        teach_words=teach_words,
                        operator_names=operators,
                        source="single_atom_replacement",
                        extra={"replace_pos": int(pos), "replace_token": token},
                    ),
                )
            )
    return candidates


def _operator_swap_neighborhood_candidates(
    context: TaskContext,
    teach_case: TeachCase,
) -> List[HintCandidate]:
    operator_names = sorted(_operator_names(context))
    teach_words = list(teach_case.example.words)
    operator_positions = [idx for idx, tok in enumerate(teach_words) if tok in operator_names]
    candidates: List[HintCandidate] = []
    for pos in operator_positions:
        original = teach_words[pos]
        for token in operator_names:
            if token == original:
                continue
            words = list(teach_words)
            words[pos] = token
            rendered = _render_hint_words(context, words)
            if rendered is None:
                continue
            candidates.append(
                _make_hint(
                    context=context,
                    words=words,
                    output=rendered,
                    kind="target_neighborhood",
                    fallback_difficulty=teach_case.difficulty,
                    metadata=_target_overlap_metadata(
                        words=words,
                        teach_words=teach_words,
                        operator_names=operator_names,
                        source="operator_swap",
                        extra={"replace_pos": int(pos), "replace_token": token},
                    ),
                )
            )
    return candidates


def _pool_overlap_neighborhood_candidates(
    context: TaskContext,
    teach_case: TeachCase,
) -> List[HintCandidate]:
    operator_names = _operator_names(context)
    teach_words = list(teach_case.example.words)
    teach_word_key = tuple(teach_words)
    teach_ops = {tok for tok in teach_words if tok in operator_names}
    teach_atoms = {tok for tok in teach_words if tok not in operator_names}

    scored: List[Tuple[float, HintCandidate]] = []
    for diff, pool in context.example_pools.items():
        for ex in pool:
            ex_key = tuple(ex.words)
            if ex_key == teach_word_key:
                continue
            ex_ops = {tok for tok in ex.words if tok in operator_names}
            ex_atoms = {tok for tok in ex.words if tok not in operator_names}
            op_overlap = len(teach_ops & ex_ops)
            atom_overlap = len(teach_atoms & ex_atoms)
            if op_overlap <= 0 and atom_overlap <= 0:
                continue
            score = (
                4.0 * float(op_overlap)
                + 1.5 * float(atom_overlap)
                - 0.25 * abs(len(ex.words) - len(teach_words))
            )
            scored.append(
                (
                    score,
                    _make_hint(
                        context=context,
                        words=ex.words,
                        output=ex.output,
                        kind="target_neighborhood",
                        fallback_difficulty=diff,
                        metadata=_target_overlap_metadata(
                            words=ex.words,
                            teach_words=teach_words,
                            operator_names=operator_names,
                            source="pool_overlap",
                            extra={
                                "operator_overlap": int(op_overlap),
                                "atom_overlap": int(atom_overlap),
                                "quality_score": float(score),
                            },
                        ),
                    ),
                )
            )
    scored.sort(key=lambda item: (item[0], -len(item[1].example.words)), reverse=True)
    return [candidate for _, candidate in scored]


def _target_neighborhood_priority(candidate: HintCandidate) -> Tuple[float, float, float, float]:
    metadata = dict(candidate.metadata)
    quality = float(metadata.get("quality_score", 0.0))
    operator_overlap = float(metadata.get("operator_overlap", 0.0))
    atom_overlap = float(metadata.get("atom_overlap", 0.0))
    source = str(metadata.get("source", ""))
    source_bonus = {
        "pool_overlap": 3.0,
        "operator_swap": 2.5,
        "single_atom_replacement": 1.5,
        "contiguous_subexpr": 1.0,
    }.get(source, 0.0)
    return (quality + source_bonus, operator_overlap, atom_overlap, -float(len(candidate.example.words)))


def build_target_neighborhood_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    candidates = _contiguous_neighborhood_candidates(context, teach_case)
    candidates.extend(_replacement_neighborhood_candidates(context, teach_case, cfg, rng))
    candidates.extend(_operator_swap_neighborhood_candidates(context, teach_case))
    candidates.extend(_pool_overlap_neighborhood_candidates(context, teach_case))
    candidates = _dedupe_candidates(candidates)
    candidates.sort(key=_target_neighborhood_priority, reverse=True)
    return candidates[: max(0, int(getattr(cfg, "target_neighborhood_limit", 16)))]


def build_answer_neighbor_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    base = build_target_neighborhood_candidates(context, teach_case, cfg, rng)
    teach_words = tuple(teach_case.example.words)
    filtered: List[HintCandidate] = []
    for candidate in base:
        words = tuple(candidate.example.words)
        if words == teach_words:
            continue
        metadata = dict(candidate.metadata)
        source = str(metadata.get("source", ""))
        operator_overlap = int(metadata.get("operator_overlap", 0))
        atom_overlap = int(metadata.get("atom_overlap", 0))
        if source not in {"single_atom_replacement", "operator_swap", "contiguous_subexpr", "pool_overlap"}:
            continue
        if operator_overlap <= 0 and atom_overlap <= 0:
            continue
        if abs(len(words) - len(teach_words)) > 2:
            continue
        filtered.append(
            HintCandidate(
                example=Example(
                    words=list(candidate.example.words),
                    output=list(candidate.example.output),
                    meta=dict(candidate.example.meta),
                ),
                difficulty=candidate.difficulty,
                kind=candidate.kind,
                source_index=candidate.source_index,
                metadata=_hint_metadata(
                    candidate.kind,
                    {
                        **dict(candidate.metadata),
                        "family": "answer_neighbor_nonanswer",
                    },
                ),
            )
        )
    filtered.sort(
        key=lambda cand: (
            float(cand.metadata.get("quality_score", 0.0)),
            float(cand.metadata.get("operator_overlap", 0.0)),
            float(cand.metadata.get("atom_overlap", 0.0)),
            -float(len(cand.example.words)),
        ),
        reverse=True,
    )
    return _dedupe_candidates(filtered[: max(0, int(getattr(cfg, "answer_neighbor_limit", 8)))])


def _retag_family(candidates: Iterable[HintCandidate], family: str) -> List[HintCandidate]:
    tagged: List[HintCandidate] = []
    for candidate in candidates:
        tagged.append(
            HintCandidate(
                example=Example(
                    words=list(candidate.example.words),
                    output=list(candidate.example.output),
                    meta=dict(candidate.example.meta),
                ),
                difficulty=candidate.difficulty,
                kind=candidate.kind,
                source_index=candidate.source_index,
                metadata=_hint_metadata(candidate.kind, {**dict(candidate.metadata), "family": family}),
            )
        )
    return tagged


def _build_family_candidates(
    family: str,
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    if family == "free":
        return build_free_hint_candidates(context, cfg, rng)
    if family == "menu_wrong":
        return build_menu_hint_candidates(
            context=context,
            teach_case=teach_case,
            allow_correct_hint=False,
            wrong_limit=int(getattr(cfg, "menu_wrong_limit", 6)),
        )
    if family == "menu_all":
        return build_menu_hint_candidates(
            context=context,
            teach_case=teach_case,
            allow_correct_hint=bool(getattr(cfg, "allow_correct_hint", False)),
            wrong_limit=int(getattr(cfg, "menu_wrong_limit", 6)),
            correct_limit=int(getattr(cfg, "menu_correct_ceiling_limit", 1)),
        )
    if family == "menu_correct_ceiling":
        return build_menu_hint_candidates(
            context=context,
            teach_case=teach_case,
            allow_correct_hint=True,
            wrong_limit=0,
            correct_limit=int(getattr(cfg, "menu_correct_ceiling_limit", 1)),
        )
    if family == "operator_probe":
        return build_operator_probe_candidates(context, teach_case, cfg, rng)
    if family == "target_neighborhood":
        return build_target_neighborhood_candidates(context, teach_case, cfg, rng)
    if family == "target_neighborhood_loose":
        return _retag_family(build_target_neighborhood_candidates(context, teach_case, cfg, rng), family)
    if family == "target_neighborhood_rank_filtered":
        return _retag_family(build_target_neighborhood_candidates(context, teach_case, cfg, rng), family)
    if family == "target_neighborhood_robust_filtered":
        return _retag_family(build_target_neighborhood_candidates(context, teach_case, cfg, rng), family)
    if family == "answer_neighbor_nonanswer":
        return build_answer_neighbor_candidates(context, teach_case, cfg, rng)
    return []


def _combined_families(cfg) -> Tuple[str, ...]:
    raw = getattr(cfg, "hint_families", ("free",))
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return tuple(parts or ["free"])
    return tuple(raw or ("free",))


def build_hint_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    mode = str(cfg.hint_mode)
    if mode == "none":
        return []
    if mode == "combined":
        candidates: List[HintCandidate] = []
        for family in _combined_families(cfg):
            candidates.extend(_build_family_candidates(family, context, teach_case, cfg, rng))
        return _dedupe_candidates(candidates)
    return _dedupe_candidates(_build_family_candidates(mode, context, teach_case, cfg, rng))


def sample_random_hint(
    context: TaskContext,
    cfg,
    rng: np.random.Generator,
) -> Optional[HintCandidate]:
    pool = list(context.example_pools.get(cfg.random_hint_difficulty, []))
    if not pool:
        return None
    ex = pool[int(rng.integers(0, len(pool)))]
    return _make_hint(
        context=context,
        words=ex.words,
        output=ex.output,
        kind="free",
        fallback_difficulty=cfg.random_hint_difficulty,
    )


def sample_random_same_pool_hint(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> Optional[HintCandidate]:
    pool = build_random_same_pool_candidates(context, teach_case, cfg, rng)
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


def build_random_same_pool_candidates(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[HintCandidate]:
    return [
        candidate
        for candidate in build_hint_candidates(context, teach_case, cfg, rng)
        if candidate.kind not in {"menu_correct_ceiling", "direct_answer"}
    ]
