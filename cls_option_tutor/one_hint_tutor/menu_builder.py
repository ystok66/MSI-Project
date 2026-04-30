from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import copy
import math

import numpy as np

from ..interfaces import Example, Option
from ..grammar.task_adapter import Grammar, TaskAdapter
from ..grammar.query_synthesizer import synthesize_queries
from ..grammar.option_generator_v2 import ProgramPool, generate_menu_v2
from ..env.state import QueryState
from .interfaces import EvalItem, ObservationCase, OperatorSpec, TaskContext, TeachCase


def extract_operator_specs(grammar: Grammar) -> List[OperatorSpec]:
    specs: OrderedDict[str, OperatorSpec] = OrderedDict()
    for pattern, template in grammar.rules:
        vars_ = [tok for tok in pattern if tok.startswith(("u", "x"))]
        lits = [tok for tok in pattern if not tok.startswith(("u", "x"))]
        if len(lits) != 1:
            continue
        literal = lits[0]
        literal_pos = pattern.index(literal)
        if len(vars_) == 1:
            if literal_pos == 0:
                placement = "prefix"
            elif literal_pos == len(pattern) - 1:
                placement = "postfix"
            else:
                placement = "wrap"
        elif len(vars_) == 2:
            placement = "infix"
        else:
            continue
        score = len(vars_) * 2 + len(template)
        specs[literal] = OperatorSpec(
            name=literal,
            arity=len(vars_),
            placement=placement,
            pattern=tuple(pattern),
            template=tuple(template),
            score=score,
        )
    return list(specs.values())


def classify_words_difficulty(words: Sequence[str], operator_names: Iterable[str]) -> str:
    op_names = set(operator_names)
    op_count = sum(1 for tok in words if tok in op_names)
    if op_count == 0 and len(words) <= 1:
        return "easy"
    if op_count >= 2 or len(words) >= 4:
        return "hard"
    return "medium"


def _dedupe_examples(examples: Iterable[Example]) -> List[Example]:
    seen = set()
    unique: List[Example] = []
    for ex in examples:
        key = tuple(ex.words)
        if key in seen:
            continue
        seen.add(key)
        unique.append(Example(words=list(ex.words), output=list(ex.output), meta=dict(ex.meta)))
    return unique


def build_task_context(task_id: str, cfg, seed: int) -> TaskContext:
    support, queries_raw, grammar = cfg._env.adapter.load_task(task_id)
    synthetic = synthesize_queries(
        grammar,
        n=max(24, cfg.n_obs + 8),
        max_depth=4,
        max_len=8,
        rng=np.random.default_rng(seed + 31),
        existing=queries_raw,
    )
    operator_specs = extract_operator_specs(grammar)
    op_names = [spec.name for spec in operator_specs]
    pools: Dict[str, List[Example]] = {"easy": [], "medium": [], "hard": []}
    combined = _dedupe_examples(list(support) + list(queries_raw) + list(synthetic))
    for ex in combined:
        diff = classify_words_difficulty(ex.words, op_names)
        pools[diff].append(ex)
    return TaskContext(
        task_id=task_id,
        grammar=grammar,
        support_examples=list(support),
        query_examples=list(queries_raw),
        synthetic_examples=list(synthetic),
        operator_specs=operator_specs,
        example_pools=pools,
        env=cfg._env,
        cfg=cfg,
    )


def _sample_examples(
    pool: Dict[str, List[Example]],
    desired: Dict[str, int],
    rng: np.random.Generator,
    exclude_words: Optional[Iterable[Tuple[str, ...]]] = None,
) -> List[Example]:
    selected: List[Example] = []
    used: Set[Tuple[str, ...]] = set(exclude_words or ())

    fallback_order = {
        "easy": ("easy", "medium", "hard"),
        "medium": ("medium", "hard", "easy"),
        "hard": ("hard", "medium", "easy"),
    }
    for diff, count in desired.items():
        remaining = int(count)
        for bucket in fallback_order[diff]:
            candidates = [ex for ex in pool[bucket] if tuple(ex.words) not in used]
            if not candidates:
                continue
            rng.shuffle(candidates)
            take = min(remaining, len(candidates))
            for ex in candidates[:take]:
                used.add(tuple(ex.words))
                selected.append(Example(words=list(ex.words), output=list(ex.output), meta=dict(ex.meta)))
            remaining -= take
            if remaining <= 0:
                break
    return selected


def sample_prelearn_examples(
    context: TaskContext,
    cfg,
    rng: np.random.Generator,
    exclude_words: Optional[Iterable[Tuple[str, ...]]] = None,
) -> List[Example]:
    desired = {
        "easy": cfg.n_pre_easy,
        "medium": cfg.n_pre_medium,
        "hard": cfg.n_pre_hard,
    }
    return _sample_examples(context.example_pools, desired, rng, exclude_words=exclude_words)


def build_observation_cases(
    context: TaskContext,
    cfg,
    rng: np.random.Generator,
    exclude_words: Optional[Iterable[Tuple[str, ...]]] = None,
) -> List[ObservationCase]:
    desired = {cfg.obs_difficulty: cfg.n_obs}
    examples = _sample_examples(context.example_pools, desired, rng, exclude_words=exclude_words)
    cases: List[ObservationCase] = []
    for ex in examples:
        menu = build_menu_for_example(
            context,
            ex,
            k=cfg.obs_menu_size,
            rng=rng,
            n_risk=0,
        )
        cases.append(ObservationCase(example=ex, menu=menu, difficulty=cfg.obs_difficulty))
    return cases


def build_teach_case(
    context: TaskContext,
    cfg,
    rng: np.random.Generator,
    exclude_words: Optional[Iterable[Tuple[str, ...]]] = None,
) -> TeachCase:
    desired = {cfg.teach_difficulty: 1}
    examples = _sample_examples(context.example_pools, desired, rng, exclude_words=exclude_words)
    if not examples:
        raise ValueError("Failed to sample a teach example")
    example = examples[0]
    menu = build_menu_for_example(
        context,
        example,
        k=cfg.teach_menu_size,
        rng=rng,
        n_risk=cfg.n_risk_options if cfg.use_risk else 0,
    )
    return TeachCase(example=example, menu=menu, difficulty=cfg.teach_difficulty)


def _assign_menu_risk(menu: List[Option], context: TaskContext, rng: np.random.Generator, n_risk: int) -> List[Option]:
    menu = copy.deepcopy(menu)
    dm = context.env.danger_model
    if dm is None:
        raise ValueError("Danger model is not initialized")
    wrong_positions = [i for i, opt in enumerate(menu) if not opt.is_correct]
    risky_positions = set(rng.choice(wrong_positions, size=min(n_risk, len(wrong_positions)), replace=False).tolist()) if n_risk > 0 and wrong_positions else set()
    for i, opt in enumerate(menu):
        if i in risky_positions:
            risk_class = int(rng.choice([1, 2, 3, 4]))
        else:
            risk_class = 0
        opt.risk_class = risk_class
        opt.danger_vec = dm.sample_danger_vec(risk_class, rng)
    return menu


def build_menu_for_example(
    context: TaskContext,
    example: Example,
    k: int,
    rng: np.random.Generator,
    n_risk: int = 0,
) -> List[Option]:
    pool: Optional[ProgramPool] = getattr(context.env, "_pool", None)
    if pool is None:
        raise ValueError("Program pool is not initialized")
    dm = context.env.danger_model
    if dm is None:
        raise ValueError("Danger model is not initialized")
    menu = generate_menu_v2(
        target_output=list(example.output),
        true_program=list(example.words),
        pool=pool,
        danger_model=dm,
        K=int(k),
        m=context.cfg.danger_dim,
        rng=rng,
    )
    return _assign_menu_risk(menu, context, rng, n_risk=n_risk)


def build_query_state(example: Example, menu: List[Option], cfg, query_id: int = 0) -> QueryState:
    return QueryState(
        query_id=query_id,
        target_output=list(example.output),
        true_program=list(example.words),
        hp=int(cfg.hp_0),
        max_rounds=int(cfg.max_attempts_main),
        max_refreshes=0,
        enforce_max_refreshes=True,
        menu=copy.deepcopy(menu),
    )


def _operator_tier_map(operator_specs: Sequence[OperatorSpec]) -> Dict[str, OperatorSpec]:
    if not operator_specs:
        return {}
    ranked = sorted(
        operator_specs,
        key=lambda spec: (int(spec.score), int(spec.arity), spec.name),
    )
    if len(ranked) == 1:
        return {"easy": ranked[0], "medium": ranked[0], "hard": ranked[0]}
    if len(ranked) == 2:
        return {"easy": ranked[0], "medium": ranked[0], "hard": ranked[1]}
    mid = len(ranked) // 2
    return {"easy": ranked[0], "medium": ranked[mid], "hard": ranked[-1]}


def _instantiate_operator(
    spec: OperatorSpec,
    left_words: Sequence[str],
    right_words: Optional[Sequence[str]] = None,
) -> List[str]:
    words: List[str] = []
    for token in spec.pattern:
        if token in ("x1", "u1"):
            words.extend(list(left_words))
        elif token in ("x2", "u2"):
            if right_words is None:
                return []
            words.extend(list(right_words))
        else:
            words.append(token)
    return words


def _fallback_eval_items(
    context: TaskContext,
    target_diff: str,
    need: int,
    seen_keys: set,
    rng: np.random.Generator,
) -> List[EvalItem]:
    if need <= 0:
        return []
    tier_map = _operator_tier_map(context.operator_specs)
    spec = tier_map.get(target_diff)
    if spec is None:
        return []
    fallback_items: List[EvalItem] = []
    extras = [
        ex
        for bucket in ("easy", "medium", "hard")
        for ex in context.example_pools[bucket]
        if tuple(ex.words) not in seen_keys
    ]
    rng.shuffle(extras)
    if spec.arity == 1:
        for ex in extras:
            words = _instantiate_operator(spec, ex.words)
            if not words or len(words) > 10:
                continue
            rendered = TaskAdapter.render(words, context.grammar)
            if not rendered:
                continue
            key = tuple(words)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fallback_items.append(
                EvalItem(
                    words=list(words),
                    output=list(rendered),
                    difficulty=target_diff,
                    source=f"fallback:{target_diff}:{spec.name}",
                )
            )
            if len(fallback_items) >= need:
                break
        return fallback_items

    if spec.arity == 2:
        for i, left in enumerate(extras):
            for j, right in enumerate(extras):
                if i == j:
                    continue
                words = _instantiate_operator(spec, left.words, right.words)
                if not words or len(words) > 10:
                    continue
                rendered = TaskAdapter.render(words, context.grammar)
                if not rendered:
                    continue
                key = tuple(words)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                fallback_items.append(
                    EvalItem(
                        words=list(words),
                        output=list(rendered),
                        difficulty=target_diff,
                        source=f"fallback:{target_diff}:{spec.name}",
                    )
                )
                if len(fallback_items) >= need:
                    return fallback_items
    return fallback_items


def build_eval_items_from_teach_menu(
    context: TaskContext,
    teach_case: TeachCase,
    cfg,
    rng: np.random.Generator,
) -> List[EvalItem]:
    menu_texts = [tuple(opt.text) for opt in teach_case.menu]
    unique_texts = [list(t) for t in OrderedDict((t, None) for t in menu_texts).keys()]
    tier_map = _operator_tier_map(context.operator_specs)
    seen_keys: set = set()
    teach_key = tuple(teach_case.example.words)

    results: List[EvalItem] = []
    for diff in ("easy", "medium", "hard"):
        spec = tier_map.get(diff)
        if spec is None:
            continue
        candidates: List[EvalItem] = []
        if spec.arity == 1:
            for base in unique_texts:
                words = _instantiate_operator(spec, base)
                if not words or len(words) > 10:
                    continue
                rendered = TaskAdapter.render(words, context.grammar)
                if not rendered:
                    continue
                key = tuple(words)
                if key in seen_keys or key == teach_key:
                    continue
                candidates.append(
                    EvalItem(
                        words=list(words),
                        output=list(rendered),
                        difficulty=diff,
                        source=f"teach_menu:{diff}:{spec.name}",
                    )
                )
        elif spec.arity == 2:
            capped = unique_texts[: min(len(unique_texts), 20)]
            for left, right in product(capped, capped):
                if left == right:
                    continue
                words = _instantiate_operator(spec, left, right)
                if not words or len(words) > 10:
                    continue
                rendered = TaskAdapter.render(words, context.grammar)
                if not rendered:
                    continue
                key = tuple(words)
                if key in seen_keys or key == teach_key:
                    continue
                candidates.append(
                    EvalItem(
                        words=list(words),
                        output=list(rendered),
                        difficulty=diff,
                        source=f"teach_menu:{diff}:{spec.name}",
                    )
                )
        rng.shuffle(candidates)
        take = min(int(cfg.eval_n_per_diff), len(candidates))
        chosen = candidates[:take]
        results.extend(chosen)
        seen_keys.update(tuple(item.words) for item in chosen)

        need = int(cfg.eval_n_per_diff) - take
        if need > 0:
            results.extend(_fallback_eval_items(context, diff, need, seen_keys, rng))

    return results


def build_exposure_sensitive_eval_items(
    context: TaskContext,
    base_words_list: Sequence[Sequence[str]],
    cfg,
    rng: np.random.Generator,
) -> List[EvalItem]:
    unique_texts = [list(t) for t in OrderedDict((tuple(words), None) for words in base_words_list if words).keys()]
    if not unique_texts:
        return []

    tier_map = _operator_tier_map(context.operator_specs)
    seen_keys: set = set()
    results: List[EvalItem] = []
    per_diff = max(1, int(getattr(cfg, "exposure_sensitive_eval_n_per_diff", 6)))

    for diff in ("easy", "medium", "hard"):
        spec = tier_map.get(diff)
        if spec is None:
            continue
        candidates: List[EvalItem] = []
        if spec.arity == 1:
            for base in unique_texts:
                words = _instantiate_operator(spec, base)
                if not words or len(words) > 10:
                    continue
                rendered = TaskAdapter.render(words, context.grammar)
                if not rendered:
                    continue
                key = tuple(words)
                if key in seen_keys:
                    continue
                candidates.append(
                    EvalItem(
                        words=list(words),
                        output=list(rendered),
                        difficulty=diff,
                        source=f"exposure_sensitive:{diff}:{spec.name}",
                    )
                )
        elif spec.arity == 2:
            capped = unique_texts[: min(len(unique_texts), 16)]
            for left, right in product(capped, capped):
                if left == right:
                    continue
                words = _instantiate_operator(spec, left, right)
                if not words or len(words) > 10:
                    continue
                rendered = TaskAdapter.render(words, context.grammar)
                if not rendered:
                    continue
                key = tuple(words)
                if key in seen_keys:
                    continue
                candidates.append(
                    EvalItem(
                        words=list(words),
                        output=list(rendered),
                        difficulty=diff,
                        source=f"exposure_sensitive:{diff}:{spec.name}",
                    )
                )
        rng.shuffle(candidates)
        chosen = candidates[:per_diff]
        results.extend(chosen)
        seen_keys.update(tuple(item.words) for item in chosen)

    return results
