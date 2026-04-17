"""
query_generator.py — Grammar-based query generation with difficulty control.

Generates new queries from a parsed Grammar by composing nouns + rules
at controlled depth. Three difficulty tiers: easy, medium, hard.

Also provides txt_resample mode: shuffle support+query pool.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .generator import Grammar, render_with_grammar
from ..interfaces import Example
from ..config import ExpConfig


# ── Difficulty definitions ────────────────────────────────────

@dataclass
class DifficultySpec:
    """Specification for one difficulty tier."""
    name: str
    min_nouns: int = 1
    max_nouns: int = 2
    min_ops: int = 0
    max_ops: int = 1
    max_output_len: int = 4


DIFFICULTY_TIERS = {
    'easy': DifficultySpec(
        name='easy', min_nouns=1, max_nouns=2,
        min_ops=0, max_ops=1, max_output_len=4),
    'medium': DifficultySpec(
        name='medium', min_nouns=1, max_nouns=3,
        min_ops=1, max_ops=2, max_output_len=8),
    'hard': DifficultySpec(
        name='hard', min_nouns=2, max_nouns=4,
        min_ops=2, max_ops=4, max_output_len=16),
}


# ── Core generator ────────────────────────────────────────────

def generate_query_from_grammar(
    grammar: Grammar,
    difficulty: str,
    rng: np.random.Generator,
    existing_inputs: Optional[Set[tuple]] = None,
    max_attempts: int = 50,
) -> Optional[Example]:
    """Generate a single query from a grammar at the given difficulty.

    Args:
        grammar: parsed Grammar
        difficulty: 'easy', 'medium', or 'hard'
        rng: random generator
        existing_inputs: set of input tuples to avoid duplicates
        max_attempts: how many attempts before giving up

    Returns:
        Example with words + output, or None if failed
    """
    spec = DIFFICULTY_TIERS.get(difficulty, DIFFICULTY_TIERS['medium'])
    nouns = list(grammar.nouns.keys())
    ops = [r for r in grammar.rules if _is_operator_rule(r)]
    existing = existing_inputs or set()

    for _ in range(max_attempts):
        words = _compose_expression(nouns, ops, spec, rng)
        if not words:
            continue

        key = tuple(words)
        if key in existing:
            continue

        # Try to render
        output = render_with_grammar(words, grammar)
        if output is None or len(output) == 0:
            continue
        if len(output) > spec.max_output_len:
            continue

        existing.add(key)
        return Example(words=words, output=output)

    return None


def _is_operator_rule(rule: Tuple[List[str], List[str]]) -> bool:
    """Check if a rule is an operator rule (has a literal keyword)."""
    pattern, _ = rule
    return any(not t.startswith(('u', 'x')) for t in pattern)


def _get_rule_keyword(rule: Tuple[List[str], List[str]]) -> Optional[str]:
    """Get the literal keyword from an operator rule."""
    pattern, _ = rule
    for t in pattern:
        if not t.startswith(('u', 'x')):
            return t
    return None


def _compose_expression(
    nouns: List[str],
    ops: List[Tuple[List[str], List[str]]],
    spec: DifficultySpec,
    rng: np.random.Generator,
) -> Optional[List[str]]:
    """Build a random expression from nouns + ops at the target difficulty."""
    n_nouns = rng.integers(spec.min_nouns, spec.max_nouns + 1)
    n_ops = rng.integers(spec.min_ops, min(spec.max_ops, len(ops)) + 1) if ops else 0

    # Pick random nouns
    selected_nouns = [rng.choice(nouns) for _ in range(n_nouns)]

    if n_ops == 0:
        # Pure noun concatenation
        return selected_nouns

    # Build expression by applying ops
    # Start with a base noun
    expr = [selected_nouns[0]]
    noun_idx = 1

    for i in range(n_ops):
        op = ops[rng.integers(0, len(ops))]
        keyword = _get_rule_keyword(op)

        if keyword is None:
            # Pure variable rule (u1 x1 -> [u1] [x1]), skip
            continue

        pattern, _ = op
        n_vars = sum(1 for t in pattern if t.startswith(('u', 'x')))

        if n_vars == 1:
            # Unary op: x1 keyword -> ... or keyword x1 -> ...
            kw_pos = next(j for j, t in enumerate(pattern) if t == keyword)
            if kw_pos == 0:
                # keyword x1
                expr = [keyword] + expr
            else:
                # x1 keyword
                expr = expr + [keyword]
        elif n_vars == 2:
            # Binary op: x1 keyword x2 or u1 keyword u2
            # Need another argument
            if noun_idx < len(selected_nouns):
                arg2 = [selected_nouns[noun_idx]]
                noun_idx += 1
            else:
                arg2 = [rng.choice(nouns)]

            kw_pos = next(j for j, t in enumerate(pattern) if t == keyword)
            if kw_pos == 1:
                # u1 keyword u2
                expr = expr + [keyword] + arg2
            elif kw_pos == 0:
                # keyword u1 u2
                expr = [keyword] + expr + arg2
            else:
                expr = expr + arg2 + [keyword]

    return expr if expr else None


# ── Batch generation ──────────────────────────────────────────

@dataclass
class GeneratedQueryBatch:
    """A batch of generated queries with metadata."""
    queries: List[Example] = field(default_factory=list)
    difficulty_tags: List[str] = field(default_factory=list)
    source: str = 'generated'

    def __len__(self):
        return len(self.queries)


def generate_query_batch(
    grammar: Grammar,
    counts: Dict[str, int],  # {'easy': 2, 'medium': 3, 'hard': 3}
    rng: np.random.Generator,
    existing_inputs: Optional[Set[tuple]] = None,
) -> GeneratedQueryBatch:
    """Generate a batch of queries at mixed difficulties.

    Args:
        grammar: parsed Grammar
        counts: dict mapping difficulty → count
        rng: random generator
        existing_inputs: set of existing input tuples to avoid

    Returns:
        GeneratedQueryBatch with queries and tags
    """
    batch = GeneratedQueryBatch()
    existing = existing_inputs or set()

    for difficulty, n in counts.items():
        for _ in range(n):
            q = generate_query_from_grammar(grammar, difficulty, rng, existing)
            if q is not None:
                batch.queries.append(q)
                batch.difficulty_tags.append(difficulty)

    return batch


def generate_episode_queries(
    grammar: Grammar,
    support: List[Example],
    rng: np.random.Generator,
    n_obs: int = 4,
    n_teach: int = 8,
    n_eval: int = 8,
) -> Tuple[List[Example], List[Example], List[Example], List[str]]:
    """Generate obs, teach, eval query batches with difficulty distribution.

    Distribution follows the user's specification:
    - Obs:   easy=1, medium=2, hard=1
    - Teach: easy=2, medium=3, hard=3
    - Eval:  easy=1, medium=3, hard=4

    Returns:
        (obs_queries, teach_queries, eval_queries, all_tags)
    """
    existing = {tuple(ex.words) for ex in support}

    # Scale to actual counts
    obs_dist = _scale_distribution({'easy': 1, 'medium': 2, 'hard': 1}, n_obs)
    teach_dist = _scale_distribution({'easy': 2, 'medium': 3, 'hard': 3}, n_teach)
    eval_dist = _scale_distribution({'easy': 1, 'medium': 3, 'hard': 4}, n_eval)

    obs_batch = generate_query_batch(grammar, obs_dist, rng, existing)
    existing.update(tuple(q.words) for q in obs_batch.queries)

    teach_batch = generate_query_batch(grammar, teach_dist, rng, existing)
    existing.update(tuple(q.words) for q in teach_batch.queries)

    eval_batch = generate_query_batch(grammar, eval_dist, rng, existing)

    all_tags = obs_batch.difficulty_tags + teach_batch.difficulty_tags + eval_batch.difficulty_tags

    return obs_batch.queries, teach_batch.queries, eval_batch.queries, all_tags


def _scale_distribution(base: Dict[str, int], target_total: int) -> Dict[str, int]:
    """Scale a difficulty distribution to a target total."""
    base_total = sum(base.values())
    if base_total == 0:
        return {'easy': target_total}
    scale = target_total / base_total
    result = {}
    assigned = 0
    keys = list(base.keys())
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            result[k] = target_total - assigned
        else:
            n = max(0, round(base[k] * scale))
            result[k] = n
            assigned += n
    return result


# ── Banked episode generation (Phase 7+) ──────────────────────
# Fixes the confound where n_obs changes teach/eval queries.
# All banks are generated once; n_obs selects an obs prefix.

@dataclass
class EpisodeBank:
    """Fixed episode query bank. teach/eval invariant to n_obs."""
    obs_bank: List[Example]        # max obs pool (e.g. 8 queries)
    teach_bank: List[Example]      # fixed teach pool
    eval_bank: List[Example]       # fixed eval pool
    obs_tags: List[str] = field(default_factory=list)
    teach_tags: List[str] = field(default_factory=list)
    eval_tags: List[str] = field(default_factory=list)


def generate_episode_bank(
    grammar: 'Grammar',
    support: List[Example],
    rng: np.random.Generator,
    n_obs_max: int = 8,
    n_teach: int = 8,
    n_eval: int = 8,
) -> 'EpisodeBank':
    """Generate all banks once. teach/eval invariant to n_obs.

    Uses separate RNG streams so obs generation doesn't affect
    teach/eval randomness.

    Args:
        grammar: parsed Grammar
        support: learner support examples (excluded from generation)
        rng: master RNG (consumed only for deriving sub-seeds)
        n_obs_max: maximum obs queries to pre-generate
        n_teach: teach query count
        n_eval: eval query count

    Returns:
        EpisodeBank with three non-overlapping query pools
    """
    # Derive independent seeds for each phase
    master_seed = int(rng.integers(0, 2**31))
    rng_obs = np.random.default_rng(master_seed + 1)
    rng_teach = np.random.default_rng(master_seed + 2)
    rng_eval = np.random.default_rng(master_seed + 3)

    existing = {tuple(ex.words) for ex in support}

    # Generate obs bank (max size)
    obs_dist = _scale_distribution(
        {'easy': 1, 'medium': 2, 'hard': 1}, n_obs_max)
    obs_batch = generate_query_batch(grammar, obs_dist, rng_obs, existing)
    existing.update(tuple(q.words) for q in obs_batch.queries)

    # Generate teach bank
    teach_dist = _scale_distribution(
        {'easy': 2, 'medium': 3, 'hard': 3}, n_teach)
    teach_batch = generate_query_batch(grammar, teach_dist, rng_teach, existing)
    existing.update(tuple(q.words) for q in teach_batch.queries)

    # Generate eval bank
    eval_dist = _scale_distribution(
        {'easy': 1, 'medium': 3, 'hard': 4}, n_eval)
    eval_batch = generate_query_batch(grammar, eval_dist, rng_eval, existing)

    return EpisodeBank(
        obs_bank=obs_batch.queries,
        teach_bank=teach_batch.queries,
        eval_bank=eval_batch.queries,
        obs_tags=obs_batch.difficulty_tags,
        teach_tags=teach_batch.difficulty_tags,
        eval_tags=eval_batch.difficulty_tags,
    )


def slice_obs_from_bank(
    bank: EpisodeBank,
    n_obs: int,
) -> Tuple[List[Example], List[Example], List[Example], List[str]]:
    """Slice obs prefix from bank. teach/eval always unchanged.

    Args:
        bank: pre-generated EpisodeBank
        n_obs: how many obs queries to use (0 to n_obs_max)

    Returns:
        (obs_queries, teach_queries, eval_queries, all_tags)
    """
    obs_q = bank.obs_bank[:n_obs]
    obs_t = bank.obs_tags[:n_obs]
    all_tags = obs_t + bank.teach_tags + bank.eval_tags
    return obs_q, list(bank.teach_bank), list(bank.eval_bank), all_tags


def make_query_rng(
    global_seed: int,
    task_id: str,
    query_id: int,
    phase: str = 'teach',
) -> np.random.Generator:
    """Create deterministic per-query RNG.

    This ensures candidate pool generation is invariant to n_obs:
    same (task_id, query_id, phase) always produces same pool.

    Args:
        global_seed: episode-level seed
        task_id: grammar/task identifier
        query_id: unique query index
        phase: 'obs', 'teach', or 'eval'

    Returns:
        independent np.random.Generator for this query
    """
    import hashlib
    key = f'{global_seed}:{task_id}:{query_id}:{phase}'
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(h)


# ── Resample mode ─────────────────────────────────────────────

def resample_queries(
    support: List[Example],
    queries: List[Example],
    n_obs: int,
    n_teach: int,
    n_eval: int,
    rng: np.random.Generator,
    allow_support_as_query: bool = True,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Resample queries from support + query pool without overflow.

    If allow_support_as_query, support inputs can be used as queries.
    Deduplication: no query appears in two phases.
    """
    pool = list(queries)
    if allow_support_as_query:
        # Add support as potential queries (inputs reused, labels are gold)
        seen_inputs = {tuple(q.words) for q in pool}
        for ex in support:
            key = tuple(ex.words)
            if key not in seen_inputs:
                pool.append(ex)
                seen_inputs.add(key)

    rng.shuffle(pool)

    total = n_obs + n_teach + n_eval
    if len(pool) < total:
        # Still not enough — allow repetition with different pool positions
        while len(pool) < total:
            pool.append(pool[rng.integers(0, len(pool))])

    obs = list(pool[:n_obs])
    teach = list(pool[n_obs:n_obs + n_teach])
    evl = list(pool[n_obs + n_teach:n_obs + n_teach + n_eval])

    return obs, teach, evl
