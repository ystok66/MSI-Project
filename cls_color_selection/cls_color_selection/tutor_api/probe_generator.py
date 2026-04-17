"""
probe_generator.py — Generate small probe sets for counterfactual evaluation.

Selects probe queries that are most informative for evaluating
whether a hint helped or hurt long-term learning.

Three probe types:
  1. same-role: queries involving same grammar roles as current query
  2. same-error: queries where learner is most likely to make similar errors
  3. same-structure: queries with similar word overlap
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


def generate_probes(
    current_words: List[str],
    available_queries: List[Tuple[List[str], List[str]]],
    n_probes: int = 3,
    strategy: str = 'word_overlap',
) -> Tuple[List[List[str]], List[List[str]]]:
    """Generate probe queries for counterfactual evaluation.

    Args:
        current_words: words of the current query
        available_queries: [(words, gold_output), ...] available queries
        n_probes: number of probes to select
        strategy: 'word_overlap' | 'random' | 'mixed'

    Returns:
        (probe_words_list, probe_golds_list)
    """
    if not available_queries:
        return [], []

    if strategy == 'random':
        return _random_probes(available_queries, n_probes)
    elif strategy == 'word_overlap':
        return _word_overlap_probes(current_words, available_queries, n_probes)
    elif strategy == 'mixed':
        return _mixed_probes(current_words, available_queries, n_probes)
    else:
        return _word_overlap_probes(current_words, available_queries, n_probes)


def _random_probes(
    available: List[Tuple[List[str], List[str]]],
    n: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Random probe selection."""
    indices = np.random.choice(len(available), size=min(n, len(available)),
                                replace=False)
    words = [available[i][0] for i in indices]
    golds = [available[i][1] for i in indices]
    return words, golds


def _word_overlap_probes(
    current_words: List[str],
    available: List[Tuple[List[str], List[str]]],
    n: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Select probes with highest word overlap to current query.

    Same-word queries are most likely to share grammar rules,
    making them informative for detecting learning changes.
    """
    current_set = set(current_words)

    scored = []
    for i, (words, gold) in enumerate(available):
        # Skip if it's the exact same query
        if words == current_words:
            continue
        overlap = len(current_set & set(words))
        scored.append((i, overlap))

    # Sort by overlap descending, break ties randomly
    scored.sort(key=lambda x: (-x[1], np.random.random()))

    selected = scored[:n]
    words = [available[i][0] for i, _ in selected]
    golds = [available[i][1] for i, _ in selected]
    return words, golds


def _mixed_probes(
    current_words: List[str],
    available: List[Tuple[List[str], List[str]]],
    n: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Mixed strategy: half word-overlap, half random."""
    n_overlap = max(n // 2, 1)
    n_random = n - n_overlap

    w1, g1 = _word_overlap_probes(current_words, available, n_overlap)
    w2, g2 = _random_probes(available, n_random)

    return w1 + w2, g1 + g2


def build_available_queries(
    teach_queries,
    current_query_idx: int,
    task_model=None,
) -> List[Tuple[List[str], List[str]]]:
    """Build list of available queries for probe generation.

    Uses remaining teach queries + their ground truth from task model.

    Args:
        teach_queries: list of query examples
        current_query_idx: index of current query (skip it)
        task_model: TutorTaskModel for ground truth

    Returns:
        [(words, gold_output), ...]
    """
    available = []
    for i, q in enumerate(teach_queries):
        if i == current_query_idx:
            continue
        words = q.words if hasattr(q, 'words') else q.get('words', [])
        if task_model:
            gold = task_model.ground_truth_output(words)
        else:
            gold = q.gold if hasattr(q, 'gold') else q.get('gold', [])
        if words and gold:
            available.append((words, gold))
    return available
