"""
query_synthesizer.py — Generate novel queries from a grammar.

Given a Grammar (nouns + rules), synthesize new (program, output) pairs
by recursively composing grammar rules. This enables "multi-task" evaluation
within a single grammar: many diverse queries from the same production system.

Key constraint: only emit programs we can verify via our renderer.
"""
from __future__ import annotations
from typing import List, Optional, Set, Tuple
import numpy as np

from ..interfaces import Example
from ..grammar.task_adapter import Grammar, TaskAdapter


def synthesize_queries(
    grammar: Grammar,
    n: int = 20,
    max_depth: int = 3,
    max_len: int = 6,
    rng: Optional[np.random.Generator] = None,
    existing: Optional[List[Example]] = None,
) -> List[Example]:
    """Synthesize n novel (program, output) pairs from grammar.

    Strategy:
    1. Enumerate all depth-1 programs (single nouns)
    2. Apply each unary rule (tufa, gazzer) to depth-1 programs
    3. Apply binary rules (fep) to pairs of programs
    4. Apply concatenation to pairs
    5. Filter: only keep programs our renderer can verify

    Args:
        grammar: parsed Grammar with nouns + rules
        n: number of queries to generate
        max_depth: max composition depth
        max_len: max program token length
        rng: random generator
        existing: existing query programs to avoid duplicates

    Returns:
        List of Example(words, output) for novel queries
    """
    rng = rng or np.random.default_rng()
    nouns = list(grammar.nouns.keys())
    if not nouns:
        return []

    # Track seen programs
    seen: Set[Tuple[str, ...]] = set()
    if existing:
        for ex in existing:
            seen.add(tuple(ex.words))

    # Bank of verified (program, output) pairs by depth
    bank: List[Tuple[List[str], List[str]]] = []

    # Depth 0: single nouns
    for w in nouns:
        prog = [w]
        out = TaskAdapter.render(prog, grammar)
        if out is not None:
            bank.append((prog, out))
            seen.add(tuple(prog))

    # Build progressively deeper compositions
    for depth in range(1, max_depth + 1):
        new_entries = []

        # Unary rules: x1 tufa → [x1] [x1] [x1], x1 gazzer → [x1] [x1]
        for pat, tmpl in grammar.rules:
            if _is_unary_rule(pat):
                operator = _get_operator(pat)
                if operator is None:
                    continue
                for base_prog, base_out in bank:
                    prog = base_prog + [operator]
                    if len(prog) > max_len:
                        continue
                    key = tuple(prog)
                    if key in seen:
                        continue
                    out = TaskAdapter.render(prog, grammar)
                    if out is not None:
                        new_entries.append((prog, out))
                        seen.add(key)

        # Binary rules: u1 fep u2 → [u1] [u2]
        for pat, tmpl in grammar.rules:
            if _is_binary_rule(pat):
                operator = _get_operator(pat)
                if operator is None:
                    continue
                # Sample pairs from bank
                if len(bank) >= 2:
                    n_pairs = min(len(bank) * 2, 30)
                    for _ in range(n_pairs):
                        i, j = rng.choice(len(bank), size=2, replace=False)
                        left, _ = bank[i]
                        right, _ = bank[j]
                        prog = left + [operator] + right
                        if len(prog) > max_len:
                            continue
                        key = tuple(prog)
                        if key in seen:
                            continue
                        out = TaskAdapter.render(prog, grammar)
                        if out is not None:
                            new_entries.append((prog, out))
                            seen.add(key)

        # Concatenation: just combine two programs
        if len(bank) >= 2:
            n_concat = min(len(bank) * 2, 20)
            for _ in range(n_concat):
                i, j = rng.choice(len(bank), size=2, replace=True)
                left, _ = bank[i]
                right, _ = bank[j]
                prog = left + right
                if len(prog) > max_len:
                    continue
                key = tuple(prog)
                if key in seen:
                    continue
                out = TaskAdapter.render(prog, grammar)
                if out is not None:
                    new_entries.append((prog, out))
                    seen.add(key)

        bank.extend(new_entries)

    # Convert to Examples (exclude depth-0 single nouns — too easy)
    candidates = [(p, o) for p, o in bank if len(p) > 1]

    # Shuffle and select n
    rng.shuffle(candidates)
    selected = candidates[:n]

    return [Example(words=p, output=o) for p, o in selected]


def _is_unary_rule(pattern: List[str]) -> bool:
    """Check if rule is unary: one variable + one literal."""
    vars_ = [p for p in pattern if p.startswith(('u', 'x'))]
    lits = [p for p in pattern if not p.startswith(('u', 'x'))]
    return len(vars_) == 1 and len(lits) == 1


def _is_binary_rule(pattern: List[str]) -> bool:
    """Check if rule has two variables + one literal."""
    vars_ = [p for p in pattern if p.startswith(('u', 'x'))]
    lits = [p for p in pattern if not p.startswith(('u', 'x'))]
    return len(vars_) == 2 and len(lits) == 1


def _get_operator(pattern: List[str]) -> Optional[str]:
    """Extract the operator (literal) from a pattern."""
    for p in pattern:
        if not p.startswith(('u', 'x', '[', ']')):
            return p
    return None
