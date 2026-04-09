"""
option_generator_v2.py — Diverse, valid-only menu generation.

Design principles:
  1. ALL options must render successfully (no NONE renders)
  2. Distractors should have PARTIAL cell overlap with target
     (some cells correct, some wrong → HIGHLIGHT can be useful)
  3. Maximize per-cell diversity across the distractor set
     (different options wrong at different cells)

Strategy:
  Phase 1: Enumerate all valid programs from grammar vocabulary
  Phase 2: Pre-render and cache the program pool
  Phase 3: For each query, select K-1 distractors that:
     a. Render to correct output length
     b. Maximize cell-diversity with target
     c. Include a mix of "easy" and "hard" distractors
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
from itertools import product
import numpy as np

from ..interfaces import Option, Example
from ..env.danger_model import DangerModel, generate_danger_vector
from .task_adapter import Grammar, TaskAdapter


class ProgramPool:
    """Pre-computed pool of all valid (renderable) programs from a grammar.

    Enumerates compositions of nouns and operators up to max_len,
    renders each one, and caches the results.
    """

    def __init__(self, grammar: Grammar, support: List[Example],
                 max_program_len: int = 5, max_output_len: int = 8):
        self.grammar = grammar
        self.max_program_len = max_program_len
        self.max_output_len = max_output_len

        # Vocabulary
        self.nouns = list(grammar.nouns.keys())
        self.operators = self._extract_operators(grammar)
        self.all_words = self.nouns + self.operators

        # Also include words from support
        for ex in support:
            for w in ex.words:
                if w not in self.all_words:
                    self.all_words.append(w)

        # Pool: {program_tuple: rendered_output_tuple}
        self.pool: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
        self._build_pool()

    def _extract_operators(self, grammar: Grammar) -> List[str]:
        """Extract operator words from grammar rules."""
        ops = set()
        for pattern, _ in grammar.rules:
            for token in pattern:
                if not token.startswith(('u', 'x')):
                    ops.add(token)
        return sorted(ops)

    def _build_pool(self) -> None:
        """Enumerate and render all valid programs."""
        # Strategy: enumerate programs of increasing length
        for length in range(1, self.max_program_len + 1):
            if length == 1:
                # Single words
                for w in self.all_words:
                    self._try_add([w])
            elif length <= 3:
                # Enumerate all combinations
                for prog in product(self.all_words, repeat=length):
                    self._try_add(list(prog))
            else:
                # For longer programs, use compositional strategy
                self._enumerate_compositions(length)

    def _enumerate_compositions(self, length: int) -> None:
        """Build longer programs by composing shorter valid ones with operators."""
        # Get valid sub-programs of various lengths
        sub_programs = {}
        for prog, out in self.pool.items():
            plen = len(prog)
            if plen not in sub_programs:
                sub_programs[plen] = []
            sub_programs[plen].append(prog)

        # Compose: subprog + operator + subprog
        for op in self.operators:
            for left_len in range(1, length - 1):
                right_len = length - left_len - 1
                if left_len not in sub_programs or right_len not in sub_programs:
                    continue
                # Sample to avoid combinatorial explosion
                lefts = sub_programs[left_len][:50]
                rights = sub_programs[right_len][:50]
                for lp in lefts:
                    for rp in rights:
                        prog = list(lp) + [op] + list(rp)
                        if len(prog) <= self.max_program_len:
                            self._try_add(prog)

        # Compose: subprog + operator (unary postfix)
        for op in self.operators:
            for sub_len in range(1, length):
                if sub_len not in sub_programs:
                    continue
                for sp in sub_programs[sub_len][:100]:
                    prog = list(sp) + [op]
                    if len(prog) == length:
                        self._try_add(prog)

        # Compose: operator + subprog (unary prefix)
        for op in self.operators:
            sub_len = length - 1
            if sub_len not in sub_programs:
                continue
            for sp in sub_programs[sub_len][:100]:
                prog = [op] + list(sp)
                self._try_add(prog)

    def _try_add(self, program: List[str]) -> None:
        """Try to render and add a program to the pool."""
        key = tuple(program)
        if key in self.pool:
            return
        rendered = TaskAdapter.render(program, self.grammar)
        if rendered is not None and 0 < len(rendered) <= self.max_output_len:
            self.pool[key] = tuple(rendered)

    def get_programs_by_output_length(self, length: int) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """Get all programs that render to a specific output length."""
        return [(prog, out) for prog, out in self.pool.items()
                if len(out) == length]

    def stats(self) -> dict:
        """Pool statistics."""
        output_lengths = {}
        for prog, out in self.pool.items():
            l = len(out)
            output_lengths[l] = output_lengths.get(l, 0) + 1
        return {
            "total_programs": len(self.pool),
            "by_output_length": output_lengths,
            "nouns": self.nouns,
            "operators": self.operators,
        }


def compute_cell_overlap(output: Tuple[str, ...], target: Tuple[str, ...]) -> Tuple[int, int]:
    """Compute (n_matching_cells, n_total_cells) between two outputs."""
    L = min(len(output), len(target))
    matches = sum(1 for i in range(L) if output[i] == target[i])
    return matches, L


def compute_error_mask(output: Tuple[str, ...], target: Tuple[str, ...]) -> Tuple[bool, ...]:
    """Binary mask: True where output differs from target."""
    L = min(len(output), len(target))
    return tuple(output[i] != target[i] for i in range(L))


def generate_menu_v2(
    target_output: List[str],
    true_program: List[str],
    pool: ProgramPool,
    danger_model: DangerModel,
    K: int = 10,
    m: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> List[Option]:
    """Generate a menu with diverse, valid-only distractors.

    Selection strategy:
      1. Filter pool to programs with matching output length
      2. Exclude programs that produce target output (duplicates of correct)
      3. Score each distractor by cell-diversity contribution
      4. Greedily select K-1 distractors maximizing coverage

    Guarantees:
      - ALL options render successfully
      - Exactly one option is correct
      - Distractors have varying cell overlaps with target
    """
    rng = rng or np.random.default_rng()
    target_tuple = tuple(target_output)
    L = len(target_output)

    # Correct option
    correct_v = generate_danger_vector(m, rng)
    correct_option = Option(
        index=0,
        text=list(true_program),
        danger_vec=correct_v,
        is_correct=True,
        rendered_output=list(target_output),
    )

    # Get candidate pool for this output length
    candidates = pool.get_programs_by_output_length(L)

    # Filter out correct answer and exact duplicates
    true_key = tuple(true_program)
    available = [(prog, out) for prog, out in candidates
                 if out != target_tuple and prog != true_key]

    if not available:
        # Fallback: also try adjacent lengths
        for delta in [-1, 1, -2, 2]:
            extra = pool.get_programs_by_output_length(L + delta)
            available.extend([(p, o) for p, o in extra if o != target_tuple])
            if len(available) >= K:
                break

    # Compute cell overlap for each candidate
    scored = []
    for prog, out in available:
        matches, total = compute_cell_overlap(out, target_tuple)
        error_mask = compute_error_mask(out, target_tuple)
        overlap_frac = matches / max(total, 1)
        scored.append({
            'prog': prog,
            'out': out,
            'overlap': overlap_frac,
            'error_mask': error_mask,
            'n_errors': sum(error_mask),
        })

    # Sort into difficulty buckets
    # Easy: overlap < 0.3 (mostly wrong)
    # Medium: 0.3 <= overlap < 0.7 (partially correct)
    # Hard: overlap >= 0.7 (mostly correct, subtle errors)
    easy = [s for s in scored if s['overlap'] < 0.3]
    medium = [s for s in scored if 0.3 <= s['overlap'] < 0.7]
    hard = [s for s in scored if s['overlap'] >= 0.7]

    # Target: ~30% easy, ~40% medium, ~30% hard
    n_need = K - 1
    n_easy = max(1, int(n_need * 0.30))
    n_hard = max(1, int(n_need * 0.30))
    n_medium = n_need - n_easy - n_hard

    # Greedy diversity selection within each bucket
    selected = []
    selected += _diverse_select(easy, target_tuple, n_easy, rng)
    selected += _diverse_select(medium, target_tuple, n_medium, rng)
    selected += _diverse_select(hard, target_tuple, n_hard, rng)

    # If we don't have enough, fill from any bucket
    if len(selected) < n_need:
        remaining = [s for s in scored if s['prog'] not in {sel['prog'] for sel in selected}]
        selected += _diverse_select(remaining, target_tuple,
                                     n_need - len(selected), rng)

    # Build options
    distractors = []
    for i, sel in enumerate(selected[:n_need]):
        v = generate_danger_vector(m, rng)
        distractors.append(Option(
            index=i + 1,
            text=list(sel['prog']),
            danger_vec=v,
            is_correct=False,
            rendered_output=list(sel['out']),
        ))

    # Assemble and shuffle
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
        ))

    return result


def _diverse_select(
    candidates: List[dict],
    target: Tuple[str, ...],
    n: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Greedily select n candidates maximizing cell-diversity.

    At each step, pick the candidate whose error_mask has the least
    overlap with the already-selected set (covers new error positions).
    """
    if not candidates or n <= 0:
        return []

    # Shuffle to break ties randomly
    candidates = list(candidates)
    rng.shuffle(candidates)

    L = len(target)
    selected = []
    covered_errors = set()  # which cell positions have been "error'd" by selected options

    for _ in range(min(n, len(candidates))):
        best_score = -1
        best_idx = 0

        for i, cand in enumerate(candidates):
            if cand in selected:
                continue

            # Score = number of NEW error positions this candidate covers
            new_errors = sum(1 for j, e in enumerate(cand['error_mask'])
                            if e and j not in covered_errors)
            # Tiebreak: prefer medium overlap
            tiebreak = 1.0 - abs(cand['overlap'] - 0.5)
            score = new_errors * 10 + tiebreak

            if score > best_score:
                best_score = score
                best_idx = i

        chosen = candidates[best_idx]
        selected.append(chosen)
        # Update coverage
        for j, e in enumerate(chosen['error_mask']):
            if e:
                covered_errors.add(j)
        # Remove from candidates
        candidates[best_idx] = candidates[-1]
        candidates.pop()
        if not candidates:
            break

    return selected


def analyze_menu_diversity(menu: List[Option], target: List[str]) -> dict:
    """Analyze the diversity of a generated menu."""
    target_tuple = tuple(target)
    L = len(target)

    stats = {
        'n_options': len(menu),
        'n_renderable': 0,
        'n_none': 0,
        'overlaps': [],
        'per_cell_error_count': [0] * L,
        'unique_outputs': set(),
    }

    for opt in menu:
        if opt.is_correct:
            continue
        rendered = opt.rendered_output
        if rendered is None:
            stats['n_none'] += 1
            continue
        stats['n_renderable'] += 1
        stats['unique_outputs'].add(tuple(rendered))

        out_tuple = tuple(rendered)
        matches, total = compute_cell_overlap(out_tuple, target_tuple)
        stats['overlaps'].append(matches / max(total, 1))

        for i in range(min(L, len(rendered))):
            if rendered[i] != target[i]:
                stats['per_cell_error_count'][i] += 1

    stats['unique_outputs'] = len(stats['unique_outputs'])
    return stats
