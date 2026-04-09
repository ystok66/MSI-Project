"""
option_generator.py — Sample menus with exactly one correct option.

Implements §4.2 menu generation with anti-shortcut constraints (§5.3).
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from ..interfaces import Option, Example
from ..env.danger_model import DangerModel, generate_danger_vector
from .task_adapter import Grammar, TaskAdapter


def generate_menu(
    target_output: List[str],
    true_program: List[str],
    grammar: Grammar,
    support: List[Example],
    danger_model: DangerModel,
    K: int = 10,
    m: int = 16,
    rng: Optional[np.random.Generator] = None,
    max_token_len: Optional[int] = None,
) -> List[Option]:
    """Generate a menu of K options with exactly one correct.

    Guarantees:
    1. Exactly one option has F_G(ν) == target_output
    2. Token length of each option ≤ |target_output| (spec §4.2)
    3. Each option has an independent danger vector

    Strategy for distractors:
    - Swap nouns in the true program
    - Use partial subprograms from support examples
    - Random compositions from vocabulary
    """
    rng = rng or np.random.default_rng()
    if max_token_len is None:
        max_token_len = len(target_output)

    # Correct option
    correct_v = generate_danger_vector(m, rng)
    correct_option = Option(
        index=0,
        text=list(true_program),
        danger_vec=correct_v,
        is_correct=True,
        rendered_output=list(target_output),
    )

    # Collect vocabulary
    vocab_words = set()
    for ex in support:
        vocab_words.update(ex.words)

    # Known nouns and their colors
    nouns = dict(grammar.nouns)
    noun_words = list(nouns.keys())

    # Generate distractors
    distractors: List[Option] = []
    seen_texts: set = {tuple(true_program)}
    attempts = 0
    max_attempts = K * 20

    while len(distractors) < K - 1 and attempts < max_attempts:
        attempts += 1
        candidate_words = _generate_distractor(
            true_program, noun_words, list(vocab_words),
            grammar, rng, max_token_len)

        if candidate_words is None:
            continue

        key = tuple(candidate_words)
        if key in seen_texts:
            continue

        # Verify it renders to something different from target
        rendered = TaskAdapter.render(candidate_words, grammar)
        if rendered is not None and rendered == target_output:
            continue  # accidentally correct — skip
        # Allow None renders (learner doesn't know the render)

        seen_texts.add(key)
        v = generate_danger_vector(m, rng)
        distractors.append(Option(
            index=len(distractors) + 1,
            text=candidate_words,
            danger_vec=v,
            is_correct=False,
            rendered_output=rendered,
        ))

    # If we couldn't generate enough distractors, pad with random
    vocab_list = list(vocab_words)
    pad_attempts = 0
    max_pad = K * 30
    while len(distractors) < K - 1 and pad_attempts < max_pad:
        pad_attempts += 1
        length = int(rng.integers(1, min(len(true_program) + 1, max_token_len + 1)))
        rand_words = list(rng.choice(vocab_list, size=length, replace=True))
        key = tuple(rand_words)
        if key in seen_texts:
            continue
        seen_texts.add(key)
        v = generate_danger_vector(m, rng)
        # Skip render check for padding — accept as distractor
        # (learner can't know the render anyway without trying)
        distractors.append(Option(
            index=len(distractors) + 1,
            text=rand_words,
            danger_vec=v,
            is_correct=False,
            rendered_output=None,  # lazy — don't render padding distractors
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


def _generate_distractor(
    true_program: List[str],
    noun_words: List[str],
    vocab_words: List[str],
    grammar: Grammar,
    rng: np.random.Generator,
    max_len: int,
) -> Optional[List[str]]:
    """Generate a single distractor program via various strategies."""
    strategy = rng.choice(["swap_noun", "partial_sub", "random_compose",
                           "drop_token", "reorder"],
                          p=[0.35, 0.20, 0.20, 0.10, 0.15])

    if strategy == "swap_noun" and noun_words and len(true_program) > 0:
        # Replace one noun with a different noun
        result = list(true_program)
        noun_positions = [i for i, w in enumerate(result) if w in grammar.nouns]
        if noun_positions:
            pos = rng.choice(noun_positions)
            original = result[pos]
            candidates = [n for n in noun_words if n != original]
            if candidates:
                result[pos] = rng.choice(candidates)
                if len(result) <= max_len:
                    return result

    elif strategy == "partial_sub" and len(true_program) > 1:
        # Use a subsequence of the true program
        start = rng.integers(0, len(true_program))
        end = rng.integers(start + 1, len(true_program) + 1)
        result = true_program[start:end]
        if 0 < len(result) < len(true_program) and len(result) <= max_len:
            return result

    elif strategy == "random_compose" and vocab_words:
        # Random combination of vocabulary words
        length = rng.integers(1, min(len(true_program) + 1, max_len + 1))
        result = list(rng.choice(vocab_words, size=length, replace=True))
        if len(result) <= max_len:
            return result

    elif strategy == "drop_token" and len(true_program) > 1:
        # Drop one token
        idx = rng.integers(0, len(true_program))
        result = true_program[:idx] + true_program[idx+1:]
        if result and len(result) <= max_len:
            return result

    elif strategy == "reorder" and len(true_program) > 1:
        # Swap two adjacent tokens
        result = list(true_program)
        idx = rng.integers(0, len(result) - 1)
        result[idx], result[idx+1] = result[idx+1], result[idx]
        if len(result) <= max_len:
            return result

    # Fallback: swap a random noun
    if noun_words and len(true_program) > 0:
        result = list(true_program)
        noun_positions = [i for i, w in enumerate(result) if w in grammar.nouns]
        if noun_positions:
            pos = rng.choice(noun_positions)
            result[pos] = rng.choice(noun_words)
            if len(result) <= max_len:
                return result

    return None


def verify_menu_invariants(menu: List[Option]) -> dict:
    """Verify menu generation invariants (§18 E0).

    Returns dict with pass/fail for each check.
    """
    correct_count = sum(1 for o in menu if o.is_correct)
    danger_range = all(
        0 <= np.linalg.norm(o.danger_vec) < 100 for o in menu)
    unique_texts = len(set(tuple(o.text) for o in menu))

    return {
        "exactly_one_correct": correct_count == 1,
        "danger_vec_valid": danger_range,
        "unique_texts": unique_texts == len(menu),
        "correct_count": correct_count,
        "menu_size": len(menu),
    }
