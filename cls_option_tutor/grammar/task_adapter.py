"""
task_adapter.py — Wraps existing CLS renderer / parser.

Provides forward rendering F_G(ν) → Y and task loading.
Operates on the BASIC/cls_learner data format.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import re

from ..interfaces import Example


@dataclass
class Grammar:
    """Parsed grammar from a CLS task file.

    Stores rules as (lhs_pattern, rhs_template) pairs.
    Nouns map directly to colors; operators have variable slots.
    """
    nouns: Dict[str, str]                  # word → COLOR
    rules: List[Tuple[List[str], List[str]]]  # (pattern, template)
    raw_text: str = ""


def parse_task_file(path: str) -> Tuple[List[Example], List[Example], Grammar]:
    """Parse a CLS data file into (support, query, grammar).

    Format:
        *SUPPORT*
        IN: w1 w2 OUT: C1 C2
        ...
        *QUERY*
        IN: w1 w2 OUT: C1 C2
        ...
        *GRAMMAR*
        word -> COLOR
        pattern -> template
    """
    with open(path, 'r') as f:
        text = f.read()

    sections = {}
    current = None
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('*') and line.endswith('*'):
            current = line.strip('*')
            sections[current] = []
        elif current and line:
            sections[current].append(line)

    def _parse_examples(lines: List[str]) -> List[Example]:
        examples = []
        for line in lines:
            m = re.match(r'IN:\s*(.*?)\s*OUT:\s*(.*)', line)
            if m:
                words = m.group(1).split()
                output = m.group(2).split()
                examples.append(Example(words=words, output=output))
        return examples

    support = _parse_examples(sections.get('SUPPORT', []))
    query = _parse_examples(sections.get('QUERY', []))

    # Parse grammar
    nouns = {}
    rules = []
    for line in sections.get('GRAMMAR', []):
        line = line.strip()
        if '->' not in line:
            continue
        lhs, rhs = line.split('->', 1)
        lhs_tokens = lhs.strip().split()
        rhs_tokens = rhs.strip().split()

        # Noun rule: single word → single COLOR
        if (len(lhs_tokens) == 1
                and len(rhs_tokens) == 1
                and rhs_tokens[0].isupper()
                and not lhs_tokens[0].startswith(('[', 'u', 'x'))):
            nouns[lhs_tokens[0]] = rhs_tokens[0]
        else:
            rules.append((lhs_tokens, rhs_tokens))

    grammar = Grammar(nouns=nouns, rules=rules, raw_text=text)
    return support, query, grammar


class TaskAdapter:
    """Wraps CLS task files for the option tutor environment.

    Provides:
        - load_task(path): parse a task file
        - render(words, grammar): execute F_G(ν) → Y
        - vocabulary: available words from support
    """

    def __init__(self, data_dir: str = ""):
        self.data_dir = data_dir
        self._cache: Dict[str, Tuple[List[Example], List[Example], Grammar]] = {}

    def load_task(self, task_id: str) -> Tuple[List[Example], List[Example], Grammar]:
        """Load and cache a task file."""
        if task_id in self._cache:
            return self._cache[task_id]

        path = os.path.join(self.data_dir, f"{task_id}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task file not found: {path}")

        result = parse_task_file(path)
        self._cache[task_id] = result
        return result

    def list_tasks(self) -> List[str]:
        """List available task IDs."""
        if not self.data_dir or not os.path.isdir(self.data_dir):
            return []
        return sorted([
            f.replace('.txt', '')
            for f in os.listdir(self.data_dir)
            if f.endswith('.txt')
        ])

    @staticmethod
    def render(words: List[str], grammar: Grammar) -> Optional[List[str]]:
        """Forward render F_G(ν) → Y using the grammar.

        Simple recursive rewriting. Returns None if no rule matches.
        """
        return _render_recursive(words, grammar, depth=0, max_depth=10)

    @staticmethod
    def get_vocabulary(support: List[Example]) -> set:
        """Extract all distinct words from support examples."""
        vocab = set()
        for ex in support:
            vocab.update(ex.words)
        return vocab

    @staticmethod
    def get_color_palette(support: List[Example]) -> set:
        """Extract all colors used in support outputs."""
        colors = set()
        for ex in support:
            colors.update(ex.output)
        return colors


def _render_recursive(
    words: List[str],
    grammar: Grammar,
    depth: int = 0,
    max_depth: int = 10,
    _memo: Optional[Dict] = None,
) -> Optional[List[str]]:
    """Recursive grammar-driven rendering.

    Tries noun lookup first, then pattern matching with variable binding.
    Uses memoization to avoid exponential blowup on concat fallback.
    """
    if depth > max_depth:
        return None
    if not words:
        return []

    # Memoization
    key = tuple(words)
    if _memo is None:
        _memo = {}
    if key in _memo:
        return _memo[key]

    # Single word: try noun lookup
    if len(words) == 1:
        w = words[0]
        if w in grammar.nouns:
            _memo[key] = [grammar.nouns[w]]
            return _memo[key]
        _memo[key] = None
        return None

    # Try each rule
    for pattern, template in grammar.rules:
        bindings = _match_pattern(words, pattern, grammar)
        if bindings is not None:
            result = _apply_template(template, bindings, grammar, depth, _memo)
            if result is not None:
                _memo[key] = result
                return result

    # Fallback: try concatenation (only at shallow depth, limited splits)
    if depth <= 5 and len(words) <= 6:
        for split in range(1, len(words)):
            left = words[:split]
            right = words[split:]
            lr = _render_recursive(left, grammar, depth + 1, max_depth, _memo)
            if lr is None:
                continue
            rr = _render_recursive(right, grammar, depth + 1, max_depth, _memo)
            if rr is not None:
                _memo[key] = lr + rr
                return _memo[key]

    _memo[key] = None
    return None


def _match_pattern(
    words: List[str],
    pattern: List[str],
    grammar: Grammar,
) -> Optional[Dict[str, List[str]]]:
    """Try to match words against a grammar pattern.

    Variables: u1, u2, x1, x2 — bind to subsequences.
    Literal words must match exactly.
    """
    if not pattern:
        return {} if not words else None

    bindings: Dict[str, List[str]] = {}

    # Simple patterns only: literal + one or two variables
    # For v1, handle common 2-token and 3-token patterns
    literals = [(i, p) for i, p in enumerate(pattern)
                if not p.startswith(('u', 'x'))]
    variables = [(i, p) for i, p in enumerate(pattern)
                 if p.startswith(('u', 'x'))]

    if len(pattern) == 2 and len(variables) == 1 and len(literals) == 1:
        # Pattern: literal + var  OR  var + literal
        lit_idx, lit_word = literals[0]
        var_idx, var_name = variables[0]

        if lit_idx == 1 and words[-1] == lit_word:
            bindings[var_name] = words[:-1]
            return bindings if bindings[var_name] else None
        elif lit_idx == 0 and words[0] == lit_word:
            bindings[var_name] = words[1:]
            return bindings if bindings[var_name] else None
        elif lit_word in words:
            # Literal is an operator in the middle
            for i, w in enumerate(words):
                if w == lit_word:
                    if var_idx == 0:
                        bindings[var_name] = words[:i]
                    else:
                        bindings[var_name] = words[i+1:]
                    if bindings[var_name]:
                        return bindings
        return None

    elif len(pattern) == 3 and len(variables) == 2 and len(literals) == 1:
        # Pattern: var + literal + var  (most common: u1 fep u2)
        lit_idx, lit_word = literals[0]
        var1_name = variables[0][1]
        var2_name = variables[1][1]

        for i, w in enumerate(words):
            if w == lit_word and i > 0 and i < len(words) - 1:
                bindings[var1_name] = words[:i]
                bindings[var2_name] = words[i+1:]
                return bindings
        return None

    elif len(pattern) == 2 and len(variables) == 2:
        # Pattern: var var (concatenation)
        if len(words) >= 2:
            for split in range(1, len(words)):
                bindings[variables[0][1]] = words[:split]
                bindings[variables[1][1]] = words[split:]
                return bindings
        return None

    return None


def _apply_template(
    template: List[str],
    bindings: Dict[str, List[str]],
    grammar: Grammar,
    depth: int,
    _memo: Optional[Dict] = None,
) -> Optional[List[str]]:
    """Apply a template with variable bindings to produce output."""
    result = []
    for token in template:
        if token.startswith('[') and token.endswith(']'):
            var_name = token[1:-1]
            if var_name in bindings:
                sub_result = _render_recursive(
                    bindings[var_name], grammar, depth + 1, _memo=_memo)
                if sub_result is None:
                    return None
                result.extend(sub_result)
            else:
                return None
        elif token.isupper():
            result.append(token)
        else:
            return None
    return result
