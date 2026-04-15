"""
generator.py — Task, candidate pool, and danger model generation.

Colors for the candidate pool come exclusively from the grammar's palette.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import os
import re
import numpy as np

from ..interfaces import CandidateBall, Example
from ..config import EnvConfig


# ── Danger Model ───────────────────────────────────────────────

@dataclass
class DangerModel:
    """Gaussian prototype model for safe/danger type classification.

    Each episode generates new prototypes; balls get vectors sampled
    from the corresponding cluster.
    """
    danger_dim: int
    n_safe_types: int
    n_danger_types: int
    cluster_sigma: float
    # Prototypes: index 0..n_safe-1 = safe, n_safe..n_safe+n_danger-1 = danger
    prototypes: np.ndarray = field(default=None)  # (n_types, danger_dim)

    @property
    def n_types(self) -> int:
        return self.n_safe_types + self.n_danger_types

    def is_danger_type(self, type_idx: int) -> bool:
        return type_idx >= self.n_safe_types


def generate_danger_model(
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> DangerModel:
    """Generate a danger model with well-separated Gaussian prototypes.

    Safe prototypes are clustered near one region; danger prototypes
    are in distinct other regions. Separation is ensured by normalizing
    and scaling.
    """
    m = cfg.danger_dim
    n_types = cfg.n_safe_types + cfg.n_danger_types

    # Generate random prototypes
    raw = rng.standard_normal((n_types, m))
    # Normalize and scale for good separation
    norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    prototypes = raw / norms * 2.0  # unit-sphere scaled by 2

    model = DangerModel(
        danger_dim=m,
        n_safe_types=cfg.n_safe_types,
        n_danger_types=cfg.n_danger_types,
        cluster_sigma=cfg.cluster_sigma,
        prototypes=prototypes,
    )
    return model


def sample_danger_vec(
    model: DangerModel,
    type_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a danger vector from the given type's cluster."""
    mu = model.prototypes[type_idx]
    return mu + model.cluster_sigma * rng.standard_normal(model.danger_dim)


# ── Candidate Pool Generation ─────────────────────────────────

def generate_candidate_pool(
    grammar_colors: List[str],
    target_output: List[str],
    n_candidates: int,
    danger_model: DangerModel,
    cfg: EnvConfig,
    rng: np.random.Generator,
) -> List[CandidateBall]:
    """Generate a candidate pool of balls.

    Colors are drawn from the grammar's color palette (not arbitrary).
    Each ball is assigned safe/danger type based on danger_ratio.
    Observed vectors have added observation noise.

    Args:
        grammar_colors: all colors in this grammar's palette
        target_output: Y* — to bias coverage toward needed colors
        n_candidates: how many balls to generate
        danger_model: prototypes for safe/danger
        cfg: environment config
        rng: random generator

    Returns:
        List of CandidateBall objects.
    """
    balls = []
    for i in range(n_candidates):
        # Color: random from grammar palette
        color = rng.choice(grammar_colors)

        # Danger assignment
        is_danger = rng.random() < cfg.danger_ratio
        if is_danger:
            # Random danger type
            danger_type_offset = rng.integers(0, cfg.n_danger_types)
            type_idx = cfg.n_safe_types + danger_type_offset
            danger_type = danger_type_offset + 1  # 1-indexed for danger
        else:
            # Random safe type
            type_idx = rng.integers(0, cfg.n_safe_types)
            danger_type = 0

        # Sample danger vector
        danger_vec = sample_danger_vec(danger_model, type_idx, rng)

        # Add observation noise for learner
        observed_vec = danger_vec + cfg.obs_sigma * rng.standard_normal(cfg.danger_dim)

        balls.append(CandidateBall(
            index=i,
            color=color,
            danger_vec=danger_vec,
            observed_vec=observed_vec,
            is_danger=is_danger,
            danger_type=danger_type,
        ))

    return balls


# ── Task File Parsing ──────────────────────────────────────────

@dataclass
class Grammar:
    """Parsed grammar with nouns and rules."""
    nouns: Dict[str, str]                    # word → COLOR
    rules: List[Tuple[List[str], List[str]]]  # (pattern, template)
    raw_text: str = ""

    @property
    def colors(self) -> List[str]:
        """All colors that appear in this grammar."""
        color_set = set(self.nouns.values())
        return sorted(color_set)


def parse_task_file(path: str) -> Tuple[List[Example], List[Example], Grammar]:
    """Parse a CLS data file into (support, query, grammar).

    Reuses the same format as BASIC/cls_learner/data/*.txt.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    sections: Dict[str, List[str]] = {}
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
    nouns: Dict[str, str] = {}
    rules: List[Tuple[List[str], List[str]]] = []
    for line in sections.get('GRAMMAR', []):
        line = line.strip()
        if '->' not in line:
            continue
        lhs, rhs = line.split('->', 1)
        lhs_tokens = lhs.strip().split()
        rhs_tokens = rhs.strip().split()
        if (len(lhs_tokens) == 1
                and len(rhs_tokens) == 1
                and rhs_tokens[0].isupper()
                and not lhs_tokens[0].startswith(('[', 'u', 'x'))):
            nouns[lhs_tokens[0]] = rhs_tokens[0]
        else:
            rules.append((lhs_tokens, rhs_tokens))

    grammar = Grammar(nouns=nouns, rules=rules, raw_text=text)
    return support, query, grammar


def render_with_grammar(words: List[str], grammar: Grammar) -> Optional[List[str]]:
    """Forward render F_G(ν) → Y using the grammar.

    Simple recursive rewriting (borrowed from cls_option_tutor/grammar/task_adapter.py).
    """
    return _render_recursive(words, grammar, depth=0, max_depth=10)


def _render_recursive(
    words: List[str],
    grammar: Grammar,
    depth: int = 0,
    max_depth: int = 10,
    _memo: Optional[Dict] = None,
) -> Optional[List[str]]:
    if depth > max_depth:
        return None
    if not words:
        return []

    key = tuple(words)
    if _memo is None:
        _memo = {}
    if key in _memo:
        return _memo[key]

    # Single word: noun lookup
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

    # Fallback: concatenation
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
    """Try to match words against a grammar pattern."""
    if not pattern:
        return {} if not words else None

    bindings: Dict[str, List[str]] = {}
    literals = [(i, p) for i, p in enumerate(pattern) if not p.startswith(('u', 'x'))]
    variables = [(i, p) for i, p in enumerate(pattern) if p.startswith(('u', 'x'))]

    if len(pattern) == 2 and len(variables) == 1 and len(literals) == 1:
        lit_idx, lit_word = literals[0]
        var_idx, var_name = variables[0]

        if lit_idx == 1 and words[-1] == lit_word:
            bindings[var_name] = words[:-1]
            return bindings if bindings[var_name] else None
        elif lit_idx == 0 and words[0] == lit_word:
            bindings[var_name] = words[1:]
            return bindings if bindings[var_name] else None
        elif lit_word in words:
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
        if len(words) >= 2:
            for split_pt in range(1, len(words)):
                bindings[variables[0][1]] = words[:split_pt]
                bindings[variables[1][1]] = words[split_pt:]
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


# ── Task Listing ───────────────────────────────────────────────

def list_task_files(data_dir: str) -> List[str]:
    """List available task IDs from a data directory."""
    if not os.path.isdir(data_dir):
        return []
    return sorted([
        f.replace('.txt', '')
        for f in os.listdir(data_dir)
        if f.endswith('.txt')
    ])
