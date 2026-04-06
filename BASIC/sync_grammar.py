"""
SyncGrammar v11e: Perceptual Chunking + Abductive Solver (Audit-Softened)

Addresses ALL 6 audit items:
 1. Soft abduction: chunk_score threshold replaces hard == in solve_*
 2. Weber's law: count noise σ = k·count + σ0 (not fixed)
 3. Soft scope: full span search + distance cost (no ±3 hard window)
 4. Direction storage: DiscoveredRule stores prefix/postfix
 5. Proper EM: batch accumulate then update (not online)
 6. Primitive distributions: P(color|word) via Counter, not hard map
"""

import os
import re
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from itertools import groupby


# ===================================================================
# 1. Perceptual Chunking (RLE)
# ===================================================================

@dataclass(frozen=True)
class Chunk:
    color: str
    count: int

    def __repr__(self):
        return f"{self.count}x{self.color}"


def compress(colors: List[str]) -> List[Chunk]:
    """RLE: [red, red, blue] -> [2xred, 1xblue]"""
    if not colors:
        return []
    chunks = []
    for color, group in groupby(colors):
        chunks.append(Chunk(color, sum(1 for _ in group)))
    return chunks


def decompress(chunks: List[Chunk]) -> List[str]:
    """Inverse RLE: [2xred, 1xblue] -> [red, red, blue]"""
    out = []
    for c in chunks:
        out.extend([c.color] * c.count)
    return out


def merge_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Merge adjacent same-color chunks: [1xR, 2xR, 1xB] -> [3xR, 1xB]"""
    if not chunks:
        return []
    merged = [Chunk(chunks[0].color, chunks[0].count)]
    for c in chunks[1:]:
        if c.color == merged[-1].color:
            merged[-1] = Chunk(c.color, merged[-1].count + c.count)
        else:
            merged.append(c)
    return merged


# ===================================================================
# 2. Weber-Law Chunk Scoring  [FIX #2]
# ===================================================================

WEBER_K = 0.3     # σ grows with count (Weber fraction)
WEBER_S0 = 0.5    # baseline σ for count=0

def chunk_score(pred: List[Chunk], target: List[Chunk]) -> float:
    """Weber-law scoring: σ = k·count + σ0, log-likelihood shape."""
    if len(pred) != len(target):
        return -5.0 * abs(len(pred) - len(target))
    score = 0.0
    for p, t in zip(pred, target):
        if p.color != t.color:
            score -= 10.0
        else:
            diff = abs(p.count - t.count)
            sigma = WEBER_K * t.count + WEBER_S0
            score -= (diff * diff) / (2.0 * sigma * sigma)
    return score


# ===================================================================
# 3. Abductive Solver (soft matching)  [FIX #1]
# ===================================================================

@dataclass
class DiscoveredRule:
    word: str
    arity: int            # 1=unary, 2=binary
    op_type: str          # 'seq_repeat', 'compose'
    position: str = 'any' # 'prefix', 'postfix', 'any'  [FIX #4]
    factor: int = 1       # for seq_repeat
    a: int = 1            # for compose: left_count
    b: int = 1            # for compose: right_count
    swapped: bool = False # for compose: right before left
    score: float = 0.0
    usage: int = 0

    def __repr__(self):
        pos = f"[{self.position}]" if self.position != 'any' else ''
        if self.op_type == 'seq_repeat':
            return f"Rule({self.word}:×{self.factor}{pos} s={self.score:.1f})"
        order = "R+L" if self.swapped else "L+R"
        return f"Rule({self.word}:{self.a}L{self.b}R({order}){pos} s={self.score:.1f})"


# Soft abduction threshold: accept rules that score above this
ABDUCTION_THRESHOLD = -0.5


class AbductiveSolver:
    """Discovers rules via soft chunk_score matching (not hard ==)."""

    @staticmethod
    def solve_unary(source: List[Chunk], target: List[Chunk]):
        """Discover unary ops that transform source → target (soft match)."""
        if not source:
            return []
        total_src = sum(c.count for c in source)
        total_tgt = sum(c.count for c in target)
        if total_src == 0:
            return []

        results = []
        # Try: target = source repeated N times
        max_n = max(1, total_tgt // total_src + 1)
        for n in range(1, min(max_n + 1, 8)):
            predicted = merge_chunks(list(source) * n)
            s = chunk_score(predicted, target)
            if s >= ABDUCTION_THRESHOLD:  # [FIX #1] soft match
                results.append(('seq_repeat', n, s))
        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return [(r[0], r[1]) for r in results]

    @staticmethod
    def solve_binary(left: List[Chunk], right: List[Chunk],
                     target: List[Chunk]):
        """Discover binary ops (soft match)."""
        results = []
        total_left = sum(c.count for c in left) if left else 0
        total_right = sum(c.count for c in right) if right else 0
        total_tgt = sum(c.count for c in target)

        if total_left == 0 and total_right == 0:
            return results

        max_a = (total_tgt // total_left + 1) if total_left > 0 else 1
        max_b = (total_tgt // total_right + 1) if total_right > 0 else 1

        for a in range(0, min(max_a + 1, 5)):
            for b in range(0, min(max_b + 1, 5)):
                if a == 0 and b == 0:
                    continue
                # Normal order: left*a + right*b
                pred = merge_chunks(list(left) * a + list(right) * b)
                s = chunk_score(pred, target)
                if s >= ABDUCTION_THRESHOLD:
                    results.append(('compose', a, b, False, s))
                # Swapped order: right*b + left*a
                if a > 0 and b > 0:
                    pred_swap = merge_chunks(
                        list(right) * b + list(left) * a)
                    s2 = chunk_score(pred_swap, target)
                    if s2 >= ABDUCTION_THRESHOLD and s2 != s:
                        results.append(('compose', a, b, True, s2))

        results.sort(key=lambda x: x[4], reverse=True)
        return [(r[0], r[1], r[2], r[3]) for r in results]


# ===================================================================
# 4. MetaGrammar Registry (Lifelong Learning)
# ===================================================================

class MetaGrammarRegistry:
    def __init__(self, alpha: float = 1.0):
        self.global_rules: Counter = Counter()
        self.alpha = alpha
        self.n_tasks = 0

    def log_prior(self, key: tuple) -> float:
        total = sum(self.global_rules.values())
        if total == 0:
            return 0.0
        count = self.global_rules.get(key, 0)
        if count > 0:
            return np.log((count + 0.1) / (total + self.alpha))
        return np.log(self.alpha / (total + self.alpha))

    def register(self, key: tuple, count: int = 1):
        self.global_rules[key] += count

    def end_task(self):
        self.n_tasks += 1

    def __repr__(self):
        tops = self.global_rules.most_common(5)
        items = ', '.join(f'{t}:{c}' for t, c in tops)
        return f"Meta(n={self.n_tasks}, [{items}])"


# ===================================================================
# 5. Data Parsing
# ===================================================================

def parse_algebraic_file(filepath):
    support, query = [], []
    current = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '*SUPPORT*':
                current = 'support'; continue
            elif line == '*QUERY*':
                current = 'query'; continue
            elif line.startswith('*GRAMMAR*'):
                current = 'grammar'; continue
            if current in ('support', 'query') and line:
                m = re.match(r'IN:\s+(.*?)\s+OUT:\s+(.*)', line)
                if m:
                    words = m.group(1).strip().split()
                    colors = m.group(2).strip().split()
                    target = support if current == 'support' else query
                    target.append({'input': words, 'output': colors})
    return support, query


# ===================================================================
# 6. SyncGrammar Agent v11e
# ===================================================================

BEAM = 12
SCOPE_COST = 0.3   # [FIX #3] soft penalty per extra scope token


class SyncGrammarAgentV11:

    def __init__(self, meta=None, verbose=False):
        self.meta = meta or MetaGrammarRegistry()
        # [FIX #6] Primitive distributions: Counter(color -> count)
        self.prim_dist: Dict[str, Counter] = defaultdict(Counter)
        self.rules: Dict[str, DiscoveredRule] = {}
        self.verbose = verbose

    # ─── Primitive Distribution Helpers ─────────────────
    #     [FIX #6]

    def _best_color(self, word) -> Optional[str]:
        """MAP estimate of P(color|word)."""
        if word not in self.prim_dist or not self.prim_dist[word]:
            return None
        return self.prim_dist[word].most_common(1)[0][0]

    def _log_prim_score(self, word, color) -> float:
        """Log P(color|word) from the distribution."""
        if word not in self.prim_dist:
            return -20.0
        dist = self.prim_dist[word]
        total = sum(dist.values())
        if total == 0:
            return -20.0
        count = dist.get(color, 0)
        return np.log((count + 0.1) / (total + 0.1 * len(dist)))

    def _add_prim_evidence(self, word, color, weight=1):
        """Add evidence for word→color mapping."""
        self.prim_dist[word][color] += weight

    def _is_known_prim(self, word) -> bool:
        return word in self.prim_dist and len(self.prim_dist[word]) > 0

    # ─── Bootstrap Primitives ───────────────────────────

    def _bootstrap_primitives(self, examples):
        """Learn primitives from data evidence.

        Uses distribution (Counter) instead of hard map.
        """
        # Phase A: 1:1 examples (strongest evidence, weight=10)
        for ex in examples:
            if len(ex['input']) == 1 and len(ex['output']) == 1:
                self._add_prim_evidence(ex['input'][0], ex['output'][0], 10)

        # Phase B: Iterative difference inference (weight=5)
        for _ in range(10):
            changed = False
            for ex in examples:
                ws, cs = ex['input'], ex['output']
                if len(ws) != len(cs):
                    continue
                unknown = [w for w in ws if not self._is_known_prim(w)]
                if len(unknown) == 1:
                    unk = unknown[0]
                    idx = ws.index(unk)
                    self._add_prim_evidence(unk, cs[idx], 5)
                    changed = True
            if not changed:
                break

        # Phase C: Validate — downgrade (not remove) suspicious prims
        confirmed = {ex['input'][0] for ex in examples
                     if len(ex['input']) == 1 and len(ex['output']) == 1}
        for _ in range(3):
            to_downgrade = set()
            for ex in examples:
                ws, cs = ex['input'], ex['output']
                if all(self._is_known_prim(w) for w in ws) and \
                   len(ws) != len(cs):
                    for w in ws:
                        if w not in confirmed:
                            to_downgrade.add(w)
            for w in to_downgrade:
                # Reduce evidence by half instead of removing
                if w in self.prim_dist:
                    for c in self.prim_dist[w]:
                        self.prim_dist[w][c] = max(1,
                                                   self.prim_dist[w][c] // 2)
            if not to_downgrade:
                break

    # ─── Parse sub-span as primitive concat ──────────────

    def _parse_prim_span(self, words) -> Optional[List[Chunk]]:
        """Parse using best MAP primitives."""
        chunks = []
        for w in words:
            c = self._best_color(w)
            if c is None:
                return None
            chunks.append(Chunk(c, 1))
        return merge_chunks(chunks)

    # ─── Context Stripping ──────────────────────────────

    def _strip_context(self, target_chunks, ctx_before, ctx_after):
        """Remove context from target to isolate operator output."""
        target_colors = decompress(target_chunks)
        before_colors = decompress(ctx_before) if ctx_before else []
        after_colors = decompress(ctx_after) if ctx_after else []

        if len(target_colors) < len(before_colors) + len(after_colors):
            return None

        for i, c in enumerate(before_colors):
            if i >= len(target_colors) or target_colors[i] != c:
                return None

        for i, c in enumerate(reversed(after_colors)):
            idx = len(target_colors) - 1 - i
            if idx < 0 or target_colors[idx] != c:
                return None

        start = len(before_colors)
        end = len(target_colors) - len(after_colors)
        if start > end:
            return None
        return compress(target_colors[start:end])

    # ─── Abductive Rule Discovery [FIX #3, #4] ─────────

    def _discover_rules(self, examples):
        """Discover rules: full-span search + distance cost + direction."""
        for _ in range(5):
            found_new = False
            for ex in examples:
                ws = ex['input']
                target_chunks = compress(ex['output'])

                for op_idx in range(len(ws)):
                    op_word = ws[op_idx]
                    if op_word in self.rules:
                        continue

                    best_rule = None
                    best_score = ABDUCTION_THRESHOLD

                    # === Unary postfix: [arg_start:op_idx] OP ===
                    # [FIX #3] Try all starts (0..op_idx-1), not ±3
                    for arg_start in range(op_idx):
                        arg_words = ws[arg_start:op_idx]
                        arg_chunks = self._parse_prim_span(arg_words)
                        if arg_chunks is None:
                            continue

                        ctx_b = ws[:arg_start]
                        ctx_a = ws[op_idx + 1:]
                        ctx_b_ch = self._parse_prim_span(ctx_b)
                        ctx_a_ch = self._parse_prim_span(ctx_a)
                        if (ctx_b and ctx_b_ch is None) or \
                           (ctx_a and ctx_a_ch is None):
                            continue

                        op_result = self._strip_context(
                            target_chunks, ctx_b_ch or [], ctx_a_ch or [])
                        if op_result is None:
                            continue

                        sols = AbductiveSolver.solve_unary(
                            arg_chunks, op_result)
                        # [FIX #3] scope cost: penalize large scopes
                        scope_len = op_idx - arg_start
                        scope_penalty = -SCOPE_COST * max(0, scope_len - 1)
                        for sol in sols:
                            cand = DiscoveredRule(
                                word=op_word, arity=1,
                                op_type=sol[0], factor=sol[1],
                                position='postfix',  # [FIX #4]
                                score=5.0 + scope_penalty)
                            if cand.score > best_score:
                                best_score = cand.score
                                best_rule = cand

                    # === Unary prefix: OP [op_idx+1:arg_end] ===
                    for arg_end in range(op_idx + 2, len(ws) + 1):
                        arg_words = ws[op_idx + 1:arg_end]
                        arg_chunks = self._parse_prim_span(arg_words)
                        if arg_chunks is None:
                            continue

                        ctx_b = ws[:op_idx]
                        ctx_a = ws[arg_end:]
                        ctx_b_ch = self._parse_prim_span(ctx_b)
                        ctx_a_ch = self._parse_prim_span(ctx_a)
                        if (ctx_b and ctx_b_ch is None) or \
                           (ctx_a and ctx_a_ch is None):
                            continue

                        op_result = self._strip_context(
                            target_chunks,
                            ctx_b_ch or [], ctx_a_ch or [])
                        if op_result is None:
                            continue

                        sols = AbductiveSolver.solve_unary(
                            arg_chunks, op_result)
                        scope_len = arg_end - op_idx - 1
                        scope_penalty = -SCOPE_COST * max(0, scope_len - 1)
                        for sol in sols:
                            cand = DiscoveredRule(
                                word=op_word, arity=1,
                                op_type=sol[0], factor=sol[1],
                                position='prefix',  # [FIX #4]
                                score=5.0 + scope_penalty)
                            if cand.score > best_score:
                                best_score = cand.score
                                best_rule = cand

                    # === Binary: [ls:op_idx] OP [op_idx+1:re] ===
                    # [FIX #3] Full span, not ±3 window
                    for ls in range(0, op_idx):
                        left_w = ws[ls:op_idx]
                        left_ch = self._parse_prim_span(left_w)
                        if left_ch is None:
                            continue
                        for re_ in range(op_idx + 2, len(ws) + 1):
                            right_w = ws[op_idx + 1:re_]
                            right_ch = self._parse_prim_span(right_w)
                            if right_ch is None:
                                continue

                            ctx_b = ws[:ls]
                            ctx_a = ws[re_:]
                            ctx_b_ch = self._parse_prim_span(ctx_b)
                            ctx_a_ch = self._parse_prim_span(ctx_a)
                            if (ctx_b and ctx_b_ch is None) or \
                               (ctx_a and ctx_a_ch is None):
                                continue

                            op_result = self._strip_context(
                                target_chunks,
                                ctx_b_ch or [], ctx_a_ch or [])
                            if op_result is None:
                                continue

                            sols = AbductiveSolver.solve_binary(
                                left_ch, right_ch, op_result)
                            total_scope = (op_idx - ls) + (re_ - op_idx - 1)
                            scope_penalty = -SCOPE_COST * max(0, total_scope - 2)
                            for sol in sols:
                                cand = DiscoveredRule(
                                    word=op_word, arity=2,
                                    op_type=sol[0], a=sol[1], b=sol[2],
                                    swapped=sol[3],
                                    position='infix',
                                    score=5.0 + scope_penalty)
                                if cand.score > best_score:
                                    best_score = cand.score
                                    best_rule = cand

                    # Commit best rule found for this word
                    if best_rule is not None:
                        self.rules[op_word] = best_rule
                        found_new = True

            # After discovering rules, try to infer more primitives
            found_new |= self._infer_prims_from_rules(examples)

            if not found_new:
                break

    def _infer_prims_from_rules(self, examples) -> bool:
        """After discovering rules, use CYK to infer remaining primitives."""
        found = False
        for ex in examples:
            ws, cs = ex['input'], ex['output']
            unknown = [w for w in ws
                       if not self._is_known_prim(w) and w not in self.rules]
            if len(unknown) != 1:
                continue
            unk = unknown[0]
            target_chunks = compress(cs)

            best_color, best_s = None, -1e9
            for color in set(cs):
                old_dist = dict(self.prim_dist.get(unk, Counter()))
                self._add_prim_evidence(unk, color, 5)
                result = self._cyk_parse(ws, target_chunks=target_chunks)
                if result is not None:
                    trace, score, hyps = result
                    s = chunk_score(trace, target_chunks)
                    if s > best_s:
                        best_s = s
                        best_color = color
                # Restore
                self.prim_dist[unk] = Counter(old_dist)

            if best_color is not None and best_s >= -0.5:
                self._add_prim_evidence(unk, best_color, 5)
                found = True
        return found

    # ─── Apply discovered rule ──────────────────────────

    def _apply_rule(self, rule: DiscoveredRule,
                    *traces: List[Chunk]) -> Optional[List[Chunk]]:
        """Apply a discovered rule to input chunk trace(s)."""
        if rule.op_type == 'seq_repeat':
            src = list(traces[0])
            return merge_chunks(src * rule.factor)

        elif rule.op_type == 'compose':
            left = list(traces[0]) if len(traces) > 0 else []
            right = list(traces[1]) if len(traces) > 1 else []
            if rule.swapped:
                return merge_chunks(right * rule.b + left * rule.a)
            else:
                return merge_chunks(left * rule.a + right * rule.b)

        return None

    # ─── CYK Chart Parser [FIX #4: direction-aware] ────

    def _cyk_parse(self, tokens, target_chunks=None):
        """Span-based CYK with direction-aware operator application."""
        n = len(tokens)
        chart = [[[] for _ in range(n + 1)] for _ in range(n + 1)]

        # === Base case: single tokens ===
        # [FIX #6] Use log P(color|word) as base score
        for i in range(n):
            w = tokens[i]
            if self._is_known_prim(w):
                color = self._best_color(w)
                trace = [Chunk(color, 1)]
                base_score = 10.0 + self._log_prim_score(w, color)
                chart[i][i + 1].append((trace, base_score, []))

        # === Recursive: spans of length 2 to n ===
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                entries = []

                # --- A. Pure concatenation ---
                for k in range(i + 1, j):
                    for lt, ls, lh in chart[i][k]:
                        for rt, rs, rh in chart[k][j]:
                            merged = merge_chunks(list(lt) + list(rt))
                            entries.append((
                                merged, ls + rs - 0.1, lh + rh))

                # --- B. Operator at each position k ---
                for k in range(i, j):
                    w = tokens[k]
                    if w not in self.rules:
                        continue
                    rule = self.rules[w]

                    if rule.arity == 1:
                        # [FIX #4] Only apply in discovered direction
                        # POSTFIX: left=[i:k], op=k
                        if rule.position in ('postfix', 'any') and k > i:
                            for lt, ls, lh in chart[i][k]:
                                applied = self._apply_rule(rule, lt)
                                if applied is None:
                                    continue
                                prior = self.meta.log_prior(
                                    self._rule_key(rule))
                                if k + 1 < j:
                                    for rt, rs, rh in chart[k+1][j]:
                                        m = merge_chunks(
                                            list(applied) + list(rt))
                                        entries.append((
                                            m, ls + rs + rule.score + prior,
                                            lh + [rule] + rh))
                                else:
                                    entries.append((
                                        applied, ls + rule.score + prior,
                                        lh + [rule]))

                        # PREFIX: op=k, right=[k+1:j]
                        if rule.position in ('prefix', 'any') and k + 1 < j:
                            for rt, rs, rh in chart[k+1][j]:
                                applied = self._apply_rule(rule, rt)
                                if applied is None:
                                    continue
                                prior = self.meta.log_prior(
                                    self._rule_key(rule))
                                if k > i:
                                    for lt, ls, lh in chart[i][k]:
                                        m = merge_chunks(
                                            list(lt) + list(applied))
                                        entries.append((
                                            m, ls + rs + rule.score + prior,
                                            lh + [rule] + rh))
                                else:
                                    entries.append((
                                        applied, rs + rule.score + prior,
                                        [rule] + rh))

                    elif rule.arity == 2:
                        # BINARY (infix)
                        if k > i and k + 1 < j:
                            for lt, ls, lh in chart[i][k]:
                                for rt, rs, rh in chart[k+1][j]:
                                    applied = self._apply_rule(
                                        rule, lt, rt)
                                    if applied is None:
                                        continue
                                    prior = self.meta.log_prior(
                                        self._rule_key(rule))
                                    entries.append((
                                        applied,
                                        ls + rs + rule.score + prior,
                                        lh + [rule] + rh))

                # --- Beam: diversity-aware pruning ---
                if target_chunks is not None and len(entries) > BEAM:
                    by_len = defaultdict(list)
                    for e in entries:
                        by_len[len(e[0])].append(e)
                    kept = []
                    for ln in sorted(by_len.keys()):
                        grp = sorted(by_len[ln],
                                     key=lambda x: x[1], reverse=True)
                        kept.extend(grp[:4])
                    kept.sort(key=lambda x: x[1], reverse=True)
                    chart[i][j] = kept[:BEAM * 2]
                else:
                    entries.sort(key=lambda x: x[1], reverse=True)
                    chart[i][j] = entries[:BEAM]

        # === Selection ===
        final = chart[0][n]
        if not final:
            return None

        if target_chunks is not None:
            best, best_score = None, -1e9
            for trace, score, hyps in final:
                match = chunk_score(trace, target_chunks) + score
                if match > best_score:
                    best_score = match
                    best = (trace, score, hyps)
            return best
        else:
            return max(final, key=lambda x: x[1])

    def _rule_key(self, rule):
        if rule.op_type == 'seq_repeat':
            return ('rep', rule.factor)
        return ('comp', rule.a, rule.b, rule.swapped)

    # ─── Proper Batch EM [FIX #5] ──────────────────────

    def _em_refine(self, examples):
        """Batch EM: accumulate usage across ALL examples, then update."""
        # E-step: parse all examples, accumulate rule usage
        usage_counts: Dict[str, int] = defaultdict(int)  # rule.word -> count
        n_ok = 0
        for ex in examples:
            tc = compress(ex['output'])
            result = self._cyk_parse(ex['input'], target_chunks=tc)
            if result is not None:
                trace, score, hyps = result
                if chunk_score(trace, tc) >= -1.0:
                    n_ok += 1
                    for h in hyps:
                        if isinstance(h, DiscoveredRule):
                            usage_counts[h.word] += 1

        # M-step: batch update all rules at once
        eta = 0.5
        for word, count in usage_counts.items():
            if word in self.rules:
                self.rules[word].usage += count
                self.rules[word].score += eta * count

        return n_ok

    # ─── Learn ──────────────────────────────────────────

    def learn(self, examples):
        self._bootstrap_primitives(examples)
        self._discover_rules(examples)

        # EM iterations (proper batch)
        for it in range(3):
            n_ok = self._em_refine(examples)
            if self.verbose and it == 0:
                print(f"  EM iter 0: {n_ok}/{len(examples)} matched")

        # Register with meta-grammar
        for w, rule in self.rules.items():
            if rule.usage > 0:
                self.meta.register(self._rule_key(rule), rule.usage)

        if self.verbose:
            prims = {w: self._best_color(w) for w in self.prim_dist}
            print(f"  Prims: {prims}")
            print(f"  Rules: {self.rules}")
            print(f"  {self.meta}")

    # ─── Predict ────────────────────────────────────────

    def predict(self, words):
        result = self._cyk_parse(list(words))
        if result is not None:
            trace, score, hyps = result
            return decompress(trace)
        # Fallback: primitives only
        out = []
        for w in words:
            c = self._best_color(w)
            if c is not None:
                out.append(c)
            else:
                out.append('?')
        return out


# ===================================================================
# 7. Evaluation
# ===================================================================

def evaluate_task(filepath, meta=None, verbose=False):
    support, query = parse_algebraic_file(filepath)
    fname = os.path.basename(filepath)
    if not support or not query:
        return None

    agent = SyncGrammarAgentV11(meta=meta, verbose=verbose)
    agent.learn(support)

    correct, total = 0, len(query)
    for qi, q in enumerate(query):
        pred = agent.predict(q['input'])
        ok = (pred == q['output'])
        if ok:
            correct += 1
        if verbose:
            mark = "OK  " if ok else "FAIL"
            inp = ' '.join(q['input'])
            print(f"  [{qi}] {mark} '{inp}'")
            if not ok:
                print(f"        got: {' '.join(str(c) for c in pred)}")
                print(f"        exp: {' '.join(q['output'])}")

    acc = correct / total if total > 0 else 0
    return {'file': fname, 'accuracy': acc, 'correct': correct, 'total': total}


def run_comparison(data_dir, n_tasks=10, lifelong=False):
    files = sorted(f for f in os.listdir(data_dir)
                   if f.endswith('.txt'))[:n_tasks]
    mode = "Lifelong" if lifelong else "Independent"
    print(f"\n{'='*60}")
    print(f"SyncGrammar v11e Evaluation ({n_tasks} tasks, {mode})")
    print(f"{'='*60}")

    meta = MetaGrammarRegistry() if lifelong else None
    results = []
    for fn in files:
        fp = os.path.join(data_dir, fn)
        task_meta = meta if lifelong else MetaGrammarRegistry()
        r = evaluate_task(fp, meta=task_meta, verbose=False)
        if r:
            results.append(r)
            print(f"  {fn}: {r['correct']}/{r['total']}"
                  f" = {r['accuracy']*100:.1f}%")
        if lifelong and meta:
            meta.end_task()

    tc = sum(r['correct'] for r in results)
    tt = sum(r['total'] for r in results)
    print(f"\n  Overall: {tc}/{tt} = {tc/tt*100:.1f}%")
    if lifelong and meta:
        print(f"  {meta}")
    return results


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'MLC', 'data_algebraic', 'data_algebraic', 'train')
    run_comparison(base, n_tasks=50, lifelong=False)
    run_comparison(base, n_tasks=50, lifelong=True)
