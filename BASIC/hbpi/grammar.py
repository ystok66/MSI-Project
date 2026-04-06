"""
grammar.py — AST nodes + serialization + node count

Fixed DSL for Phase-0 HBPI:
  Expr -> Prim(w)                          # leaf: emit color
  Expr -> Concat(Expr, Expr)               # concatenation
  Expr -> Unary(w, n, Expr)                # postfix repeat: w at end of span
  Expr -> Binary(w, Expr, Expr)            # infix op: w between two spans

Execution semantics (see executor.py):
  Prim(w)          -> [MAP_color(w)]
  Concat(a, b)     -> exec(a) + exec(b)
  Unary(w, n, arg) -> exec(arg) * n
  Binary(w, a, b)  -> exec(b) + exec(a)     [swap; extend later]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── AST node types ──────────────────────────────────────────────

@dataclass(frozen=True)
class Prim:
    """Leaf: a single primitive word emitting one color."""
    word: str

    @property
    def node_count(self) -> int:
        return 1

    def canonical(self) -> str:
        return f"P({self.word})"


@dataclass(frozen=True)
class Concat:
    """Concatenation of two sub-expressions."""
    left: AST
    right: AST

    @property
    def node_count(self) -> int:
        return 1 + self.left.node_count + self.right.node_count

    def canonical(self) -> str:
        return f"C({self.left.canonical()},{self.right.canonical()})"


@dataclass(frozen=True)
class Unary:
    """Postfix unary operator: repeat the argument n times.
    The operator word appears at the END of the span."""
    op_word: str
    repeat_n: int
    arg: AST

    @property
    def node_count(self) -> int:
        return 1 + self.arg.node_count

    def canonical(self) -> str:
        return f"U({self.op_word},{self.repeat_n},{self.arg.canonical()})"


# Binary modes
BINARY_MODES = ['swap', 'concat']  # swap=R+L, concat=L+R
N_BINARY_MODES = len(BINARY_MODES)


@dataclass(frozen=True)
class Binary:
    """Infix binary operator: the operator word sits BETWEEN two sub-expressions.
    binary_mode: 'swap' (emit R then L) or 'concat' (emit L then R)."""
    op_word: str
    binary_mode: str    # 'swap' or 'concat'
    left: AST
    right: AST

    @property
    def node_count(self) -> int:
        return 1 + self.left.node_count + self.right.node_count

    def canonical(self) -> str:
        return f"B({self.op_word},{self.binary_mode},{self.left.canonical()},{self.right.canonical()})"


# Union type for convenience
AST = Prim | Concat | Unary | Binary
