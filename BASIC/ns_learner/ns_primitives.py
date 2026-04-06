"""
ns_primitives.py — Probabilistic Stack Machine engine.

Defines the execution substrate for BPL program traces.
5 primitives operate on an immutable stack of color-vector sequences:

  EMIT          push [μ]              — leaf noun emitting one color
  REPEAT        pop_n(arity) → X, push X*k  — postfix unary repetition
  SWAP_INFIX    pop_n(arity) as A, consume B → push B,A  — infix binary swap
  CONCAT_INFIX  pop_n(arity) as A, consume B → push A,B  — infix binary concat
  OVER_INFIX    pop_n(arity) as A, consume B → push A,B,A — infix binary surround

Key BPL property: `arity` (how many stack items to bind) is a LATENT
VARIABLE. This corresponds to MLC's variable x₁ binding to expressions
of arbitrary length.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import numpy as np

# Type alias
Vector = np.ndarray  # shape (d,)


# ── Stack Data Structures ───────────────────────────────────────

@dataclass(frozen=True)
class StackItem:
    """Immutable sequence of color vectors (one logical unit on the stack)."""
    content: Tuple[Vector, ...]  # use tuple for immutability

    @staticmethod
    def from_list(vecs: List[Vector]) -> 'StackItem':
        return StackItem(content=tuple(vecs))

    @staticmethod
    def single(vec: Vector) -> 'StackItem':
        return StackItem(content=(vec,))

    def repeat(self, k: int) -> 'StackItem':
        """Return a new StackItem with content repeated k times."""
        return StackItem(content=self.content * k)

    def __len__(self):
        return len(self.content)


def concat_items(items: List[StackItem]) -> StackItem:
    """Concatenate multiple StackItems into one (preserving order)."""
    vecs = []
    for item in items:
        vecs.extend(item.content)
    return StackItem.from_list(vecs)


class StackState:
    """
    Immutable execution state — a stack of StackItems.
    
    All operations return new StackState instances (functional style)
    to support beam search branching without mutation.
    """

    __slots__ = ('_stack',)

    def __init__(self, stack: Tuple[StackItem, ...] = ()):
        self._stack = stack

    def push(self, item: StackItem) -> 'StackState':
        """Push item on top, return new state."""
        return StackState(self._stack + (item,))

    def pop(self) -> Tuple[Optional[StackItem], 'StackState']:
        """Pop top item. Returns (item, new_state) or (None, self) if empty."""
        if not self._stack:
            return None, self
        return self._stack[-1], StackState(self._stack[:-1])

    def pop_n(self, n: int) -> Tuple[Optional[StackItem], 'StackState']:
        """
        Pop top n items, concatenate them in bottom-to-top order
        into a single StackItem.
        
        Returns (merged_item, remaining_state) or (None, self) if
        fewer than n items on stack.
        
        This is the key mechanism for variable-length expression binding:
        arity=n means "bind the top n stack items as one expression".
        """
        if n <= 0 or len(self._stack) < n:
            return None, self
        if n == 1:
            return self.pop()
        # Take top n items (they're in bottom-to-top order in _stack)
        taken = self._stack[-n:]  # bottom-to-top slice
        remaining = self._stack[:-n]
        merged = concat_items(list(taken))
        return merged, StackState(remaining)

    def peek(self) -> Optional[StackItem]:
        """Look at top without popping."""
        return self._stack[-1] if self._stack else None

    def flatten(self) -> List[Vector]:
        """Flatten entire stack bottom-to-top into output sequence."""
        result = []
        for item in self._stack:
            result.extend(item.content)
        return result

    @property
    def depth(self) -> int:
        return len(self._stack)

    @property 
    def total_length(self) -> int:
        """Total number of vectors across all stack items."""
        return sum(len(item) for item in self._stack)

    def __repr__(self):
        items = [f"[{len(it)}v]" for it in self._stack]
        return f"Stack({' '.join(items)})"


# ── Primitive Base ──────────────────────────────────────────────

class Primitive(ABC):
    """Abstract base for stack machine primitives."""
    name: str = "BASE"
    consumes_next: bool = False  # If True, consumes next token from instruction
    uses_arity: bool = False     # If True, has variable arity (expression binding)

    @abstractmethod
    def execute(self, state: StackState, param: Any,
                arity: int = 1) -> Optional[StackState]:
        """
        Execute this primitive on the given stack state.
        
        Args:
            state: current stack
            param: primitive-specific parameter
                   EMIT: Vector (color vec to push)
                   REPEAT: int (repetition count k)
                   SWAP/CONCAT/OVER: Vector (next word's color vec B)
            arity: how many stack items to bind as the expression
                   (only meaningful when uses_arity=True)
        
        Returns:
            New StackState, or None if execution is invalid.
        """
        raise NotImplementedError


# ── Concrete Primitives ────────────────────────────────────────

class PrimEmit(Primitive):
    """
    EMIT: Push a single color vector onto the stack.
    
    Stack effect: ... → ... [μ]
    MLC analogue: dax → BLUE (leaf noun)
    """
    name = "EMIT"
    consumes_next = False
    uses_arity = False

    def execute(self, state: StackState, param: Vector,
                arity: int = 1) -> Optional[StackState]:
        return state.push(StackItem.single(param))


class PrimRepeat(Primitive):
    """
    REPEAT: Pop top `arity` items as expression X, push X repeated k times.
    
    Stack effect: ... [A₁] [A₂] ... [Aₙ] → ... [(A₁+...+Aₙ) × k]
    MLC analogue: x1 gazzer → [x1][x1]  (k=2)
    
    `arity` determines how many stack items form the expression x1.
    This is a LATENT VARIABLE inferred by beam search.
    """
    name = "REPEAT"
    consumes_next = False
    uses_arity = True

    def execute(self, state: StackState, param: int,
                arity: int = 1) -> Optional[StackState]:
        k = param
        if k < 1:
            return None
        expr, rest = state.pop_n(arity)
        if expr is None:
            return None
        return rest.push(expr.repeat(k))


class PrimSwapInfix(Primitive):
    """
    SWAP_INFIX: Pop `arity` items as A, consume next word B, push B then A.
    
    Stack effect: ... [A] + consume B → ... [B] [A]
    Output order: B, A (reversed)
    MLC analogue: x1 kiki u1 → [u1][x1]
    """
    name = "SWAP_INFIX"
    consumes_next = True
    uses_arity = True

    def execute(self, state: StackState, param: Vector,
                arity: int = 1) -> Optional[StackState]:
        b_vec = param
        expr_a, rest = state.pop_n(arity)
        if expr_a is None:
            return None
        b_item = StackItem.single(b_vec)
        return rest.push(b_item).push(expr_a)


class PrimConcatInfix(Primitive):
    """
    CONCAT_INFIX: Pop `arity` items as A, consume next word B, push A then B.
    
    Stack effect: ... [A] + consume B → ... [A] [B]
    Output order: A, B (preserved — identity concat)
    MLC analogue: u1 fep u2 → [u1][u2]
    """
    name = "CONCAT_INFIX"
    consumes_next = True
    uses_arity = True

    def execute(self, state: StackState, param: Vector,
                arity: int = 1) -> Optional[StackState]:
        b_vec = param
        expr_a, rest = state.pop_n(arity)
        if expr_a is None:
            return None
        b_item = StackItem.single(b_vec)
        return rest.push(expr_a).push(b_item)


class PrimOverInfix(Primitive):
    """
    OVER_INFIX: Pop `arity` items as A, consume next word B, push A, B, A.
    
    Stack effect: ... [A] + consume B → ... [A] [B] [A]
    Output order: A, B, A (surround)
    MLC analogue: u1 surround u2 → [u1][u2][u1]
    """
    name = "OVER_INFIX"
    consumes_next = True
    uses_arity = True

    def execute(self, state: StackState, param: Vector,
                arity: int = 1) -> Optional[StackState]:
        b_vec = param
        expr_a, rest = state.pop_n(arity)
        if expr_a is None:
            return None
        b_item = StackItem.single(b_vec)
        return rest.push(expr_a).push(b_item).push(expr_a)


# ── Registry ───────────────────────────────────────────────────

PRIMITIVES = {
    'EMIT':          PrimEmit(),
    'REPEAT':        PrimRepeat(),
    'SWAP_INFIX':    PrimSwapInfix(),
    'CONCAT_INFIX':  PrimConcatInfix(),
    'OVER_INFIX':    PrimOverInfix(),
}

ROLES = list(PRIMITIVES.keys())
N_ROLES = len(ROLES)

# Max arity to enumerate in beam search (caps branching factor)
MAX_ARITY = 6

# Convenience: which roles consume the next instruction token
INFIX_ROLES = [r for r, p in PRIMITIVES.items() if p.consumes_next]
