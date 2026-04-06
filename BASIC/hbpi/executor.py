"""
executor.py — Execute an AST to produce (predicted_seq, provenance_seq).

Execution semantics:
  Prim(w)          -> ([MAP_color(w)], [w])
  Concat(a, b)     -> exec(a) + exec(b)
  Unary(w, n, arg) -> exec(arg) * n  (repeats both pred and provenance)
  Binary(w, a, b)  -> exec(b) + exec(a)  [swap semantics]

Provenance tracks which leaf word generated each output token,
so the EM M-step can credit colors to the right word.
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .grammar import AST
    from .model import HBPIModel


def execute(ast: 'AST', model: 'HBPIModel') -> Tuple[List[str], List[str]]:
    """
    Execute an AST under the current model.

    Returns:
        pred: list of predicted color tokens
        prov: list of provenance words (same length as pred)
              prov[i] = the leaf word that generated pred[i]
    """
    from .grammar import Prim, Concat, Unary, Binary

    if isinstance(ast, Prim):
        color = model.map_color(ast.word)
        return [color], [ast.word]

    elif isinstance(ast, Concat):
        lp, lprov = execute(ast.left, model)
        rp, rprov = execute(ast.right, model)
        return lp + rp, lprov + rprov

    elif isinstance(ast, Unary):
        arg_pred, arg_prov = execute(ast.arg, model)
        n = ast.repeat_n
        return arg_pred * n, arg_prov * n

    elif isinstance(ast, Binary):
        lp, lprov = execute(ast.left, model)
        rp, rprov = execute(ast.right, model)
        if ast.binary_mode == 'swap':
            return rp + lp, rprov + lprov
        else:  # 'concat'
            return lp + rp, lprov + rprov

    else:
        raise TypeError(f"Unknown AST node: {type(ast)}")
