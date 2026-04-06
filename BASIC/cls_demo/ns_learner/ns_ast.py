"""
ns_ast.py — AST-based latent parse for compositional inference.

Builds a latent AST via beam search, then evaluates.
Handles hierarchical scoping via "deferred hole-fill":

When a hole is open, EMIT produces TWO beam branches:
  A) Fill the hole immediately (cascade)
  B) Add to roots WITHOUT filling (hole stays open)
     → Next REPEAT/INFIX can wrap this root
     → Then the wrapped result fills the pending hole

Example: "3 after DAX thrice"
  Step 1: 3 → EMIT(GREEN) → root[0]
  Step 2: after → SWAP(root[0], hole) → holes=[SWAP.right]
  Step 3: DAX →
     Branch A: fill hole → SWAP(3,DAX) root[0]
     Branch B: add root → roots=[DAX], hole still open
  Step 4 (Branch A): thrice → REPEAT(SWAP(3,DAX),k=3) ← WRONG
  Step 4 (Branch B): thrice → REPEAT(roots[-1]=DAX) →
     roots=[REPEAT(DAX,k=3)], hole still open → auto-close fills it →
     SWAP(3, REPEAT(DAX,k=3)) ← CORRECT!
"""
import copy
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

from ns_learner.ns_concept import (
    NeuroConcept, NIGParams, ROLES, REPEAT_RANGE, N_COLORS,
    COLOR_VECS, color_to_vec, vec_to_color
)
from ns_learner.ns_primitives import Vector
from ns_learner.ns_inference import (
    TraceStep, soft_edit_distance, context_role_prior, log_span_prior
)


# ── AST Node ───────────────────────────────────────────────────

@dataclass
class ASTNode:
    kind: str
    word: str
    children: List['ASTNode'] = field(default_factory=list)
    emit_vec: Optional[np.ndarray] = None
    repeat_k: Optional[int] = None

    def __repr__(self):
        if self.kind == 'EMIT':
            c = vec_to_color(self.emit_vec) if self.emit_vec is not None else '?'
            return f"{self.word}→{c}"
        elif self.kind == 'REPEAT':
            return f"REPEAT({self.children[0]},k={self.repeat_k})"
        else:
            parts = ','.join(str(c) for c in self.children if c is not None)
            return f"{self.kind}({parts})"

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children if c is not None)


# ── AST Evaluation ─────────────────────────────────────────────

def eval_ast(node: ASTNode) -> List[np.ndarray]:
    if node.kind == 'EMIT':
        return [node.emit_vec] if node.emit_vec is not None else []
    elif node.kind == 'REPEAT':
        return eval_ast(node.children[0]) * (node.repeat_k or 1)
    elif node.kind == 'SWAP':
        return eval_ast(node.children[1]) + eval_ast(node.children[0])
    elif node.kind == 'CONCAT':
        return eval_ast(node.children[0]) + eval_ast(node.children[1])
    elif node.kind == 'OVER':
        a = eval_ast(node.children[0])
        return a + eval_ast(node.children[1]) + a
    return []


# ── Parse State ────────────────────────────────────────────────

@dataclass
class Hole:
    parent: ASTNode
    slot: int

@dataclass
class ParseState:
    roots: List[ASTNode] = field(default_factory=list)
    holes: List[Hole]    = field(default_factory=list)

    def deep_copy(self) -> 'ParseState':
        old_to_new = {}
        def _cp(n):
            if n is None: return None
            nid = id(n)
            if nid in old_to_new: return old_to_new[nid]
            nn = ASTNode(kind=n.kind, word=n.word, children=[],
                         emit_vec=n.emit_vec.copy() if n.emit_vec is not None else None,
                         repeat_k=n.repeat_k)
            old_to_new[nid] = nn
            nn.children = [_cp(c) for c in n.children]
            return nn
        nr = [_cp(r) for r in self.roots]
        nh = []
        for h in self.holes:
            np_ = old_to_new.get(id(h.parent))
            if np_ is None: np_ = _cp(h.parent)
            nh.append(Hole(parent=np_, slot=h.slot))
        return ParseState(roots=nr, holes=nh)

@dataclass
class ASTBeamEntry:
    log_score: float
    instr_idx: int
    state: ParseState

    def __lt__(self, other):
        return self.log_score > other.log_score


# ── AST → TraceSteps ───────────────────────────────────────────

def ast_to_trace_steps(roots):
    steps = []
    for root in roots:
        _ast_dfs(root, steps)
    return steps

def _ast_dfs(node, steps):
    if node.kind == 'EMIT':
        steps.append(TraceStep(word=node.word, role='EMIT', emit_vec=node.emit_vec))
    elif node.kind == 'REPEAT':
        _ast_dfs(node.children[0], steps)
        steps.append(TraceStep(word=node.word, role='REPEAT',
                               repeat_k=node.repeat_k,
                               arity=node.children[0].node_count()))
    elif node.kind in ('SWAP', 'CONCAT', 'OVER'):
        rm = {'SWAP':'SWAP_INFIX','CONCAT':'CONCAT_INFIX','OVER':'OVER_INFIX'}
        _ast_dfs(node.children[0], steps)
        _ast_dfs(node.children[1], steps)
        steps.append(TraceStep(word=node.word, role=rm[node.kind], arity=1))


# ── Hole filling ───────────────────────────────────────────────

def _fill_hole(state, node):
    """Fill innermost hole; cascade if parent completes."""
    if not state.holes:
        state.roots.append(node)
        return state
    hole = state.holes[-1]
    state.holes = state.holes[:-1]
    hole.parent.children[hole.slot] = node
    if all(c is not None for c in hole.parent.children):
        return _fill_hole(state, hole.parent)
    return state

def _auto_close(state):
    """Close remaining holes by pulling from roots."""
    state = state.deep_copy()
    while state.holes and state.roots:
        filler = state.roots.pop()
        state = _fill_hole(state, filler)
    return state


def _get_emit_candidates(concept, nig, eps_obj, tau_inc, k_b=3, delta=None,
                         gauss=False):
    return concept.top_k_emit_candidates(nig, k_b=k_b,
                                          eps_obj=eps_obj, tau_inc=tau_inc,
                                          delta=delta, gauss=gauss)


# ── Core AST Beam Search ──────────────────────────────────────

def infer_top_k_ast(
    instruction: List[str],
    target: Optional[List[np.ndarray]],
    library: Dict[str, NeuroConcept],
    priors: object,
    k: int = 10,
    beam_width: int = 100,
    k_b: int = 3,
    mem_bias=None,  # MemBias from ns_hpc (HPC Layer 2)
) -> List[Tuple[float, List[ASTNode], List[TraceStep]]]:
    """Beam search with deferred hole-fill for correct scoping."""
    alpha = priors.alpha
    gamma = priors.gamma
    nig = priors.nig
    beta = priors.beta
    tau_span = priors.tau_span
    eps_obj = priors.eps_obj
    tau_inc = priors.tau_inc
    delta = getattr(priors, 'delta', None)  # None = continuous, dict = discrete
    gauss = getattr(priors, 'gauss', False)  # True = Gaussian log-lik
    n_instr = len(instruction)

    DEFER_PENALTY = -0.3  # slight penalty for deferring a hole-fill

    beam = [ASTBeamEntry(log_score=0.0, instr_idx=0, state=ParseState())]

    for token_step in range(n_instr):
        next_beam = []
        for entry in beam:
            if entry.instr_idx >= n_instr:
                next_beam.append(entry)
                continue

            idx = entry.instr_idx
            w = instruction[idx]
            if w not in library:
                library[w] = NeuroConcept(w)
            concept = library[w]

            eff_alpha = context_role_prior(
                w, idx, n_instr, len(entry.state.roots), alpha
            )
            has_hole = len(entry.state.holes) > 0

            # ── EMIT ──────────────────────────────────────────
            role_score = concept.log_role_prob('EMIT', eff_alpha)
            # HPC prior boost: log-linear fusion for EMIT (S5)
            if mem_bias is not None and w in mem_bias.role_boost:
                role_score += mem_bias.lam_mem * mem_bias.role_boost[w].get('EMIT', 0.0)
            emit_cands = _get_emit_candidates(concept, nig, eps_obj, tau_inc,
                                               k_b, delta=delta, gauss=gauss)
            for vec, emit_score in emit_cands:
                new_score = entry.log_score + role_score + emit_score

                if has_hole:
                    # Branch A: fill hole (cascade)
                    ns_a = entry.state.deep_copy()
                    emit_a = ASTNode(kind='EMIT', word=w, emit_vec=vec.copy())
                    filled = _fill_hole(ns_a, emit_a)
                    next_beam.append(ASTBeamEntry(
                        log_score=new_score, instr_idx=idx+1, state=filled
                    ))

                    # Branch B: defer fill — add to roots, keep hole open
                    # Only if more tokens remain (so REPEAT can wrap)
                    if idx + 1 < n_instr:
                        ns_b = entry.state.deep_copy()
                        emit_b = ASTNode(kind='EMIT', word=w, emit_vec=vec.copy())
                        ns_b.roots = ns_b.roots + [emit_b]
                        next_beam.append(ASTBeamEntry(
                            log_score=new_score + DEFER_PENALTY,
                            instr_idx=idx+1, state=ns_b
                        ))
                else:
                    ns = entry.state.deep_copy()
                    emit_n = ASTNode(kind='EMIT', word=w, emit_vec=vec.copy())
                    ns.roots = ns.roots + [emit_n]
                    next_beam.append(ASTBeamEntry(
                        log_score=new_score, instr_idx=idx+1, state=ns
                    ))

            # ── REPEAT: postfix, needs preceding root ─────────
            # Can fire even with holes open (wraps root, not hole)
            if entry.state.roots:
                role_score = concept.log_role_prob('REPEAT', eff_alpha)
                # HPC prior boost: log-linear fusion for REPEAT (S5)
                if mem_bias is not None and w in mem_bias.role_boost:
                    role_score += mem_bias.lam_mem * mem_bias.role_boost[w].get('REPEAT', 0.0)
                last = entry.state.roots[-1]
                sp = log_span_prior(last.node_count(), tau_span)
                for rep_k in REPEAT_RANGE:
                    if rep_k == 1: continue
                    k_score = concept.log_repeat_prob(rep_k, gamma)
                    ns = entry.state.deep_copy()
                    last_copy = ns.roots[-1]
                    rep_node = ASTNode(kind='REPEAT', word=w,
                                       children=[last_copy], repeat_k=rep_k)
                    ns.roots = ns.roots[:-1] + [rep_node]
                    next_beam.append(ASTBeamEntry(
                        log_score=entry.log_score + role_score + k_score + sp,
                        instr_idx=idx+1, state=ns
                    ))

            # ── INFIX: needs preceding root ───────────────────
            # Can fire even with holes open (creates nested scope)
            if entry.state.roots:
                for kind, role_name in [
                    ('SWAP', 'SWAP_INFIX'),
                    ('CONCAT', 'CONCAT_INFIX'),
                    ('OVER', 'OVER_INFIX'),
                ]:
                    role_score = concept.log_role_prob(role_name, eff_alpha)
                    # HPC prior boost: log-linear fusion for INFIX (S5)
                    if mem_bias is not None and w in mem_bias.role_boost:
                        role_score += mem_bias.lam_mem * mem_bias.role_boost[w].get(role_name, 0.0)
                    left = entry.state.roots[-1]
                    sp = log_span_prior(left.node_count(), tau_span)
                    ns = entry.state.deep_copy()
                    left_copy = ns.roots[-1]
                    infix_node = ASTNode(kind=kind, word=w,
                                         children=[left_copy, None])
                    hole = Hole(parent=infix_node, slot=1)
                    ns.roots = ns.roots[:-1]
                    ns.holes = ns.holes + [hole]
                    next_beam.append(ASTBeamEntry(
                        log_score=entry.log_score + role_score + sp,
                        instr_idx=idx+1, state=ns
                    ))

        next_beam.sort()
        beam = next_beam[:beam_width]

    # ── Phase 2: close, evaluate, score ────────────────────────
    final = []
    for entry in beam:
        state = _auto_close(entry.state)
        if not state.roots: continue

        output_vecs = []
        for root in state.roots:
            output_vecs.extend(eval_ast(root))
        if not output_vecs: continue

        score = entry.log_score
        if target is not None and len(target) > 0:
            sed = soft_edit_distance(output_vecs, target)
            score += -beta * sed

        trace = ast_to_trace_steps(state.roots)
        final.append((score, state.roots, trace))

    final.sort(key=lambda x: x[0], reverse=True)
    return final[:k]
