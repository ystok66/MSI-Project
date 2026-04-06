"""
ns_inference.py — Abductive beam search with soft alignment + arity latent + RSA.

Core BPL inference: enumerate candidate program traces that could have
generated the target output, scored by:

  log w(trace) = log P(trace | Φ) + log P(target | trace, Φ)

Key BPL properties:
  - Alignment is a latent variable (not hard-bound to target position)
  - Arity/span is a latent variable (how many stack items an op binds)
  - INFIX ops keep top-K candidate B interpretations (not just MAP)
  - Unseen words use context-conditioned role priors
  - Soft edit-distance likelihood allows insertion/deletion errors

RSA extension:
  - Pragmatic scoring via speaker S1 alternatives competition
  - log P_S1(u_obs | m) penalizes traces if meaning m could be
    expressed more efficiently by alternative utterances
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.special import logsumexp

from .ns_primitives import (
    StackState, StackItem, PRIMITIVES, ROLES, INFIX_ROLES,
    MAX_ARITY, Vector, concat_items
)
from .ns_concept import (
    NeuroConcept, NIGParams, COLOR_VECS, COLORS, REPEAT_RANGE,
    vec_to_color, color_to_vec
)


# ── Trace Step ──────────────────────────────────────────────────

@dataclass
class TraceStep:
    """One step in a program trace."""
    word: str
    role: str
    emit_vec: Optional[np.ndarray] = None  # for EMIT: which color was emitted
    repeat_k: Optional[int] = None          # for REPEAT: how many times
    arity: int = 1                          # for REPEAT/INFIX: expression span
    b_word: Optional[str] = None            # for INFIX: which next word was consumed
    b_vec: Optional[np.ndarray] = None      # for INFIX: B's color interpretation


@dataclass
class BeamEntry:
    """One candidate in the beam search."""
    log_score: float
    instr_idx: int          # position in instruction stream
    target_idx: int         # position in target sequence (alignment pointer)
    state: StackState
    trace: List[TraceStep]

    def __lt__(self, other):
        return self.log_score > other.log_score


# ── Soft Edit Distance ──────────────────────────────────────────

def soft_edit_distance(pred: List[Vector], target: List[Vector],
                       sigma: float = 0.5) -> float:
    """
    Soft Levenshtein distance with Gaussian substitution cost.
    
    Substitution cost = 1 - exp(-||a-b||² / (2σ²))
    """
    n, m = len(pred), len(target)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)

    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = pred[i - 1] - target[j - 1]
            dist_sq = float(np.dot(diff, diff))
            sub_cost = 1.0 - np.exp(-dist_sq * inv_2s2)
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + sub_cost
            )

    return float(dp[n][m])


# ── Context-Conditioned Role Prior ──────────────────────────────

def context_role_prior(word: str, instr_idx: int, n_instr: int,
                       stack_depth: int,
                       base_alpha: Dict[str, float]) -> Dict[str, float]:
    """
    Compute context-conditioned role prior for a word.
    
    Unseen words (or words with weak posteriors) benefit from
    positional cues:
      - Sentence-final / postfix → boost REPEAT
      - Between two expressions (infix) → boost SWAP/CONCAT
      - Sentence-initial / no left context → boost EMIT
    
    Returns adjusted alpha dict for Dirichlet predictive.
    """
    alpha = dict(base_alpha)  # copy

    has_left = stack_depth > 0
    has_right = instr_idx < n_instr - 1
    is_final = instr_idx == n_instr - 1

    if has_left and (is_final or not has_right):
        # Postfix position: boost REPEAT
        alpha['REPEAT'] *= 3.0
        # Suppress EMIT slightly (probably an operator, not a noun)
        alpha['EMIT'] *= 0.5
    elif has_left and has_right:
        # Infix position: boost SWAP/CONCAT
        alpha['SWAP_INFIX'] *= 2.0
        alpha['CONCAT_INFIX'] *= 2.0
        alpha['REPEAT'] *= 1.5  # could still be postfix with more to the right
    else:
        # No left context: almost certainly EMIT
        alpha['EMIT'] *= 3.0
        alpha['REPEAT'] *= 0.1

    return alpha


# ── Span Prior ──────────────────────────────────────────────────

def log_span_prior(arity: int, tau_span: float) -> float:
    """
    Log prior on arity/span: prefer shorter bindings.
    
    log P(arity=n) = -τ_span * (n - 1)
    """
    return -tau_span * (arity - 1)


# ── RSA Pragmatic Scoring ──────────────────────────────────────

def generate_alternatives(instruction: List[str],
                          library: Dict[str, NeuroConcept],
                          priors: object) -> List[List[str]]:
    """
    Generate alternative utterances for RSA competition.
    
    Strategy:
      - For REPEAT words: substitute with other REPEAT words  
        (different k implications) + try deletion (→ k=1 identity)
      - For INFIX words: substitute with other INFIX words  
        (SWAP ↔ CONCAT ↔ OVER paradigmatic set)
        Do NOT delete — removing an infix changes arity, not meaning type
      - Noun-only strip removed: too aggressive, distorts competition
    
    Key principle: alternatives should be utterances a rational speaker
    *could have said* to express the same structural meaning. Deletions
    are only valid for REPEAT (speaker could just not repeat).
    """
    alternatives = [list(instruction)]  # always include observed utterance
    alpha = priors.alpha

    # Collect words by role for substitution pools
    repeat_words = []
    infix_words = []
    for w, concept in library.items():
        mr = concept.map_role(alpha)
        if mr == 'REPEAT':
            repeat_words.append(w)
        elif mr in ('SWAP_INFIX', 'CONCAT_INFIX', 'OVER_INFIX'):
            infix_words.append(w)

    # Also include "function words" from instruction not yet in library
    # (they'll get default role priors — could be REPEAT or INFIX)
    func_words = set(repeat_words + infix_words)

    for i, w in enumerate(instruction):
        if w not in library:
            continue
        concept = library[w]
        map_role = concept.map_role(alpha)

        if map_role == 'REPEAT':
            # Alt 1: deletion (→ identity / k=1 implicit)
            alt = list(instruction[:i]) + list(instruction[i+1:])
            if alt:
                alternatives.append(alt)

            # Alt 2: substitute with other REPEAT words
            for rw in repeat_words:
                if rw != w:
                    alt = list(instruction)
                    alt[i] = rw
                    alternatives.append(alt)

        elif map_role in ('SWAP_INFIX', 'CONCAT_INFIX', 'OVER_INFIX'):
            # Paradigmatic substitution: swap with other INFIX words
            for iw in infix_words:
                if iw != w:
                    alt = list(instruction)
                    alt[i] = iw
                    alternatives.append(alt)

            # Also try substituting INFIX with REPEAT words
            # (speaker chose infix instead of repeat — different structure)
            for rw in repeat_words:
                alt = list(instruction)
                alt[i] = rw
                alternatives.append(alt)

    # Deduplicate
    seen = set()
    unique = []
    for alt in alternatives:
        key = tuple(alt)
        if key not in seen:
            seen.add(key)
            unique.append(alt)

    return unique


def rsa_pragmatic_term(u_obs: List[str],
                       m_vecs: List[np.ndarray],
                       library: Dict[str, NeuroConcept],
                       priors: object,
                       alt_us: List[List[str]],
                       _depth: int = 0) -> float:
    """
    Compute log P_S1(u_obs | m) — speaker probability under alternatives.
    
    S1(u | m) ∝ exp(α · log L0(m | u) - cost · |u|)
    
    L0(m | u) is approximated by running infer_top_k(u, target=m) and
    taking logsumexp of trace scores.
    
    _depth prevents infinite recursion (RSA call inside RSA).
    """
    if _depth > 0:
        return 0.0  # no nested RSA

    rsa_alpha = priors.rsa_alpha
    rsa_cost = priors.rsa_cost

    def l0_logprob(u):
        """Log P_L0(m | u) — literal listener probability."""
        traces = _infer_top_k_inner(
            u, m_vecs, library, priors,
            k=3, beam_width=10, use_rsa=False
        )
        if not traces:
            return -1e9
        return logsumexp([t[0] for t in traces])

    # S1 utility for observed utterance
    l0_obs = l0_logprob(u_obs)
    util_obs = rsa_alpha * l0_obs - rsa_cost * len(u_obs)

    # S1 utilities for all alternatives (including u_obs)
    utils = []
    for u in alt_us:
        l0 = l0_logprob(u)
        utils.append(rsa_alpha * l0 - rsa_cost * len(u))

    # log P_S1 = util_obs - log Σ exp(util)
    return util_obs - logsumexp(utils)


# ── Main Inference ──────────────────────────────────────────────

def infer_top_k(
    instruction: List[str],
    target: Optional[List[np.ndarray]],
    library: Dict[str, NeuroConcept],
    priors: object,  # GlobalPriors from ns_learner
    k: int = 10,
    beam_width: int = 30,
    k_b: int = 3,
    mem_bias=None,  # MemBias from ns_hpc (HPC Layer 2)
) -> List[Tuple[float, List[TraceStep]]]:
    """Public API: beam search with optional RSA scoring."""
    return _infer_top_k_inner(
        instruction, target, library, priors,
        k=k, beam_width=beam_width, k_b=k_b, use_rsa=True,
        mem_bias=mem_bias
    )


def _infer_top_k_inner(
    instruction: List[str],
    target: Optional[List[np.ndarray]],
    library: Dict[str, NeuroConcept],
    priors: object,
    k: int = 10,
    beam_width: int = 30,
    k_b: int = 3,
    use_rsa: bool = True,
    mem_bias=None,  # MemBias from ns_hpc (HPC Layer 2)
) -> List[Tuple[float, List[TraceStep]]]:
    """
    Abductive beam search over program traces.
    
    Latent variables enumerated:
      - Role (per word): EMIT / REPEAT / SWAP_INFIX / CONCAT_INFIX / OVER_INFIX
      - Alignment offset (for EMIT): target[idx + offset]
      - Arity/span (for REPEAT/INFIX): how many stack items to bind
      - Repeat k (for REPEAT): {2, 3, 4}
      - B interpretation (for INFIX): top-K_b candidate colors
    """
    n_instr = len(instruction)
    if n_instr == 0:
        return []

    alpha = priors.alpha
    gamma = priors.gamma
    nig = priors.nig
    lam = priors.lam
    beta_temp = priors.beta
    tau_span = priors.tau_span
    eps_obj = priors.eps_obj
    tau_inc = priors.tau_inc
    delta = getattr(priors, 'delta', None)  # None = continuous, dict = discrete
    gauss = getattr(priors, 'gauss', False)  # True = Gaussian log-lik (for Lab)

    # Ensure all words have concepts
    for w in instruction:
        if w not in library:
            library[w] = NeuroConcept(w, d=nig.mu0.shape[0])

    initial = BeamEntry(
        log_score=0.0, instr_idx=0, target_idx=0,
        state=StackState(), trace=[]
    )
    beam = [initial]

    # Process instruction tokens
    while True:
        active = [e for e in beam if e.instr_idx < n_instr]
        finished = [e for e in beam if e.instr_idx >= n_instr]

        if not active:
            break

        next_beam = list(finished)

        for entry in active:
            idx = entry.instr_idx
            w = instruction[idx]
            concept = library[w]

            # Use context-conditioned prior for words with very little data
            total_obs = sum(concept.role_counts.values())
            if total_obs < 1.0:
                eff_alpha = context_role_prior(
                    w, idx, n_instr, entry.state.depth, alpha)
            else:
                eff_alpha = alpha

            for role in ROLES:
                role_score = concept.log_role_prob(role, eff_alpha)

                # HPC prior boost: log-linear fusion (M4)
                if mem_bias is not None and w in mem_bias.role_boost:
                    role_score += mem_bias.lam_mem * mem_bias.role_boost[w].get(role, 0.0)

                if role == 'EMIT':
                    _expand_emit(
                        entry, w, concept, role_score,
                        target, nig, lam, eps_obj, tau_inc, next_beam,
                        delta=delta, gauss=gauss
                    )
                elif role == 'REPEAT':
                    _expand_repeat(
                        entry, w, concept, role_score,
                        target, gamma, nig, tau_span, next_beam
                    )
                elif role in INFIX_ROLES:
                    _expand_infix(
                        entry, w, role, concept, role_score,
                        instruction, target, library,
                        eff_alpha, nig, lam, tau_span,
                        eps_obj, tau_inc, k_b, next_beam,
                        delta=delta, gauss=gauss
                    )

        # Prune to top beam_width
        if len(next_beam) > beam_width:
            next_beam.sort(key=lambda e: e.log_score, reverse=True)
            next_beam = next_beam[:beam_width]

        beam = next_beam

    # Final scoring with soft likelihood + RSA
    results = []
    for entry in beam:
        if entry.instr_idx < n_instr:
            continue

        output_vecs = entry.state.flatten()
        total_score = entry.log_score

        if target is not None and target:
            edit_dist = soft_edit_distance(output_vecs, target)
            likelihood = -beta_temp * edit_dist
            total_score += likelihood

        # RSA pragmatic term (only at final scoring, not during beam)
        if use_rsa and priors.rsa_alpha > 0 and target is not None:
            alt_us = generate_alternatives(instruction, library, priors)
            if len(alt_us) > 1:
                rsa_score = rsa_pragmatic_term(
                    instruction, output_vecs, library, priors, alt_us,
                    _depth=0  # L0 calls inside use use_rsa=False
                )
                total_score += rsa_score

        results.append((total_score, entry.trace))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]


# ── Expansion Helpers ───────────────────────────────────────────

def _expand_emit(entry, word, concept, role_score, target, nig, lam,
                 eps_obj, tau_inc, beam_out, delta=None, gauss=False):
    """Expand EMIT branches with soft alignment."""
    if target is not None and target:
        for offset in [0, 1, -1]:
            aligned_idx = entry.target_idx + offset
            if aligned_idx < 0 or aligned_idx >= len(target):
                continue
            vec = target[aligned_idx]
            emit_score = concept.log_emit_prob(vec, nig, eps_obj, tau_inc,
                                               delta=delta, gauss=gauss)
            align_penalty = -lam * abs(offset)
            new_state = entry.state.push(StackItem.single(vec))
            step = TraceStep(word=word, role='EMIT', emit_vec=vec)
            beam_out.append(BeamEntry(
                log_score=entry.log_score + role_score + emit_score + align_penalty,
                instr_idx=entry.instr_idx + 1,
                target_idx=aligned_idx + 1,
                state=new_state,
                trace=entry.trace + [step]
            ))
    else:
        # Query time: still include log_emit_prob so non-nouns are penalized
        vec_name = concept.map_color(nig, eps_obj, tau_inc, delta=delta,
                                      gauss=gauss)
        # Use Lab vec when gauss mode is active
        if gauss:
            from ns_learner.ns_colors import lab_vec as _lv
            vec = _lv(vec_name)
        else:
            vec = COLOR_VECS[vec_name]
        emit_score = concept.log_emit_prob(vec, nig, eps_obj, tau_inc,
                                           delta=delta, gauss=gauss)
        new_state = entry.state.push(StackItem.single(vec))
        step = TraceStep(word=word, role='EMIT', emit_vec=vec)
        beam_out.append(BeamEntry(
            log_score=entry.log_score + role_score + emit_score,
            instr_idx=entry.instr_idx + 1,
            target_idx=entry.target_idx + 1,
            state=new_state,
            trace=entry.trace + [step]
        ))


def _expand_repeat(entry, word, concept, role_score,
                   target, gamma, nig, tau_span, beam_out):
    """Expand REPEAT branches with arity enumeration."""
    max_a = min(MAX_ARITY, entry.state.depth)
    if max_a < 1:
        return  # nothing to repeat

    for arity in range(1, max_a + 1):
        expr, rest = entry.state.pop_n(arity)
        if expr is None:
            continue

        span_score = log_span_prior(arity, tau_span)

        for k in REPEAT_RANGE:
            if k < 2:
                continue  # k=1 is identity

            k_score = concept.log_repeat_prob(k, gamma)
            repeated = expr.repeat(k)
            new_state = rest.push(repeated)

            # How many extra output tokens does this produce?
            extra = (k - 1) * len(expr)
            new_target_idx = entry.target_idx + extra

            step = TraceStep(word=word, role='REPEAT',
                             repeat_k=k, arity=arity)
            beam_out.append(BeamEntry(
                log_score=(entry.log_score + role_score +
                           k_score + span_score),
                instr_idx=entry.instr_idx + 1,
                target_idx=new_target_idx,
                state=new_state,
                trace=entry.trace + [step]
            ))


def _expand_infix(entry, word, role, concept, role_score,
                  instruction, target, library,
                  alpha, nig, lam, tau_span,
                  eps_obj, tau_inc, k_b, beam_out, delta=None, gauss=False):
    """Expand SWAP_INFIX / CONCAT_INFIX / OVER_INFIX with arity."""
    next_idx = entry.instr_idx + 1
    if next_idx >= len(instruction):
        return

    next_word = instruction[next_idx]
    if next_word not in library:
        library[next_word] = NeuroConcept(next_word, d=nig.mu0.shape[0])
    next_concept = library[next_word]

    # Top-K_b candidate B interpretations
    b_candidates = next_concept.top_k_emit_candidates(
        nig, k_b=k_b, eps_obj=eps_obj, tau_inc=tau_inc, delta=delta,
        gauss=gauss)
    next_emit_score = next_concept.log_role_prob('EMIT', alpha)

    primitive = PRIMITIVES[role]

    max_a = min(MAX_ARITY, entry.state.depth)
    if max_a < 1:
        return

    for arity in range(1, max_a + 1):
        span_score = log_span_prior(arity, tau_span)

        for b_vec, b_emit_score in b_candidates:
            # Execute primitive with arity
            new_state = primitive.execute(entry.state, b_vec, arity=arity)
            if new_state is None:
                continue

            # Estimate extra output tokens
            expr, _ = entry.state.pop_n(arity)
            if role == 'OVER_INFIX':
                extra = 1 + len(expr)  # B + copy of A
            elif role in ('SWAP_INFIX', 'CONCAT_INFIX'):
                extra = 1  # just B
            else:
                extra = 0

            new_target_idx = entry.target_idx + extra

            step = TraceStep(
                word=word, role=role, arity=arity,
                b_word=next_word, b_vec=b_vec
            )
            total = (entry.log_score + role_score +
                     b_emit_score + next_emit_score + span_score)

            beam_out.append(BeamEntry(
                log_score=total,
                instr_idx=next_idx + 1,
                target_idx=new_target_idx,
                state=new_state,
                trace=entry.trace + [step]
            ))


# ── Utility: Execute a trace to get output ──────────────────────

def execute_trace(trace: List[TraceStep], library: Dict[str, NeuroConcept],
                  nig: NIGParams, delta=None, gauss=False) -> List[np.ndarray]:
    """Re-execute a trace on a fresh stack to get output vectors."""
    # Use Lab vectors when gauss mode is active
    if gauss:
        from ns_learner.ns_colors import lab_vec as _lv
        _color_vecs = {c: _lv(c) for c in COLORS}
    else:
        _color_vecs = COLOR_VECS

    state = StackState()
    for step in trace:
        prim = PRIMITIVES[step.role]

        if step.role == 'EMIT':
            if step.emit_vec is not None:
                param = step.emit_vec
            else:
                concept = library.get(step.word)
                if concept:
                    param = _color_vecs[concept.map_color(nig, delta=delta, gauss=gauss)]
                else:
                    param = np.zeros(nig.mu0.shape[0])
            state = prim.execute(state, param, arity=1)

        elif step.role == 'REPEAT':
            state = prim.execute(state, step.repeat_k or 2,
                                 arity=step.arity)

        elif step.role in INFIX_ROLES:
            if step.b_vec is not None:
                param = step.b_vec
            else:
                b_concept = library.get(step.b_word)
                if b_concept:
                    param = _color_vecs[b_concept.map_color(nig, delta=delta, gauss=gauss)]
                else:
                    param = np.zeros(nig.mu0.shape[0])
            state = prim.execute(state, param, arity=step.arity)

        if state is None:
            return []

    return state.flatten()
