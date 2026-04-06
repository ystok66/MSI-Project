"""
em.py — EM loop for HBPI Phase-0.

E-step: For each SUPPORT example, enumerate top-K parses and compute
        posterior weights q(p) ∝ exp(log_prior(p) + log_likelihood(p, y)).

M-step: Reset counts to priors, then accumulate expected counts from
        all examples weighted by q(p). This updates:
        - P(type | w)   for each word
        - P(color | w)  for each primitive
        - P(repeat | w) for each unary operator

Convergence diagnostics:
  - avg max_q: average of max q(p) over examples (higher = sharper posterior)
  - log_evidence: Σ_e log Σ_p w(p) (proxy for corpus fit)
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional

from .grammar import AST
from .model import HBPIModel, HBPIHyperparams
from .parser import enumerate_parses
from .executor import execute


def softmax_weights(log_scores: List[float]) -> List[float]:
    """Numerically stable softmax over log scores → posterior weights."""
    if not log_scores:
        return []
    max_s = max(log_scores)
    exps = [np.exp(s - max_s) for s in log_scores]
    total = sum(exps)
    if total < 1e-30:
        return [1.0 / len(exps)] * len(exps)
    return [e / total for e in exps]


def em_step(dataset: List[Dict],
            model: HBPIModel,
            verbose: bool = False
            ) -> Dict:
    """
    One full EM iteration.

    Args:
        dataset: list of {'input': [words], 'output': [colors]}
        model: current HBPI model (modified in place)
        verbose: print diagnostics

    Returns:
        dict with convergence diagnostics:
        - 'avg_max_q': average sharpness of parse posterior
        - 'log_evidence': corpus log evidence proxy
        - 'n_parses_avg': average number of parses per example
    """
    # ── E-step: enumerate top-K parses per example ──
    all_e_results = []  # list of (example, [(score, ast, weight)])

    total_log_evidence = 0.0
    total_max_q = 0.0
    total_n_parses = 0

    for ex in dataset:
        tokens = ex['input']
        gold = ex['output']

        # Enumerate and score
        scored_parses = enumerate_parses(tokens, model, gold=gold)

        if not scored_parses:
            all_e_results.append((ex, []))
            continue

        # Compute posterior weights
        log_scores = [s for s, a in scored_parses]
        weights = softmax_weights(log_scores)

        # Log evidence for this example: log Σ_p exp(log w(p))
        max_s = max(log_scores)
        log_ev = max_s + np.log(sum(np.exp(s - max_s) for s in log_scores))
        total_log_evidence += log_ev

        total_max_q += max(weights) if weights else 0
        total_n_parses += len(scored_parses)

        parsed = [(scored_parses[i][0], scored_parses[i][1], weights[i])
                  for i in range(len(scored_parses))]
        all_e_results.append((ex, parsed))

    n_examples = len(dataset)

    # ── M-step: reset counts and accumulate weighted expected counts ──
    model.reset_counts()

    for ex, parsed in all_e_results:
        gold = ex['output']
        for score, ast, weight in parsed:
            if weight > 1e-8:  # skip negligible parses
                model.accumulate_counts(ast, gold, weight)

    # ── Diagnostics ──
    diagnostics = {
        'avg_max_q': total_max_q / max(n_examples, 1),
        'log_evidence': total_log_evidence,
        'n_parses_avg': total_n_parses / max(n_examples, 1),
    }

    if verbose:
        print(f"    EM diagnostics: avg_max_q={diagnostics['avg_max_q']:.3f}, "
              f"log_ev={diagnostics['log_evidence']:.1f}, "
              f"n_parses_avg={diagnostics['n_parses_avg']:.0f}")

    return diagnostics


def run_em(dataset: List[Dict],
           hp: Optional[HBPIHyperparams] = None,
           verbose: bool = False
           ) -> HBPIModel:
    """
    Full EM training loop.

    Args:
        dataset: SUPPORT examples
        hp: hyperparameters (or use defaults)
        verbose: print per-iteration diagnostics

    Returns:
        Trained HBPIModel
    """
    if hp is None:
        hp = HBPIHyperparams()

    model = HBPIModel(hp)

    # Initialize: ensure all words from dataset are in the model
    for ex in dataset:
        for w in ex['input']:
            model.ensure(w)

    if verbose:
        print(f"  HBPI EM: {hp.em_iters} iterations, "
              f"K_span={hp.K_span}, K_full={hp.K_full}")

    for it in range(hp.em_iters):
        diag = em_step(dataset, model, verbose=verbose)
        if verbose:
            print(f"    Iter {it}: avg_max_q={diag['avg_max_q']:.3f}, "
                  f"log_ev={diag['log_evidence']:.1f}")

            # Show type posteriors
            snap = model.snapshot()
            type_summary = {w: info['map_type'] for w, info in snap.items()}
            print(f"      Types: {type_summary}")

    return model
