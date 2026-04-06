"""
ns_learner.py — NSLearner: the BPL Agent with Meta-Learning.

Two-loop architecture:
  Inner Loop (study_episode):  Few-shot learning via Soft-EM on SUPPORT
  Outer Loop (meta_train):     Empirical Bayes on background episodes
                                to learn inductive biases (priors Φ)

The inner loop runs for each new task:
  E-step: infer_top_k → get weighted program traces
  M-step: accumulate weighted stats into NeuroConcepts

The outer loop learns Φ = {α, γ, NIG, λ, β, τ_span, ε_obj, τ_inc, RSA} from background tasks:
  Objective: max_Φ  Σ_episodes  log P_Φ(QUERY | SUPPORT)
  Method: Smoothed MLE for α/γ, moment matching for NIG,
          coordinate grid search for λ/β
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from scipy.special import logsumexp
import copy

from .ns_primitives import ROLES, INFIX_ROLES, StackState
from .ns_concept import (
    NeuroConcept, NIGParams, COLORS, COLOR_VECS, N_COLORS,
    REPEAT_RANGE, vec_to_color, color_to_vec
)
from .ns_inference import infer_top_k, execute_trace, soft_edit_distance
from .ns_hpc import EpisodeHPC


# ── Global Priors (meta-learned) ────────────────────────────────

@dataclass
class GlobalPriors:
    """
    Meta-learned inductive biases Φ.
    
    These are NOT per-word — they are shared across all words and
    define the "shape" of the learning space.
    """
    alpha: Dict[str, float] = field(default_factory=lambda: {
        'EMIT': 2.0,           # Most words are nouns
        'REPEAT': 1.0,
        'SWAP_INFIX': 0.5,
        'CONCAT_INFIX': 0.5,
        'OVER_INFIX': 0.2,     # Rare — higher MDL cost via low prior
    })
    gamma: Dict[int, float] = field(default_factory=lambda: {
        1: 0.5, 2: 1.5, 3: 1.0, 4: 0.5  # k=2 slightly preferred
    })
    nig: NIGParams = field(default_factory=lambda: NIGParams(d=N_COLORS))
    delta: Optional[Dict[str, float]] = None  # None=continuous, dict=discrete Dirichlet
    gauss: bool = False  # True=Gaussian log-likelihood (for Lab color space)
    lam: float = 0.3          # MDL / alignment penalty
    beta: float = 2.0         # Likelihood temperature
    tau_span: float = 0.5     # Arity/span prior penalty: -τ_span*(arity-1)
    eps_obj: float = 0.1      # Object observation uncertainty (ε in KL)
    tau_inc: float = 1.0      # KL-inclusion temperature
    rsa_alpha: float = 0.5    # RSA speaker rationality α
    rsa_cost: float = 0.2     # RSA per-token utterance cost


# ── NSLearner ───────────────────────────────────────────────────

class NSLearner:
    """
    Neuro-Symbolic BPL Agent.
    
    Usage:
        learner = NSLearner()
        learner.study_episode(support_examples)  # learn from SUPPORT
        pred = learner.predict(query_words)       # predict QUERY
    
    For meta-learning:
        learner.meta_train(background_episodes)   # learn priors
    """

    def __init__(self, priors: Optional[GlobalPriors] = None,
                 n_em: int = 3, beam_k: int = 10, beam_width: int = 30,
                 use_hpc: bool = False):
        self.priors = priors or GlobalPriors()
        self.library: Dict[str, NeuroConcept] = {}
        self.n_em = n_em
        self.beam_k = beam_k
        self.beam_width = beam_width
        self.use_hpc = use_hpc
        self.hpc = EpisodeHPC() if use_hpc else None

    def _ensure_concept(self, word: str):
        """Lazily create a NeuroConcept for a new word."""
        if word not in self.library:
            d = len(self.priors.nig.mu0)  # 3 for Lab, 6 for one-hot
            self.library[word] = NeuroConcept(word, d=d)

    def _color_to_vecs(self, colors):
        """Convert color names to target vectors (Lab or one-hot)."""
        _gauss = getattr(self.priors, 'gauss', False)
        if _gauss:
            from ns_learner.ns_colors import lab_vec
            return [lab_vec(c) for c in colors]
        else:
            return [COLOR_VECS.get(c, np.zeros(N_COLORS)) for c in colors]

    def _extract_trace_summary(self, example: Dict,
                                traces: Optional[List] = None) -> Dict:
        """
        Build HPC payload from best trace or bootstrap info.

        Returns dict with per_word_role, per_word_color, trace_roles.
        """
        per_word_role: Dict[str, str] = {}
        per_word_color: Dict[str, str] = {}
        trace_roles: Dict[str, Dict[str, float]] = {}

        if traces and len(traces) > 0:
            # Use weighted traces (soft distribution)
            scores = np.array([t[0] for t in traces])
            if len(scores) > 1:
                log_w = scores - logsumexp(scores)
                weights = np.exp(log_w)
            else:
                weights = np.array([1.0])

            for (score, trace), w in zip(traces, weights):
                for step in trace:
                    word = step.word
                    role = step.role
                    if word not in trace_roles:
                        trace_roles[word] = {r: 0.0 for r in ROLES}
                    trace_roles[word][role] += w

            # MAP: pick most frequent role per word
            for word, role_dist in trace_roles.items():
                best_role = max(role_dist, key=role_dist.get)
                per_word_role[word] = best_role
                if best_role == 'EMIT' and word in self.library:
                    per_word_color[word] = self.library[word].map_color(
                        self.priors.nig, self.priors.eps_obj,
                        self.priors.tau_inc, delta=self.priors.delta)
        else:
            # Fallback: use current concept MAP roles
            for w in example.get('input', []):
                if w in self.library:
                    c = self.library[w]
                    mr = c.map_role(self.priors.alpha)
                    per_word_role[w] = mr
                    trace_roles[w] = {
                        r: c.role_counts.get(r, 0.0) for r in ROLES
                    }
                    if mr == 'EMIT':
                        per_word_color[w] = c.map_color(
                            self.priors.nig, self.priors.eps_obj,
                            self.priors.tau_inc, delta=self.priors.delta)

        return {
            'per_word_role': per_word_role,
            'per_word_color': per_word_color,
            'trace_roles': trace_roles,
        }

    # ── Inner Loop: Few-Shot Learning ───────────────────────────

    def study_episode(self, examples: List[Dict[str, list]],
                      verbose: bool = False):
        """
        Learn from SUPPORT examples via Soft-EM.
        
        Each example: {'input': [words], 'output': [colors]}
        
        EM iterations:
          E-step: For each example, infer top-K traces, compute softmax weights
          M-step: Accumulate weighted sufficient statistics
        """
        # Ensure concepts for all vocabulary
        for ex in examples:
            for w in ex['input']:
                self._ensure_concept(w)

        # HPC: reset at episode start
        if self.hpc is not None:
            self.hpc.reset()

        # Phase 0: Bootstrap — learn obvious 1:1 mappings first
        self._bootstrap_nouns(examples, verbose)

        # HPC: write examples after bootstrap (M3: store returned indices)
        hpc_indices: List[int] = []
        if self.hpc is not None:
            for ex in examples:
                summary = self._extract_trace_summary(ex)
                idx = self.hpc.write_example(
                    ex['input'], ex['output'], summary)
                hpc_indices.append(idx)
            self.hpc.calibrate_gate()  # S1: auto-calibrate CA1 thresholds

        # EM iterations
        for em_iter in range(self.n_em):
            if verbose:
                print(f"  EM iter {em_iter}:")

            # Reset counts for fresh E-step
            # (keep visual stats from bootstrap — don't reset emit_stats)
            if em_iter > 0:
                for concept in self.library.values():
                    # Only reset role/repeat counts, preserve emit stats
                    for r in ROLES:
                        concept.role_counts[r] *= 0.5  # decay, don't zero
                    for k in REPEAT_RANGE:
                        concept.repeat_counts[k] *= 0.5

            # E-step + M-step
            total_score = 0.0
            all_traces_per_ex: List[Optional[List]] = []
            for ex in examples:
                words = ex['input']
                target_colors = ex['output']
                target_vecs = self._color_to_vecs(target_colors)

                # E-step: get top-K traces
                traces = infer_top_k(
                    words, target_vecs, self.library, self.priors,
                    k=self.beam_k, beam_width=self.beam_width
                )
                all_traces_per_ex.append(traces if traces else None)

                if not traces:
                    continue

                # Compute softmax weights
                scores = np.array([t[0] for t in traces])
                if len(scores) > 1:
                    log_weights = scores - logsumexp(scores)
                    weights = np.exp(log_weights)
                else:
                    weights = np.array([1.0])

                total_score += scores[0] if len(scores) > 0 else 0.0

                # M-step: accumulate weighted stats
                for (score, trace), weight in zip(traces, weights):
                    for step in trace:
                        concept = self.library[step.word]
                        concept.soft_update(
                            weight=weight,
                            role=step.role,
                            vec=step.emit_vec,
                            k=step.repeat_k
                        )
                        # Also update consumed B word if INFIX
                        if step.b_word and step.b_vec is not None:
                            b_concept = self.library.get(step.b_word)
                            if b_concept:
                                b_concept.soft_update(
                                    weight=weight,
                                    role='EMIT',
                                    vec=step.b_vec
                                )

            # HPC: reconsolidation — update payload with improved traces
            if self.hpc is not None:
                for i, (ex, traces) in enumerate(
                        zip(examples, all_traces_per_ex)):
                    if i < len(hpc_indices):
                        summary = self._extract_trace_summary(ex, traces)
                        self.hpc.update_trace(hpc_indices[i], summary)

            # HPC: replay — soft M-step with trace_roles (S3)
            if self.hpc is not None:
                replays = self.hpc.sample_replay(batch_size=3)
                for payload in replays:
                    for word, role_dist in payload.trace_roles.items():
                        if word in self.library:
                            concept = self.library[word]
                            total_r = sum(role_dist.values())
                            if total_r > 0:
                                for role, count in role_dist.items():
                                    if count > 0 and role in ROLES:
                                        concept.soft_update(
                                            weight=0.2 * count / total_r,
                                            role=role)

            if verbose:
                print(f"    total_score={total_score:.2f}")
                for w, c in sorted(self.library.items()):
                    mr = c.map_role(self.priors.alpha)
                    mc = c.map_color(self.priors.nig,
                                     self.priors.eps_obj,
                                     self.priors.tau_inc) if mr == 'EMIT' else ''
                    print(f"    {w:>12}: {mr:15s} {mc}")

    def _bootstrap_nouns(self, examples: List[Dict], verbose: bool = False):
        """
        Bootstrap phase: identify obvious nouns from simple examples.
        
        1:1 mappings (single word → single color) are unambiguous.
        Length-matched examples with verified alignment also help.
        """
        if verbose:
            print("  Bootstrap:")

        # Select color vec source based on emission model
        _gauss = getattr(self.priors, 'gauss', False)
        if _gauss:
            from ns_learner.ns_colors import lab_vec as _lv
            _cvecs = {c_name: _lv(c_name) for c_name in COLORS}
            _d = 3
        else:
            _cvecs = COLOR_VECS
            _d = N_COLORS

        def _get_vec(color_name):
            return _cvecs.get(color_name, np.zeros(_d))

        # Pass 1: single input → single output
        for ex in examples:
            if len(ex['input']) == 1 and len(ex['output']) == 1:
                w = ex['input'][0]
                c = ex['output'][0]
                vec = _get_vec(c)
                concept = self.library[w]
                # Strong EMIT evidence
                concept.soft_update(weight=3.0, role='EMIT', vec=vec)
                if verbose:
                    print(f"    {w} → {c} (1:1)")

        # Pass 2: length-matched multi-word examples
        confirmed = {}  # word → color
        for concept in self.library.values():
            if concept.emit_stats['sum_w'] > 1.0:
                confirmed[concept.name] = concept.map_color(
                    self.priors.nig, self.priors.eps_obj, self.priors.tau_inc,
                    delta=self.priors.delta, gauss=_gauss)

        for _ in range(5):  # iterate to propagate
            changed = False
            for ex in examples:
                words, colors = ex['input'], ex['output']
                if len(words) != len(colors):
                    continue

                unknowns = [i for i, w in enumerate(words)
                            if w not in confirmed]
                knowns_ok = all(
                    confirmed.get(words[i]) == colors[i]
                    for i in range(len(words)) if i not in unknowns
                )

                if knowns_ok and len(unknowns) == 1:
                    idx = unknowns[0]
                    w = words[idx]
                    c = colors[idx]
                    vec = _get_vec(c)
                    self.library[w].soft_update(weight=2.0, role='EMIT', vec=vec)
                    confirmed[w] = c
                    changed = True
                    if verbose:
                        print(f"    {w} → {c} (align)")

            if not changed:
                break

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, words: List[str], verbose: bool = False) -> List[str]:
        """
        Predict output colors for a query input.
        
        Uses beam search to find best program trace, then executes it.
        """
        for w in words:
            self._ensure_concept(w)

        # HPC: retrieve bias for query
        mem_bias = self.hpc.get_bias(words) if self.hpc is not None else None

        traces = infer_top_k(
            words, None, self.library, self.priors,
            k=self.beam_k, beam_width=self.beam_width,
            mem_bias=mem_bias
        )

        if not traces:
            # Fallback: emit MAP color for each word
            return [self.library[w].map_color(
                self.priors.nig, self.priors.eps_obj, self.priors.tau_inc,
                delta=self.priors.delta
            ) for w in words]

        best_score, best_trace = traces[0]

        # Execute trace to get output
        output_vecs = execute_trace(best_trace, self.library, self.priors.nig,
                                     delta=self.priors.delta)

        if not output_vecs:
            return [self.library[w].map_color(
                self.priors.nig, self.priors.eps_obj, self.priors.tau_inc,
                delta=self.priors.delta
            ) for w in words]

        result = [vec_to_color(v) for v in output_vecs]

        if verbose:
            trace_str = ' → '.join(
                f"{s.word}:{s.role}" +
                (f"(k={s.repeat_k})" if s.repeat_k else '') +
                (f"(a={s.arity})" if s.arity > 1 else '') +
                (f"(B={s.b_word})" if s.b_word else '')
                for s in best_trace
            )
            print(f"    trace: {trace_str}")
            print(f"    output: {' '.join(result)}")

        return result

    # ── AST-Based Learning & Prediction ────────────────────────

    def study_episode_ast(self, examples: List[Dict[str, list]],
                          verbose: bool = False):
        """Learn from SUPPORT using AST-based beam search.
        
        Same interface as study_episode but uses infer_top_k_ast
        for hierarchical scope handling.
        """
        from ns_learner.ns_ast import infer_top_k_ast

        for ex in examples:
            for w in ex['input']:
                self._ensure_concept(w)

        # HPC: reset at episode start
        if self.hpc is not None:
            self.hpc.reset()

        # Bootstrap nouns (same as stack-machine version)
        self._bootstrap_nouns(examples, verbose)

        # HPC: write examples after bootstrap
        hpc_indices: List[int] = []
        if self.hpc is not None:
            for ex in examples:
                summary = self._extract_trace_summary(ex)
                idx = self.hpc.write_example(
                    ex['input'], ex['output'], summary)
                hpc_indices.append(idx)
            self.hpc.calibrate_gate()

        # EM iterations
        for em_iter in range(self.n_em):
            if verbose:
                print(f"  EM iter {em_iter} (AST):")

            if em_iter > 0:
                for concept in self.library.values():
                    for r in ROLES:
                        concept.role_counts[r] *= 0.5
                    for k in REPEAT_RANGE:
                        concept.repeat_counts[k] *= 0.5

            total_score = 0.0
            all_results_per_ex: List[Optional[List]] = []
            for ex in examples:
                words = ex['input']
                target_colors = ex['output']
                target_vecs = self._color_to_vecs(target_colors)

                # AST beam search
                results = infer_top_k_ast(
                    words, target_vecs, self.library, self.priors,
                    k=self.beam_k, beam_width=self.beam_width
                )
                all_results_per_ex.append(results if results else None)

                if not results:
                    continue

                scores = np.array([r[0] for r in results])
                if len(scores) > 1:
                    log_weights = scores - logsumexp(scores)
                    weights = np.exp(log_weights)
                else:
                    weights = np.array([1.0])

                total_score += scores[0]

                # M-step: use trace_steps (AST → TraceStep conversion)
                for (score, roots, trace), weight in zip(results, weights):
                    for step in trace:
                        if step.word not in self.library:
                            continue
                        concept = self.library[step.word]
                        concept.soft_update(
                            weight=weight,
                            role=step.role,
                            vec=step.emit_vec,
                            k=step.repeat_k
                        )

            # HPC: reconsolidation — update payload with improved traces
            if self.hpc is not None:
                for i, (ex, results) in enumerate(
                        zip(examples, all_results_per_ex)):
                    if i < len(hpc_indices):
                        traces = None
                        if results:
                            traces = [(r[0], r[2]) for r in results]
                        summary = self._extract_trace_summary(ex, traces)
                        self.hpc.update_trace(hpc_indices[i], summary)

            # HPC: replay — soft M-step with trace_roles (S3)
            if self.hpc is not None:
                replays = self.hpc.sample_replay(batch_size=3)
                for payload in replays:
                    for word, role_dist in payload.trace_roles.items():
                        if word in self.library:
                            concept = self.library[word]
                            total_r = sum(role_dist.values())
                            if total_r > 0:
                                for role, count in role_dist.items():
                                    if count > 0 and role in ROLES:
                                        concept.soft_update(
                                            weight=0.2 * count / total_r,
                                            role=role)

            if verbose:
                print(f"    total_score={total_score:.2f}")
                for w, c in sorted(self.library.items()):
                    mr = c.map_role(self.priors.alpha)
                    mc = c.map_color(self.priors.nig,
                                     self.priors.eps_obj,
                                     self.priors.tau_inc) if mr == 'EMIT' else ''
                    print(f"    {w:>12}: {mr:15s} {mc}")

    def predict_ast(self, words: List[str], verbose: bool = False) -> List[str]:
        """Predict output using AST-based beam search.
        
        Same interface as predict but uses hierarchical parsing.
        """
        from ns_learner.ns_ast import infer_top_k_ast, eval_ast

        for w in words:
            self._ensure_concept(w)

        # HPC: retrieve bias for query
        mem_bias = self.hpc.get_bias(words) if self.hpc is not None else None

        results = infer_top_k_ast(
            words, None, self.library, self.priors,
            k=self.beam_k, beam_width=self.beam_width,
            mem_bias=mem_bias
        )

        if not results:
            return [self.library[w].map_color(
                self.priors.nig, self.priors.eps_obj, self.priors.tau_inc,
                delta=self.priors.delta
            ) for w in words]

        best_score, best_roots, best_trace = results[0]

        # Evaluate AST to get output
        output_vecs = []
        for root in best_roots:
            output_vecs.extend(eval_ast(root))

        if not output_vecs:
            return [self.library[w].map_color(
                self.priors.nig, self.priors.eps_obj, self.priors.tau_inc,
                delta=self.priors.delta
            ) for w in words]

        result = [vec_to_color(v) for v in output_vecs]

        if verbose:
            print(f"    AST roots: {best_roots}")
            print(f"    output: {' '.join(result)}")

        return result

    # ── Outer Loop: Meta-Learning (Empirical Bayes) ─────────────

    def meta_train(self, background_episodes: List[Dict],
                   n_epochs: int = 3, verbose: bool = False):
        """
        Learn inductive biases Φ from background tasks.
        
        Objective: max_Φ  Σ_episodes  log P_Φ(QUERY | SUPPORT)
        
        Each episode: {'support': [...], 'query': [...]}
        
        Method:
          - Smoothed MLE for α (role Dirichlet) and γ (repeat Dirichlet)
          - Moment matching for NIG visual prior
          - Coordinate grid search for λ, β, tau_span, eps_obj
          - Train/val split: only keep updates that improve val accuracy
        """
        if verbose:
            print(f"Meta-training on {len(background_episodes)} episodes")

        # Split: 70% train, 30% val for grid search
        n_train = max(1, int(len(background_episodes) * 0.7))
        train_eps = background_episodes[:n_train]
        val_eps = background_episodes[n_train:]
        if verbose:
            print(f"  train={len(train_eps)}, val={len(val_eps)} episodes")

        # Track best priors via validation
        best_priors = copy.deepcopy(self.priors)
        best_val_acc = self._eval_accuracy(val_eps)
        if verbose:
            print(f"  Initial val_acc={best_val_acc:.1%}")

        for epoch in range(n_epochs):
            # Aggregated sufficient statistics
            agg_role = {r: 0.0 for r in ROLES}
            agg_repeat = {k: 0.0 for k in REPEAT_RANGE}
            agg_visual_w = 0.0
            agg_visual_wx = np.zeros(N_COLORS)
            agg_visual_wx2 = np.zeros(N_COLORS)

            total_query_loglik = 0.0
            n_correct = 0
            n_total = 0

            for episode in train_eps:
                support = episode['support']
                query = episode.get('query', [])

                # Fresh library per episode
                ep_learner = NSLearner(
                    priors=copy.deepcopy(self.priors),
                    n_em=self.n_em,
                    beam_k=self.beam_k,
                    beam_width=self.beam_width
                )

                # Inner loop: learn from support
                ep_learner.study_episode(support)

                # Evaluate on query
                for q in query:
                    pred = ep_learner.predict(q['input'])
                    n_total += 1
                    if pred == q['output']:
                        n_correct += 1

                    target_vecs = self._color_to_vecs(q['output'])
                    traces = infer_top_k(
                        q['input'], target_vecs, ep_learner.library,
                        ep_learner.priors, k=5
                    )
                    if traces:
                        total_query_loglik += logsumexp(
                            [t[0] for t in traces])

                # Aggregate posterior counts
                for concept in ep_learner.library.values():
                    for r in ROLES:
                        agg_role[r] += concept.role_counts[r]
                    for k in REPEAT_RANGE:
                        agg_repeat[k] += concept.repeat_counts[k]
                    sw = concept.emit_stats['sum_w']
                    if sw > 0:
                        agg_visual_w += sw
                        agg_visual_wx += concept.emit_stats['sum_wx']
                        agg_visual_wx2 += concept.emit_stats['sum_wx2']

            # M-step: conservative update (η=0.1)
            candidate_priors = copy.deepcopy(self.priors)
            eta = 0.1

            sum_roles = sum(agg_role.values())
            if sum_roles > 0:
                for r in ROLES:
                    target_alpha = 0.1 + (agg_role[r] / sum_roles) * 5.0
                    candidate_priors.alpha[r] = (
                        (1 - eta) * candidate_priors.alpha[r] +
                        eta * target_alpha
                    )

            sum_repeats = sum(agg_repeat.values())
            if sum_repeats > 0:
                for k in REPEAT_RANGE:
                    target_gamma = 0.1 + (agg_repeat[k] / sum_repeats) * 4.0
                    candidate_priors.gamma[k] = (
                        (1 - eta) * candidate_priors.gamma[k] +
                        eta * target_gamma
                    )

            # NIG moment matching (conservative) — skip in discrete mode
            if agg_visual_w > 1.0 and self.priors.delta is None:
                mu_agg = agg_visual_wx / agg_visual_w
                var_agg = (agg_visual_wx2 / agg_visual_w) - mu_agg ** 2
                var_agg = np.maximum(var_agg, 1e-6)
                candidate_priors.nig.mu0 = (
                    (1 - eta) * candidate_priors.nig.mu0 + eta * mu_agg)
                candidate_priors.nig.beta0 = (
                    (1 - eta) * candidate_priors.nig.beta0 +
                    eta * float(np.mean(var_agg)))

            # Validate candidate priors — only accept if val_acc improves
            old_priors = copy.deepcopy(self.priors)
            self.priors = candidate_priors
            val_acc = self._eval_accuracy(val_eps)

            train_acc = n_correct / max(n_total, 1)
            if verbose:
                print(f"  Epoch {epoch}: "
                      f"train_acc={train_acc:.1%}, "
                      f"val_acc={val_acc:.1%}, "
                      f"loglik={total_query_loglik:.1f}")
                print(f"    α={dict((r, round(v,2)) for r,v in self.priors.alpha.items())}")
                print(f"    γ={dict((k, round(v,2)) for k,v in self.priors.gamma.items())}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_priors = copy.deepcopy(self.priors)
                if verbose:
                    print(f"    ✓ accepted (val improved to {val_acc:.1%})")
            else:
                self.priors = old_priors  # revert
                if verbose:
                    print(f"    ✗ reverted (val {val_acc:.1%} < best {best_val_acc:.1%})")

        # Grid search for λ, β, tau_span on validation set
        self.priors = copy.deepcopy(best_priors)
        if val_eps:
            self._grid_search_temps(val_eps, verbose)

    def _eval_accuracy(self, episodes: List[Dict]) -> float:
        """Evaluate query accuracy with current priors on a set of episodes."""
        n_correct, n_total = 0, 0
        for ep in episodes:
            ep_learner = NSLearner(
                priors=copy.deepcopy(self.priors),
                n_em=2, beam_k=5, beam_width=15
            )
            ep_learner.study_episode(ep['support'])
            for q in ep.get('query', []):
                pred = ep_learner.predict(q['input'])
                n_total += 1
                if pred == q['output']:
                    n_correct += 1
        return n_correct / max(n_total, 1)

    def _grid_search_temps(self, val_episodes: List[Dict],
                           verbose: bool = False):
        """
        Grid search for λ, β, tau_span on validation episodes.
        Only accepts if improvement over current accuracy.
        """
        lam_grid = [0.1, 0.2, 0.3, 0.5]
        beta_grid = [1.0, 2.0, 3.0, 5.0]
        tau_span_grid = [0.2, 0.5, 1.0]

        current_acc = self._eval_accuracy(val_episodes)
        best_acc = current_acc
        best_lam = self.priors.lam
        best_beta = self.priors.beta
        best_tau_span = self.priors.tau_span

        for lam in lam_grid:
            for beta_t in beta_grid:
                for tau_s in tau_span_grid:
                    test_priors = copy.deepcopy(self.priors)
                    test_priors.lam = lam
                    test_priors.beta = beta_t
                    test_priors.tau_span = tau_s

                    n_correct, n_total = 0, 0
                    for ep in val_episodes:
                        ep_learner = NSLearner(
                            priors=test_priors,
                            n_em=2, beam_k=5, beam_width=15
                        )
                        ep_learner.study_episode(ep['support'])
                        for q in ep.get('query', []):
                            pred = ep_learner.predict(q['input'])
                            n_total += 1
                            if pred == q['output']:
                                n_correct += 1

                    acc = n_correct / max(n_total, 1)
                    if acc > best_acc:
                        best_acc = acc
                        best_lam = lam
                        best_beta = beta_t
                        best_tau_span = tau_s

        # Only update if grid search found improvement
        if best_acc > current_acc:
            self.priors.lam = best_lam
            self.priors.beta = best_beta
            self.priors.tau_span = best_tau_span

        if verbose:
            print(f"  Grid search: λ={best_lam}, β={best_beta}, "
                  f"τ_span={best_tau_span}, "
                  f"val_acc={best_acc:.1%} (was {current_acc:.1%})")

    # ── Convenience ─────────────────────────────────────────────

    def learn(self, examples: List[Dict], verbose: bool = False):
        """Alias for study_episode (compatibility with test harness)."""
        self.study_episode(examples, verbose=verbose)

    def snapshot(self) -> str:
        """Human-readable summary of learned concepts."""
        lines = ["NSLearner snapshot:"]
        for w in sorted(self.library):
            c = self.library[w]
            lines.append(c.snapshot(self.priors.alpha))
        return '\n'.join(lines)
