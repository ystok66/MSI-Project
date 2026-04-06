# NS Learner

A **Neuro-Symbolic BPL Agent** (Bayesian Program Learning) for few-shot compositional concept learning, inspired by Lake et al.'s "human-like" generalization framework.

## Overview

NS Learner is the core inference engine that learns compositional programs from a handful of input→output examples. Given a few demonstrations like `dax → RED`, `lug → BLUE`, `dax fep → RED RED RED`, it induces a symbolic program that generalizes to novel queries (e.g., `lug fep → BLUE BLUE BLUE`).

**Key idea**: Each word is modeled as a probabilistic concept (`NeuroConcept`) with Dirichlet-smoothed posteriors over **roles** (what it does) and **emissions** (what it produces). A beam search enumerates candidate program traces, and Soft-EM iteratively refines the posteriors from data.

### Two-Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Outer Loop: Meta-Training                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Empirical Bayes on background episodes               │  │
│  │  Learn: α (role priors), δ (emission priors),         │  │
│  │         λ (softmax temp), β (alignment sharpness),    │  │
│  │         τ_span (arity penalty), rsa_alpha, ...        │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓ Φ (inductive biases)             │
│  Inner Loop: Episode Learning                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Per-Episode Few-Shot Soft-EM                         │  │
│  │  1. Bootstrap: detect 1:1 noun mappings               │  │
│  │  2. EM iterations:                                    │  │
│  │     E-step: beam search → top-K program traces        │  │
│  │     M-step: accumulate weighted sufficient statistics  │  │
│  │  3. Predict: beam search → execute best trace          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Five Primitive Operations

The system learns to assign each word one of five **roles**, corresponding to primitives on an immutable stack machine:

| Primitive        | Role                          | Stack Effect                    | MLC Example  |
| ---------------- | ----------------------------- | ------------------------------- | ------------ |
| **EMIT**         | Push a color vector           | `... → ... [μ]`                 | `dax → BLUE` |
| **REPEAT**       | Pop top, push k copies        | `... [X] → ... [X×k]`           | `fep → ×3`   |
| **SWAP_INFIX**   | Pop 2, push in reversed order | `... [A] [B] → ... [B] [A]`     | `after`      |
| **CONCAT_INFIX** | Pop 2, concatenate            | `... [A] [B] → ... [A·B]`       | `and`        |
| **OVER_INFIX**   | Pop 2, surround               | `... [A] [B] → ... [A] [B] [A]` | `surround`   |

Each infix operation supports **variable arity**: `arity=n` means "bind the top n stack items as one expression before applying the operation". This is a latent variable inferred by beam search with a geometric span prior `P(arity=n) ∝ exp(-τ_span · (n-1))`.

## Module Reference

```
ns_learner/
├── ns_learner.py      # NSLearner: main agent (Soft-EM + meta-training)
├── ns_concept.py      # NeuroConcept: per-word probabilistic model
├── ns_inference.py    # Stack-based beam search + RSA pragmatics
├── ns_ast.py          # AST-based beam search (hierarchical scoping)
├── ns_primitives.py   # Stack machine engine (5 primitives)
├── ns_colors.py       # CIELAB color space utilities
└── ns_hpc.py          # Hippocampal fast-memory module (DG→CA3→CA1)
```

### `ns_learner.py` — Main Agent

The orchestrator. Contains two main classes:

**`GlobalPriors`** — Meta-learned inductive biases Φ shared across all words:

| Parameter   | Type              | Default                  | Description                                     |
| ----------- | ----------------- | ------------------------ | ----------------------------------------------- |
| `alpha`     | `Dict[str,float]` | EMIT:2.0, others:0.3–1.0 | Dirichlet prior over roles                      |
| `nig`       | `NIGParams`       | κ₀=0.1, α₀=1.0, β₀=1.0   | Normal-Inverse-Gamma prior for emission         |
| `delta`     | `Dict[str,float]` | None                     | Discrete emission prior (Dirichlet over colors) |
| `gauss`     | `bool`            | False                    | Use Gaussian emission model                     |
| `lam`       | `float`           | 0.3                      | Softmax temperature for alignment               |
| `beta`      | `float`           | 2.0                      | Alignment sharpness (KL weighting)              |
| `tau_span`  | `float`           | 0.5                      | Arity/span geometric penalty                    |
| `rsa_alpha` | `float`           | 0.5                      | RSA rationality parameter                       |
| `rsa_cost`  | `float`           | 0.2                      | RSA utterance cost per word                     |

**`NSLearner`** — The agent itself:

| Method                        | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| `study_episode(examples)`     | Learn from support via bootstrap + online Soft-EM         |
| `study_episode_ast(examples)` | Same but using AST-based beam search                      |
| `predict(words)`              | Predict output using stack-based beam search              |
| `predict_ast(words)`          | Predict output using AST-based beam search                |
| `meta_train(episodes)`        | Outer loop: learn Φ from background tasks via grid search |
| `learn(examples)`             | Alias for `study_episode` (compatibility)                 |
| `snapshot()`                  | Human-readable concept summary                            |

### `ns_concept.py` — Per-Word Probabilistic Model

**`NeuroConcept`** — Each word carries:

1. **Role posterior**: `P(role | word)` — Dirichlet over {EMIT, REPEAT, SWAP_INFIX, CONCAT_INFIX, OVER_INFIX}
2. **Repeat posterior**: `P(k | word)` — Dirichlet over {1, 2, 3, 4}
3. **Emission posterior** (three modes):
   - **Discrete** (`delta≠None`): Dirichlet over color names
   - **Continuous NIG** (`gauss=False`): Normal-Inverse-Gamma → Student-t predictive
   - **Gaussian** (`gauss=True`): Direct Gaussian log-likelihood

Key operations:

- `log_role_prob(role, alpha)` — Dirichlet predictive: `(α + count) / Σ`
- `log_emit_prob(vec, nig, ...)` — Emission score under chosen model
- `map_color(nig, ...)` — MAP color (argmax over palette)
- `soft_update(weight, role, vec, k)` — M-step: accumulate weighted sufficient statistics
- `emit_entropy(delta)` — Shannon entropy `H[P(color|word)]` for diagnostics

**`NIGParams`** — Normal-Inverse-Gamma hyperparameters (μ₀, κ₀, α₀, β₀) defining the prior over Gaussian emission means and variances.

### `ns_inference.py` — Stack-Based Beam Search

Abductive beam search over program traces scored by:

```
S(z) = Σ_i [ log P(role_i | word_i)     # role prior
           + log P(emit_i | word_i)      # emission likelihood
           + log P(arity_i | τ_span)     # span prior
           + λ · align(target, state)    # soft alignment bonus
           + mem_bias_i ]                # HPC memory boost (optional)
```

Key components:

| Function                                                     | Description                                         |
| ------------------------------------------------------------ | --------------------------------------------------- |
| `infer_top_k(instruction, target, library, priors, ...)`     | Public API: beam search with optional RSA           |
| `_infer_top_k_inner(...)`                                    | Core beam search implementation                     |
| `_expand_emit(...)`                                          | Expand EMIT branches with soft alignment            |
| `_expand_repeat(...)`                                        | Expand REPEAT branches with arity enumeration       |
| `_expand_infix(...)`                                         | Expand infix operations (SWAP/CONCAT/OVER)          |
| `execute_trace(trace, library, nig, ...)`                    | Re-execute a trace to produce output vectors        |
| `soft_edit_distance(pred, target, σ)`                        | Gaussian-kernel soft Levenshtein distance           |
| `context_role_prior(word, idx, n, depth, alpha)`             | Positional context boosts for unseen words          |
| `log_span_prior(arity, τ_span)`                              | Geometric penalty on arity                          |
| `rsa_pragmatic_term(u_obs, m_vecs, library, priors, alt_us)` | RSA S1 speaker log-probability                      |
| `generate_alternatives(instruction, library, priors)`        | Generate alternative utterances for RSA competition |

**RSA (Rational Speech Acts)**: After beam search finds top-K candidates, an optional pragmatic reranking computes `log P_S1(u | m)` — the probability that a rational speaker would choose utterance `u` to convey meaning `m`, given alternatives. This favors interpretations where each word contributes meaningfully.

### `ns_ast.py` — AST-Based Beam Search

An alternative beam search that constructs a **latent Abstract Syntax Tree** instead of flat stack operations. This correctly handles **hierarchical scoping** (e.g., `A surround B after C thrice` where `thrice` binds only to `C`).

Key mechanism — **deferred hole-fill**:

- When an infix word opens a scope, it creates a "hole" in the AST
- Subsequent EMITs can either: (a) fill the hole (operand binding) or (b) create a new root (independent)
- This naturally handles nested composition without explicit bracket parsing

```
Example: "2 after 3 surround DAX"
  AST: SWAP(2→RED, OVER(3→GREEN, DAX→YELLOW))
  Output: 3 DAX 3 2  ← correct nested interpretation

  Stack decoder gets: 3 2 DAX 2  ← wrong (flat left-to-right)
```

| Class/Function              | Description                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `ASTNode`                   | Tree node: kind (EMIT/REPEAT/SWAP/CONCAT/OVER), word, children, emit_vec, repeat_k |
| `ASTBeamEntry`              | Beam entry: log_score, instruction index, parse state                              |
| `ParseState`                | Partial parse: list of roots + list of open holes                                  |
| `Hole`                      | An unfilled operand slot in the AST                                                |
| `eval_ast(node)`            | Recursively evaluate an AST node to produce color vectors                          |
| `infer_top_k_ast(...)`      | Main AST beam search with deferred hole-fill                                       |
| `ast_to_trace_steps(roots)` | Convert AST to flat TraceStep list (for M-step compatibility)                      |

### `ns_primitives.py` — Stack Machine Engine

Immutable, functional-style stack machine supporting beam search branching without mutation:

| Class             | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `StackItem`       | Immutable sequence of color vectors (one logical unit)    |
| `StackState`      | Immutable stack of StackItems with push/pop/pop_n/flatten |
| `Primitive`       | Abstract base for all primitives                          |
| `PrimEmit`        | Push one color vector                                     |
| `PrimRepeat`      | Pop `arity` items as X, push X×k                          |
| `PrimSwapInfix`   | Pop `arity` as B, pop `arity` as A, push B then A         |
| `PrimConcatInfix` | Pop `arity` as B, pop `arity` as A, push A·B concatenated |
| `PrimOverInfix`   | Pop `arity` as B, pop `arity` as A, push A·B·A            |

All operations return **new** `StackState` instances (never mutate), enabling safe branching in beam search.

### `ns_colors.py` — CIELAB Color Space

Provides continuous color representations for the Gaussian emission model:

| Function                | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `rgb_to_lab(rgb)`       | sRGB (0-255) → CIELAB [L*, a*, b*] via D65 white     |
| `normalize_lab(raw)`    | Normalize CIELAB to [0,1] per dimension              |
| `add_noise(vec, sigma)` | Gaussian noise injection (sigma in raw Lab units)    |
| `nearest_color(vec)`    | Nearest-palette quantization by L2 in normalized Lab |
| `lab_palette_mean()`    | Mean palette vector (for NIG μ₀ initialization)      |

The 6-color MLC palette in normalized Lab space:

| Color  | RGB           |  L\* |   a\* |    b\* |
| ------ | ------------- | ---: | ----: | -----: |
| BLUE   | (0,0,255)     | 32.3 |  79.2 | -107.9 |
| RED    | (255,0,0)     | 53.2 |  80.1 |   67.2 |
| GREEN  | (0,128,0)     | 46.2 | -51.7 |   49.9 |
| YELLOW | (255,255,0)   | 97.1 | -21.6 |   94.5 |
| PURPLE | (128,0,128)   | 29.8 |  58.9 |  -36.5 |
| PINK   | (255,192,203) | 83.6 |  24.1 |    3.3 |

### `ns_hpc.py` — Hippocampal Fast Memory

An episode-level hippocampal system following the DG→CA3→CA1 anatomy. This is the **monolithic** version; `cls_learner/layer2_hpc/` provides a modularized refactor.

**Pipeline**: EventEncoder → DGEncoder → CA3Memory → CA1Comparator → MemBias

| Component       | Brain Region      | Function                                                |
| --------------- | ----------------- | ------------------------------------------------------- |
| `EventEncoder`  | Entorhinal Cortex | Hash-BOW (64d) + bigram (64d) → 128d L2-normalized      |
| `DGEncoder`     | Dentate Gyrus     | Random projection + kWTA sparse coding (512d, k=30)     |
| `CA3Memory`     | CA3               | Hopfield auto-associative memory + pattern completion   |
| `CA1Comparator` | CA1               | Blockwise Mahalanobis mismatch → sigmoid gating         |
| `EpisodeHPC`    | Full HPC          | Top-level wrapper: write → retrieve → gate → role boost |

Output: `MemBias(role_boost, lam_mem, delta, mode)` where:

- `role_boost`: per-word log-softmax role distributions from retrieved memories
- `lam_mem`: gating strength ∈ [0, 1] (0 = ignore, 1 = full trust)
- `mode`: {retrieve, mixed, explore} based on novelty

## Quick Start

```python
from ns_learner.ns_learner import NSLearner

# Create learner with default priors
learner = NSLearner(n_em=3, beam_k=10, beam_width=30)

# Few-shot learning from support examples
learner.study_episode([
    {'input': ['dax'],        'output': ['RED']},
    {'input': ['lug'],        'output': ['BLUE']},
    {'input': ['dax', 'fep'], 'output': ['RED', 'RED', 'RED']},
])

# Predict query
print(learner.predict(['lug', 'fep']))
# → ['BLUE', 'BLUE', 'BLUE']

# Inspect learned concepts
learner.snapshot()
# → dax: EMIT RED | lug: EMIT BLUE | fep: REPEAT k=3
```

### Using the AST decoder

```python
learner = NSLearner()

# AST handles nested composition correctly
learner.study_episode_ast([
    {'input': ['1'],               'output': ['BLUE']},
    {'input': ['2'],               'output': ['RED']},
    {'input': ['3'],               'output': ['GREEN']},
    {'input': ['DAX'],             'output': ['YELLOW']},
    {'input': ['DAX', 'after', '1'], 'output': ['BLUE', 'YELLOW']},
    {'input': ['DAX', 'surround', '2'], 'output': ['YELLOW', 'RED', 'YELLOW']},
])

print(learner.predict_ast(['2', 'after', '3', 'surround', 'DAX']))
# → ['GREEN', 'YELLOW', 'GREEN', 'RED']  ← correct nested scoping
```

### Meta-training (outer loop)

```python
learner = NSLearner()

# Learn priors from background tasks
learner.meta_train(
    background_episodes=[
        {'support': [...], 'query': [...]},
        {'support': [...], 'query': [...]},
        ...  # 50–100 episodes
    ],
    n_epochs=3,
    verbose=True,
)

# Now use learned priors for new episodes
learner.study_episode(new_support)
pred = learner.predict(new_query)
```

## Inference Scoring Details

The beam search scores each candidate trace `z = (z₁, ..., z_n)` as:

```
S(z) = Σᵢ [ log P(roleᵢ | wordᵢ; α)           # Dirichlet role prior
           + 𝟙[role=EMIT] · log P(vecᵢ | wordᵢ; NIG)  # emission likelihood
           + 𝟙[role=REPEAT] · log P(kᵢ | wordᵢ; γ)    # repeat prior
           + log P(arityᵢ | τ_span)              # span prior: -τ·(arity-1)
           + λ · softmax_align(target, state)     # soft alignment bonus
           + mem_biasᵢ ]                          # HPC role boost (optional)
```

At the end of beam search, an optional **RSA pragmatic term** reranks the top-K:

```
S_RSA(u, m) = α · log L₀(m | u) - cost · |u|
            = α · [best_trace_score(u → m)] - cost · |u|
```

where alternatives `u'` are generated by substituting/deleting REPEAT words, and the pragmatic listener probability is:

```
P_L1(m | u) ∝ exp(S_RSA(u, m)) / Σ_{u'} exp(S_RSA(u', m))
```

## Design Decisions

1. **Online EM (interleaved per-example)**: Each example's M-step immediately updates concept counts, benefiting subsequent examples' E-steps within the same iteration. This is more sample-efficient than batch EM for few-shot settings.

2. **Immutable stack machine**: All `StackState` operations return new instances, enabling safe beam search branching without deep-copy overhead.

3. **Dual decoder**: The **stack decoder** is faster and handles most tasks, while the **AST decoder** is needed for nested composition (achieves 90% vs 70% on mini-SCAN).

4. **Soft alignment**: Uses a Gaussian-kernel soft edit distance rather than hard string matching, enabling gradient-like signal even for partial matches.

5. **Context-conditioned role prior**: Unseen words get positional boosts (e.g., sentence-final → REPEAT, between two known words → infix role), reducing the search space.

6. **Three emission models**: Discrete (BPL-style Dirichlet), Continuous NIG (Student-t predictive), and Gaussian — selectable per experiment for noise robustness comparisons.

## Relationship to CLS Learner

`cls_learner` wraps `ns_learner` with a three-layer CLS architecture:

- **Layer 1 (Cortex)**: Uses `NeuroConcept` library and `ns_inference`/`ns_ast` beam search
- **Layer 2 (HPC)**: Modularized version of `ns_hpc.py` (split into encoder, dg, ca3, ca1, replay)
- **Layer 3 (Control)**: Orchestrates search and applies cortex-only reranking at prediction time

When CLS is configured with `use_hpc=False`, it produces **identical** results to NSLearner (verified on 100 tasks, 0 divergences).
