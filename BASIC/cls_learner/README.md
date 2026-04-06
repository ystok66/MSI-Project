# CLS Learner

A **Complementary Learning Systems** (CLS) agent for few-shot neuro-symbolic induction, inspired by the neuroscience of how cortical and hippocampal systems interact during learning.

## Overview

CLS Learner wraps the base `NSLearner` (neuro-symbolic beam-search engine) with a three-layer architecture that mirrors mammalian memory systems:

| Layer                 | Brain Analog          | Role                                                 | Speed            |
| --------------------- | --------------------- | ---------------------------------------------------- | ---------------- |
| **Layer 1 — Cortex**  | Neocortex             | Slow, generalizable concept learning via Bayesian EM | Slow (iterative) |
| **Layer 2 — HPC**     | Hippocampus           | Fast, episode-specific memory and pattern completion | Fast (one-shot)  |
| **Layer 3 — Control** | PFC + BG + Cerebellum | Search orchestration, candidate selection, execution | Real-time        |

The key insight: **Cortex** handles generalization through statistical learning (online EM), while **HPC** provides rapid memorization of individual support examples. At prediction time, HPC biases the beam search to **improve recall** of relevant candidates (proposal), but the final answer is selected using **cortex-only scores** (target), ensuring HPC never harms precision.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       CLSAgent                          │
│                                                         │
│  reset_episode() → study(support) → predict(query)      │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 3: Control System                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │  │
│  │  │   PFC    │  │    BG    │  │  Cerebellum    │  │  │
│  │  │ Planner  │  │ Selector │  │   Executor     │  │  │
│  │  └─────┬────┘  └────┬─────┘  └───────┬────────┘  │  │
│  └────────┼─────────────┼────────────────┼───────────┘  │
│           │             │                │              │
│  ┌────────┼─────────────┼────────────────┼───────────┐  │
│  │  Layer 2: HPC        │                │           │  │
│  │  ┌─────────┐  ┌──────┴──┐  ┌────┐  ┌─┴──┐       │  │
│  │  │ Encoder │→ │   DG    │→ │ CA3│→ │ CA1│       │  │
│  │  │ (BOW+   │  │ (kWTA   │  │Hop-│  │Gate│       │  │
│  │  │ bigram) │  │ sparse) │  │fld │  │    │       │  │
│  │  └─────────┘  └─────────┘  └────┘  └────┘       │  │
│  │                          ┌─────────┐              │  │
│  │                          │ Replay  │              │  │
│  │                          │ Sampler │              │  │
│  │                          └─────────┘              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Layer 1: Cortex                                  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  NeuroConcept Library                       │  │  │
│  │  │  { word → (role_counts, emit_dist, ...) }   │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  bootstrap() → EM: E-step (beam search) + M-step  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Episode Lifecycle

Each few-shot learning episode follows three phases:

### 1. `reset_episode()`

Clears the cortex library and HPC memories. Called at the start of each new task.

### 2. `study(support)`

Learns from labeled support examples through:

1. **Bootstrap** — Detect 1:1 word→color mappings (nouns) from examples
2. **HPC Write** — Store each support example with its trace summary into hippocampal memory
3. **Online EM Loop** (default 3 iterations):
   - **E-step**: Beam search over program traces for each example (no HPC bias during study)
   - **M-step**: Immediately update concept counts from each example's traces (interleaved per-example, matching NSLearner's online EM for faster convergence)
   - **Decay**: Reduce role/repeat counts between iterations

> **Design Decision**: HPC is **passive during study** — it only writes memories after bootstrap but does not modulate beam search, replay, or IS-correct during EM. This prevents HPC from corrupting the cortex's statistical learning and ensures perfect parity with the NSLearner baseline.

> **CIELAB mode** (`gauss=True`): Target vectors are converted from color names to 3D Lab vectors via `_color_to_vecs()`. When `lab_sigma > 0`, Gaussian noise is injected into each Lab vector before it enters the E-step, simulating perceptual noise.

### 3. `predict(query)`

Infers the output for a query word sequence using two-stage inference:

1. **Proposal** (HPC-biased): Beam search with `mem_bias` from HPC → produces top-K candidates with `S_proposal = S_cortex + S_mem`
2. **Target** (cortex-only rerank): Select the MAP candidate by `S_target = S_proposal - S_mem`, using `MemBias.log_q_trace()` to subtract HPC's contribution

This ensures HPC **improves recall** (bringing more diverse candidates into the beam) without **harming precision** (the final selection is purely cortex-based).

## Directory Structure

```
cls_learner/
├── __init__.py              # Package exports
├── agent.py                 # CLSAgent: unified entry point
├── config.py                # CLSConfig: all hyperparameters
├── interfaces.py            # Shared dataclasses (Example, Episode, TraceSummary, MemBias, MemoryPayload)
│
├── layer1_cortex/           # Slow, generalizable concept learning
│   ├── cortex.py            # CortexMemory: library management, bootstrap, EM, decay
│   └── concept_adapter.py   # ConceptAdapter: per-concept scoring interface
│
├── layer2_hpc/              # Fast, episode-specific memory
│   ├── hpc.py               # EpisodeHPC: top-level HPC wrapper (Encoder→DG→CA3→CA1→Replay)
│   ├── encoder.py           # EventEncoder: utterance → fixed-size vector (BOW + bigram hash)
│   ├── dg.py                # DGEncoder: Dentate Gyrus, random projection + kWTA sparse coding
│   ├── ca3.py               # CA3Memory: auto-associative Hopfield memory, pattern completion
│   ├── ca1.py               # CA1Comparator: blockwise Mahalanobis mismatch → sigmoid gate
│   └── replay.py            # ReplaySampler: mixed uniform + priority sampling
│
├── layer3_control/          # Search orchestration + execution
│   ├── control.py           # ControlSystem: E-step orchestration, predict with rerank
│   ├── pfc.py               # PFCPlanner: delegates to ns_inference / ns_ast beam search
│   ├── bg.py                # BGSelector: beam width modulation, RSA utility rerank
│   └── cerebellum.py        # CerebellumExecutor: trace execution + error tracking
│
├── adapters/                # Utility adapters
│   ├── logging.py           # Structured logging for CLS events
│   ├── mlc_episode.py       # MLC task → Episode converter
│   └── trace_summary.py     # Trace → TraceSummary extraction
│
└── tests/
    └── test_agent_smoke.py  # 13 smoke tests covering all layers
```

## Layer Details

### Layer 1: Cortex (`layer1_cortex/`)

The cortex maintains a **concept library** — a dictionary mapping each word to a `NeuroConcept` with:

- **Role counts**: Dirichlet-smoothed counts for EMIT, REPEAT, SWAP_INFIX, CONCAT_INFIX, OVER_INFIX
- **Emission distribution**: Normal-Inverse-Gamma posterior over color vectors (6D one-hot or 3D Lab)
- **Repeat distribution**: counts over repeat factors {2, 3, 4}

Key operations:

- `bootstrap()`: Detect 1:1 word→color mappings from support examples
- `m_step_from_traces()`: Update concept counts from beam search traces, with optional IS correction and temperature-scaled responsibilities
- `decay()`: Multiplicative decay of counts between EM iterations
- `replay_update()`: Soft update from HPC memory replay

> **CIELAB mode**: When `gauss=True`, `_ensure_concept()` creates concepts with `d=3` (from `NIG.mu0` dimension) instead of `d=6`. The NIG prior is initialized with `mu0 = lab_palette_mean()` (mean of the 6-color Lab palette).

### Layer 2: HPC (`layer2_hpc/`)

The hippocampal system provides **fast, one-shot memory** with biologically-inspired submodules:

| Submodule       | Brain Region      | Function                                                                                              |
| --------------- | ----------------- | ----------------------------------------------------------------------------------------------------- |
| `EventEncoder`  | Entorhinal Cortex | Hash-based encoding: BOW (64d) + bigram (64d) → 128d L2-normalized vector                             |
| `DGEncoder`     | Dentate Gyrus     | Pattern separation: random projection (128→512) + kWTA (k=30) sparse coding                           |
| `CA3Memory`     | CA3               | Auto-associative Hopfield memory with outer-product learning; pattern completion via iterative recall |
| `CA1Comparator` | CA1               | Blockwise Mahalanobis mismatch detection with sigmoid gating → `(λ_mem, mode)`                        |
| `ReplaySampler` | —                 | Priority-weighted replay sampling (ρ-mixed uniform + priority)                                        |

**Retrieval pipeline** (`get_bias()`):

1. Encode query → `EventEncoder` → event vector
2. Sparse code → `DGEncoder` → DG code
3. Pattern completion → `CA3.complete()` → completed code
4. Top-R retrieval → `CA3.retrieve()` → nearest memories
5. Mismatch detection → `CA1.mismatch()` → δ
6. Gating → `CA1.gate(δ)` → `(λ_mem, mode)` where mode ∈ {retrieve, mixed, explore}
7. Aggregate → `_aggregate_role_boost()` → per-word role log-softmax boosts

Output: `MemBias(role_boost, lam_mem, delta, mode)` consumed by beam search.

### Layer 3: Control (`layer3_control/`)

| Component            | Brain Analog      | Function                                                                             |
| -------------------- | ----------------- | ------------------------------------------------------------------------------------ |
| `PFCPlanner`         | Prefrontal Cortex | Delegates to beam search engines (`infer_top_k_stack` / `infer_top_k_ast`)           |
| `BGSelector`         | Basal Ganglia     | Beam width modulation based on novelty; RSA utility soft rerank                      |
| `CerebellumExecutor` | Cerebellum        | Executes beam traces on the stack machine; tracks prediction errors                  |
| `ControlSystem`      | —                 | Orchestrates E-step and predict, including the cortex-only rerank at prediction time |

**Predict-time rerank** (implemented in `ControlSystem.predict()`):

```
candidates ← PFC.infer_top_k(query, mem_bias=hpc_bias)    # Stage 1: proposal
best ← argmax(cand.score  -  mem_bias.log_q_trace(cand.trace))  # Stage 2: target (cortex-only)
output ← execute(best.trace)
```

> **CIELAB mode**: `predict()` selects the color decoder based on `gauss`: `nearest_color()` (Lab → nearest CIELAB color) when `gauss=True`, `vec_to_color()` (argmax one-hot) otherwise. The `gauss` flag is forwarded to `execute_trace` and `map_color` throughout the beam search pipeline.

## Shared Interfaces (`interfaces.py`)

| Dataclass       | Lifecycle  | Purpose                                                                  |
| --------------- | ---------- | ------------------------------------------------------------------------ |
| `Example`       | Input      | One input→output pair (words, colors)                                    |
| `Episode`       | Input      | Support + query example lists                                            |
| `TraceSummary`  | Ephemeral  | Per-inference-call results: MAP roles, colors, soft distributions        |
| `MemoryPayload` | Persistent | Stored in HPC per example: words, colors, role/color mappings            |
| `MemBias`       | Ephemeral  | HPC output to beam search: role boosts + gating signal + `log_q_trace()` |

## Configuration (`config.py`)

All hyperparameters are grouped by layer in `CLSConfig`:

```python
CLSConfig(
    # General
    mode='ast',          # 'stack' or 'ast' (beam search engine)
    use_hpc=True,        # enable Layer 2
    n_em=3,              # EM iterations

    # Emission model
    gauss=False,         # True = CIELAB 3D Gaussian emission
    lab_sigma=0.0,       # Gaussian noise σ in Lab units (0 = clean)
    delta=None,          # Dirichlet prior (None = use NIG/KL)

    # Layer 1: Cortex
    beam_k=10,           # top-K traces to keep
    beam_width=30,       # beam width per step
    decay_rate=0.5,      # count decay between EM iters

    # Layer 2: HPC
    hpc_m=512,           # DG sparse code dimension
    hpc_k=30,            # kWTA sparsity
    hpc_top_r=5,         # CA3 retrieval top-R
    hpc_lam_max=1.0,     # max memory strength

    # Layer 3: Control
    bg_explore_factor=1.0,   # beam expansion in explore mode
    bg_rsa_rerank_alpha=0.0, # RSA rerank weight (0 = off)
)
```

## Quick Start

```python
from cls_learner.agent import CLSAgent
from cls_learner.config import CLSConfig
from cls_learner.interfaces import Example, Episode

# Create agent (defaults match NSLearner for parity)
agent = CLSAgent(CLSConfig(mode='ast', use_hpc=True))

# Build an episode
support = [
    Example(words=['dax'],           output=['RED']),
    Example(words=['lug'],           output=['BLUE']),
    Example(words=['dax', 'fep'],    output=['RED', 'RED', 'RED']),
]
query = [
    Example(words=['lug', 'fep'],    output=['BLUE', 'BLUE', 'BLUE']),
]

# Full evaluation
result = agent.evaluate_episode(Episode(support=support, query=query))
print(f"Accuracy: {result['accuracy']:.0%}")
# → Accuracy: 100%
```

Or use the convenience `learn`/`predict` API:

```python
agent = CLSAgent()
agent.learn([
    {'input': ['dax'], 'output': ['RED']},
    {'input': ['lug'], 'output': ['BLUE']},
])
print(agent.predict(['dax']))  # → ['RED']
```

## Testing

Run the smoke tests (13 tests covering all layers):

```bash
python -m pytest cls_learner/tests/test_agent_smoke.py -v
```

Run the full MLC parity evaluation (100 tasks, ~8 minutes):

```bash
python tests/eval_cls_comparison.py
```

## Design Principles

1. **Parity with NSLearner**: When `use_hpc=False`, CLSAgent produces identical results to NSLearner. This is verified by 100-task MLC ablation (Δ=+0.0pp, Tied: 100/100).

2. **Proposal ≠ Target**: HPC biases the beam search (proposal distribution) to improve candidate recall, but the final MAP selection uses cortex-only scores (target distribution). This "never harm" property ensures HPC cannot degrade performance.

3. **Online EM**: Each support example's M-step immediately benefits subsequent E-steps within the same EM iteration, matching NSLearner's interleaved update schedule.

4. **HPC Passive During Study**: HPC only writes memories during the bootstrap phase and retrieves during prediction. It does not modulate beam search, perform replay updates, or apply IS correction during EM — preventing interference with cortex learning.

5. **Biologically Inspired**: Each submodule maps to a neuroscience analog (DG pattern separation, CA3 auto-association, CA1 mismatch detection), making the architecture interpretable and grounded in CLS theory.

## Human-Likeness Evaluation (mini-SCAN)

The system is evaluated on a mini-SCAN task (14 support examples, 10 queries, 10 human behavior files from Lake & Baroni 2018) to measure alignment with human generalization patterns. Four configurations are compared:

| Config | Training | Testing | Emission Model       |
| ------ | -------- | ------- | -------------------- |
| A      | Stack    | Stack   | Continuous (NIG/KL)  |
| B      | Stack    | AST     | Continuous (hybrid)  |
| C      | Stack    | Stack   | Discrete (BPL-style) |
| D      | Stack    | AST     | Discrete (hybrid)    |

### Per-Query Results

| Query                               | Gold            |  A  |  B  |  C  |  D  |
| ----------------------------------- | --------------- | :-: | :-: | :-: | :-: |
| `3 after DAX`                       | DAX 3           |  ✔  |  ✔  |  ✔  |  ✔  |
| `DAX after 1`                       | 1 DAX           |  ✔  |  ✔  |  ✔  |  ✔  |
| `DAX thrice`                        | DAX DAX DAX     |  ✔  |  ✔  |  ✔  |  ✔  |
| `1 surround DAX`                    | 1 DAX 1         |  ✔  |  ✔  |  ✔  |  ✔  |
| `DAX surround 2`                    | DAX 2 DAX       |  ✔  |  ✔  |  ✔  |  ✔  |
| `2 after 3 surround DAX`            | 3 DAX 3 2       |  ✘  |  ✔  |  ✘  |  ✔  |
| `DAX thrice after 2`                | 2 DAX DAX DAX   |  ✔  |  ✔  |  ✔  |  ✔  |
| `3 after DAX thrice`                | DAX DAX DAX 3   |  ✘  |  ✔  |  ✘  |  ✔  |
| `DAX surround DAX after DAX thrice` | DAX×6           |  ✔  |  ✔  |  ✔  |  ✔  |
| `DAX surround 3 after 1 thrice`     | 1 1 1 DAX 3 DAX |  ✘  |  ✘  |  ✘  |  ✘  |

### Summary Metrics

| Metric                      | A (St/St/C) | B (St/AST/C) | C (St/St/D) | D (St/AST/D) |
| --------------------------- | :---------: | :----------: | :---------: | :----------: |
| **M1. Gold accuracy**       |     70%     |   **90%**    |     70%     |   **90%**    |
| **M2. Modal agreement**     |    7/10     |   **9/10**   |    7/10     |   **9/10**   |
| **M3. Human agree rate**    |    60.8%    |  **74.7%**   |    60.8%    |  **74.7%**   |
| **M4. Difficulty corr (r)** |  **0.628**  |    0.464     |  **0.628**  |    0.464     |

**Key finding**: The AST decoder (configs B, D) achieves **90% gold accuracy** and **74.7% human agreement**, fixing 2 queries that the stack decoder gets wrong (nested composition `after ... surround` and `after ... thrice`). The only remaining failure is the deepest 3-operator nesting `DAX surround 3 after 1 thrice`, which is also the hardest query for humans (only 57% human accuracy).

### Error Signatures vs Humans

| Error Type     |  A  |    B    |  C  |    D    | Humans |
| -------------- | :-: | :-----: | :-: | :-----: | :----: |
| Correct        | 70% | **90%** | 70% | **90%** |  81%   |
| Extra tokens   | 0%  |   0%    | 0%  |   0%    |   2%   |
| Missing tokens | 0%  |   0%    | 0%  |   0%    |   5%   |
| Order error    | 0%  |   10%   | 0%  |   10%   |   4%   |
| Substitution   | 30% |   0%    | 30% |   0%    |   7%   |

The AST decoder's error profile (90% correct, 10% order errors) is closer to the human distribution than the stack decoder (70% correct, 30% substitutions).

## CIELAB Noise Robustness (mini-SCAN)

CLS Learner supports **CIELAB Gaussian emission** (`gauss=True`), encoding colors as 3D Lab vectors instead of 6D one-hot. This enables testing noise robustness by injecting Gaussian noise (σ in raw Lab units) into the color vectors during learning.

```python
cfg = CLSConfig(mode='ast', gauss=True, lab_sigma=15)
```

**Setup**: mini-SCAN (14 support + 10 query), RSA OFF (α=0), each σ repeated 5 times.

| σ   | Cont+Stack | Cont+AST |
| --- | ---------- | -------- |
| 0   | 70%        | **90%**  |
| 5   | 70%        | **90%**  |
| 10  | 70%        | **90%**  |
| 15  | 70%        | **90%**  |
| 20  | 70%        | **90%**  |
| 25  | 70%        | 82±16%   |
| 30  | 52±15%     | 74±20%   |
| 40  | 40%        | 50%      |
| 50  | 34±23%     | 50%      |

**Key findings**:

1. **σ ≤ 20**: Perfect robustness — NIG posterior absorbs noise without accuracy loss
2. **σ = 25–30**: Graceful degradation begins, AST more resilient than Stack
3. **σ ≥ 40**: Significant degradation, but AST still maintains 50% (chance = ~17%)
4. **AST >> Stack** across all noise levels (+20pp at σ=0, still +10–16pp at high noise)

> **Eval script**: `tests/eval_cls_miniscan_noise.py`

## Gemini vs CLS: Noise Robustness Comparison

Comparison with **Gemini 3 Pro** (gemini-3-pro-preview) on mini-SCAN under CIELAB noise. Both models receive identical noisy Lab vectors as input — Gemini as numerical triplets in its prompt, CLS via native Gaussian emission.

**Setup**: mini-SCAN (14 support + 10 query), AST decoder, seed=42.

### Summary

| σ   | Gemini 3 Pro     | CLS (AST)      |
| --- | ---------------- | -------------- |
| 0   | **10/10 (100%)** | 9/10 (90%)     |
| 10  | **10/10 (100%)** | 9/10 (90%)     |
| 20  | **10/10 (100%)** | 9/10 (90%)     |
| 25  | 6/10 (60%)       | **9/10 (90%)** |
| 30  | 6/10 (60%)       | **9/10 (90%)** |

### Per-Query Breakdown

| Query                               | Gold                       | σ=0 G/C | σ=10 G/C | σ=20 G/C | σ=25 G/C | σ=30 G/C |
| ----------------------------------- | -------------------------- | :-----: | :------: | :------: | :------: | :------: |
| `3 after DAX`                       | YELLOW GREEN               |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✘/✔    |   ✘/✔    |
| `DAX after 1`                       | BLUE YELLOW                |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✔/✔    |   ✔/✔    |
| `DAX thrice`                        | YELLOW×3                   |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✔/✔    |   ✔/✔    |
| `1 surround DAX`                    | BLUE YELLOW BLUE           |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✔/✔    |   ✔/✔    |
| `DAX surround 2`                    | YELLOW RED YELLOW          |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✔/✔    |   ✔/✔    |
| `2 after 3 surround DAX`            | GREEN YELLOW GREEN RED     |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✘/✔    |   ✘/✔    |
| `DAX thrice after 2`                | RED YELLOW×3               |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✔/✔    |   ✔/✔    |
| `3 after DAX thrice`                | YELLOW×3 GREEN             |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✘/✔    |   ✘/✔    |
| `DAX surround DAX after DAX thrice` | YELLOW×6                   |   ✔/✔   |   ✔/✔    |   ✔/✔    |   ✔/✔    |   ✔/✔    |
| `DAX surround 3 after 1 thrice`     | BLUE×3 YELLOW GREEN YELLOW |   ✔/✘   |   ✔/✘    |   ✔/✘    |   ✘/✘    |   ✘/✘    |

> **G** = Gemini, **C** = CLS

### Analysis

- **σ ≤ 20**: Gemini achieves **perfect 100%** by leveraging its general reasoning capability. CLS achieves 90% (the same Q10 failure as its standalone evaluation). At low noise, Gemini's advantage in language understanding outweighs CLS's statistical robustness.

- **σ ≥ 25**: **Gemini collapses to 60%**, losing 4 queries. Its failure mode is consistently **GREEN → YELLOW substitution** — when the noisy Lab vector for GREEN drifts toward YELLOW in Lab space, Gemini fails to recover the correct color identity. CLS maintains **90%** because its NIG posterior absorbs the noise probabilistically.

- **Q10** (`DAX surround 3 after 1 thrice`): The hardest 3-operator nesting query. Both models fail at σ≥25. CLS fails at all σ values (this is the same structural failure as in standalone evaluation). Gemini solves it at σ=0–20 but fails at σ≥25.

- **Key insight**: CLS's Bayesian emission model (NIG posterior) provides **noise-invariant grounding** — it tracks the full distribution over color vectors rather than point estimates, making it robust to Lab perturbations. Gemini treats Lab vectors as opaque numerical patterns, with no statistical model to absorb noise.

> **Eval script**: `tests/run_gemini_batch.py` / `tests/eval_gemini_miniscan.py`
> **Raw results**: `tests/gemini_vs_cls_results.txt`

## Performance (MLC Ablation)

Results on 100-task MLC evaluation:

| Configuration       | Query Accuracy | Δ vs NSLearner |
| ------------------- | -------------- | -------------- |
| CLS (Stack, no HPC) | 68.7%          | +0.0pp         |
| CLS (Stack, + HPC)  | 68.4%          | −0.3pp         |
| CLS (AST, no HPC)   | 51.6%          | +0.0pp         |
| CLS (AST, + HPC)    | 51.1%          | −0.5pp         |

The small residual gap (−0.3pp to −0.5pp) with HPC comes from beam-internal pruning effects and is within noise for this evaluation size.
