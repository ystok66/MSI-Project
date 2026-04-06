# HBPI — Hierarchical Bayesian Program Induction (Phase 0)

## Equations

### Scoring (per parse p)

```
log w(p) = log P(p | Θ) + log P(y | p, Θ)
```

**MDL Prior:**

```
log P(p | Θ) = -λ_len * |p|  +  Σ_nodes log P(type(node) | w)  +  Σ_unary log P(n | w)
```

**Soft Likelihood:**

```
log P(y | p, Θ) = -α_edit * edit_distance(ŷ, y)
```

### EM Updates

**E-step:** q(p) = softmax{ log w(p) } over top-K parses per example

**M-step (Dirichlet posterior mean):**

```
P(type=t | w) = (γ₀[t] + Σ_p q(p) · #_p(w used as t)) / Σ_t' (γ₀[t'] + count[t'])
P(color=c | w) = (α₀[c] + Σ aligned color credits) / Σ_c' (α₀[c'] + credit[c'])
P(repeat=n | w) = (δ₀[n] + Σ_p q(p) · #_p(Unary(w,n))) / Σ_n' (δ₀[n'] + count[n'])
```

### Hyperparameter Defaults

| Param      | Default | Meaning                           |
| ---------- | ------- | --------------------------------- |
| γ₀         | 1.0     | Dirichlet prior for word type     |
| α₀         | 1.0     | Dirichlet prior for color         |
| δ₀         | 1.0     | Dirichlet prior for repeat        |
| λ_len      | 0.2     | MDL length penalty                |
| α_edit     | 1.0     | Soft likelihood sharpness         |
| sub_weight | 0.3     | Alignment credit for substitution |
| K_span     | 20      | Top-K parses per chart span       |
| K_full     | 50      | Top-K full parses                 |
| em_iters   | 5       | EM iterations                     |
| REPEAT_SET | {2,3,4} | Allowed repeat factors            |

## Module Layout

```
hbpi/
  grammar.py   — AST nodes (Prim, Concat, Unary, Binary) + canonical form
  executor.py  — Execute AST → (pred_seq, provenance_seq)
  align.py     — Levenshtein distance + alignment backtrace
  model.py     — HBPIModel: Dirichlet posteriors + scoring
  parser.py    — Chart-based parse enumeration + top-K pruning
  em.py        — EM loop (E: softmax posterior, M: Dirichlet update)
  eval.py      — MLC evaluation harness
```
