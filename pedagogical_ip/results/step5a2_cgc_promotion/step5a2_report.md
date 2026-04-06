# Step 5A.2: CGC-v2 Multi-Path Promotion Audit

**Episodes**: 300 | **Elapsed**: 0.3s

## Phase A: Candidate-Path Audit

**Total episodes evaluated**: 300

### Path Count Distribution

| N_paths | Count | Fraction |
|---------|-------|----------|
| 2 | 300 | 1.000 |

**Mean hazard variance across paths**: 0.0131
**Mean uncertainty variance across paths**: 0.0035

**GateDiffRate** (N2 changes top1 vs A2): 0.000 (0/300)

## Phase B: Replay Ranking Audit

### Agreement Rates

| Comparison | Agreement |
|------------|----------|
| A2 vs baseline | 0.560 |
| A2+N2 vs A2 | 1.000 |
| A2+N2 vs baseline | 0.560 |

### Path Choice Distribution

| Planner | safe_branch | risky_branch | safe_then_risky | risky_then_safe |
|---------|-------------|-------------|-----------------|----------------|
| baseline | 0.560 | 0.440 | 0.000 | 0.000 |
| A2 | 1.000 | 0.000 | 0.000 | 0.000 |
| A2+N2 | 1.000 | 0.000 | 0.000 | 0.000 |

### SafeTop1Rate

| Planner | SafeTop1Rate |
|---------|-------------|
| baseline | 0.560 |
| A2 | 1.000 |
| A2+N2 | 1.000 |

## Phase C: Promotion Criteria

| Criterion | Threshold | Result | Pass? |
|-----------|-----------|--------|-------|
| GateDiffRate > 0 | > 0.01 | 0.000 | ✗ |
| A2 SafeRate ≥ baseline | ≥ 0.560 | 1.000 | ✓ |
| A2+N2 SafeRate ≥ A2 | ≥ 1.000 | 1.000 | ✓ |
| >30% episodes have ≥3 paths | > 0.30 | 0.000 | ✗ |

### Verdict

**A2 improves safe selection but N2 gate has limited additional value → A2 SHADOW-READY, N2 OPTIONAL**
