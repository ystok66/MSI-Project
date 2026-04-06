# Next Agent Quickstart

## 30-Second Context

Pedagogical robot assistance in grid worlds. Agent navigates lattice with traps + deadline. Robot tutor chooses WAIT/WARN/UNLOCK/ITEM_DROP. System is **model-based Bayesian**, not RL.

## Current State

Phase 10 complete. Three intervention families validated:

| Scenario | Best lever | SR (no_tutor → best) |
|----------|-----------|---------------------|
| deadline_gate | **UNLOCK** | 70% → **100%** |
| hazard_belt | **ITEM_DROP** | 30% → **60%** |
| fork_trap | **robot_belief** | 5% → **65%** |

406/406 tests pass.

## Read First

1. `task_list/step1_new_scene/project_handoff_summary.md` — full context
2. `src/envs/lattice_v2_runner.py` — main loop
3. `src/agents/planner_astar.py` — `cell_cost_v2_latent()` is the core formula
4. `src/agents/feature_belief.py` — belief + provenance

## Run First

```bash
# Tests
python -m pytest tests/ -q

# 3-family experiment
python scripts/stage3_experiment.py
```

## Key Formula

```
J = λ_c·ĉ + φ(r̂)·[α + (1-α)(1-n)] + λ_uc·(1-n)·u_c + λ_ur·(1-n)·u_r
```

- `α = min(1, n_updates/10)` — learning factor
- `n` — route necessity (BFS)
- `u_c = w_c^T Σ w_c` — directional uncertainty

## Next Steps

1. Cross-difficulty sweep (easy/medium/hard × 3 families × 20 seeds)
2. Stage 4: Tutor perceptual model
3. Transfer evaluation (tutor-assisted → no-tutor)
4. Paper-facing tables/figures

## Do Not Touch

- `w=0, b=0` prior in risk_model.py (learning factor depends on it)
- NaN safety in `cell_cost_v2_latent`
- Legacy V0 code (`bounded_agent.py`, `belief.py`)
- `CellMemoryMeta` backward-compat aliases
