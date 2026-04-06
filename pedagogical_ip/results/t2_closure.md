# Task 2 Closure: Persistent Learner Profile

**Status**: Core Complete | Extensions Deferred to Task 3
**Date**: 2026-03-29

---

## 1. Summary

Task 2 implemented and validated a persistent learner profile system for the 5D Observer Architecture. The system enables cross-session state carry-over without modifying the frozen 3D observer core or the 2-act canonical decision logic.

**Core finding**: Persistent carry-over is the validated main effect. It reduces tutor WarnRate during teaching (0.033 vs 0.050–0.067 reset) while maintaining transfer performance (Phase B/C ≥ reset). Profile-aware curriculum consumption (need hook) is mechanically correct but does not yet produce measurable incremental gain over raw carry-over.

---

## 2. Evidence Chain

### Exp-2A: PP-MRB Persistent vs Reset
- Pure carry-over produces no step-level differential
- wait_clean / wait_lure WarnRate already at 0.000 floor
- **Conclusion**: Carry-over value is at curriculum level, not step level

### Exp-2B: TIC 3-Phase Transfer (3-arm)
- Persistent Phase A WarnRate: **0.033** vs reset **0.050–0.067** ← main signal
- Phase B/C transfer: **≥ reset in 2/2 θ**
- needhook ≈ nohook
- **Conclusion**: ✅ Carry-over works. "Learned → talk less, without hurting transfer"

### Exp-2C: TIC-v4 Longitudinal (3-arm)
- WR_unnecessary = **0.000** across all modes (no overwarning)
- warn_rescue: persistent **0.094** vs reset **0.125**
- E_calib: persistent **0.10** vs reset **0.03** at session 4 ← calibration drift
- SatRate_ν: 0.55–0.88, SatRate_γ_gen: 0.98–1.00 (ALL modes including reset)
- needhook ≈ nohook
- **Conclusion**: ✅ Carry-over safe, ⚠️ calibration needs correction for 3+ sessions

---

## 3. Deliverables

### New Source Files
| File | Lines | Purpose |
|------|:-----:|---------|
| `src/teachers/profile_state.py` | 97 | ProfileState + SessionSummary dataclasses |
| `src/teachers/profile_manager.py` | 166 | In-memory manager, EMA probe weakness, JSON export |
| `src/teachers/profile_bootstrap.py` | 165 | bootstrap_observer/agent_state, finalize_session, make_need_hook |

### Modified Source Files
| File | Change |
|------|--------|
| `src/teachers/internalization_observer.py` | +35 lines: bootstrap_from_profile, finalize_to_profile |
| `src/curriculum/curriculum_controller_v13.py` | +20 lines: _profile_hook, install/remove, hook call site |
| `src/curriculum/pairwise_response_model.py` | N_LESSONS 12→13 |

### Test Files
| File | Tests | Status |
|------|:-----:|:------:|
| `tests/test_profile_system.py` | 19 | ✅ |
| `tests/test_profile_curriculum_hook.py` | 11 | ✅ |

### Experiment Results
| File | Experiment |
|------|-----------|
| `results/t2_exp2a_persistent_vs_reset.md` | PP-MRB 5-session |
| `results/t2_exp2b_tic_transfer.md` | TIC 3-phase 3-arm |
| `results/t2_exp2c_ticv4_longitudinal.md` | TIC-v4 longitudinal 3-arm |

---

## 4. Regression

| Suite | Count | Status |
|-------|:-----:|:------:|
| Canonical observer | 55/55 | ✅ |
| Profile system | 19/19 | ✅ |
| Curriculum hook | 11/11 | ✅ |
| T1 smoke check | 26/26 | ✅ |

---

## 5. Frozen Invariants

The following are confirmed as immutable going forward:

- 5D three-layer architecture (estimate → micro → macro)
- Micro Q: only (τ̂, ν̂, γ̂_gen), action space {WAIT, WARN}
- κ̂ macro bonus β=0.02
- γ̂_spec / κ̂ do NOT enter micro Q
- A1MtObserverFrozen 3D core update logic
- Corrected active mask
- Dead-zone ε_Q = 0.05
- Profile bootstrap/finalize hook pattern

---

## 6. Known Limitations → Task 3

| Issue | Root Cause | Priority |
|-------|-----------|:--------:|
| E_calib grows under persistent | η=1 compounds estimation error | **3A** |
| needhook ≈ nohook | λ_need=0.3 too small or probe signal too weak | **3B** |
| ν/γ_gen saturation | State dynamics + rescue lesson side effects | **3C** |
| Drift robustness unknown | Not yet tested | **3D** |
