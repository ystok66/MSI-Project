"""GTET-L / DTMB-L Family 3 — PRS Baseline Experiments (12 workers).

Exp A: tutor-trained vs no-tutor-trained session transfer
Exp B: DTMB-only vs GTET-only vs mixed curriculum
Exp C: IID vs topology shift vs semantic shift
"""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

N_WORKERS = min(12, os.cpu_count() or 4)
N_SESSIONS = 6  # sessions per condition (parallelized)


def _run_session(args):
    """Worker: run a full PRS session."""
    session_seed, curriculum, tutor_strategy = args
    from src.envs.prs_session import PRSSession, SessionConfig

    cfg = SessionConfig(
        session_seed=session_seed,
        curriculum=curriculum,
        tutor_strategy=tutor_strategy,
        difficulty="hard",
        shift_difficulty="hard",
        persist_agent_memory=True,
    )
    session = PRSSession(cfg)
    result = session.run_session()
    result["session_seed"] = session_seed
    result["curriculum"] = curriculum
    result["tutor_strategy"] = tutor_strategy
    return result


def run_sessions(jobs):
    """Run sessions in parallel."""
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_session, j): j for j in jobs}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  Session failed: {e}")
    return results


def print_metrics(label, results):
    """Print mean metrics across sessions."""
    if not results:
        print(f"  {label}: NO RESULTS")
        return

    keys = ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D",
            "surv_A", "surv_B", "surv_C", "surv_D",
            "transfer_gap_C", "transfer_gap_D",
            "dependence_proxy"]

    print(f"\n  {label} ({len(results)} sessions)")
    print(f"  {'Metric':25s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for k in keys:
        vals = [r.get(k, 0.0) for r in results]
        print(f"  {k:25s} {np.mean(vals):8.3f} {np.std(vals):8.3f}")

    # Learning curve
    upd_a = [r.get("predictor_updates_end_A", 0) for r in results]
    upd_end = [r.get("predictor_updates_end_session", 0) for r in results]
    if any(u > 0 for u in upd_a):
        print(f"  {'predictor_updates_A':25s} {np.mean(upd_a):8.1f}")
        print(f"  {'predictor_updates_end':25s} {np.mean(upd_end):8.1f}")


# ═══════════════════════════════════════════════════════════════
# Exp A: Tutor-trained vs No-tutor transfer
# ═══════════════════════════════════════════════════════════════

def exp_a():
    print("=" * 70)
    print(f"Exp A: Tutor-on vs No-tutor Transfer ({N_SESSIONS} sessions, {N_WORKERS}w)")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        # Condition 1: tutor-on training → tutor-off eval
        jobs.append((seed * 100, "mixed", "selective"))
        # Condition 2: no tutor at all (control)
        jobs.append((seed * 100, "mixed", "no_tutor"))

    all_results = run_sessions(jobs)

    tutor_results = [r for r in all_results if r["tutor_strategy"] == "selective"]
    no_tutor_results = [r for r in all_results if r["tutor_strategy"] == "no_tutor"]

    print_metrics("tutor-trained (selective)", tutor_results)
    print_metrics("no-tutor (control)", no_tutor_results)

    # APD calculation
    if tutor_results and no_tutor_results:
        print("\n--- Agent Performance Delta (APD) ---")
        for block in ["B", "C", "D"]:
            t_perf = np.mean([r.get(f"tbsr_{block}", 0) for r in tutor_results])
            c_perf = np.mean([r.get(f"tbsr_{block}", 0) for r in no_tutor_results])
            apd = t_perf - c_perf
            print(f"  APD_{block} = {apd:+.3f} (tutor={t_perf:.3f}, control={c_perf:.3f})")

        # Transfer gap comparison
        print("\n--- Transfer Gap Comparison ---")
        for block in ["C", "D"]:
            t_gap = np.mean([r.get(f"transfer_gap_{block}", 0) for r in tutor_results])
            c_gap = np.mean([r.get(f"transfer_gap_{block}", 0) for r in no_tutor_results])
            print(f"  TransferGap_{block}: tutor={t_gap:+.3f}, control={c_gap:+.3f}")


# ═══════════════════════════════════════════════════════════════
# Exp B: Curriculum comparison
# ═══════════════════════════════════════════════════════════════

def exp_b():
    print("\n" + "=" * 70)
    print(f"Exp B: DTMB-only vs GTET-only vs Mixed ({N_SESSIONS} sessions)")
    print("=" * 70)

    curricula = ["dtmb_only", "gtet_only", "mixed"]
    jobs = []
    for cur in curricula:
        for seed in range(N_SESSIONS):
            jobs.append((seed * 100 + 1, cur, "selective"))

    all_results = run_sessions(jobs)

    for cur in curricula:
        cur_results = [r for r in all_results if r["curriculum"] == cur]
        print_metrics(f"curriculum={cur}", cur_results)


# ═══════════════════════════════════════════════════════════════
# Exp C: IID vs Shift
# ═══════════════════════════════════════════════════════════════

def exp_c():
    print("\n" + "=" * 70)
    print(f"Exp C: Block B (IID) vs C (topology) vs D (semantic)")
    print("=" * 70)

    # Already computed in Exp A tutor-trained sessions
    # Just provide explicit comparison
    jobs = []
    for seed in range(N_SESSIONS):
        jobs.append((seed * 100 + 2, "mixed", "selective"))

    results = run_sessions(jobs)

    if results:
        print("\n  Block-by-block survival & TBSR:")
        for block in ["A", "B", "C", "D"]:
            surv = np.mean([r.get(f"surv_{block}", 0) for r in results])
            tbsr = np.mean([r.get(f"tbsr_{block}", 0) for r in results])
            print(f"  Block {block}: surv={surv:.3f} tbsr={tbsr:.3f}")

        print("\n  Transfer pattern:")
        tbsr_b = np.mean([r.get("tbsr_B", 0) for r in results])
        tbsr_c = np.mean([r.get("tbsr_C", 0) for r in results])
        tbsr_d = np.mean([r.get("tbsr_D", 0) for r in results])
        gap_c = tbsr_b - tbsr_c
        gap_d = tbsr_b - tbsr_d
        print(f"  B→C drop: {gap_c:+.3f} (topology shift)")
        print(f"  B→D drop: {gap_d:+.3f} (semantic shift)")

        if gap_c < 0.05 and gap_d < 0.05:
            print("  → Robust transfer: learning generalizes across shifts")
        elif gap_c > gap_d:
            print("  → Topology shift hurts more than semantic shift")
        else:
            print("  → Semantic shift hurts more than topology shift")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Workers: {N_WORKERS}, CPU: {os.cpu_count()}")

    exp_a()
    exp_b()
    exp_c()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL PRS EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
