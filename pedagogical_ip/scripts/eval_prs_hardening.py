"""PRS v1 Hardening — 4 work packages in one script (12 workers).

WP-A: Stateful vs Stateless (semantic foundation)
WP-B: Curriculum rebalance (fix DTMB-hard issue)
WP-C: Scale up to 12 sessions (statistical hardening)
WP-D: Aggressive-help baseline (selective vs always_warn)
"""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

N_WORKERS = min(12, os.cpu_count() or 4)
N_SESSIONS = 12  # per condition for statistical power


def _run_session(args):
    """Worker: run a full PRS session."""
    session_seed, curriculum, tutor_strategy, persist, difficulty, extra_label = args
    from src.envs.prs_session import PRSSession, SessionConfig

    cfg = SessionConfig(
        session_seed=session_seed,
        curriculum=curriculum,
        tutor_strategy=tutor_strategy,
        difficulty=difficulty,
        shift_difficulty="hard",
        persist_agent_memory=persist,
        block_a_size=30,
        block_b_size=15,
        block_c_size=15,
        block_d_size=15,
    )
    session = PRSSession(cfg)
    result = session.run_session()
    result["session_seed"] = session_seed
    result["curriculum"] = curriculum
    result["tutor_strategy"] = tutor_strategy
    result["persist"] = persist
    result["difficulty"] = difficulty
    result["label"] = extra_label
    return result


def run_sessions(jobs):
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_session, j): j for j in jobs}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  Session failed: {e}")
    return results


def bootstrap_ci(vals, n_boot=1000, alpha=0.05):
    """Bootstrap 95% CI for mean."""
    if len(vals) < 2:
        return np.mean(vals), np.mean(vals), np.mean(vals)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(vals), lo, hi


def print_metrics(label, results, ref_results=None):
    """Print metrics with bootstrap CI."""
    if not results:
        print(f"  {label}: NO RESULTS")
        return

    print(f"\n  {label} (n={len(results)} sessions)")
    print(f"  {'Metric':25s} {'Mean':>7s} {'[95% CI]':>16s}")
    print(f"  {'-'*25} {'-'*7} {'-'*16}")

    for k in ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D"]:
        vals = [r.get(k, 0.0) for r in results]
        m, lo, hi = bootstrap_ci(vals)
        print(f"  {k:25s} {m:7.3f}  [{lo:.3f}, {hi:.3f}]")

    # APD vs reference
    if ref_results:
        print(f"  --- APD (vs {len(ref_results)} ref sessions) ---")
        for block in ["B", "C", "D"]:
            t_vals = [r.get(f"tbsr_{block}", 0) for r in results]
            c_vals = [r.get(f"tbsr_{block}", 0) for r in ref_results]
            t_m = np.mean(t_vals)
            c_m = np.mean(c_vals)
            apd = t_m - c_m
            # Bootstrap APD CI
            rng = np.random.default_rng(42)
            apds = []
            for _ in range(1000):
                ts = rng.choice(t_vals, size=len(t_vals), replace=True)
                cs = rng.choice(c_vals, size=len(c_vals), replace=True)
                apds.append(np.mean(ts) - np.mean(cs))
            lo = np.percentile(apds, 2.5)
            hi = np.percentile(apds, 97.5)
            sig = "✓" if lo > 0 else "✗"
            print(f"  APD_{block:s}{'':<20s} {apd:+7.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")

    # Transfer gaps
    for block in ["C", "D"]:
        vals = [r.get(f"transfer_gap_{block}", 0) for r in results]
        m, lo, hi = bootstrap_ci(vals)
        print(f"  {'gap_'+block:25s} {m:7.3f}  [{lo:.3f}, {hi:.3f}]")

    # Dependence
    vals = [r.get("dependence_proxy", 0) for r in results]
    m, lo, hi = bootstrap_ci(vals)
    print(f"  {'dependence':25s} {m:7.3f}  [{lo:.3f}, {hi:.3f}]")


# ═══════════════════════════════════════════════════════════════
# WP-A: Stateful vs Stateless
# ═══════════════════════════════════════════════════════════════

def wp_a():
    print("=" * 70)
    print(f"WP-A: Stateful vs Stateless ({N_SESSIONS} sessions each, {N_WORKERS}w)")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        # Stateful tutor-trained
        jobs.append((seed * 100, "mixed", "selective", True, "hard", "stateful_tutor"))
        # Stateless tutor-trained
        jobs.append((seed * 100, "mixed", "selective", False, "hard", "stateless_tutor"))
        # Stateful no-tutor (control)
        jobs.append((seed * 100, "mixed", "no_tutor", True, "hard", "stateful_notutor"))

    all_results = run_sessions(jobs)

    stateful_tutor = [r for r in all_results if r["label"] == "stateful_tutor"]
    stateless_tutor = [r for r in all_results if r["label"] == "stateless_tutor"]
    stateful_notutor = [r for r in all_results if r["label"] == "stateful_notutor"]

    print_metrics("Stateful + Tutor", stateful_tutor, ref_results=stateful_notutor)
    print_metrics("Stateless + Tutor", stateless_tutor, ref_results=stateful_notutor)
    print_metrics("Stateful + NoTutor (ctrl)", stateful_notutor)

    # Key comparison
    if stateful_tutor and stateless_tutor:
        print("\n--- Statefulness Validity ---")
        for block in ["B", "C", "D"]:
            sf = np.mean([r.get(f"tbsr_{block}", 0) for r in stateful_tutor])
            sl = np.mean([r.get(f"tbsr_{block}", 0) for r in stateless_tutor])
            print(f"  Block {block}: stateful={sf:.3f} stateless={sl:.3f} "
                  f"Δ={sf-sl:+.3f} {'✓ stateful wins' if sf > sl + 0.02 else ''}")


# ═══════════════════════════════════════════════════════════════
# WP-B: Curriculum Rebalance
# ═══════════════════════════════════════════════════════════════

def wp_b():
    print("\n" + "=" * 70)
    print(f"WP-B: Curriculum Rebalance ({N_SESSIONS} sessions each)")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        # C1: GTET-only hard (current strongest)
        jobs.append((seed * 100 + 10, "gtet_only", "selective", True, "hard", "gtet_hard"))
        # C2: Mixed-balanced (DTMB medium + GTET hard)
        jobs.append((seed * 100 + 10, "mixed", "selective", True, "medium", "mixed_balanced"))
        # C3: GTET-only medium (easier baseline)
        jobs.append((seed * 100 + 10, "gtet_only", "selective", True, "medium", "gtet_medium"))

    all_results = run_sessions(jobs)

    for label in ["gtet_hard", "mixed_balanced", "gtet_medium"]:
        recs = [r for r in all_results if r["label"] == label]
        print_metrics(f"Curriculum={label}", recs)


# ═══════════════════════════════════════════════════════════════
# WP-D: Aggressive-help Baseline
# ═══════════════════════════════════════════════════════════════

def wp_d():
    print("\n" + "=" * 70)
    print(f"WP-D: Selective vs Aggressive-Help ({N_SESSIONS} sessions each)")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        # Selective tutor (canonical)
        jobs.append((seed * 100 + 20, "gtet_only", "selective", True, "hard", "selective"))
        # Always-warn (aggressive)
        jobs.append((seed * 100 + 20, "gtet_only", "always_warn", True, "hard", "always_warn"))
        # No-tutor (control)
        jobs.append((seed * 100 + 20, "gtet_only", "no_tutor", True, "hard", "no_tutor"))

    all_results = run_sessions(jobs)

    no_tutor = [r for r in all_results if r["label"] == "no_tutor"]

    for label in ["selective", "always_warn", "no_tutor"]:
        recs = [r for r in all_results if r["label"] == label]
        ref = no_tutor if label != "no_tutor" else None
        print_metrics(f"Strategy={label}", recs, ref_results=ref)

    # Key comparison: selective vs always_warn transfer
    selective = [r for r in all_results if r["label"] == "selective"]
    aggressive = [r for r in all_results if r["label"] == "always_warn"]
    if selective and aggressive:
        print("\n--- Selective vs Aggressive in Transfer ---")
        for block in ["A", "B", "C", "D"]:
            s = np.mean([r.get(f"tbsr_{block}", 0) for r in selective])
            a = np.mean([r.get(f"tbsr_{block}", 0) for r in aggressive])
            winner = "selective" if s > a + 0.01 else ("aggressive" if a > s + 0.01 else "tie")
            print(f"  Block {block}: selective={s:.3f} aggressive={a:.3f} → {winner}")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Workers: {N_WORKERS}, CPU: {os.cpu_count()}")

    wp_a()
    wp_b()
    wp_d()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL HARDENING EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
