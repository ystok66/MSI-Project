"""PRS-2: Transfer Identifiability Repair — main experiments (12 workers).

Exp A: Negative control (episode_random) vs Transfer regime (session_shared)
       × Stateful vs Stateless
       × Tutor vs NoTutor
Exp B: Curriculum rebalance under session_shared
Exp C: Always-warn baseline under session_shared
"""
import sys
sys.path.insert(0, ".")

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import hashlib

N_WORKERS = min(12, os.cpu_count() or 4)
N_SESSIONS = 12


def _run_session(args):
    """Worker: run a full PRS session."""
    (session_seed, curriculum, tutor_strategy,
     persist, weight_mode, difficulty, label) = args
    from src.envs.prs_session import PRSSession, SessionConfig

    cfg = SessionConfig(
        session_seed=session_seed,
        curriculum=curriculum,
        tutor_strategy=tutor_strategy,
        difficulty=difficulty,
        shift_difficulty="hard",
        persist_agent_memory=persist,
        weight_mode=weight_mode,
        block_a_size=30,
        block_b_size=15,
        block_c_size=15,
        block_d_size=15,
    )
    session = PRSSession(cfg)
    result = session.run_session()
    result["label"] = label
    result["weight_mode"] = weight_mode
    result["persist"] = persist
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


def bootstrap_ci(vals, n_boot=2000, alpha=0.05):
    if len(vals) < 2:
        return np.mean(vals), np.mean(vals), np.mean(vals)
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(vals, size=len(vals), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(vals), lo, hi


def print_block(label, results, ref=None):
    if not results:
        print(f"  {label}: NO RESULTS")
        return

    print(f"\n  {label} (n={len(results)})")
    print(f"  {'Metric':25s} {'Mean':>7s} {'[95% CI]':>16s}")
    print(f"  {'-'*25} {'-'*7} {'-'*16}")

    for k in ["tbsr_A", "tbsr_B", "tbsr_C", "tbsr_D"]:
        vals = [r.get(k, 0.0) for r in results]
        m, lo, hi = bootstrap_ci(vals)
        print(f"  {k:25s} {m:7.3f}  [{lo:.3f}, {hi:.3f}]")

    if ref:
        for block in ["B", "C", "D"]:
            t_vals = [r.get(f"tbsr_{block}", 0) for r in results]
            c_vals = [r.get(f"tbsr_{block}", 0) for r in ref]
            apd = np.mean(t_vals) - np.mean(c_vals)
            rng = np.random.default_rng(42)
            apds = [np.mean(rng.choice(t_vals, len(t_vals), True)) -
                    np.mean(rng.choice(c_vals, len(c_vals), True))
                    for _ in range(2000)]
            lo, hi = np.percentile(apds, 2.5), np.percentile(apds, 97.5)
            sig = "✓" if lo > 0 else ("✗" if hi < 0 else "~")
            print(f"  APD_{block:s}{'':<20s} {apd:+7.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")

    for k in ["transfer_gap_C", "transfer_gap_D", "dependence_proxy"]:
        vals = [r.get(k, 0.0) for r in results]
        m, lo, hi = bootstrap_ci(vals)
        short = k.replace("transfer_gap_", "gap_").replace("dependence_proxy", "depend")
        print(f"  {short:25s} {m:7.3f}  [{lo:.3f}, {hi:.3f}]")


def print_state_gain(sf_results, sl_results, label=""):
    """StateGain = Perf(stateful) - Perf(stateless)."""
    if not sf_results or not sl_results:
        return
    print(f"\n  [StateGain] {label}")
    for block in ["B", "C", "D"]:
        sf = [r.get(f"tbsr_{block}", 0) for r in sf_results]
        sl = [r.get(f"tbsr_{block}", 0) for r in sl_results]
        gain = np.mean(sf) - np.mean(sl)
        rng = np.random.default_rng(42)
        gains = [np.mean(rng.choice(sf, len(sf), True)) -
                 np.mean(rng.choice(sl, len(sl), True))
                 for _ in range(2000)]
        lo, hi = np.percentile(gains, 2.5), np.percentile(gains, 97.5)
        sig = "✓" if lo > 0 else ("✗ neg" if hi < 0 else "~")
        print(f"    Block {block}: {gain:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")


# ═══════════════════════════════════════════════════════════════
# Exp A: episode_random vs session_shared × stateful/stateless × tutor/no_tutor
# ═══════════════════════════════════════════════════════════════

def exp_a():
    print("=" * 70)
    print(f"Exp A: Transfer Identifiability (12 sessions, {N_WORKERS}w)")
    print("   R0=episode_random R1=session_shared")
    print("   SF=stateful SL=stateless T=tutor NT=no_tutor")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        s = seed * 100
        # GTET-medium only (current best curriculum)
        for wm in ["episode_random", "session_shared"]:
            tag = "R0" if wm == "episode_random" else "R1"
            # Stateful + Tutor
            jobs.append((s, "gtet_only", "selective", True, wm, "medium",
                         f"{tag}_SF_T"))
            # Stateless + Tutor
            jobs.append((s, "gtet_only", "selective", False, wm, "medium",
                         f"{tag}_SL_T"))
            # Stateful + NoTutor
            jobs.append((s, "gtet_only", "no_tutor", True, wm, "medium",
                         f"{tag}_SF_NT"))

    all_results = run_sessions(jobs)

    # Group
    groups = {}
    for label in ["R0_SF_T", "R0_SL_T", "R0_SF_NT",
                   "R1_SF_T", "R1_SL_T", "R1_SF_NT"]:
        groups[label] = [r for r in all_results if r["label"] == label]

    # Print per-group
    for label in ["R0_SF_T", "R0_SL_T", "R0_SF_NT"]:
        ref = groups["R0_SF_NT"] if "NT" not in label else None
        print_block(f"[episode_random] {label}", groups[label], ref)

    for label in ["R1_SF_T", "R1_SL_T", "R1_SF_NT"]:
        ref = groups["R1_SF_NT"] if "NT" not in label else None
        print_block(f"[session_shared] {label}", groups[label], ref)

    # StateGain comparison — the KEY metric
    print("\n" + "=" * 50)
    print("  KEY METRIC: StateGain by regime")
    print("=" * 50)
    print_state_gain(groups["R0_SF_T"], groups["R0_SL_T"],
                     "R0 (episode_random) — expect ~0")
    print_state_gain(groups["R1_SF_T"], groups["R1_SL_T"],
                     "R1 (session_shared) — expect >0 if transfer works")

    # APD comparison across regimes
    print("\n" + "=" * 50)
    print("  APD: Tutor benefit by regime")
    print("=" * 50)
    for regime, tag in [("episode_random", "R0"), ("session_shared", "R1")]:
        sf_t = groups[f"{tag}_SF_T"]
        sf_nt = groups[f"{tag}_SF_NT"]
        if sf_t and sf_nt:
            print(f"\n  [{regime}] APD(stateful+tutor vs stateful+notutor):")
            for b in ["B", "C", "D"]:
                t = np.mean([r.get(f"tbsr_{b}", 0) for r in sf_t])
                n = np.mean([r.get(f"tbsr_{b}", 0) for r in sf_nt])
                print(f"    Block {b}: APD={t-n:+.3f} (tutor={t:.3f}, ctrl={n:.3f})")


# ═══════════════════════════════════════════════════════════════
# Exp B: Curriculum under session_shared
# ═══════════════════════════════════════════════════════════════

def exp_b():
    print("\n" + "=" * 70)
    print(f"Exp B: Curriculum under session_shared ({N_SESSIONS} sessions)")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        s = seed * 100 + 50
        # C1: GTET-medium
        jobs.append((s, "gtet_only", "selective", True, "session_shared",
                      "medium", "gtet_med"))
        # C2: Mixed-balanced (medium difficulty for both)
        jobs.append((s, "mixed", "selective", True, "session_shared",
                      "medium", "mixed_med"))

    all_results = run_sessions(jobs)

    for label in ["gtet_med", "mixed_med"]:
        recs = [r for r in all_results if r["label"] == label]
        print_block(f"Curriculum={label}", recs)


# ═══════════════════════════════════════════════════════════════
# Exp C: Always-warn under session_shared
# ═══════════════════════════════════════════════════════════════

def exp_c():
    print("\n" + "=" * 70)
    print(f"Exp C: Selective vs Always-warn (session_shared, {N_SESSIONS})")
    print("=" * 70)

    jobs = []
    for seed in range(N_SESSIONS):
        s = seed * 100 + 80
        for strat in ["selective", "always_warn", "no_tutor"]:
            jobs.append((s, "gtet_only", strat, True, "session_shared",
                          "medium", f"ss_{strat}"))

    all_results = run_sessions(jobs)

    no_tutor = [r for r in all_results if r["label"] == "ss_no_tutor"]
    for label in ["ss_selective", "ss_always_warn", "ss_no_tutor"]:
        recs = [r for r in all_results if r["label"] == label]
        ref = no_tutor if "no_tutor" not in label else None
        print_block(f"Strategy={label}", recs, ref)

    # Transfer isolation
    sel = [r for r in all_results if r["label"] == "ss_selective"]
    aw = [r for r in all_results if r["label"] == "ss_always_warn"]
    nt = [r for r in all_results if r["label"] == "ss_no_tutor"]
    if sel and aw and nt:
        print("\n  --- Transfer vs Continued Help ---")
        for b in ["A", "B", "C", "D"]:
            s_v = np.mean([r.get(f"tbsr_{b}", 0) for r in sel])
            a_v = np.mean([r.get(f"tbsr_{b}", 0) for r in aw])
            n_v = np.mean([r.get(f"tbsr_{b}", 0) for r in nt])
            print(f"  Block {b}: sel={s_v:.3f} aw={a_v:.3f} nt={n_v:.3f}")
        print("\n  If sel_B > nt_B under session_shared:")
        print("    → genuine transfer (tutor taught something reusable)")
        print("  If aw_B >> sel_B >> nt_B:")
        print("    → tutor helps in-the-moment; session_shared enables SOME transfer")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Workers: {N_WORKERS}, CPU: {os.cpu_count()}")
    print(f"Sessions/condition: {N_SESSIONS}")

    exp_a()
    exp_b()
    exp_c()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL PRS-2 EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
