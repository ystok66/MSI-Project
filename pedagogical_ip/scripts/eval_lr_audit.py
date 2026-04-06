"""Small LR Audit: does lowering learning rate create StateGain > 0?

Grid: risk_lr × cost_lr × {session_shared+stateful, session_shared+stateless}
Only 6 sessions each (speed); GTET-medium curriculum.
"""
import sys; sys.path.insert(0, ".")
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

N_WORKERS = min(12, os.cpu_count() or 4)
N_SESSIONS = 6  # small for audit

LR_GRID = [
    # (cost_lr, risk_lr)
    (0.10, 0.30),   # current default
    (0.05, 0.15),   # moderate reduction
    (0.02, 0.05),   # aggressive reduction
    (0.01, 0.02),   # very slow
]


def _run_session(args):
    """Worker: run a PRS session with custom lr."""
    session_seed, cost_lr, risk_lr, persist, label = args
    from src.envs.prs_session import PRSSession, SessionConfig
    from src.agents.cost_risk_model import LatentCostRiskHead

    cfg = SessionConfig(
        session_seed=session_seed,
        curriculum="gtet_only",
        tutor_strategy="selective",
        difficulty="medium",
        shift_difficulty="hard",
        persist_agent_memory=persist,
        weight_mode="session_shared",
        block_a_size=30,
        block_b_size=15,
        block_c_size=15,
        block_d_size=15,
    )

    # Monkey-patch: create session runner, then override lr
    session = PRSSession(cfg)

    # We need to override the LatentCostRiskHead creation.
    # The cleanest way: override run_session to use custom lr.
    from src.envs.lattice_v2_runner import LatticeV2Runner

    runner_cfg = cfg
    rng = np.random.default_rng(runner_cfg.session_seed)
    from src.envs.prs_session import SessionState, EpisodeSpec
    from src.agents.cost_risk_model import generate_world_weights

    runner = LatticeV2Runner()
    state = SessionState()

    # Init with custom lr
    if persist:
        state.latent_predictor = LatentCostRiskHead(
            d=4, cost_lr=cost_lr, risk_lr=risk_lr)
    state.session_world_weights = generate_world_weights(
        np.random.default_rng(runner_cfg.session_seed * 7 + 3), d=4)

    schedule = session._build_schedule(rng)

    for block_id, episodes in schedule.items():
        state.block_id = block_id
        state.block_episode_index = 0

        if runner_cfg.tutor_strategy == "no_tutor":
            tutor_enabled = False
        elif runner_cfg.tutor_strategy == "always_warn":
            tutor_enabled = True
        else:
            tutor_enabled = (block_id == "A")

        for ep_spec in episodes:
            result = session._run_episode(runner, state, ep_spec, tutor_enabled)
            state.block_results[block_id].append(result)
            state.episode_index += 1
            state.block_episode_index += 1

    from src.envs.prs_metrics import compute_session_metrics
    metrics = compute_session_metrics(state.block_results)
    metrics["label"] = label
    metrics["cost_lr"] = cost_lr
    metrics["risk_lr"] = risk_lr
    metrics["persist"] = persist
    metrics["block_results"] = state.block_results
    return metrics


def run_sessions(jobs):
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_session, j): j for j in jobs}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  Failed: {e}")
    return results


def bootstrap_ci(vals, n_boot=2000):
    if len(vals) < 2:
        return np.mean(vals), np.mean(vals), np.mean(vals)
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(vals, len(vals), True)) for _ in range(n_boot)]
    return np.mean(vals), np.percentile(means, 2.5), np.percentile(means, 97.5)


if __name__ == "__main__":
    t0 = time.time()
    print(f"=== Small LR Audit ({N_WORKERS}w, {N_SESSIONS} sessions/cond) ===")
    print(f"Grid: {len(LR_GRID)} lr combos × SF/SL = {len(LR_GRID)*2} conditions")

    jobs = []
    for seed in range(N_SESSIONS):
        s = seed * 100
        for cost_lr, risk_lr in LR_GRID:
            tag = f"lr_c{cost_lr}_r{risk_lr}"
            # Stateful
            jobs.append((s, cost_lr, risk_lr, True, f"{tag}_SF"))
            # Stateless
            jobs.append((s, cost_lr, risk_lr, False, f"{tag}_SL"))

    print(f"Total jobs: {len(jobs)}")
    all_results = run_sessions(jobs)

    # Print results grouped by lr
    print(f"\n{'='*70}")
    print(f"{'cost_lr':>8} {'risk_lr':>8} | "
          f"{'SF_B':>7} {'SL_B':>7} {'ΔB':>7} | "
          f"{'SF_C':>7} {'SL_C':>7} {'ΔC':>7} | "
          f"{'SF_D':>7} {'SL_D':>7} {'ΔD':>7}")
    print(f"{'='*70}")

    for cost_lr, risk_lr in LR_GRID:
        tag = f"lr_c{cost_lr}_r{risk_lr}"
        sf = [r for r in all_results if r["label"] == f"{tag}_SF"]
        sl = [r for r in all_results if r["label"] == f"{tag}_SL"]

        row = f"{cost_lr:8.3f} {risk_lr:8.3f} |"
        for block in ["B", "C", "D"]:
            sf_vals = [r.get(f"tbsr_{block}", 0) for r in sf]
            sl_vals = [r.get(f"tbsr_{block}", 0) for r in sl]
            sf_m = np.mean(sf_vals) if sf_vals else 0
            sl_m = np.mean(sl_vals) if sl_vals else 0
            delta = sf_m - sl_m
            row += f" {sf_m:7.3f} {sl_m:7.3f} {delta:+7.3f} |"
        print(row)

    # Detailed StateGain with CI
    print(f"\n{'='*70}")
    print("StateGain with Bootstrap 95% CI")
    print(f"{'='*70}")

    for cost_lr, risk_lr in LR_GRID:
        tag = f"lr_c{cost_lr}_r{risk_lr}"
        sf = [r for r in all_results if r["label"] == f"{tag}_SF"]
        sl = [r for r in all_results if r["label"] == f"{tag}_SL"]

        print(f"\n  lr=(cost={cost_lr}, risk={risk_lr})")
        if not sf or not sl:
            print(f"    NO DATA")
            continue

        for block in ["A", "B", "C", "D"]:
            sf_vals = [r.get(f"tbsr_{block}", 0) for r in sf]
            sl_vals = [r.get(f"tbsr_{block}", 0) for r in sl]
            gain = np.mean(sf_vals) - np.mean(sl_vals)
            rng = np.random.default_rng(42)
            gains = [np.mean(rng.choice(sf_vals, len(sf_vals), True)) -
                     np.mean(rng.choice(sl_vals, len(sl_vals), True))
                     for _ in range(2000)]
            lo, hi = np.percentile(gains, 2.5), np.percentile(gains, 97.5)
            sig = "✓" if lo > 0 else ("✗" if hi < 0 else "~")
            print(f"    Block {block}: StateGain={gain:+.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")

        # Also print Block A performance (to check lr doesn't break training)
        sf_a = np.mean([r.get("tbsr_A", 0) for r in sf])
        sl_a = np.mean([r.get("tbsr_A", 0) for r in sl])
        print(f"    [check] Block A: SF={sf_a:.3f} SL={sl_a:.3f}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"LR AUDIT COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
