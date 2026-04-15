"""
Observation-depth experiment: does longer observation help the tutor?

Design:
  - N_obs = 10 (tutor gets 10 frozen-learner queries to observe)
  - N_teach = 1 or 2 (tutor intervenes on 1-2 queries)
  - N_eval = 3 (frozen learner, no tutor)
  - M_queries = N_obs + N_teach + N_eval = 14 or 15
  - n_sup = {2, 4, 6, 8}
  - 20 seeds x 3 grammars = 60 samples per cell
  - Conditions: baseline, ban_only, hl_only, full_tutor, oracle

Total: 4 n_sup x 2 teach x 5 conds x 20 seeds x 3 grammars = 2400 jobs
"""
from __future__ import annotations
import os, sys
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cls_option_tutor.config import FullConfig
from cls_option_tutor.env.option_env import OptionEnv
from cls_option_tutor.learner.learner_agent import LearnerAgent
from cls_option_tutor.tutor.tutor_agent import TutorAgent

DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'BASIC', 'cls_learner', 'data')

N_SEEDS = 20
GRAMMARS = ["000001", "000002", "000003"]
N_SUP_VALUES = [2, 4, 6, 8]
N_TEACH_VALUES = [1, 2]
N_OBS = 10
N_EVAL = 3
MAX_WORKERS = 12

CONDITIONS = {
    "baseline": {},
    "ban_only": {"c_hint": 100.0, "c_skip": 100.0, "c_hl": 100.0},
    "hl_only":  {"c_hint": 100.0, "c_skip": 100.0, "c_ban": 100.0},
    "full_tutor": {},
    "oracle": {"use_oracle": True},
}


def run_one(condition, grammar_id, seed, n_sup, n_teach):
    cond_cfg = CONDITIONS[condition]
    cfg = FullConfig()
    cfg.learner.n_sup = n_sup

    # Set phase sizes
    cfg.env.N_obs = N_OBS
    cfg.env.N_teach = n_teach
    cfg.env.N_eval = N_EVAL
    cfg.env.M_queries = N_OBS + n_teach + N_EVAL

    use_oracle = cond_cfg.get("use_oracle", False)
    cfg.learner.use_cls = not use_oracle

    for k in ["c_hint", "c_skip", "c_hl", "c_ban"]:
        if k in cond_cfg:
            setattr(cfg.tutor, k, cond_cfg[k])

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    tutor_enabled = condition not in ("baseline", "oracle")

    if tutor_enabled:
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)
        block = tutor.run_block(env, learner, grammar_id, seed=seed)
    else:
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=(not use_oracle))
        block = learner.run_block(env, grammar_id, seed=seed)

    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    eval_end = len(block.queries)

    def phase_sr(start, end):
        qs_list = block.queries[start:end]
        if not qs_list:
            return 0.0
        return sum(1 for q in qs_list if q.success) / len(qs_list)

    def phase_dmg(start, end):
        return sum(max(0, cfg.env.H_0 - q.hp) for q in block.queries[start:end])

    # Tutor actions in teaching phase only
    teach_actions = defaultdict(int)
    for s in block.tutor_trace:
        # Check if this step was in teaching phase queries
        qi = s.query_id
        if obs_end <= qi < teach_end:
            teach_actions[s.action] += 1

    n_teach_ex = len(getattr(learner, '_teaching_examples', []))

    return {
        'cond': condition, 'grammar': grammar_id, 'seed': seed,
        'n_sup': n_sup, 'n_teach': n_teach,
        'obs_sr': phase_sr(0, obs_end),
        'teach_sr': phase_sr(obs_end, teach_end),
        'eval_sr': phase_sr(teach_end, eval_end),
        'eval_dmg': phase_dmg(teach_end, eval_end),
        'total_dmg': sum(max(0, cfg.env.H_0 - q.hp) for q in block.queries),
        'n_teach_ex': n_teach_ex,
        'teach_actions': dict(teach_actions),
    }


def main():
    jobs = []
    for gid in GRAMMARS:
        for seed in range(N_SEEDS):
            for cond in CONDITIONS:
                for ns in N_SUP_VALUES:
                    for nt in N_TEACH_VALUES:
                        jobs.append((cond, gid, seed, ns, nt))

    total = len(jobs)
    print(f"Running {total} jobs ({len(CONDITIONS)} conds x {len(N_SUP_VALUES)} n_sup "
          f"x {len(N_TEACH_VALUES)} n_teach x {N_SEEDS} seeds x {len(GRAMMARS)} grammars)")
    print(f"Phases: Obs({N_OBS}) -> Teach(1 or 2) -> Eval({N_EVAL})")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one, *j): j for j in jobs}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                results.append(f.result())
                if done % 100 == 0:
                    print(f"  [{done}/{total}]")
            except Exception as e:
                print(f"  FAILED {futures[f]}: {e}")

    N = N_SEEDS * len(GRAMMARS)
    sep = "=" * 120

    # ═══ TABLE 1: EVAL-PHASE SR by n_sup x n_teach x condition ═══
    for nt in N_TEACH_VALUES:
        print(f"\n{sep}")
        print(f"TABLE 1-{nt}: EVAL SR with N_teach={nt} (N_obs={N_OBS}, N_eval={N_EVAL})")
        print(sep)
        header = f"{'n_sup':>5s}"
        for c in CONDITIONS:
            header += f"  {c:>18s}"
        print(header)
        print("-" * 120)
        for ns in N_SUP_VALUES:
            row = f"{ns:5d}"
            for c in CONDITIONS:
                cr = [r['eval_sr'] for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                m = np.mean(cr) if cr else 0
                se = np.std(cr)/np.sqrt(len(cr)) if cr else 0
                row += f"  {m:9.3f}+/-{se:.3f}"
            print(row)

    # ═══ TABLE 2: DELTA vs baseline ═══
    for nt in N_TEACH_VALUES:
        print(f"\n{sep}")
        print(f"TABLE 2-{nt}: dEVAL_SR vs baseline (N_teach={nt})")
        print(sep)
        header = f"{'n_sup':>5s}"
        for c in ["ban_only", "hl_only", "full_tutor", "oracle"]:
            header += f"  {c:>12s}"
        print(header)
        print("-" * 120)
        for ns in N_SUP_VALUES:
            bl = [r['eval_sr'] for r in results
                  if r['cond'] == 'baseline' and r['n_sup'] == ns and r['n_teach'] == nt]
            bl_m = np.mean(bl) if bl else 0
            row = f"{ns:5d}"
            for c in ["ban_only", "hl_only", "full_tutor", "oracle"]:
                cr = [r['eval_sr'] for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                d = np.mean(cr) - bl_m if cr else 0
                row += f"  {d:+12.3f}"
            print(row)

    # ═══ TABLE 3: OBSERVATION vs EVAL comparison ═══
    print(f"\n{sep}")
    print(f"TABLE 3: OBS_SR vs EVAL_SR (does teaching create learning gain?)")
    print(sep)
    print(f"{'n_sup':>5s} {'nt':>3s} {'cond':>12s} {'OBS_SR':>8s} {'TEACH_SR':>9s} {'EVAL_SR':>8s} {'dEval':>7s} {'#TeachEx':>9s}")
    print("-" * 120)
    for ns in N_SUP_VALUES:
        for nt in N_TEACH_VALUES:
            for c in ["baseline", "full_tutor"]:
                cr = [r for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                obs = np.mean([r['obs_sr'] for r in cr])
                teach = np.mean([r['teach_sr'] for r in cr])
                ev = np.mean([r['eval_sr'] for r in cr])
                te = np.mean([r['n_teach_ex'] for r in cr])
                d = ev - obs  # learning gain
                print(f"{ns:5d} {nt:3d} {c:>12s} {obs:8.3f} {teach:9.3f} {ev:8.3f} {d:+7.3f} {te:9.1f}")
        print()

    # ═══ TABLE 4: TUTOR ACTIONS in teaching phase ═══
    print(f"\n{sep}")
    print(f"TABLE 4: TUTOR ACTIONS in teaching phase (full_tutor only)")
    print(sep)
    print(f"{'n_sup':>5s} {'nt':>3s} {'BAN':>6s} {'HL':>6s} {'HINT':>6s} {'SKIP':>6s} {'WAIT':>6s}")
    for ns in N_SUP_VALUES:
        for nt in N_TEACH_VALUES:
            cr = [r for r in results
                  if r['cond'] == 'full_tutor' and r['n_sup'] == ns and r['n_teach'] == nt]
            acts = defaultdict(float)
            for r in cr:
                for k, v in r['teach_actions'].items():
                    acts[k] += v
            n = max(len(cr), 1)
            print(f"{ns:5d} {nt:3d} {acts.get('BAN',0)/n:6.2f} {acts.get('HIGHLIGHT',0)/n:6.2f} "
                  f"{acts.get('RISK_HINT',0)/n:6.2f} {acts.get('SKIP',0)/n:6.2f} "
                  f"{acts.get('WAIT',0)/n:6.2f}")

    # ═══ TABLE 5: EVAL DMG comparison ═══
    print(f"\n{sep}")
    print(f"TABLE 5: EVAL-PHASE DAMAGE (lower = better)")
    print(sep)
    for nt in N_TEACH_VALUES:
        print(f"\n--- N_teach={nt} ---")
        header = f"{'n_sup':>5s}"
        for c in CONDITIONS:
            header += f"  {c:>12s}"
        print(header)
        for ns in N_SUP_VALUES:
            row = f"{ns:5d}"
            for c in CONDITIONS:
                cr = [r['eval_dmg'] for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                row += f"  {np.mean(cr):12.1f}" if cr else f"  {'N/A':>12s}"
            print(row)


if __name__ == "__main__":
    main()
