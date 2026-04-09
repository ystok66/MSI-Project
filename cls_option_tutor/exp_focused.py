"""
Focused 4-phase experiment: n_sup = {0, 2, 4, 6}.

Reports:
  - Per-phase SR, damage, refresh counts
  - Tutor action breakdown
  - Per-query trace summary
  - n_teach_examples accumulated

N=10 seeds x 3 grammars per cell.
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

N_SEEDS = 10
GRAMMARS = ["000001", "000002", "000003"]
N_SUP_VALUES = [0, 2, 4, 6]
MAX_WORKERS = 12

CONDITIONS = {
    "baseline": {},
    "ban_only": {"c_hint": 100.0, "c_skip": 100.0, "c_hl": 100.0},
    "hl_only":  {"c_hint": 100.0, "c_skip": 100.0, "c_ban": 100.0},
    "full_tutor": {},
    "oracle": {"use_oracle": True},
}


def run_one(condition, grammar_id, seed, n_sup):
    cond_cfg = CONDITIONS[condition]
    cfg = FullConfig()
    cfg.learner.n_sup = n_sup

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

    def phase_metrics(start, end):
        qs_list = block.queries[start:end]
        if not qs_list:
            return {'sr': 0.0, 'dmg': 0, 'n': 0, 'refreshes': 0, 'rounds': 0}
        correct = sum(1 for q in qs_list if q.success)
        dmg = sum(max(0, cfg.env.H_0 - q.hp) for q in qs_list)
        refreshes = sum(q.refreshes_used for q in qs_list)
        rounds = sum(q.rounds_used for q in qs_list)
        return {'sr': correct / len(qs_list), 'dmg': dmg, 'n': len(qs_list),
                'refreshes': refreshes, 'rounds': rounds}

    obs_m = phase_metrics(0, obs_end)
    teach_m = phase_metrics(obs_end, teach_end)
    eval_m = phase_metrics(teach_end, eval_end)

    action_counts = defaultdict(int)
    for s in block.tutor_trace:
        action_counts[s.action] += 1

    n_teach_ex = len(getattr(learner, '_teaching_examples', []))
    metrics = OptionEnv.get_block_metrics(block)

    return {
        'cond': condition, 'grammar': grammar_id, 'seed': seed, 'n_sup': n_sup,
        'obs_sr': obs_m['sr'], 'obs_dmg': obs_m['dmg'],
        'obs_refresh': obs_m['refreshes'], 'obs_rounds': obs_m['rounds'],
        'teach_sr': teach_m['sr'], 'teach_dmg': teach_m['dmg'],
        'teach_refresh': teach_m['refreshes'], 'teach_rounds': teach_m['rounds'],
        'eval_sr': eval_m['sr'], 'eval_dmg': eval_m['dmg'],
        'eval_refresh': eval_m['refreshes'], 'eval_rounds': eval_m['rounds'],
        'overall_sr': metrics['solve_rate'], 'total_dmg': metrics['total_damage'],
        'total_refreshes': block.total_refreshes,
        'n_teach_ex': n_teach_ex,
        'actions': dict(action_counts),
    }


def fmt(val, se=None):
    if se is not None:
        return f"{val:.3f}+/-{se:.3f}"
    return f"{val:.3f}"


def main():
    jobs = []
    for gid in GRAMMARS:
        for seed in range(N_SEEDS):
            for cond in CONDITIONS:
                for ns in N_SUP_VALUES:
                    jobs.append((cond, gid, seed, ns))

    total = len(jobs)
    print(f"Running {total} jobs...")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one, *j): j for j in jobs}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                results.append(f.result())
                if done % 50 == 0:
                    print(f"  [{done}/{total}]")
            except Exception as e:
                print(f"  FAILED {futures[f]}: {e}")

    N = N_SEEDS * len(GRAMMARS)
    sep = "=" * 110

    # ═══ TABLE 1: EVAL-PHASE SR ═══
    print(f"\n{sep}")
    print(f"TABLE 1: EVAL-PHASE SR (Phase 4, frozen learner, no tutor)")
    print(sep)
    header = f"{'n_sup':>5s}"
    for c in CONDITIONS:
        header += f"  {c:>18s}"
    print(header)
    print("-" * 110)
    for ns in N_SUP_VALUES:
        row = f"{ns:5d}"
        for c in CONDITIONS:
            cr = [r['eval_sr'] for r in results if r['cond'] == c and r['n_sup'] == ns]
            row += f"  {np.mean(cr):9.3f}+/-{np.std(cr)/np.sqrt(len(cr)):.3f}"
        print(row)

    # ═══ TABLE 2: ALL PHASES ═══
    print(f"\n{sep}")
    print(f"TABLE 2: PHASE BREAKDOWN (baseline vs full_tutor)")
    print(sep)
    for ns in N_SUP_VALUES:
        print(f"\n--- n_sup={ns} ---")
        print(f"{'Cond':>12s} {'Phase':>6s} {'SR':>8s} {'DMG':>6s} {'Refr':>6s} {'Rnds':>6s}")
        for c in ["baseline", "full_tutor"]:
            for phase, key_prefix in [("OBS", "obs"), ("TEACH", "teach"), ("EVAL", "eval")]:
                cr = [r for r in results if r['cond'] == c and r['n_sup'] == ns]
                sr = np.mean([r[f'{key_prefix}_sr'] for r in cr])
                dmg = np.mean([r[f'{key_prefix}_dmg'] for r in cr])
                ref = np.mean([r[f'{key_prefix}_refresh'] for r in cr])
                rnd = np.mean([r[f'{key_prefix}_rounds'] for r in cr])
                print(f"{c:>12s} {phase:>6s} {sr:8.3f} {dmg:6.1f} {ref:6.1f} {rnd:6.1f}")

    # ═══ TABLE 3: DELTA ═══
    print(f"\n{sep}")
    print(f"TABLE 3: dEVAL_SR vs baseline")
    print(sep)
    header3 = f"{'n_sup':>5s}"
    for c in ["ban_only", "hl_only", "full_tutor", "oracle"]:
        header3 += f"  {c:>12s}"
    print(header3)
    print("-" * 110)
    for ns in N_SUP_VALUES:
        bl = np.mean([r['eval_sr'] for r in results if r['cond'] == 'baseline' and r['n_sup'] == ns])
        row = f"{ns:5d}"
        for c in ["ban_only", "hl_only", "full_tutor", "oracle"]:
            cr = np.mean([r['eval_sr'] for r in results if r['cond'] == c and r['n_sup'] == ns])
            row += f"  {cr - bl:+12.3f}"
        print(row)

    # ═══ TABLE 4: TUTOR ACTIONS ═══
    print(f"\n{sep}")
    print(f"TABLE 4: TUTOR ACTIONS (full_tutor, mean per block)")
    print(sep)
    print(f"{'n_sup':>5s} {'BAN':>6s} {'HL':>6s} {'HINT':>6s} {'SKIP':>6s} {'WAIT':>6s} {'#TeachEx':>9s}")
    for ns in N_SUP_VALUES:
        cr = [r for r in results if r['cond'] == 'full_tutor' and r['n_sup'] == ns]
        acts = defaultdict(float)
        for r in cr:
            for k, v in r['actions'].items():
                acts[k] += v
        n = max(len(cr), 1)
        te = np.mean([r['n_teach_ex'] for r in cr])
        print(f"{ns:5d} {acts.get('BAN',0)/n:6.1f} {acts.get('HIGHLIGHT',0)/n:6.1f} "
              f"{acts.get('RISK_HINT',0)/n:6.1f} {acts.get('SKIP',0)/n:6.1f} "
              f"{acts.get('WAIT',0)/n:6.1f} {te:9.1f}")

    # ═══ TABLE 5: REFRESH BEHAVIOR ═══
    print(f"\n{sep}")
    print(f"TABLE 5: REFRESH BEHAVIOR (mean refreshes per block)")
    print(sep)
    header5 = f"{'n_sup':>5s}"
    for c in CONDITIONS:
        header5 += f"  {c:>12s}"
    print(header5)
    for ns in N_SUP_VALUES:
        row = f"{ns:5d}"
        for c in CONDITIONS:
            cr = [r['total_refreshes'] for r in results if r['cond'] == c and r['n_sup'] == ns]
            row += f"  {np.mean(cr):12.1f}"
        print(row)


if __name__ == "__main__":
    main()
