"""
4-phase experiment: measures eval-phase SR specifically.

4 phases per block:
  Phase 1 (pre-train): CLS studies n_sup support examples
  Phase 2 (observe):   Tutor watches frozen learner on 2 queries
  Phase 3 (teach):     Tutor intervenes, learner CLS learns from reveals (3 queries)
  Phase 4 (evaluate):  Frozen learner, no tutor (3 queries) ← THIS IS THE METRIC

N=10 seeds x 3 grammars × {baseline, highlight, full_tutor} conditions.
"""
from __future__ import annotations
import os, sys
import numpy as np
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
MAX_WORKERS = 12


def run_one(condition, grammar_id, seed, n_sup=5):
    """Run one 4-phase block."""
    cfg = FullConfig()
    cfg.learner.use_cls = (condition != "oracle")
    cfg.learner.n_sup = n_sup

    if condition == "highlight":
        cfg.tutor.c_hint = 100.0
        cfg.tutor.c_skip = 100.0
    elif condition == "oracle":
        cfg.learner.use_cls = False

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)

    if condition == "baseline" or condition == "oracle":
        learner = LearnerAgent(cfg=cfg, seed=seed,
                               use_cls=(condition != "oracle"))
        block = learner.run_block(env, grammar_id, seed=seed)
    else:
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)
        block = tutor.run_block(env, learner, grammar_id, seed=seed)

    # Per-phase metrics
    obs_start = 0
    obs_end = block.obs_phase_queries
    teach_start = obs_end
    teach_end = teach_start + block.teach_phase_queries
    eval_start = teach_end
    eval_end = len(block.queries)

    def phase_metrics(start, end):
        qs_list = block.queries[start:end]
        if not qs_list:
            return {'sr': 0.0, 'dmg': 0, 'n': 0}
        correct = sum(1 for q in qs_list if q.success)
        dmg = sum(max(0, cfg.env.H_0 - q.hp) for q in qs_list)
        return {'sr': correct / len(qs_list), 'dmg': dmg, 'n': len(qs_list)}

    obs_m = phase_metrics(obs_start, obs_end)
    teach_m = phase_metrics(teach_start, teach_end)
    eval_m = phase_metrics(eval_start, eval_end)

    # Teaching examples accumulated
    n_teach_examples = len(getattr(learner, '_teaching_examples', []))

    metrics = OptionEnv.get_block_metrics(block)
    return {
        'condition': condition,
        'grammar': grammar_id,
        'seed': seed,
        'n_sup': n_sup,
        'overall_sr': metrics['solve_rate'],
        'overall_dmg': metrics['total_damage'],
        'obs_sr': obs_m['sr'],
        'teach_sr': teach_m['sr'],
        'eval_sr': eval_m['sr'],
        'eval_dmg': eval_m['dmg'],
        'n_teach_examples': n_teach_examples,
    }


def main():
    conditions = ["baseline", "highlight", "full_tutor", "oracle"]
    n_sups = [5]  # L3 only for now

    jobs = []
    for gid in GRAMMARS:
        for seed in range(N_SEEDS):
            for cond in conditions:
                for ns in n_sups:
                    jobs.append((cond, gid, seed, ns))

    total = len(jobs)
    print(f"Running {total} jobs with {MAX_WORKERS} workers...")
    print(f"4-Phase: Obs(2) -> Teach(3) -> Eval(3)")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one, *j): j for j in jobs}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                r = f.result()
                results.append(r)
                if done % 20 == 0:
                    print(f"  [{done}/{total}]")
            except Exception as e:
                j = futures[f]
                print(f"  FAILED: {j}: {e}")

    # Print results
    print(f"\n{'='*80}")
    print("4-PHASE EXPERIMENT RESULTS (N={} per condition)".format(
        N_SEEDS * len(GRAMMARS)))
    print(f"{'='*80}")
    print(f"{'Condition':20s} {'Obs SR':>8s} {'Teach SR':>9s} {'EVAL SR':>8s} "
          f"{'Eval DMG':>9s} {'Overall':>8s} {'#Teaches':>9s}")
    print("-" * 80)

    for cond in conditions:
        cr = [r for r in results if r['condition'] == cond]
        if not cr:
            continue
        obs = np.mean([r['obs_sr'] for r in cr])
        teach = np.mean([r['teach_sr'] for r in cr])
        eval_sr = np.mean([r['eval_sr'] for r in cr])
        eval_se = np.std([r['eval_sr'] for r in cr]) / np.sqrt(len(cr))
        eval_dmg = np.mean([r['eval_dmg'] for r in cr])
        overall = np.mean([r['overall_sr'] for r in cr])
        n_te = np.mean([r['n_teach_examples'] for r in cr])

        print(f"{cond:20s} {obs:8.3f} {teach:9.3f} "
              f"{eval_sr:8.3f}+/-{eval_se:.3f} {eval_dmg:9.1f} "
              f"{overall:8.3f} {n_te:9.1f}")

    # Compute deltas
    print(f"\n{'='*80}")
    print("EVAL-PHASE DELTAS (vs baseline)")
    print(f"{'='*80}")
    baseline_eval = [r['eval_sr'] for r in results if r['condition'] == 'baseline']
    if baseline_eval:
        bl_mean = np.mean(baseline_eval)
        for cond in ["highlight", "full_tutor", "oracle"]:
            cr = [r['eval_sr'] for r in results if r['condition'] == cond]
            if cr:
                delta = np.mean(cr) - bl_mean
                print(f"  {cond:20s}: dEVAL_SR = {delta:+.3f}")


if __name__ == "__main__":
    main()
