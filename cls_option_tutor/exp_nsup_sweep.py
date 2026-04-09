"""
n_sup sweep: pre-training 1-10 examples.

4 phases per block:
  Phase 1: CLS studies n_sup support examples
  Phase 2: Tutor watches frozen learner (2 queries)
  Phase 3: Tutor can {WAIT, BAN, HIGHLIGHT, RISK_HINT, SKIP} (3 queries)
  Phase 4: Frozen learner, no tutor (3 queries) <-- eval metric

Conditions:
  baseline:     no tutor (WAIT only)
  ban_only:     only BAN enabled
  highlight_only: only HIGHLIGHT enabled
  full_tutor:   all interventions enabled
  oracle:       deterministic scorer (upper bound)

N=5 seeds x 3 grammars per condition x n_sup.
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

N_SEEDS = 5
GRAMMARS = ["000001", "000002", "000003"]
N_SUP_VALUES = list(range(1, 11))  # 1-10
MAX_WORKERS = 12

CONDITIONS = {
    "baseline": {},
    "ban_only": {"c_hint": 100.0, "c_skip": 100.0, "c_hl": 100.0},
    "hl_only":  {"c_hint": 100.0, "c_skip": 100.0, "c_ban": 100.0},
    "full_tutor": {},
    "oracle": {"use_oracle": True},
}


def run_one(condition, grammar_id, seed, n_sup):
    """Run one 4-phase block."""
    cond_cfg = CONDITIONS[condition]
    cfg = FullConfig()
    cfg.learner.n_sup = n_sup

    use_oracle = cond_cfg.get("use_oracle", False)
    if use_oracle:
        cfg.learner.use_cls = False
    else:
        cfg.learner.use_cls = True

    # Apply tutor cost overrides
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
        learner = LearnerAgent(cfg=cfg, seed=seed,
                               use_cls=(not use_oracle))
        block = learner.run_block(env, grammar_id, seed=seed)

    # Per-phase metrics
    obs_end = block.obs_phase_queries
    teach_end = obs_end + block.teach_phase_queries
    eval_end = len(block.queries)

    def phase_metrics(start, end):
        qs_list = block.queries[start:end]
        if not qs_list:
            return {'sr': 0.0, 'dmg': 0, 'n': 0}
        correct = sum(1 for q in qs_list if q.success)
        dmg = sum(max(0, cfg.env.H_0 - q.hp) for q in qs_list)
        return {'sr': correct / len(qs_list), 'dmg': dmg, 'n': len(qs_list)}

    eval_m = phase_metrics(teach_end, eval_end)
    teach_m = phase_metrics(obs_end, teach_end)

    # Count tutor actions
    action_counts = defaultdict(int)
    for s in block.tutor_trace:
        action_counts[s.action] += 1

    n_teach_ex = len(getattr(learner, '_teaching_examples', []))

    return {
        'condition': condition,
        'grammar': grammar_id,
        'seed': seed,
        'n_sup': n_sup,
        'eval_sr': eval_m['sr'],
        'eval_dmg': eval_m['dmg'],
        'teach_sr': teach_m['sr'],
        'n_teach_ex': n_teach_ex,
        'actions': dict(action_counts),
    }


def main():
    jobs = []
    for gid in GRAMMARS:
        for seed in range(N_SEEDS):
            for cond in CONDITIONS:
                for ns in N_SUP_VALUES:
                    jobs.append((cond, gid, seed, ns))

    total = len(jobs)
    print(f"Running {total} jobs ({len(CONDITIONS)} conds x {len(N_SUP_VALUES)} n_sup x "
          f"{N_SEEDS} seeds x {len(GRAMMARS)} grammars)")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one, *j): j for j in jobs}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                r = f.result()
                results.append(r)
                if done % 50 == 0:
                    print(f"  [{done}/{total}]")
            except Exception as e:
                j = futures[f]
                print(f"  FAILED: {j}: {e}")

    # Print results table
    print(f"\n{'='*100}")
    print("EVAL-PHASE SR by n_sup and condition")
    print(f"{'='*100}")

    header = f"{'n_sup':>5s}"
    for cond in CONDITIONS:
        header += f"  {cond:>12s}"
    print(header)
    print("-" * 100)

    for ns in N_SUP_VALUES:
        row = f"{ns:5d}"
        for cond in CONDITIONS:
            cr = [r for r in results
                  if r['condition'] == cond and r['n_sup'] == ns]
            if cr:
                mean = np.mean([r['eval_sr'] for r in cr])
                se = np.std([r['eval_sr'] for r in cr]) / np.sqrt(len(cr))
                row += f"  {mean:7.3f}+/-{se:.3f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # Tutor action breakdown
    print(f"\n{'='*100}")
    print("TUTOR ACTION COUNTS (full_tutor condition, summed)")
    print(f"{'='*100}")
    header2 = f"{'n_sup':>5s}  {'BAN':>5s} {'HL':>5s} {'HINT':>5s} {'SKIP':>5s} {'WAIT':>5s}"
    print(header2)
    for ns in N_SUP_VALUES:
        cr = [r for r in results
              if r['condition'] == 'full_tutor' and r['n_sup'] == ns]
        acts = defaultdict(int)
        for r in cr:
            for k, v in r['actions'].items():
                acts[k] += v
        n = max(len(cr), 1)
        row2 = (f"{ns:5d}  {acts.get('BAN',0)/n:5.1f} {acts.get('HIGHLIGHT',0)/n:5.1f} "
                f"{acts.get('RISK_HINT',0)/n:5.1f} {acts.get('SKIP',0)/n:5.1f} "
                f"{acts.get('WAIT',0)/n:5.1f}")
        print(row2)

    # Delta table
    print(f"\n{'='*100}")
    print("dEVAL_SR vs baseline")
    print(f"{'='*100}")
    header3 = f"{'n_sup':>5s}"
    for cond in ["ban_only", "hl_only", "full_tutor", "oracle"]:
        header3 += f"  {cond:>12s}"
    print(header3)
    print("-" * 100)

    for ns in N_SUP_VALUES:
        bl = [r['eval_sr'] for r in results
              if r['condition'] == 'baseline' and r['n_sup'] == ns]
        bl_mean = np.mean(bl) if bl else 0.0
        row3 = f"{ns:5d}"
        for cond in ["ban_only", "hl_only", "full_tutor", "oracle"]:
            cr = [r['eval_sr'] for r in results
                  if r['condition'] == cond and r['n_sup'] == ns]
            if cr:
                d = np.mean(cr) - bl_mean
                row3 += f"  {d:+12.3f}"
            else:
                row3 += f"  {'N/A':>12s}"
        print(row3)


if __name__ == "__main__":
    main()
