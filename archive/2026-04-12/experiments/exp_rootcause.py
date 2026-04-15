"""
Root-cause disentangling experiment.

8 conditions:
  C0: baseline (no tutor)
  C1: current_tutor (full_tutor with cortex_em)
  C2: no_incr_study (full_tutor, reveal_mode=off)
  C3: hl_no_incr (highlight-only, reveal_mode=off)
  C4: neg_memory (full_tutor, reveal_mode=negative_memory)
  C5: persistent_attn (full_tutor, attention_init=persistent_prior) [η sweep]
  D1: cheat_sem (tutor reads learner semantic state)
  D2: cheat_full (tutor reads learner full utility state)

Sweep: n_sup={2,4,6,8} × N_teach={1,2} × 20 seeds × 3 grammars
Smoke: 3 seeds × 2 grammars × n_sup={2,6} × N_teach={1}
"""
from __future__ import annotations
import os, sys, json, argparse
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

# ── Condition definitions ──
# Each maps to config overrides
CONDITIONS = {
    "C0_baseline": {
        "tutor": False,
        "reveal_learning_mode": "cortex_em",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "proxy_oracle",
    },
    "C1_current_tutor": {
        "tutor": True,
        "reveal_learning_mode": "cortex_em",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "proxy_oracle",
    },
    "C2_no_incr_study": {
        "tutor": True,
        "reveal_learning_mode": "off",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "proxy_oracle",
    },
    "C3_hl_no_incr": {
        "tutor": True,
        "reveal_learning_mode": "off",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "proxy_oracle",
        "hl_only": True,  # disable BAN/HINT/SKIP via high cost
    },
    "C4_neg_memory": {
        "tutor": True,
        "reveal_learning_mode": "negative_memory",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "proxy_oracle",
    },
    "C5_persistent_attn": {
        "tutor": True,
        "reveal_learning_mode": "cortex_em",
        "attention_init_mode": "persistent_prior",
        "tutor_access_mode": "proxy_oracle",
        # η_attn will be swept externally
    },
    "D1_cheat_sem": {
        "tutor": True,
        "reveal_learning_mode": "cortex_em",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "cheat_sem",
    },
    "D2_cheat_full": {
        "tutor": True,
        "reveal_learning_mode": "cortex_em",
        "attention_init_mode": "uniform",
        "tutor_access_mode": "cheat_full",
    },
}

ETA_ATTN_VALUES = [0.1, 0.3, 0.5, 1.0]


def run_one(condition, grammar_id, seed, n_sup, n_teach,
            n_obs=10, n_eval=3, eta_attn=0.3):
    """Run one block and return result dict."""
    cond_cfg = CONDITIONS[condition]
    cfg = FullConfig()
    cfg.learner.n_sup = n_sup
    cfg.learner.use_cls = True

    # Phase sizes
    cfg.env.N_obs = n_obs
    cfg.env.N_teach = n_teach
    cfg.env.N_eval = n_eval
    cfg.env.M_queries = n_obs + n_teach + n_eval

    # Root-cause modes
    cfg.learner.reveal_learning_mode = cond_cfg["reveal_learning_mode"]
    cfg.learner.attention_init_mode = cond_cfg["attention_init_mode"]
    cfg.learner.eta_attn = eta_attn
    cfg.tutor.tutor_access_mode = cond_cfg["tutor_access_mode"]

    # HL-only: disable other actions via high cost
    if cond_cfg.get("hl_only"):
        cfg.tutor.c_ban = 100.0
        cfg.tutor.c_hint = 100.0
        cfg.tutor.c_skip = 100.0

    env = OptionEnv(cfg=cfg, data_dir=DATA_DIR)
    tutor_enabled = cond_cfg["tutor"]

    if tutor_enabled:
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
        tutor = TutorAgent(cfg=cfg)
        block = tutor.run_block(env, learner, grammar_id, seed=seed)
    else:
        learner = LearnerAgent(cfg=cfg, seed=seed, use_cls=True)
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

    # Tutor actions
    teach_actions = defaultdict(int)
    for s in block.tutor_trace:
        qi = s.query_id
        if obs_end <= qi < teach_end:
            teach_actions[s.action] += 1

    n_teach_ex = len(getattr(learner, '_teaching_examples', []))
    neg_mem_size = (learner._negative_memory.size
                    if hasattr(learner, '_negative_memory')
                    and learner._negative_memory is not None
                    else 0)

    return {
        'cond': condition, 'grammar': grammar_id, 'seed': seed,
        'n_sup': n_sup, 'n_teach': n_teach, 'eta_attn': eta_attn,
        'obs_sr': phase_sr(0, obs_end),
        'teach_sr': phase_sr(obs_end, teach_end),
        'eval_sr': phase_sr(teach_end, eval_end),
        'eval_dmg': phase_dmg(teach_end, eval_end),
        'n_teach_ex': n_teach_ex,
        'neg_mem_size': neg_mem_size,
        'teach_actions': dict(teach_actions),
    }


def print_tables(results, conditions, n_sup_values, n_teach_values):
    """Print all result tables."""
    sep = "=" * 130
    N = len(set((r['grammar'], r['seed']) for r in results
                if r['cond'] == list(conditions)[0]))

    # TABLE 1: EVAL SR
    for nt in n_teach_values:
        print(f"\n{sep}")
        print(f"TABLE 1 (N_teach={nt}): EVAL_SR by condition x n_sup")
        print(sep)
        header = f"{'n_sup':>5}"
        for c in conditions:
            header += f"  {c:>20}"
        print(header)
        print("-" * 130)
        for ns in n_sup_values:
            row = f"{ns:5d}"
            for c in conditions:
                cr = [r['eval_sr'] for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                if cr:
                    m, se = np.mean(cr), np.std(cr) / np.sqrt(len(cr))
                    row += f"  {m:11.3f}+/-{se:.3f}"
                else:
                    row += f"  {'N/A':>20}"
            print(row)

    # TABLE 2: DELTA vs C0_baseline
    for nt in n_teach_values:
        print(f"\n{sep}")
        print(f"TABLE 2 (N_teach={nt}): dEVAL_SR vs C0_baseline")
        print(sep)
        others = [c for c in conditions if c != "C0_baseline"]
        header = f"{'n_sup':>5}"
        for c in others:
            header += f"  {c:>20}"
        print(header)
        print("-" * 130)
        for ns in n_sup_values:
            bl = [r['eval_sr'] for r in results
                  if r['cond'] == 'C0_baseline' and r['n_sup'] == ns and r['n_teach'] == nt]
            bl_m = np.mean(bl) if bl else 0
            row = f"{ns:5d}"
            for c in others:
                cr = [r['eval_sr'] for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                d = np.mean(cr) - bl_m if cr else 0
                row += f"  {d:+20.3f}"
            print(row)

    # TABLE 3: TransferGap
    print(f"\n{sep}")
    print(f"TABLE 3: TransferGap = EVAL_SR - OBS_SR")
    print(sep)
    print(f"{'n_sup':>5} {'nt':>3} {'cond':>22} {'OBS_SR':>8} {'EVAL_SR':>8} "
          f"{'TransGap':>9} {'NegMem':>7}")
    print("-" * 130)
    for ns in n_sup_values:
        for nt in n_teach_values:
            for c in conditions:
                cr = [r for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                if not cr:
                    continue
                obs = np.mean([r['obs_sr'] for r in cr])
                ev = np.mean([r['eval_sr'] for r in cr])
                nm = np.mean([r['neg_mem_size'] for r in cr])
                print(f"{ns:5d} {nt:3d} {c:>22} {obs:8.3f} {ev:8.3f} "
                      f"{ev - obs:+9.3f} {nm:7.1f}")
            print()

    # TABLE 4: Tutor Actions
    print(f"\n{sep}")
    print(f"TABLE 4: TUTOR ACTIONS in teaching phase")
    print(sep)
    print(f"{'cond':>22} {'n_sup':>5} {'nt':>3} {'BAN':>6} {'HL':>6} "
          f"{'HINT':>6} {'SKIP':>6} {'WAIT':>6}")
    tutor_conds = [c for c in conditions if CONDITIONS[c]["tutor"]]
    for c in tutor_conds:
        for ns in n_sup_values:
            for nt in n_teach_values:
                cr = [r for r in results
                      if r['cond'] == c and r['n_sup'] == ns and r['n_teach'] == nt]
                if not cr:
                    continue
                acts = defaultdict(float)
                for r in cr:
                    for k, v in r['teach_actions'].items():
                        acts[k] += v
                n = max(len(cr), 1)
                print(f"{c:>22} {ns:5d} {nt:3d} "
                      f"{acts.get('BAN', 0) / n:6.2f} "
                      f"{acts.get('HIGHLIGHT', 0) / n:6.2f} "
                      f"{acts.get('RISK_HINT', 0) / n:6.2f} "
                      f"{acts.get('SKIP', 0) / n:6.2f} "
                      f"{acts.get('WAIT', 0) / n:6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 3 seeds x 2 grammars x n_sup={2,6} x nt={1}")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=str, default="results/rootcause_results.txt")
    parser.add_argument("--eta-sweep", action="store_true",
                        help="Also sweep eta_attn for C5 condition")
    args = parser.parse_args()

    if args.smoke:
        n_seeds = 3
        grammars = ["000001", "000002"]
        n_sup_values = [2, 6]
        n_teach_values = [1]
        conditions = CONDITIONS
    else:
        n_seeds = 20
        grammars = ["000001", "000002", "000003"]
        n_sup_values = [2, 4, 6, 8]
        n_teach_values = [1, 2]
        conditions = CONDITIONS

    # Build job list
    jobs = []
    for gid in grammars:
        for seed in range(n_seeds):
            for cond in conditions:
                for ns in n_sup_values:
                    for nt in n_teach_values:
                        if cond == "C5_persistent_attn" and args.eta_sweep:
                            for eta in ETA_ATTN_VALUES:
                                jobs.append((cond, gid, seed, ns, nt, eta))
                        else:
                            jobs.append((cond, gid, seed, ns, nt, 0.3))

    total = len(jobs)
    print(f"Running {total} jobs ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"Conditions: {list(conditions.keys())}")
    print(f"n_sup={n_sup_values}, N_teach={n_teach_values}")
    print(f"Seeds={n_seeds}, Grammars={grammars}")
    if args.eta_sweep:
        print(f"eta_attn sweep: {ETA_ATTN_VALUES}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for j in jobs:
            cond, gid, seed, ns, nt, eta = j
            f = executor.submit(run_one, cond, gid, seed, ns, nt, eta_attn=eta)
            futures[f] = j
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                results.append(f.result())
                if done % 50 == 0:
                    print(f"  [{done}/{total}]")
            except Exception as e:
                print(f"  FAILED {futures[f]}: {e}")

    # Save raw results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Print tables
    import io
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        print_tables(results, conditions, n_sup_values, n_teach_values)

        # eta_attn sweep table (if applicable)
        if args.eta_sweep:
            print("\n" + "=" * 130)
            print("TABLE 5: eta_attn sweep for C5_persistent_attn")
            print("=" * 130)
            print(f"{'eta':>6} {'n_sup':>5} {'nt':>3} {'EVAL_SR':>10} {'TransGap':>10}")
            for eta in ETA_ATTN_VALUES:
                for ns in n_sup_values:
                    for nt in n_teach_values:
                        cr = [r['eval_sr'] for r in results
                              if r['cond'] == 'C5_persistent_attn'
                              and r['n_sup'] == ns and r['n_teach'] == nt
                              and abs(r['eta_attn'] - eta) < 0.01]
                        obs_cr = [r['obs_sr'] for r in results
                                  if r['cond'] == 'C5_persistent_attn'
                                  and r['n_sup'] == ns and r['n_teach'] == nt
                                  and abs(r['eta_attn'] - eta) < 0.01]
                        if cr:
                            m = np.mean(cr)
                            tg = m - np.mean(obs_cr)
                            print(f"{eta:6.2f} {ns:5d} {nt:3d} {m:10.3f} {tg:+10.3f}")

    output_text = buf.getvalue()
    print(output_text)

    with open(args.output, "w") as f:
        f.write(output_text)

    # Save JSON for downstream analysis
    json_path = args.output.replace('.txt', '.json')
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output} and {json_path}")


if __name__ == "__main__":
    main()
