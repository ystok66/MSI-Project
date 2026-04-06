"""Analyze Phase 2A results."""
import csv
import numpy as np
from collections import defaultdict

rows = list(csv.DictReader(open('results/step2_phase2a/phase2a_episodes.csv')))
groups = defaultdict(list)
for r in rows:
    key = (r['family'], r['difficulty'], r['variant'])
    groups[key].append(r)

fams = ['fork_trap', 'elcb_po', 'baseline_v2']
variants = ['legacy_bias', 'rsa_obs_l0', 'rsa_obs_s1', 'rsa_obs_s1_trust', 'rsa_plus_phase10']

for diff in ['easy', 'medium', 'hard']:
    print(f"\n=== Difficulty: {diff} ===")
    header = f"{'Family':<14} {'Variant':<22} {'n':>3} {'SBCR':>6} {'TBSR':>6} {'dH':>7} {'M_b':>6} {'M_a':>6} {'dNLL':>7}"
    print(header)
    print("-" * 82)
    for fam in fams:
        for v in variants:
            k = (fam, diff, v)
            if k not in groups:
                continue
            rs = groups[k]
            n = len(rs)
            sbcr = sum(1 for r in rs if r['safe_branch_chosen'] == 'True') / n
            tbsr = sum(1 for r in rs if r['survived'] == 'True' and r['reached_goal'] == 'True') / n
            dh = np.mean([float(r['delta_H']) for r in rs])
            m_b = np.mean([float(r['M_true_before']) for r in rs])
            m_a = np.mean([float(r['M_true_after']) for r in rs])
            dnll = np.mean([float(r['delta_nll_local']) for r in rs])
            print(f"{fam:<14} {v:<22} {n:>3} {sbcr:>6.3f} {tbsr:>6.3f} {dh:>7.4f} {m_b:>6.4f} {m_a:>6.4f} {dnll:>7.4f}")
        print()

# Summary: dM_true per variant (medium difficulty)
print("\n=== Response Selectivity Diagnostic (medium) ===")
print(f"{'Family':<14} {'Variant':<22} {'dM_true':>8}")
print("-" * 46)
for fam in fams:
    for v in variants:
        k = (fam, 'medium', v)
        if k not in groups:
            continue
        rs = groups[k]
        m_b = np.mean([float(r['M_true_before']) for r in rs])
        m_a = np.mean([float(r['M_true_after']) for r in rs])
        dm = m_a - m_b
        print(f"{fam:<14} {v:<22} {dm:>8.4f}")
    print()
