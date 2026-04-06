"""Compact Phase 2A analysis."""
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

# Focus on medium difficulty
diff = 'medium'
print(f"Phase 2A Results ({diff})")
print()

for fam in fams:
    print(f"[{fam}]")
    for v in variants:
        k = (fam, diff, v)
        if k not in groups: continue
        rs = groups[k]
        n = len(rs)
        sbcr = sum(1 for r in rs if r['safe_branch_chosen'] == 'True') / n
        tbsr = sum(1 for r in rs if r['survived'] == 'True' and r['reached_goal'] == 'True') / n
        surv = sum(1 for r in rs if r['survived'] == 'True') / n
        dh = np.mean([float(r['delta_H']) for r in rs])
        m_a = np.mean([float(r['M_true_after']) for r in rs])
        dnll = np.mean([float(r['delta_nll_local']) for r in rs])
        tag = v.replace('rsa_obs_', 'r_').replace('rsa_plus_', 'r+')
        print(f"  {tag:<16} SBCR={sbcr:.2f} TBSR={tbsr:.2f} dH={dh:.3f} Ma={m_a:.3f} dNLL={dnll:.3f}")
    print()

# Cross-difficulty for fork_trap
print("--- fork_trap across difficulties ---")
for diff2 in ['easy', 'medium', 'hard']:
    for v in ['legacy_bias', 'rsa_obs_s1']:
        k = (fams[0], diff2, v)
        if k not in groups: continue
        rs = groups[k]
        n = len(rs)
        sbcr = sum(1 for r in rs if r['safe_branch_chosen'] == 'True') / n
        tbsr = sum(1 for r in rs if r['survived'] == 'True' and r['reached_goal'] == 'True') / n
        dnll = np.mean([float(r['delta_nll_local']) for r in rs])
        print(f"  {diff2:<8} {v:<16} SBCR={sbcr:.2f} TBSR={tbsr:.2f} dNLL={dnll:.3f}")
    print()
