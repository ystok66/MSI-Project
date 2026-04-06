"""Final Phase 2A analysis - per-line output for readability."""
import csv, numpy as np
from collections import defaultdict

rows = list(csv.DictReader(open('results/step2_phase2a/phase2a_episodes.csv')))
groups = defaultdict(list)
for r in rows:
    key = (r['family'], r['difficulty'], r['variant'])
    groups[key].append(r)

fams = ['fork_trap', 'elcb_po', 'baseline_v2']
vs = ['legacy_bias', 'rsa_obs_l0', 'rsa_obs_s1', 'rsa_obs_s1_trust', 'rsa_plus_phase10']

lines = []
lines.append("Phase 2A Final Results (50 seeds)")
lines.append("")

for diff in ['medium']:
    for fam in fams:
        lines.append(f"[{fam} / {diff}]")
        for v in vs:
            k = (fam, diff, v)
            if k not in groups: continue
            rs = groups[k]
            n = len(rs)
            sbcr = sum(1 for r in rs if r['safe_branch_chosen']=='True') / n
            tbsr = sum(1 for r in rs if r['survived']=='True' and r['reached_goal']=='True') / n
            dh = np.mean([float(r['delta_H']) for r in rs])
            m_b = np.mean([float(r['M_true_before']) for r in rs])
            m_a = np.mean([float(r['M_true_after']) for r in rs])
            dnll = np.mean([float(r['delta_nll_local']) for r in rs])
            dm = m_a - m_b
            tag = v[:16]
            lines.append(f"  {tag:<18} SBCR={sbcr:.2f}  TBSR={tbsr:.2f}  dH={dh:+.3f}  dM={dm:+.4f}  dNLL={dnll:+.4f}")
        lines.append("")

# Cross-difficulty for S1 vs legacy on fork_trap
lines.append("fork_trap: S1 vs legacy across difficulties")
for d in ['easy', 'medium', 'hard']:
    for v in ['legacy_bias', 'rsa_obs_s1']:
        k = ('fork_trap', d, v)
        if k not in groups: continue
        rs = groups[k]
        n = len(rs)
        sbcr = sum(1 for r in rs if r['safe_branch_chosen']=='True') / n
        tbsr = sum(1 for r in rs if r['survived']=='True' and r['reached_goal']=='True') / n
        dnll = np.mean([float(r['delta_nll_local']) for r in rs])
        m_a = np.mean([float(r['M_true_after']) for r in rs])
        lines.append(f"  {d:<8} {v:<18} SBCR={sbcr:.2f}  TBSR={tbsr:.2f}  Ma={m_a:.3f}  dNLL={dnll:+.4f}")
    lines.append("")

with open('results/step2_phase2a/final_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
