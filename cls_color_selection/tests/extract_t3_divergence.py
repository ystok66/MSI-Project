"""Extract T3 per-query divergence logs from raw results."""
import json, os, sys
import numpy as np

results_dir = os.path.join(
    os.path.dirname(__file__), '..', 'cls_color_selection', 'results', 'phase3_t3_hint')
raw_path = os.path.join(results_dir, 'raw_results.jsonl')

lines = []
lines.append("# T3 Per-Query Divergence Log (T3's grammar vs Real learner's grammar)")
lines.append("")

for cond_name in ['T3_hint_on', 'T3_hint_off']:
    data = [json.loads(l) for l in open(raw_path, encoding='utf-8')
            if json.loads(l).get('condition') == cond_name and json.loads(l).get('status') != 'error']

    # Re-read properly
    all_runs = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d.get('condition') == cond_name and d.get('status') != 'error':
                all_runs.append(d)

    lines.append(f"## {cond_name} ({len(all_runs)} runs)")
    lines.append("")

    # Collect divergence logs
    div_by_phase_qi = {}
    for run in all_runs:
        bstats = run.get('behavioral_stats', {})
        divlog = bstats.get('divergence_log', [])
        if not divlog:
            continue
        for rec in divlog:
            if rec is None:
                continue
            key = (rec.get('phase', '?'), rec.get('query_idx', -1))
            if key not in div_by_phase_qi:
                div_by_phase_qi[key] = []
            div_by_phase_qi[key].append(rec)

    lines.append("| Phase | Query | top1_agree | JS_div | T3_acc | Real_acc | Acc_gap | gram_comp | risk_comp | N |")
    lines.append("|-------|:-----:|:----------:|:------:|:------:|:--------:|:-------:|:---------:|:---------:|:--:|")

    for key in sorted(div_by_phase_qi.keys(), key=lambda x: (x[0], x[1])):
        phase, qi = key
        recs = div_by_phase_qi[key]
        n = len(recs)

        def safe_mean(field):
            vals = [r[field] for r in recs if r.get(field) is not None]
            return f"{np.mean(vals):.4f}" if vals else "—"

        lines.append(
            f"| {phase} | {qi} | {safe_mean('top1_agreement')} | "
            f"{safe_mean('js_divergence')} | {safe_mean('my_accuracy')} | "
            f"{safe_mean('real_accuracy')} | {safe_mean('accuracy_gap')} | "
            f"{safe_mean('behavioral_gram_competence')} | "
            f"{safe_mean('behavioral_risk_competence')} | {n} |"
        )
    lines.append("")

    # Summary stats
    if all_runs:
        bstats_all = [r.get('behavioral_stats', {}) for r in all_runs]
        lines.append(f"### Behavioral Stats Summary")
        for key in ['gram_competence', 'risk_competence', 'stuck_tendency',
                     'n_confirm_success', 'n_confirm_fail', 'n_danger_selected',
                     'n_safe_selected', 'n_retries']:
            vals = [b.get(key, 0) for b in bstats_all if key in b]
            if vals:
                lines.append(f"- {key}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
        lines.append("")

out_path = os.path.join(results_dir, 'divergence_analysis.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Written to {out_path}")
