"""
Quick sanity check: run one episode end-to-end and dump results.
Writes output to a UTF-8 .md file for view_file reading.
"""
import sys
import os
import json

# Paths
proj_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, '..', 'BASIC'))

from cls_color_selection.experiments.run_phase1 import run_episode

data_dir = os.path.normpath(os.path.join(proj_root, '..', 'BASIC', 'cls_learner', 'data'))
task_path = os.path.join(data_dir, '000001.txt')

# Run single episode
result = run_episode(
    task_path=task_path,
    task_id='000001',
    seed=42,
    cfg=None,  # will use defaults
    condition_overrides={},
)

# Write result to file
out_dir = os.path.join(proj_root, 'results', 'sanity')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sanity_result.md')

lines = []
lines.append("# Sanity Check: Single Episode (task 000001, seed 42)\n")
lines.append(f"**Status**: {result.get('status', 'unknown')}\n")

if result.get('status') == 'error':
    lines.append(f"**Error**: {result.get('error', 'unknown')}\n")
else:
    lines.append("## Teach Metrics")
    for k, v in result.get('teach_metrics', {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Eval Metrics")
    for k, v in result.get('eval_metrics', {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Teach Details")
    for qr in result.get('teach_details', []):
        lines.append(f"  - Q{qr['query_id']}: {qr['outcome']} "
                      f"(confirms={qr['confirm_count']}, retries={qr['retry_count']}, "
                      f"danger_sel={qr['danger_select_count']})")

    lines.append("\n## Eval Details")
    for qr in result.get('eval_details', []):
        lines.append(f"  - Q{qr['query_id']}: {qr['outcome']} "
                      f"(confirms={qr['confirm_count']}, retries={qr['retry_count']}, "
                      f"danger_sel={qr['danger_select_count']})")

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Result written to: {out_path}")
