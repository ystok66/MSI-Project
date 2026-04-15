"""
Quick Phase 2 end-to-end sanity check.
Runs 3 conditions × 1 task × 1 seed to verify the pipeline works.
"""
import sys
import os

proj_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, '..', 'BASIC'))

from cls_color_selection.config import FullConfig
from cls_color_selection.experiments.run_phase2 import run_episode_phase2

data_dir = os.path.normpath(os.path.join(proj_root, '..', 'BASIC', 'cls_learner', 'data'))
task_path = os.path.join(data_dir, '000001.txt')

conditions = {
    'no_tutor': {
        'tutor.tutor_policy_mode': 'none',
        'tutor.use_observation_phase': False,
    },
    'tutor_rule': {
        'tutor.tutor_policy_mode': 'rule',
        'tutor.use_observation_phase': True,
    },
    'tutor_proxy': {
        'tutor.tutor_policy_mode': 'proxy',
        'tutor.use_observation_phase': True,
    },
}

out_dir = os.path.join(proj_root, 'results', 'phase2_sanity')
os.makedirs(out_dir, exist_ok=True)

lines = ["# Phase 2 Sanity Check: task 000001, seed 42\n"]

for cond_name, overrides in conditions.items():
    print(f"Running {cond_name}...")
    try:
        result = run_episode_phase2(
            task_path=task_path, task_id='000001', seed=42,
            cfg=None, condition_overrides=overrides,
        )
        lines.append(f"\n## {cond_name} (status: {result.get('status', 'ok')})\n")

        if 'obs_summary' in result:
            lines.append("### Observation Summary")
            for k, v in result['obs_summary'].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if 'belief_summary' in result:
            lines.append("### Belief Summary")
            for k, v in result['belief_summary'].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("### Teach Metrics")
        for k, v in result.get('teach_metrics', {}).items():
            lines.append(f"- {k}: {v}")

        lines.append("\n### Eval Metrics")
        for k, v in result.get('eval_metrics', {}).items():
            lines.append(f"- {k}: {v}")

        lines.append("\n### Teach Details")
        for qr in result.get('teach_details', []):
            lines.append(f"  - Q{qr['query_id']}: {qr['outcome']} "
                          f"(confirms={qr['confirm_count']}, retries={qr['retry_count']})")

    except Exception as e:
        import traceback
        lines.append(f"\n## {cond_name} — ERROR\n")
        lines.append(f"```\n{traceback.format_exc()}\n```\n")

out_path = os.path.join(out_dir, 'sanity_result.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nResults written to: {out_path}")
