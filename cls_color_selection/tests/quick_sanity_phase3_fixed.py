"""
Phase 3 sanity check after P0/P1/P2 fixes.
Verify: divergence is no longer 0, query counts match config.
"""
import sys, os
import copy
import numpy as np

proj_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, '..', 'BASIC'))

from cls_color_selection.config import FullConfig
from cls_color_selection.experiments.run_phase3 import run_episode_phase3
from cls_color_selection.experiments.registry_phase3 import REGISTRY_P3

data_dir = os.path.normpath(os.path.join(proj_root, '..', 'BASIC', 'cls_learner', 'data'))
task_path = os.path.join(data_dir, '000001.txt')

conditions = ['no_tutor_generated', 'T0_generated', 'T1_generated', 'T2_generated', 'T2_txt_only', 'T2_exact', 'T2_compressed']

out_dir = os.path.join(proj_root, 'results', 'phase3_sanity_fixed')
os.makedirs(out_dir, exist_ok=True)

lines = ["# Phase 3 Sanity Check (FIXED): task 000001, seed 42\n"]

for cond_name in conditions:
    overrides = dict(REGISTRY_P3.get(cond_name, {}))
    print(f"Running {cond_name}...")
    try:
        result = run_episode_phase3(
            task_path=task_path, task_id='000001', seed=42,
            cfg=FullConfig(), condition_overrides=overrides,
        )
        lines.append(f"\n## {cond_name} (status: {result.get('status', 'ok')})")
        lines.append(f"- query_source: {result.get('query_source', '?')}")
        lines.append(f"- n_obs={result.get('n_obs')}, n_teach={result.get('n_teach')}, n_eval={result.get('n_eval')}")
        lines.append(f"- shadow_fidelity: {result.get('shadow_fidelity', 'none')}")

        lines.append("\n### Teach Metrics")
        for k, v in result.get('teach_metrics', {}).items():
            lines.append(f"- {k}: {v}")

        lines.append("\n### Eval Metrics")
        for k, v in result.get('eval_metrics', {}).items():
            lines.append(f"- {k}: {v}")

        if 'joint_debug' in result:
            lines.append("\n### Joint Debug (DIVERGENCE)")
            for k, v in result['joint_debug'].items():
                lines.append(f"- **{k}**: {v}")

    except Exception as e:
        import traceback
        lines.append(f"\n## {cond_name} — ERROR\n")
        lines.append(f"```\n{traceback.format_exc()}\n```\n")

out_path = os.path.join(out_dir, 'sanity_fixed.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nResults: {out_path}")
