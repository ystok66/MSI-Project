"""Debug v11e regressions on 000002, 000003."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sync_grammar import *

base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'MLC', 'data_algebraic', 'data_algebraic', 'train')

for task_id in ['000002', '000003']:
    fp = os.path.join(base, f'{task_id}.txt')
    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"{'='*60}")
    r = evaluate_task(fp, meta=MetaGrammarRegistry(), verbose=True)
