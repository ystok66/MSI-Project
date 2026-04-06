"""
eval.py — Evaluation harness for HBPI on MLC algebraic tasks.

Workflow per task:
  1. Parse the task file → SUPPORT + QUERY
  2. Run EM on SUPPORT → trained model
  3. For each QUERY: enumerate parses (no gold), take MAP parse, execute
  4. Compare predicted output with expected output
"""

from __future__ import annotations
import os, re
from typing import List, Dict, Tuple, Optional

from .model import HBPIModel, HBPIHyperparams
from .em import run_em
from .parser import enumerate_parses
from .executor import execute


def parse_algebraic_file(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse MLC algebraic task file → (support, query) examples."""
    support, query = [], []
    current = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '*SUPPORT*': current = 'support'; continue
            elif line == '*QUERY*': current = 'query'; continue
            elif line == '*GRAMMAR*': current = 'grammar'; continue
            if current in ('support', 'query') and line:
                m = re.match(r'IN:\s+(.*?)\s+OUT:\s+(.*)', line)
                if m:
                    words = m.group(1).strip().split()
                    colors = m.group(2).strip().split()
                    (support if current == 'support' else query).append(
                        {'input': words, 'output': colors})
    return support, query


def evaluate_task(filepath: str,
                  hp: Optional[HBPIHyperparams] = None,
                  verbose: bool = False
                  ) -> Optional[Dict]:
    """
    Evaluate HBPI on one MLC task.

    Returns:
        dict with 'file', 'accuracy', 'correct', 'total'
    """
    support, query = parse_algebraic_file(filepath)
    if not support or not query:
        return None

    fname = os.path.basename(filepath)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[HBPI] Task: {fname}")
        print(f"{'='*60}")

    # ── Train via EM ──
    model = run_em(support, hp=hp, verbose=verbose)

    if verbose:
        snap = model.snapshot()
        print(f"  Learned model:")
        for w, info in sorted(snap.items()):
            print(f"    {w}: {info['map_type']} "
                  f"(color={info['map_color']}) "
                  f"type_p={info['type_probs']}")

    # ── Predict on QUERY ──
    correct, total = 0, len(query)

    if verbose:
        print(f"\n  QUERY ({total}):")

    for qi, q in enumerate(query):
        tokens = q['input']
        gold = q['output']

        # Enumerate parses using learned model (constrained to MAP types, no gold)
        scored = enumerate_parses(tokens, model, gold=None, constrained=True)

        if scored:
            # Take MAP parse (highest prior score)
            best_score, best_ast = scored[0]
            pred, prov = execute(best_ast, model)
        else:
            pred = ['?'] * len(tokens)

        ok = (pred == gold)
        if ok:
            correct += 1

        if verbose:
            status = "OK" if ok else "FAIL"
            extra = f" [len {len(pred)}vs{len(gold)}]" if len(pred) != len(gold) else ""
            print(f"    [{qi}] {status:4s} '{' '.join(tokens)}'{extra}")
            if not ok:
                print(f"         got: {' '.join(str(x) for x in pred)}")
                print(f"         exp: {' '.join(gold)}")

    acc = correct / total if total else 0
    if verbose:
        print(f"  => {correct}/{total} = {acc*100:.1f}%")

    return {'file': fname, 'accuracy': acc, 'correct': correct, 'total': total}


def run_comparison(data_dir: str,
                   n_tasks: int = 10,
                   hp: Optional[HBPIHyperparams] = None,
                   verbose_tasks: Optional[set] = None):
    """
    Run HBPI on multiple tasks and print comparison table.
    """
    files = sorted(f for f in os.listdir(data_dir)
                   if f.endswith('.txt'))[:n_tasks]
    if verbose_tasks is None:
        verbose_tasks = set()

    results = []
    for fn in files:
        r = evaluate_task(os.path.join(data_dir, fn),
                          hp=hp,
                          verbose=(fn in verbose_tasks))
        if r:
            results.append(r)

    # Print comparison
    print(f"\n\n{'='*60}")
    print(f"HBPI Phase-0 Results ({n_tasks} tasks)")
    print(f"{'='*60}")
    print(f"{'File':<14} {'Score':>10}")
    print('-' * 30)
    for r in results:
        print(f"{r['file']:<14} {r['correct']:>3}/{r['total']:<3} "
              f"({r['accuracy']*100:3.0f}%)")
    print('-' * 30)
    tc = sum(r['correct'] for r in results)
    tt = sum(r['total'] for r in results)
    print(f"{'Overall':<14} {tc:>3}/{tt:<3} ({tc/tt*100 if tt else 0:3.0f}%)")

    return results
