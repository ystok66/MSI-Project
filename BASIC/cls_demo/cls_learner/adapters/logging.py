"""
logging.py — Ablation and diagnostic logging for CLS system.

Captures delta/mode/gate decisions for analysis.
"""
from __future__ import annotations
import json
from collections import defaultdict
from typing import Dict, List, Optional
from cls_learner.interfaces import MemBias


class CLSLogger:
    """
    Diagnostic logger for CLS system decisions.

    Captures per-query HPC gating decisions, CA1 delta values,
    and mode distributions for ablation analysis.
    """

    def __init__(self):
        self.entries: List[Dict] = []
        self.mode_counts = defaultdict(int)
        self.delta_values: List[float] = []

    def log_query(self, words: List[str], mem_bias: Optional[MemBias],
                  predicted: List[str], expected: Optional[List[str]] = None):
        """Log one prediction event."""
        entry = {
            'words': words,
            'predicted': predicted,
            'expected': expected,
            'correct': (predicted == expected) if expected else None,
        }

        if mem_bias is not None:
            entry['delta'] = mem_bias.delta
            entry['mode'] = mem_bias.mode
            entry['lam_mem'] = mem_bias.lam_mem
            entry['n_role_boosts'] = len(mem_bias.role_boost)

            self.mode_counts[mem_bias.mode] += 1
            if mem_bias.delta != float('inf'):
                self.delta_values.append(mem_bias.delta)

        self.entries.append(entry)

    def summary(self) -> Dict:
        """Return summary statistics."""
        import numpy as np
        total = len(self.entries)
        correct = sum(1 for e in self.entries if e.get('correct'))

        result = {
            'total_queries': total,
            'correct': correct,
            'accuracy': correct / max(1, total),
            'mode_distribution': dict(self.mode_counts),
        }

        if self.delta_values:
            arr = np.array(self.delta_values)
            result['delta_stats'] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'median': float(np.median(arr)),
            }

        return result

    def to_json(self, filepath: str):
        """Save all entries to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'entries': self.entries,
                'summary': self.summary(),
            }, f, indent=2, default=str)

    def reset(self):
        """Clear all logged data."""
        self.entries.clear()
        self.mode_counts.clear()
        self.delta_values.clear()
