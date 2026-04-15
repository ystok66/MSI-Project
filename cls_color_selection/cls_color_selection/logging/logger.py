"""
logger.py — Structured logging for Phase 1 experiments.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, Optional


class ExperimentLogger:
    """JSON-lines logger for experiment events."""

    def __init__(self, output_dir: str, exp_name: str, seed: int):
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(output_dir, f'{ts}_{exp_name}_{seed}.jsonl')
        self._f = open(self.path, 'w', encoding='utf-8')

    def log(self, event_type: str, data: Dict[str, Any]):
        entry = {'ts': time.time(), 'event': event_type, **data}
        self._f.write(json.dumps(entry, default=str, ensure_ascii=False) + '\n')

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
